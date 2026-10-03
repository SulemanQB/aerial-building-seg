"""Sliding-window inference and visual diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _normalise_rgb(image: np.ndarray) -> np.ndarray:
    """Return an RGB array in float32 ``[0, 1]`` form.

    CLI inputs are uint8 images in ``[0, 255]``. Float arrays are also
    accepted when they are already in ``[0, 1]``.
    """
    if not np.issubdtype(image.dtype, np.number):
        raise ValueError("image must contain numeric RGB values")
    image_float = image.astype(np.float32, copy=False)
    minimum, maximum = float(image_float.min()), float(image_float.max())
    if minimum < 0 or maximum > 255:
        raise ValueError("image values must be in the range [0, 255]")
    if np.issubdtype(image.dtype, np.integer) or maximum > 1:
        image_float = image_float / 255.0
    return image_float


def sliding_window_predict(model: torch.nn.Module, image: np.ndarray, tile_size: int = 256, stride: int = 192, device: str = "cpu") -> np.ndarray:
    """Predict an RGB image by averaging probabilities from overlapping tiles."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must have positive height and width")
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if stride > tile_size:
        raise ValueError("stride must not exceed tile_size")
    image_float = _normalise_rgb(image)
    height, width = image.shape[:2]
    probabilities = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for top in range(0, height, stride):
            for left in range(0, width, stride):
                bottom, right = min(top + tile_size, height), min(left + tile_size, width)
                tile = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
                tile[: bottom - top, : right - left] = image_float[top:bottom, left:right]
                tensor = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).to(device)
                prediction = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
                probabilities[top:bottom, left:right] += prediction[: bottom - top, : right - left]
                weights[top:bottom, left:right] += 1.0
    return probabilities / np.maximum(weights, 1e-6)


def save_overlay(image: np.ndarray, probability: np.ndarray, output_path: str | Path, threshold: float = 0.5) -> None:
    """Save an RGB image with thresholded building probabilities highlighted."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an HxWx3 RGB image")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must have positive height and width")
    if probability.shape != image.shape[:2]:
        raise ValueError("probability must have the image height and width")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in the range [0, 1]")

    image = _normalise_rgb(image)
    mask = probability >= threshold
    overlay = image.copy()
    overlay[mask] = overlay[mask] * 0.45 + np.array([1.0, 0.35, 0.05]) * 0.55
    Image.fromarray(np.uint8(np.clip(overlay, 0, 1) * 255)).save(output_path)
