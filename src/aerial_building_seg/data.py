"""Dataset and tiling utilities for RGB aerial imagery."""

from __future__ import annotations

import csv
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _find_matching_mask(mask_dir: Path, image_path: Path) -> Path:
    """Return the mask matching an image stem, regardless of extension."""
    for extension in IMAGE_EXTENSIONS:
        candidate = mask_dir / f"{image_path.stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No mask found for {image_path.name} in {mask_dir}")


def _source_from_tile_name(tile_name: str) -> str:
    """Extract the source stem from a generated row/column tile name."""
    source, separator, coordinates = Path(tile_name).stem.rpartition("_r")
    if not separator or not re.fullmatch(r"\d+_c\d+", coordinates):
        raise ValueError(f"Invalid generated tile name: {tile_name}")
    return source


def make_tiles(
    image_dir: str | Path,
    mask_dir: str | Path,
    output_dir: str | Path,
    tile_size: int = 256,
    stride: int = 192,
) -> int:
    """Tile paired source images, retaining edge tiles via zero padding."""
    if tile_size <= 0 or stride <= 0:
        raise ValueError("tile_size and stride must be positive")
    if stride > tile_size:
        raise ValueError("stride must not exceed tile_size")

    image_dir, mask_dir, output_dir = Path(image_dir), Path(mask_dir), Path(output_dir)
    image_out, mask_out = output_dir / "images", output_dir / "masks"
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    count = 0
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
        mask_path = _find_matching_mask(mask_dir, image_path)
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image/mask size mismatch for {image_path.name}")
        height, width = image.shape[:2]
        for top in range(0, height, stride):
            for left in range(0, width, stride):
                image_tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                mask_tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
                bottom, right = min(top + tile_size, height), min(left + tile_size, width)
                image_tile[: bottom - top, : right - left] = image[top:bottom, left:right]
                mask_tile[: bottom - top, : right - left] = mask[top:bottom, left:right]
                tile_id = f"{image_path.stem}_r{top:05d}_c{left:05d}"
                Image.fromarray(image_tile).save(image_out / f"{tile_id}.png")
                Image.fromarray((mask_tile > 0).astype(np.uint8) * 255).save(mask_out / f"{tile_id}.png")
                count += 1
    return count


class BuildingTileDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load paired RGB image and binary mask tiles as PyTorch tensors."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, augment: bool = False) -> None:
        self.image_dir, self.mask_dir, self.augment = Path(image_dir), Path(mask_dir), augment
        self.images = sorted(p for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.images:
            raise FileNotFoundError(f"No image tiles found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.images[index]
        mask_path = _find_matching_mask(self.mask_dir, image_path)
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        if self.augment and random.random() < 0.5:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        if self.augment and random.random() < 0.5:
            image, mask = image[::-1].copy(), mask[::-1].copy()
        return torch.from_numpy(image.transpose(2, 0, 1)), torch.from_numpy(mask[None])


def write_split_manifest(tile_dir: str | Path, output_path: str | Path, validation_fraction: float = 0.2, seed: int = 42) -> None:
    """Split by source image stem, so adjacent tiles cannot cross the split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be in the range (0, 1)")

    image_dir = Path(tile_dir) / "images"
    rows = [(p.name, _source_from_tile_name(p.name)) for p in sorted(image_dir.iterdir())]
    sources = sorted({source for _, source in rows})
    if len(sources) < 2:
        raise ValueError("At least two source images are required for a train/validation split")
    random.Random(seed).shuffle(sources)
    split_at = min(len(sources) - 1, max(1, math.ceil(len(sources) * (1 - validation_fraction))))
    train_sources = set(sources[:split_at])
    with Path(output_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tile", "split", "source"])
        writer.writeheader()
        for tile, source in rows:
            writer.writerow({"tile": tile, "split": "train" if source in train_sources else "val", "source": source})
