import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from aerial_building_seg.data import (
    BuildingTileDataset,
    make_tiles,
    write_split_manifest,
)
from aerial_building_seg.inference import save_overlay, sliding_window_predict
from aerial_building_seg.metrics import dice_iou
from aerial_building_seg.model import UNet


def test_unet_preserves_spatial_shape():
    model = UNet(base_channels=8)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 1, 64, 64)


def test_metrics_perfect_prediction():
    logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    dice, iou = dice_iou(logits, targets)
    assert dice == 1.0 and iou == 1.0


def test_metrics_reject_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        dice_iou(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), threshold=1.5)


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        dice_iou(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 1, 2))


def test_tiling_and_dataset(tmp_path: Path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    image[2:6, 3:8] = 255
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:6, 3:8] = 255
    Image.fromarray(image).save(images / "city.png")
    Image.fromarray(mask).save(masks / "city.png")
    tiles = tmp_path / "tiles"
    assert make_tiles(images, masks, tiles, tile_size=8, stride=8) == 4
    dataset = BuildingTileDataset(tiles / "images", tiles / "masks")
    assert dataset[0][0].shape == (3, 8, 8)


def test_manifest_keeps_sources_in_one_split(tmp_path: Path):
    image_dir, mask_dir = tmp_path / "images", tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    for source in ("alpha_ridge", "beta"):
        for column in (0, 8):
            tile_name = f"{source}_r00000_c{column:05d}.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / tile_name)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(mask_dir / tile_name)

    manifest_path = tmp_path / "manifest.csv"
    write_split_manifest(tmp_path, manifest_path, validation_fraction=0.5, seed=42)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assignments = {}
    for row in rows:
        assignments.setdefault(row["source"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in assignments.values())
    assert {row["split"] for row in rows} == {"train", "val"}


def test_manifest_rejects_one_source_split(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(image_dir / "only_r00000_c00000.png")

    with pytest.raises(ValueError, match="At least two source images"):
        write_split_manifest(tmp_path, tmp_path / "manifest.csv")


class _ConstantLogitModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]))


def test_sliding_window_predict_averages_overlaps(tmp_path: Path):
    image = np.full((5, 7, 3), 128, dtype=np.uint8)
    with pytest.raises(ValueError, match="stride"):
        sliding_window_predict(_ConstantLogitModel(), image, tile_size=4, stride=5)
    probability = sliding_window_predict(_ConstantLogitModel(), image, tile_size=4, stride=3)
    normalized_probability = sliding_window_predict(
        _ConstantLogitModel(), image.astype(np.float32) / 255.0, tile_size=4, stride=3
    )
    assert probability.shape == (5, 7)
    assert np.allclose(probability, 0.5)
    assert np.allclose(normalized_probability, probability)

    output_path = tmp_path / "overlay.png"
    save_overlay(image, probability, output_path)
    assert Image.open(output_path).size == (7, 5)
