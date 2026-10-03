"""Evaluate a checkpoint on held-out tiles and write JSON metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aerial_building_seg.data import BuildingTileDataset
from aerial_building_seg.metrics import dice_iou
from aerial_building_seg.model import load_checkpoint_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tiles/val"))
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model = load_checkpoint_model(args.checkpoint, args.device)
    loader = DataLoader(BuildingTileDataset(args.data / "images", args.data / "masks"), batch_size=4)
    scores = []
    with torch.inference_mode():
        for images, masks in loader:
            scores.append(dice_iou(model(images.to(args.device)), masks.to(args.device)))
    result = {"tiles": len(loader.dataset), "dice": sum(x[0] for x in scores) / len(scores), "iou": sum(x[1] for x in scores) / len(scores)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
