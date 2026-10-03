"""Run tiled inference on a large RGB image and save a mask plus overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from aerial_building_seg.inference import save_overlay, sliding_window_predict
from aerial_building_seg.model import load_checkpoint_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/prediction"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    model = load_checkpoint_model(args.checkpoint, args.device)
    image = np.asarray(Image.open(args.image).convert("RGB"))
    probability = sliding_window_predict(model, image, args.tile_size, args.stride, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.uint8(probability >= args.threshold) * 255).save(args.output / "mask.png")
    save_overlay(image, probability, args.output / "overlay.png", args.threshold)
    print(f"Saved {args.output / 'mask.png'} and {args.output / 'overlay.png'}")


if __name__ == "__main__":
    main()
