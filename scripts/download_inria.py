"""Prepare the INRIA dataset directory after the official download."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from aerial_building_seg.data import IMAGE_EXTENSIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy a capped INRIA subset into the project layout.")
    parser.add_argument("--source", type=Path, required=True, help="Extracted INRIA root containing train/images and train/gt")
    parser.add_argument("--output", type=Path, default=Path("data/raw/inria"))
    parser.add_argument("--max-images", type=int, default=20)
    args = parser.parse_args()
    if args.max_images <= 0:
        raise ValueError("max-images must be positive")
    image_dir = args.source / "train" / "images"
    mask_dir = args.source / "train" / "gt"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("Expected <source>/train/images and <source>/train/gt from the official INRIA archive")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    destination_images = args.output / "images"
    destination_masks = args.output / "masks"
    if destination_images.exists():
        shutil.rmtree(destination_images)
    if destination_masks.exists():
        shutil.rmtree(destination_masks)
    destination_images.mkdir(parents=True, exist_ok=True)
    destination_masks.mkdir(parents=True, exist_ok=True)
    for image in images:
        mask = next((mask_dir / f"{image.stem}{extension}" for extension in sorted(IMAGE_EXTENSIONS)), None)
        if mask is None or not mask.exists():
            raise FileNotFoundError(f"No matching mask for {image.name}")
        shutil.copy2(image, destination_images / image.name)
        shutil.copy2(mask, destination_masks / mask.name)
    print(f"Copied {len(images)} source images to {args.output}")


if __name__ == "__main__":
    main()
