"""Create capped, padded tiles and source-level train/validation splits."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from aerial_building_seg.data import make_tiles, write_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/tiles"))
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=192)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    all_dir = args.output / "all"
    count = make_tiles(args.images, args.masks, all_dir, args.tile_size, args.stride)
    manifest = args.output / "manifest.csv"
    write_split_manifest(all_dir, manifest, args.val_fraction)
    with manifest.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for split in ("train", "val"):
        for kind in ("images", "masks"):
            (args.output / split / kind).mkdir(parents=True, exist_ok=True)
    for row in rows:
        split = row["split"]
        shutil.copy2(all_dir / "images" / row["tile"], args.output / split / "images" / row["tile"])
        shutil.copy2(all_dir / "masks" / row["tile"], args.output / split / "masks" / row["tile"])
    print(f"Wrote {count} tiles to {args.output} ({sum(row['split'] == 'train' for row in rows)} train, {sum(row['split'] == 'val' for row in rows)} val)")


if __name__ == "__main__":
    main()
