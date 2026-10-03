"""Train the U-Net and save the best validation checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from aerial_building_seg.data import BuildingTileDataset
from aerial_building_seg.metrics import dice_iou
from aerial_building_seg.model import build_model


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/tiles"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--pretrained", action="store_true", help="Use an ImageNet-pretrained ResNet-34 encoder")
    parser.add_argument("--max-train-tiles", type=int, default=None)
    parser.add_argument("--max-val-tiles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_train_tiles is not None and args.max_train_tiles <= 0:
        raise ValueError("max-train-tiles must be positive when provided")
    if args.max_val_tiles is not None and args.max_val_tiles <= 0:
        raise ValueError("max-val-tiles must be positive when provided")
    set_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = BuildingTileDataset(args.data / "train/images", args.data / "train/masks", augment=True)
    val_dataset = BuildingTileDataset(args.data / "val/images", args.data / "val/masks")
    if args.max_train_tiles:
        train_dataset = Subset(train_dataset, range(min(args.max_train_tiles, len(train_dataset))))
    if args.max_val_tiles:
        val_dataset = Subset(val_dataset, range(min(args.max_val_tiles, len(val_dataset))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(base_channels=args.base_channels, pretrained=args.pretrained).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    best_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images, masks = images.to(args.device), masks.to(args.device)
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            dice_loss = 1 - (2 * (probabilities * masks).sum() + 1e-6) / (probabilities.sum() + masks.sum() + 1e-6)
            loss = bce(logits, masks) + dice_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        model.eval()
        with torch.inference_mode():
            scores = [dice_iou(model(images.to(args.device)), masks.to(args.device)) for images, masks in val_loader]
        val_dice = sum(score[0] for score in scores) / len(scores)
        val_iou = sum(score[1] for score in scores) / len(scores)
        row = {"epoch": epoch, "train_loss": running_loss / len(train_loader.dataset), "val_dice": val_dice, "val_iou": val_iou}
        history.append(row)
        print(json.dumps(row))
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(), "config": vars(args), "metrics": row}, args.run_dir / "best.pt")
    (args.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
