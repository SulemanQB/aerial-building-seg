"""A compact U-Net suitable for CPU smoke tests and small GPU runs."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Apply two convolution, normalization, and activation layers."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNet(nn.Module):
    """Compact binary U-Net with transposed-convolution upsampling and skips."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        self.down1 = DoubleConv(in_channels, widths[0])
        self.down2 = DoubleConv(widths[0], widths[1])
        self.down3 = DoubleConv(widths[1], widths[2])
        self.bottleneck = DoubleConv(widths[2], widths[3])
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(widths[3], widths[2], 2, stride=2)
        self.conv3 = DoubleConv(widths[3], widths[2])
        self.up2 = nn.ConvTranspose2d(widths[2], widths[1], 2, stride=2)
        self.conv2 = DoubleConv(widths[2], widths[1])
        self.up1 = nn.ConvTranspose2d(widths[1], widths[0], 2, stride=2)
        self.conv1 = DoubleConv(widths[1], widths[0])
        self.head = nn.Conv2d(widths[0], num_classes, 1)

    @staticmethod
    def _pad_to_match(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        height_delta = target.shape[-2] - source.shape[-2]
        width_delta = target.shape[-1] - source.shape[-1]
        return nn.functional.pad(source, [width_delta // 2, width_delta - width_delta // 2, height_delta // 2, height_delta - height_delta // 2])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(inputs)
        skip2 = self.down2(self.pool(skip1))
        skip3 = self.down3(self.pool(skip2))
        bottleneck = self.bottleneck(self.pool(skip3))
        x = self.up3(bottleneck)
        x = self.conv3(torch.cat([self._pad_to_match(x, skip3), skip3], dim=1))
        x = self.up2(x)
        x = self.conv2(torch.cat([self._pad_to_match(x, skip2), skip2], dim=1))
        x = self.up1(x)
        x = self.conv1(torch.cat([self._pad_to_match(x, skip1), skip1], dim=1))
        return self.head(x)


def build_model(
    base_channels: int = 32,
    pretrained: bool = False,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build the scratch baseline or an ImageNet-pretrained encoder model."""
    if not pretrained:
        return UNet(base_channels=base_channels)
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RuntimeError("Install the transfer extra with `pip install -e .[transfer]` to use --pretrained") from error
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )


def load_checkpoint_model(checkpoint_path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Rebuild a model from a training checkpoint and load its weights.

    Checkpoints are expected to contain the ``model`` state dictionary and the
    training ``config`` written by ``scripts/train.py``. Only load checkpoints
    that come from a trusted source because PyTorch's full checkpoint format
    uses Python deserialization.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = build_model(
        base_channels=config.get("base_channels", 32),
        pretrained=config.get("pretrained", False),
        encoder_weights=None,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model
