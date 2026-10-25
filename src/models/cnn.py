from typing import Any
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Convolutional neural network for Fashion-MNIST classification."""

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        conv_channels: list[int] | None = None,
        fc_dim: int = 256,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        channels = conv_channels or [32, 64, 128]

        layers: list[nn.Module] = []
        cur_in = in_channels
        for c in channels:
            layers.extend([
                nn.Conv2d(cur_in, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ])
            cur_in = c

        self.features = nn.Sequential(*layers)

        # 28 -> 14 -> 7 -> 3 for 3 downsampling blocks
        spatial_dim = 28 // (2 ** len(channels))
        flattened_dim = channels[-1] * spatial_dim * spatial_dim

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


def build_cnn(model_cfg: dict[str, Any] | None = None) -> SimpleCNN:
    """Factory creating SimpleCNN from configuration dictionary."""
    cfg = model_cfg or {}
    return SimpleCNN(
        in_channels=int(cfg.get("in_channels", 1)),
        num_classes=int(cfg.get("num_classes", 10)),
        conv_channels=cfg.get("conv_channels", [32, 64, 128]),
        fc_dim=int(cfg.get("fc_dim", 256)),
        dropout=float(cfg.get("dropout", 0.5))
    )

