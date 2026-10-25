from typing import Any
import torch
import torch.nn as nn
from src.utils import weights_init


class ConditionalDiscriminator(nn.Module):
    """Conditional DCGAN Discriminator evaluating image-label compatibility."""

    def __init__(
        self,
        nc: int = 1,
        ndf: int = 64,
        num_classes: int = 10,
        leaky_relu_slope: float = 0.2,
        extra_layers: int = 0,
        kernel_size: int = 3
    ) -> None:
        super().__init__()
        if extra_layers > 0 and kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd when extra_layers > 0 (got {kernel_size}).")
        self.embed = nn.Embedding(num_classes, 28 * 28)


        pad = kernel_size // 2
        in_channels = nc + 1
        layers: list[nn.Module] = [
            # 28x28 -> 14x14
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(leaky_relu_slope, inplace=True)
        ]
        for _ in range(extra_layers):
            layers.extend([
                nn.Conv2d(ndf, ndf, kernel_size=kernel_size, stride=1, padding=pad, bias=False),
                nn.BatchNorm2d(ndf),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            ])

        # 14x14 -> 7x7
        layers.extend([
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(leaky_relu_slope, inplace=True)
        ])
        for _ in range(extra_layers):
            layers.extend([
                nn.Conv2d(ndf * 2, ndf * 2, kernel_size=kernel_size, stride=1, padding=pad, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(leaky_relu_slope, inplace=True)
            ])

        # 7x7 -> 1x1
        layers.append(nn.Conv2d(ndf * 2, 1, kernel_size=7, stride=1, padding=0, bias=False))

        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # project label to 28x28 spatial map and concatenate as extra channel
        batch_size = x.size(0)
        label_map = self.embed(labels).view(batch_size, 1, 28, 28)
        x_cat = torch.cat([x, label_map], dim=1)
        out = self.net(x_cat)
        return out.view(batch_size, -1)


def build_discriminator(model_cfg: dict[str, Any] | None = None) -> ConditionalDiscriminator:
    """Factory creating ConditionalDiscriminator from configuration dictionary."""
    cfg = model_cfg or {}
    return ConditionalDiscriminator(
        nc=int(cfg.get("nc", 1)),
        ndf=int(cfg.get("ndf", 64)),
        num_classes=int(cfg.get("num_classes", 10)),
        leaky_relu_slope=float(cfg.get("leaky_relu_slope", 0.2)),
        extra_layers=int(cfg.get("extra_layers", 0)),
        kernel_size=int(cfg.get("kernel_size", 3))
    )

