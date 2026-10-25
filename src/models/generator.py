from typing import Any
import torch
import torch.nn as nn
from src.utils import weights_init


class ConditionalGenerator(nn.Module):
    """Conditional DCGAN Generator producing 1x28x28 images from noise and class labels."""

    def __init__(
        self,
        nz: int = 100,
        ngf: int = 64,
        nc: int = 1,
        num_classes: int = 10,
        embed_dim: int = 32,
        extra_layers: int = 0,
        kernel_size: int = 3
    ) -> None:
        super().__init__()
        if extra_layers > 0 and kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd when extra_layers > 0 (got {kernel_size}).")
        self.nz = nz

        self.embed = nn.Embedding(num_classes, embed_dim)

        pad = kernel_size // 2
        in_channels = nz + embed_dim
        layers: list[nn.Module] = [
            # 1x1 -> 7x7
            nn.ConvTranspose2d(in_channels, ngf * 4, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        ]
        for _ in range(extra_layers):
            layers.extend([
                nn.Conv2d(ngf * 4, ngf * 4, kernel_size=kernel_size, stride=1, padding=pad, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
            ])

        # 7x7 -> 14x14
        layers.extend([
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        ])
        for _ in range(extra_layers):
            layers.extend([
                nn.Conv2d(ngf * 2, ngf * 2, kernel_size=kernel_size, stride=1, padding=pad, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True)
            ])

        # 14x14 -> 28x28
        layers.extend([
            nn.ConvTranspose2d(ngf * 2, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        ])

        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # concatenate latent vector and label embedding along channel dimension
        label_embedding = self.embed(labels)
        latent = torch.cat([z, label_embedding], dim=1)
        latent_spatial = latent.unsqueeze(-1).unsqueeze(-1)
        return self.net(latent_spatial)


def build_generator(model_cfg: dict[str, Any] | None = None) -> ConditionalGenerator:
    """Factory creating ConditionalGenerator from configuration dictionary."""
    cfg = model_cfg or {}
    return ConditionalGenerator(
        nz=int(cfg.get("nz", 100)),
        ngf=int(cfg.get("ngf", 64)),
        nc=int(cfg.get("nc", 1)),
        num_classes=int(cfg.get("num_classes", 10)),
        embed_dim=int(cfg.get("embed_dim", 32)),
        extra_layers=int(cfg.get("extra_layers", 0)),
        kernel_size=int(cfg.get("kernel_size", 3))
    )

