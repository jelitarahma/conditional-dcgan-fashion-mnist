import argparse
import csv
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

from src.data import get_gan_dataloader
from src.models import build_discriminator, build_generator

from src.utils import (
    get_device,
    load_config,
    plot_loss_curves,
    save_checkpoint,
    save_config,
    set_seed,
    setup_logger
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Conditional DCGAN on Fashion-MNIST")
    parser.add_argument("--config", type=str, default="configs/dcgan.yaml", help="Path to YAML config")
    parser.add_argument("--run-name", type=str, default="dcgan_full_run", help="Experiment run name")
    parser.add_argument("--override", type=str, action="append", default=[], help="Config overrides (key=value)")
    return parser.parse_args()


def train_gan(config_path: str, run_name: str, overrides: list[str] | None = None) -> None:
    cfg = load_config(config_path, overrides)
    out_dir = os.path.join("outputs", run_name)
    samples_dir = os.path.join(out_dir, "samples")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    save_config(cfg, os.path.join(out_dir, "config.yaml"))
    logger = setup_logger("train_gan", os.path.join(out_dir, "train.log"))
    logger.info("Starting Conditional DCGAN run: %s", run_name)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("device"))
    logger.info("Using device: %s", device)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    dataloader = get_gan_dataloader(
        data_dir=data_cfg.get("data_dir", "data"),
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        seed=seed
    )

    net_g = build_generator(model_cfg).to(device)
    net_d = build_discriminator(model_cfg).to(device)
    nz = net_g.nz
    num_classes = net_g.embed.num_embeddings


    lr_g = float(train_cfg.get("lr_g", train_cfg.get("lr", 0.0002)))
    lr_d = float(train_cfg.get("lr_d", train_cfg.get("lr", 0.0002)))
    beta1 = float(train_cfg.get("beta1", 0.5))
    beta2 = float(train_cfg.get("beta2", 0.999))
    epochs = int(train_cfg.get("epochs", 40))
    label_smoothing = bool(train_cfg.get("label_smoothing", True))
    smooth_val = float(train_cfg.get("smooth_val", 0.9))
    sample_interval = int(train_cfg.get("sample_interval", 5))
    ckpt_interval = int(train_cfg.get("checkpoint_interval", 10))

    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    opt_g = optim.Adam(net_g.parameters(), lr=lr_g, betas=(beta1, beta2))
    opt_d = optim.Adam(net_d.parameters(), lr=lr_d, betas=(beta1, beta2))
    criterion = nn.BCEWithLogitsLoss()

    # fixed noise and labels for reproducible grid evaluation across epochs
    fixed_noise = torch.randn(num_classes * 10, nz, device=device)
    fixed_labels = torch.arange(num_classes, device=device).repeat_interleave(10)

    log_csv_path = os.path.join(out_dir, "train_log.csv")
    with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss_g", "loss_d"])

    history: dict[str, list[float]] = {"Generator": [], "Discriminator": []}

    for epoch in range(1, epochs + 1):
        net_g.train()
        net_d.train()
        running_loss_g = 0.0
        running_loss_d = 0.0
        total_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for real_imgs, real_labels in pbar:
            batch_sz = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # update discriminator
            opt_d.zero_grad()

            target_real = torch.full((batch_sz, 1), smooth_val if label_smoothing else 1.0, device=device)
            target_fake = torch.zeros((batch_sz, 1), device=device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out_real = net_d(real_imgs, real_labels)
                loss_d_real = criterion(out_real, target_real)

                noise = torch.randn(batch_sz, nz, device=device)
                fake_labels = torch.randint(0, num_classes, (batch_sz,), device=device)
                fake_imgs = net_g(noise, fake_labels)

                # detach so d step does not backprop into g
                out_fake = net_d(fake_imgs.detach(), fake_labels)
                loss_d_fake = criterion(out_fake, target_fake)
                loss_d = loss_d_real + loss_d_fake

            if use_amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
            else:
                loss_d.backward()
                opt_d.step()

            # update generator
            opt_g.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                out_fake_for_g = net_d(fake_imgs, fake_labels)
                loss_g = criterion(out_fake_for_g, torch.ones((batch_sz, 1), device=device))

            if use_amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g)
                scaler_g.update()
            else:
                loss_g.backward()
                opt_g.step()

            running_loss_d += loss_d.item()
            running_loss_g += loss_g.item()
            total_batches += 1
            pbar.set_postfix(loss_d=f"{loss_d.item():.4f}", loss_g=f"{loss_g.item():.4f}")

        epoch_loss_d = running_loss_d / max(total_batches, 1)
        epoch_loss_g = running_loss_g / max(total_batches, 1)
        history["Discriminator"].append(epoch_loss_d)
        history["Generator"].append(epoch_loss_g)

        logger.info("Epoch [%d/%d] Loss_D: %.4f Loss_G: %.4f", epoch, epochs, epoch_loss_d, epoch_loss_g)

        with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_loss_g, epoch_loss_d])

        # save sample grid in eval mode
        if epoch % sample_interval == 0 or epoch == epochs:
            net_g.eval()
            with torch.no_grad():
                samples = net_g(fixed_noise, fixed_labels)
            grid_path = os.path.join(samples_dir, f"epoch_{epoch:03d}.png")
            save_image(samples, grid_path, nrow=10, normalize=True, value_range=(-1, 1))

        # save periodic checkpoint
        if epoch % ckpt_interval == 0:
            save_checkpoint(
                {"epoch": epoch, "state_dict": net_g.state_dict(), "model_cfg": model_cfg},
                os.path.join(ckpt_dir, f"generator_epoch_{epoch:03d}.pt")
            )

    # save final checkpoints and loss plot
    save_checkpoint(
        {"epoch": epochs, "state_dict": net_g.state_dict(), "model_cfg": model_cfg},
        os.path.join(ckpt_dir, "generator_final.pt")
    )
    save_checkpoint(
        {"epoch": epochs, "state_dict": net_d.state_dict(), "model_cfg": model_cfg},
        os.path.join(ckpt_dir, "discriminator_final.pt")
    )
    plot_loss_curves(history, os.path.join(plots_dir, "loss_curve.png"), title=f"DCGAN Loss - {run_name}")
    logger.info("Training complete. Artifacts saved to: %s", out_dir)


if __name__ == "__main__":
    args = parse_args()
    train_gan(config_path=args.config, run_name=args.run_name, overrides=args.override)
