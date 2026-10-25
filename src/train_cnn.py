import argparse
import csv
import json
import os
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data import VARIANTS, VariantType, get_scenario_dataloader, get_test_dataloader
from src.evaluate import evaluate_all, evaluate_model, plot_confusion_matrix
from src.models import build_cnn
from src.utils import (
    get_device,
    load_config,
    save_checkpoint,
    set_seed,
    setup_logger
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN classifier across training-data scenarios")
    parser.add_argument("--config", type=str, default="configs/cnn.yaml", help="Path to CNN config")
    parser.add_argument("--run-name", type=str, default="dcgan_full_run", help="Experiment run name")
    parser.add_argument("--synthetic-path", type=str, default=None, help="Path to synthetic_data.pt")
    parser.add_argument("--variant", type=str, default="all", choices=[*VARIANTS, "all"], help="Scenario variant")

    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="List of random seeds")
    parser.add_argument("--override", type=str, action="append", default=[], help="Config overrides (key=value)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda, mps)")
    return parser.parse_args()


def train_single_run(
    variant: VariantType,
    seed: int,
    cfg: dict[str, Any],
    run_name: str,
    synthetic_path: str | None,
    device_name: str
) -> None:
    out_dir = os.path.join("outputs", run_name)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    metrics_dir = os.path.join(out_dir, "metrics")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger = setup_logger(f"cnn_{variant}_{seed}", os.path.join(metrics_dir, "cnn_training.log"))
    logger.info("Training variant: %s | Seed: %d", variant, seed)

    set_seed(seed)
    device = get_device(device_name, fallback=cfg.get("device"))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    train_loader = get_scenario_dataloader(
        variant=variant,
        data_dir=data_cfg.get("data_dir", "data"),
        synthetic_path=synthetic_path,
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        seed=seed
    )

    test_batch_size = int(data_cfg.get("test_batch_size", data_cfg.get("batch_size", 128)))
    test_loader = get_test_dataloader(
        data_dir=data_cfg.get("data_dir", "data"),
        batch_size=test_batch_size,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False))
    )

    model = build_cnn(model_cfg).to(device)


    epochs = int(train_cfg.get("epochs", 15))
    lr = float(train_cfg.get("lr", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))

    use_amp = bool(train_cfg.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    log_csv = os.path.join(metrics_dir, f"train_log_{variant}_seed{seed}.csv")
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc"])

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"[{variant}-s{seed}] Epoch {epoch}/{epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = total_loss / max(total_samples, 1)
        epoch_acc = correct / max(total_samples, 1)
        logger.info("Epoch [%d/%d] Loss: %.4f Acc: %.4f", epoch, epochs, epoch_loss, epoch_acc)

        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, epoch_loss, epoch_acc])

    # save trained weights
    ckpt_path = os.path.join(ckpt_dir, f"cnn_{variant}_seed{seed}.pt")
    save_checkpoint({"state_dict": model.state_dict(), "model_cfg": model_cfg, "variant": variant, "seed": seed}, ckpt_path)

    # evaluate on official test set and persist metrics
    metrics, cm = evaluate_model(model, test_loader, device)
    json_path = os.path.join(metrics_dir, f"metrics_{variant}_seed{seed}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    cm_path = os.path.join(plots_dir, f"confusion_matrix_{variant}_seed{seed}.png")
    plot_confusion_matrix(cm, cm_path, title=f"Confusion Matrix: {variant} (seed {seed})")
    logger.info("Variant %s (seed %d) Test Accuracy: %.4f", variant, seed, metrics["accuracy"])


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.override)

    variants: list[VariantType] = list(VARIANTS) if args.variant == "all" else [args.variant]
    synth_path = args.synthetic_path or os.path.join("outputs", args.run_name, "synthetic", "synthetic_data.pt")

    for v in variants:
        if v != "v1_real" and not os.path.exists(synth_path):
            raise FileNotFoundError(
                f"Synthetic data file not found at {synth_path}. Run generate.py first before variant '{v}'."
            )

    for v in variants:
        for seed in args.seeds:
            train_single_run(
                variant=v,
                seed=seed,
                cfg=cfg,
                run_name=args.run_name,
                synthetic_path=synth_path if v != "v1_real" else None,
                device_name=args.device
            )

    # aggregate results from saved metrics without duplicate evaluation
    evaluate_all(
        config_path=args.config,
        run_name=args.run_name,
        variants=variants,
        seeds=args.seeds,
        device_name=args.device
    )


if __name__ == "__main__":
    main()
