import argparse
import csv
import json
import os
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import SyntheticDataset, VARIANTS, get_test_dataloader
from src.models import build_cnn
from src.utils import get_device, load_checkpoint, load_config, setup_logger

FASHION_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]
METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN models on official Fashion-MNIST test set")
    parser.add_argument("--config", type=str, default="configs/cnn.yaml", help="Path to CNN config")
    parser.add_argument("--run-name", type=str, default="dcgan_full_run", help="Experiment run name")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), help="Variants to evaluate")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="List of seeds")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda, mps)")
    parser.add_argument("--force-eval", action="store_true", help="Re-evaluate from checkpoints even if JSON exists")
    parser.add_argument("--eval-synthetic-with-real", action="store_true", help="Measure real model accuracy on synthetic dataset")
    parser.add_argument("--real-run-name", type=str, default=None, help="Run containing the real-trained CNN checkpoint")
    return parser.parse_args()


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], np.ndarray]:
    """Compute test metrics and confusion matrix in no_grad mode."""
    model.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(labels.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    precisions, recalls, f1s, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    per_class: dict[str, dict[str, float]] = {}
    for i, name in enumerate(FASHION_CLASSES):
        per_class[name] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i])
        }

    cm = confusion_matrix(y_true, y_pred)
    metrics = {
        "accuracy": acc,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "per_class": per_class
    }
    return metrics, cm


def plot_confusion_matrix(cm: np.ndarray, output_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(FASHION_CLASSES))
    plt.xticks(tick_marks, FASHION_CLASSES, rotation=45, ha="right")
    plt.yticks(tick_marks, FASHION_CLASSES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, "d"), horizontalalignment="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def aggregate_and_save_summary(records: list[dict[str, Any]], output_csv_path: str) -> None:
    """Group metrics by variant and output mean +/- sample std to summary CSV."""
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    grouped: dict[str, dict[str, list[float]]] = {}
    for r in records:
        v = r["variant"]
        if v not in grouped:
            grouped[v] = {m: [] for m in METRIC_KEYS}
        for m in METRIC_KEYS:
            grouped[v][m].append(r["metrics"][m])

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant", "seeds_count", "accuracy", "macro_precision", "macro_recall", "macro_f1"])

        for v, metrics_dict in grouped.items():
            n_seeds = len(metrics_dict["accuracy"])
            row: list[Any] = [v, n_seeds]
            for m in METRIC_KEYS:
                vals = metrics_dict[m]
                mean_val = float(np.mean(vals))
                if n_seeds > 1:
                    std_val = float(np.std(vals, ddof=1))
                    row.append(f"{mean_val:.4f} +/- {std_val:.4f}")
                else:
                    row.append(f"{mean_val:.4f} +/- n/a")

            writer.writerow(row)


def evaluate_synthetic_quality(
    config_path: str,
    run_name: str,
    synthetic_path: str,
    seed: int = 0,
    device_name: str = "auto",
    real_run_name: str | None = None
) -> dict[str, Any]:
    """Measure how accurately a real-trained model classifies synthetic samples and save JSON."""
    cfg = load_config(config_path)
    out_dir = os.path.join("outputs", run_name)
    metrics_dir = os.path.join(out_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    logger = setup_logger("synthetic_quality", os.path.join(metrics_dir, "synthetic_quality.log"))

    device = get_device(device_name, fallback=cfg.get("device"))
    ckpt_dir = os.path.join("outputs", real_run_name or run_name, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"cnn_v1_real_seed{seed}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Real checkpoint {ckpt_path} not found.")

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    m_cfg = ckpt.get("model_cfg", cfg.get("model", {}))
    model = build_cnn(m_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    synth_dataset = SyntheticDataset(synthetic_path)
    test_batch_sz = int(cfg.get("data", {}).get("test_batch_size", 128))
    loader = DataLoader(synth_dataset, batch_size=test_batch_sz, shuffle=False)
    metrics, _ = evaluate_model(model, loader, device)

    # persist synthetic quality results
    out_json = os.path.join(metrics_dir, "synthetic_quality.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Synthetic Data Quality (Model: v1_real, Seed: %d): Accuracy=%.4f, Macro-F1=%.4f", seed, metrics["accuracy"], metrics["macro_f1"])
    logger.info("Saved synthetic quality metrics to: %s", out_json)
    return metrics


def evaluate_all(
    config_path: str,
    run_name: str,
    variants: list[str],
    seeds: list[int],
    device_name: str = "auto",
    force_eval: bool = False
) -> None:
    cfg = load_config(config_path)
    out_dir = os.path.join("outputs", run_name)
    metrics_dir = os.path.join(out_dir, "metrics")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    logger = setup_logger("evaluate", os.path.join(metrics_dir, "evaluation.log"))
    device = get_device(device_name, fallback=cfg.get("device"))

    test_loader = None
    records: list[dict[str, Any]] = []

    for v in variants:
        for seed in seeds:
            json_path = os.path.join(metrics_dir, f"metrics_{v}_seed{seed}.json")

            # use existing json metrics if available to avoid redundant computation
            if not force_eval and os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                records.append({"variant": v, "seed": seed, "metrics": metrics})
                continue

            ckpt_path = os.path.join(out_dir, "checkpoints", f"cnn_{v}_seed{seed}.pt")
            if not os.path.exists(ckpt_path):
                logger.warning("Checkpoint %s does not exist, skipping.", ckpt_path)
                continue

            if test_loader is None:
                data_cfg = cfg["data"]
                test_batch_sz = int(data_cfg.get("test_batch_size", data_cfg.get("batch_size", 128)))
                test_loader = get_test_dataloader(
                    data_dir=data_cfg.get("data_dir", "data"),
                    batch_size=test_batch_sz,
                    num_workers=int(data_cfg.get("num_workers", 0)),
                    pin_memory=bool(data_cfg.get("pin_memory", False))
                )

            ckpt = load_checkpoint(ckpt_path, map_location=device)
            m_cfg = ckpt.get("model_cfg", cfg.get("model", {}))
            model = build_cnn(m_cfg).to(device)
            model.load_state_dict(ckpt["state_dict"])

            metrics, cm = evaluate_model(model, test_loader, device)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            cm_path = os.path.join(plots_dir, f"confusion_matrix_{v}_seed{seed}.png")
            plot_confusion_matrix(cm, cm_path, title=f"Confusion Matrix: {v} (seed {seed})")

            records.append({"variant": v, "seed": seed, "metrics": metrics})
            logger.info("Evaluated %s (seed %d): Accuracy=%.4f F1=%.4f", v, seed, metrics["accuracy"], metrics["macro_f1"])

    if records:
        summary_csv = os.path.join(metrics_dir, "results_summary.csv")
        aggregate_and_save_summary(records, summary_csv)
        logger.info("Saved aggregated evaluation summary to: %s", summary_csv)
    else:
        logger.warning("No metrics or checkpoints found.")


if __name__ == "__main__":
    args = parse_args()
    variant_list = list(VARIANTS) if "all" in args.variants else args.variants

    if args.eval_synthetic_with_real:
        synth_file = os.path.join("outputs", args.run_name, "synthetic", "synthetic_data.pt")
        evaluate_synthetic_quality(args.config, args.run_name, synth_file, seed=args.seeds[0], device_name=args.device, real_run_name=args.real_run_name)
    else:
        evaluate_all(
            config_path=args.config,
            run_name=args.run_name,
            variants=variant_list,
            seeds=args.seeds,
            device_name=args.device,
            force_eval=args.force_eval
        )
