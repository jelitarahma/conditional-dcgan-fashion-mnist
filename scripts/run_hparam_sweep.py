import argparse
import csv
import json
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
SWEEP_CSV = os.path.join(OUTPUTS_DIR, "hparam_sweep_summary.csv")

SWEEP_CONFIGS = [
    {
        "name": "hparam_baseline_nz100",
        "factor": "Baseline (Control)",
        "param_changed": "Default reference",
        "overrides": ["training.epochs=5", "data.batch_size=256", "training.sample_interval=5", "training.checkpoint_interval=5"]
    },
    {
        "name": "hparam_nz_50",
        "factor": "Latent Dimension (nz)",
        "param_changed": "model.nz=50",
        "overrides": ["model.nz=50", "training.epochs=5", "data.batch_size=256", "training.sample_interval=5", "training.checkpoint_interval=5"]
    },
    {
        "name": "hparam_nz_200",
        "factor": "Latent Dimension (nz)",
        "param_changed": "model.nz=200",
        "overrides": ["model.nz=200", "training.epochs=5", "data.batch_size=256", "training.sample_interval=5", "training.checkpoint_interval=5"]
    },
    {
        "name": "hparam_filters_32",
        "factor": "Capacity (ngf/ndf)",
        "param_changed": "model.ngf=32, model.ndf=32",
        "overrides": ["model.ngf=32", "model.ndf=32", "training.epochs=5", "data.batch_size=256", "training.sample_interval=5", "training.checkpoint_interval=5"]
    },
    {
        "name": "hparam_extra_layers_1",
        "factor": "Conv Architecture",
        "param_changed": "model.extra_layers=1",
        "overrides": ["model.extra_layers=1", "training.epochs=5", "data.batch_size=256", "training.sample_interval=5", "training.checkpoint_interval=5"]
    },
    {
        "name": "hparam_slope_01",
        "factor": "Activation Slope",
        "param_changed": "model.leaky_relu_slope=0.1",
        "overrides": ["model.leaky_relu_slope=0.1", "training.epochs=5", "data.batch_size=256", "training.sample_interval=5", "training.checkpoint_interval=5"]
    },
    {
        "name": "hparam_lr_5e4",
        "factor": "Learning Rate",
        "param_changed": "lr_g=0.0005, lr_d=0.0005",
        "overrides": ["training.lr_g=0.0005", "training.lr_d=0.0005", "training.epochs=5", "data.batch_size=256", "training.sample_interval=5", "training.checkpoint_interval=5"]
    },
    {
        "name": "hparam_batch_64",
        "factor": "Batch Size",
        "param_changed": "data.batch_size=64",
        "overrides": ["data.batch_size=64", "training.epochs=5", "training.sample_interval=5", "training.checkpoint_interval=5"]
    },
    {
        "name": "hparam_kernel_5",
        "factor": "Kernel Size (Extra Layers)",
        "param_changed": "model.extra_layers=1, model.kernel_size=5",
        "overrides": ["model.extra_layers=1", "model.kernel_size=5", "training.epochs=5", "data.batch_size=256", "training.sample_interval=5", "training.checkpoint_interval=5"]
    }
]


def run_command(cmd: list[str]) -> None:
    print(f"\n[RUNNING]: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=BASE_DIR)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DCGAN hyperparameter exploration sweep.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per sweep run (default: 5)")
    parser.add_argument("--indices", nargs="+", type=int, default=None, help="Indices of configs to run")
    parser.add_argument("--output-csv", type=str, default=SWEEP_CSV, help="Output path for summary CSV")
    args = parser.parse_args()

    target_csv = args.output_csv
    configs_to_run = SWEEP_CONFIGS
    if args.indices is not None:
        configs_to_run = [SWEEP_CONFIGS[i] for i in args.indices]

    results = []

    for item in configs_to_run:
        run_name = item["name"]
        print(f"\n=======================================================")
        print(f"Executing Sweep Trial: {run_name} | Factor: {item['factor']}")
        print(f"=======================================================")

        # 1. Train GAN
        gan_cmd = [
            sys.executable, "-m", "src.train_gan",
            "--config", "configs/dcgan.yaml",
            "--run-name", run_name,
        ]
        for ov in item["overrides"]:
            if ov.startswith("training.epochs="):
                ov = f"training.epochs={args.epochs}"
            gan_cmd.extend(["--override", ov])
        run_command(gan_cmd)

        # 2. Generate 1000 samples per class (10k total) for rapid evaluation
        ckpt_path = os.path.join(OUTPUTS_DIR, run_name, "checkpoints", "generator_final.pt")
        gen_cmd = [
            sys.executable, "-m", "src.generate",
            "--checkpoint", ckpt_path,
            "--run-name", run_name,
            "--samples-per-class", "1000"
        ]
        run_command(gen_cmd)

        # 3. Evaluate synthetic quality using the real model from dcgan_full_run
        eval_cmd = [
            sys.executable, "-m", "src.evaluate",
            "--config", "configs/cnn.yaml",
            "--run-name", run_name,
            "--eval-synthetic-with-real",
            "--real-run-name", "dcgan_full_run"
        ]
        run_command(eval_cmd)

        # 4. Extract synthetic quality metrics
        quality_json = os.path.join(OUTPUTS_DIR, run_name, "metrics", "synthetic_quality.json")
        acc, f1 = 0.0, 0.0
        if os.path.exists(quality_json):
            with open(quality_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                acc = data.get("accuracy", 0.0)
                f1 = data.get("macro_f1", 0.0)

        # Extract final G/D loss from train_log.csv
        train_csv = os.path.join(OUTPUTS_DIR, run_name, "train_log.csv")
        loss_g, loss_d = 0.0, 0.0
        if os.path.exists(train_csv):
            with open(train_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    loss_g = float(row.get("loss_g", 0.0))
                    loss_d = float(row.get("loss_d", 0.0))

        results.append({
            "run_name": run_name,
            "factor": item["factor"],
            "param_changed": item["param_changed"],
            "loss_g": f"{loss_g:.4f}",
            "loss_d": f"{loss_d:.4f}",
            "synthetic_accuracy": f"{acc * 100:.2f}%",
            "synthetic_f1": f"{f1 * 100:.2f}%"
        })

    # Save summary CSV (merge with existing rows if target_csv already exists)
    os.makedirs(os.path.dirname(os.path.abspath(target_csv)), exist_ok=True)
    existing_rows = {}
    if os.path.exists(target_csv):
        with open(target_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "run_name" in row:
                    existing_rows[row["run_name"]] = row
    for r in results:
        existing_rows[r["run_name"]] = r

    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_name", "factor", "param_changed", "loss_g", "loss_d", "synthetic_accuracy", "synthetic_f1"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(list(existing_rows.values()))

    print(f"\n[DONE] Saved sweep summary table to {target_csv}")


if __name__ == "__main__":
    main()
