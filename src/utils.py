import logging
import os
import random
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def get_device(device_override: str | None = None, fallback: str | None = None) -> torch.device:
    """Return torch.device based on override, config fallback, or automatic detection."""
    target = device_override if (device_override and device_override != "auto") else fallback
    if target and target != "auto":
        return torch.device(target)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def set_seed(seed: int) -> torch.Generator:
    """Set global random seeds for reproducibility and return seeded generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def weights_init(m: torch.nn.Module) -> None:
    # dcgan weight initialization
    classname = m.__class__.__name__
    if "Conv" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif "BatchNorm" in classname:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif "Embedding" in classname:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)



def seed_worker(worker_id: int) -> None:
    # seeded worker initialization for DataLoader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _cast_value(value_str: str) -> Any:
    # parse typed scalar from command line override string
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() in ("null", "none"):
        return None
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def _set_nested_key(cfg: dict[str, Any], keys: list[str], val: Any) -> None:
    curr = cfg
    for k in keys[:-1]:
        if k not in curr or not isinstance(curr[k], dict):
            raise KeyError(f"Unknown config section '{k}' in path '{'.'.join(keys)}'.")
        curr = curr[k]
    if keys[-1] not in curr:
        raise KeyError(f"Unknown config key '{keys[-1]}' in path '{'.'.join(keys)}'.")
    curr[keys[-1]] = val


def _apply_override(cfg: dict[str, Any], override_str: str) -> None:
    # apply single key=value override with strict validation
    if "=" not in override_str:
        raise ValueError(f"Invalid override format: {override_str}. Expected key=value.")
    key, val_str = override_str.split("=", 1)
    val = _cast_value(val_str)

    if "." in key:
        _set_nested_key(cfg, key.split("."), val)
        return

    # check top-level key
    if key in cfg:
        cfg[key] = val
        return

    # check first-level sub-dictionaries
    matching = [sub for sub in cfg.values() if isinstance(sub, dict) and key in sub]
    if len(matching) == 1:
        matching[0][key] = val
        return
    if len(matching) > 1:
        raise KeyError(f"Ambiguous override key '{key}' found in multiple sections. Use dot notation (e.g. section.{key}=value).")

    raise KeyError(f"Unknown config key '{key}' in override '{override_str}'.")



def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config and apply list of key=value overrides."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        for item in overrides:
            _apply_override(cfg, item)

    return cfg


def save_config(config: dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def setup_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def save_checkpoint(state: dict[str, Any], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, map_location: torch.device | str = "cpu") -> dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    return torch.load(filepath, map_location=map_location)


def plot_loss_curves(history: dict[str, list[float]], output_path: str, title: str = "Training Loss") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.figure(figsize=(8, 5))
    for label, losses in history.items():
        plt.plot(range(1, len(losses) + 1), losses, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
