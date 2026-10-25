import os
from typing import Literal
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets
from src.utils import seed_worker

VARIANTS = ("v0_real_30k", "v1_real", "v2_half_real_half_synth", "v3_real_plus_synth")
VariantType = Literal["v0_real_30k", "v1_real", "v2_half_real_half_synth", "v3_real_plus_synth"]


class FastFashionMNIST(Dataset):
    """In-memory tensor dataset for Fashion-MNIST to avoid per-sample PIL transforms."""

    def __init__(self, data_dir: str, train: bool = True) -> None:
        raw_dataset = datasets.FashionMNIST(root=data_dir, train=train, download=True)
        # convert uint8 [0, 255] to float32 [-1, 1] tensor in memory
        self.data = (raw_dataset.data.float().unsqueeze(1) / 127.5) - 1.0
        self.targets = raw_dataset.targets.long()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data[idx], int(self.targets[idx].item())


class SyntheticDataset(Dataset):
    """Loads pre-generated synthetic images and integer labels from disk."""

    def __init__(self, data_path: str) -> None:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Synthetic data file not found: {data_path}")

        payload = torch.load(data_path, map_location="cpu")
        raw_images = payload["images"]
        raw_labels = payload["labels"]

        # convert uint8 [0, 255] to float [-1, 1] scale matching tanh output
        if raw_images.dtype == torch.uint8:
            self.images = (raw_images.float() / 127.5) - 1.0
        else:
            self.images = raw_images.float()

        self.labels = raw_labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], int(self.labels[idx].item())


def get_fashion_mnist_datasets(data_dir: str) -> tuple[FastFashionMNIST, FastFashionMNIST]:
    """Load cached in-memory Fashion-MNIST train and test splits."""
    os.makedirs(data_dir, exist_ok=True)
    train_set = FastFashionMNIST(data_dir=data_dir, train=True)
    test_set = FastFashionMNIST(data_dir=data_dir, train=False)
    return train_set, test_set


def get_balanced_subset(dataset: Dataset, samples_per_class: int, seed: int) -> Subset:
    """Draw a deterministic class-balanced subset from dataset."""
    targets = dataset.targets
    if isinstance(targets, torch.Tensor):
        targets_arr = targets.cpu().numpy()
    else:
        targets_arr = np.asarray(targets)

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []
    unique_classes = np.unique(targets_arr)

    for c in unique_classes:
        class_indices = np.where(targets_arr == c)[0]
        if len(class_indices) < samples_per_class:
            raise ValueError(f"Requested {samples_per_class} samples for class {c}, only {len(class_indices)} exist.")
        perm = rng.permutation(class_indices)
        selected_indices.extend(perm[:samples_per_class].tolist())

    rng.shuffle(selected_indices)
    return Subset(dataset, selected_indices)


def get_scenario_dataset(
    variant: VariantType,
    data_dir: str,
    synthetic_path: str | None = None,
    seed: int = 42
) -> Dataset:
    """Construct dataset for one of the three benchmark scenarios."""
    os.makedirs(data_dir, exist_ok=True)
    real_train_set = FastFashionMNIST(data_dir=data_dir, train=True)

    if variant == "v0_real_30k":
        # baseline: 30,000 real images only (3,000 per class balanced)
        return get_balanced_subset(real_train_set, samples_per_class=3000, seed=seed)

    if variant == "v1_real":
        return real_train_set

    if not synthetic_path:
        raise ValueError(f"Synthetic data path is required for variant '{variant}'.")

    synthetic_set = SyntheticDataset(synthetic_path)

    if variant == "v2_half_real_half_synth":
        # 30,000 real (3,000 per class) + 30,000 synthetic
        num_classes = len(torch.unique(synthetic_set.labels))
        samples_per_class = len(synthetic_set) // max(num_classes, 1) if len(synthetic_set) != 30000 else 3000
        real_subset = get_balanced_subset(real_train_set, samples_per_class=samples_per_class, seed=seed)
        return ConcatDataset([real_subset, synthetic_set])

    if variant == "v3_real_plus_synth":
        # 60,000 real + synthetic
        return ConcatDataset([real_train_set, synthetic_set])

    raise ValueError(f"Unknown variant '{variant}'. Choose from {VARIANTS}.")



def get_scenario_dataloader(
    variant: VariantType,
    data_dir: str,
    synthetic_path: str | None,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int
) -> DataLoader:
    """Return reproducible DataLoader for selected training scenario."""
    dataset = get_scenario_dataset(variant, data_dir, synthetic_path, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator
    )


def get_gan_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int
) -> DataLoader:
    """Return training DataLoader for GAN on full 60,000 train set."""
    train_set, _ = get_fashion_mnist_datasets(data_dir)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator
    )


def get_test_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    pin_memory: bool
) -> DataLoader:
    """Return evaluation DataLoader for official 10,000 test set."""
    _, test_set = get_fashion_mnist_datasets(data_dir)
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
