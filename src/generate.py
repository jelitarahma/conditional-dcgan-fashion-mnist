import argparse
import os
import torch
from torchvision.utils import save_image


from src.models import build_generator
from src.utils import get_device, load_checkpoint, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labeled synthetic Fashion-MNIST dataset using trained Generator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint (.pt)")
    parser.add_argument("--run-name", type=str, default="dcgan_full_run", help="Experiment run name")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory for synthetic data")
    parser.add_argument("--samples-per-class", type=int, default=3000, help="Number of samples to generate per class")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (defaults to checkpoint config)")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for image generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, cuda, mps)")
    return parser.parse_args()


def generate_synthetic_data(
    checkpoint_path: str,
    output_dir: str,
    samples_per_class: int = 3000,
    num_classes: int | None = None,
    batch_size: int = 256,
    seed: int = 42,
    device_name: str = "auto"
) -> str:
    """Generate class-balanced synthetic images and save compact .pt dataset with preview grid."""
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("generate", os.path.join(output_dir, "generation.log"))
    logger.info("Loading generator checkpoint from: %s", checkpoint_path)

    set_seed(seed)
    device = get_device(device_name)
    logger.info("Using device: %s", device)

    ckpt = load_checkpoint(checkpoint_path, map_location=device)
    model_cfg = ckpt.get("model_cfg", {})

    net_g = build_generator(model_cfg).to(device)
    net_g.load_state_dict(ckpt["state_dict"])
    net_g.eval()

    nz = net_g.nz
    resolved_num_classes = num_classes if num_classes is not None else net_g.embed.num_embeddings


    all_images: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    logger.info("Generating %d samples per class (%d total)", samples_per_class, samples_per_class * resolved_num_classes)

    with torch.no_grad():
        for c in range(resolved_num_classes):
            generated_class_count = 0
            while generated_class_count < samples_per_class:
                cur_batch = min(batch_size, samples_per_class - generated_class_count)
                z = torch.randn(cur_batch, nz, device=device)
                labels = torch.full((cur_batch,), c, dtype=torch.long, device=device)

                fake = net_g(z, labels).cpu()
                # convert [-1, 1] tanh scale to compact uint8 [0, 255]
                fake_uint8 = ((fake.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)

                all_images.append(fake_uint8)
                all_labels.append(labels.cpu())
                generated_class_count += cur_batch

    final_images = torch.cat(all_images, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    dataset_path = os.path.join(output_dir, "synthetic_data.pt")
    torch.save({"images": final_images, "labels": final_labels}, dataset_path)
    logger.info("Saved synthetic dataset to %s (shape: %s, dtype: %s)", dataset_path, final_images.shape, final_images.dtype)

    # generate preview grid: classes x 10 samples
    preview_z = torch.randn(resolved_num_classes * 10, nz, device=device)
    preview_labels = torch.arange(resolved_num_classes, device=device).repeat_interleave(10)
    with torch.no_grad():
        preview_samples = net_g(preview_z, preview_labels).cpu()

    preview_path = os.path.join(output_dir, "preview_grid.png")
    save_image(preview_samples, preview_path, nrow=10, normalize=True, value_range=(-1, 1))
    logger.info("Saved preview grid to: %s", preview_path)

    return dataset_path


if __name__ == "__main__":
    args = parse_args()
    target_out = args.output_dir or os.path.join("outputs", args.run_name, "synthetic")
    generate_synthetic_data(
        checkpoint_path=args.checkpoint,
        output_dir=target_out,
        samples_per_class=args.samples_per_class,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device
    )
