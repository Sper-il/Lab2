"""Sinh biểu đồ kiểm chứng data pipeline (M1).

Chạy: .\\.venv\\Scripts\\python.exe scripts\\plot_data_report.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Cho phép chạy trực tiếp file này: thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.dataloader import DataLoaderConfig, build_dataloaders  # noqa: E402
from src.utils.runs import create_run_dir, setup_logging  # noqa: E402

LOGGER = logging.getLogger("plot_data_report")


def _denormalize(batch: torch.Tensor) -> np.ndarray:
    """Đưa tensor đã normalize về [0, 1] để vẽ.

    Args:
        batch: Tensor ảnh đã normalize, shape [B, 3, H, W].

    Returns:
        Mảng numpy uint8 shape [B, H, W, 3].
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = (batch * std + mean).clamp(0, 1)
    return (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)


def _counts_of(loader) -> dict[int, int]:
    """Đếm số mẫu theo lớp của một DataLoader.

    Args:
        loader: DataLoader cần đếm.

    Returns:
        Dict {class_index: count}.
    """
    dataset = loader.dataset
    base = dataset.dataset if hasattr(dataset, "dataset") else dataset
    targets = base.targets
    indices = dataset.indices if hasattr(dataset, "indices") else range(len(targets))
    selected = targets[np.asarray(list(indices))]
    labels, counts = np.unique(selected, return_counts=True)
    return {int(label): int(count) for label, count in zip(labels, counts)}


def _plot_distribution(counts_by_split, class_names, output: Path) -> None:
    """Vẽ biểu đồ cột phân bố lớp của 3 split.

    Args:
        counts_by_split: Dict {split_name: {class_index: count}}.
        class_names: Tên các lớp.
        output: Đường dẫn file PNG output.
    """
    indices = sorted(counts_by_split["train"])
    width = 0.27
    positions = np.arange(len(indices))
    styles = [
        ("train", "Train (45,000)", "#4C78A8"),
        ("val", "Validation (5,000)", "#F58518"),
        ("test", "Test (10,000)", "#54A24B"),
    ]

    figure, axis = plt.subplots(figsize=(12, 5.5))
    for offset, (key, label, color) in zip((-width, 0.0, width), styles):
        axis.bar(
            positions + offset,
            [counts_by_split[key][i] for i in indices],
            width,
            label=label,
            color=color,
        )

    axis.set_xticks(positions)
    axis.set_xticklabels([class_names[i] for i in indices], rotation=35, ha="right")
    axis.set_ylabel("So mau")
    axis.set_title("Phan bo lop CIFAR-10 sau khi chia tap (stratified, seed=42)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    LOGGER.info("Da luu %s", output)


def _plot_samples(images: torch.Tensor, labels: list[int], class_names, output: Path) -> None:
    """Vẽ lưới ảnh mẫu kèm nhãn.

    Args:
        images: Batch ảnh đã normalize.
        labels: Nhãn tương ứng.
        class_names: Tên các lớp.
        output: Đường dẫn file PNG output.
    """
    count = min(12, len(labels))
    array = _denormalize(images[:count])

    figure, axes = plt.subplots(3, 4, figsize=(13, 10))
    for index, axis in enumerate(axes.flatten()):
        if index < count:
            axis.imshow(array[index])
            axis.set_title(class_names[labels[index]], fontsize=10)
        axis.axis("off")

    figure.suptitle(
        "Anh mau sau transform (resize 32 -> 224 + normalize ImageNet)", fontsize=13
    )
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    LOGGER.info("Da luu %s", output)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Danh sách argument.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Sinh bao cao truc quan cho M1")
    parser.add_argument("--run-dir", default=None, help="Thu muc run de ghi vao")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else create_run_dir()
    setup_logging(run_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    config = DataLoaderConfig(
        data_root="data/raw",
        image_size=224,
        batch_size=64,
        num_workers=2,
        val_size=5_000,
        download=False,
    )

    train_loader, val_loader, test_loader = build_dataloaders(config)
    class_names = train_loader.dataset.dataset.classes

    _plot_distribution(
        {
            "train": _counts_of(train_loader),
            "val": _counts_of(val_loader),
            "test": _counts_of(test_loader),
        },
        class_names,
        plots_dir / "class_distribution.png",
    )

    sample_images, sample_labels = next(iter(val_loader))
    _plot_samples(
        sample_images,
        [int(label) for label in sample_labels],
        class_names,
        plots_dir / "sample_images.png",
    )

    LOGGER.info("Hoan tat bao cao truc quan: %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
