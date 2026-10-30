"""Tạo DataLoader cho train/validation/test theo config."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from src.data.dataset import Cifar10Dataset
from src.data.split import split_train_val
from src.data.transforms import build_transform
from src.utils.config import resolve_path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataLoaderConfig:
    """Cấu hình data pipeline.

    Attributes:
        data_root: Thư mục chứa CIFAR-10 (mặc định `data/raw`).
        image_size: Kích thước ảnh đầu vào sau resize.
        batch_size: Kích thước batch.
        num_workers: Số tiến trình DataLoader.
        val_size: Số mẫu validation lấy từ tập train.
        seed: Seed cho split và shuffle.
        augment: Có dùng augmentation cho tập train hay không.
        pin_memory: Bật `pin_memory` để tăng tốc chuyển tensor sang GPU.
        persistent_workers: Giữ worker alive giữa các epoch.
        download: Cho phép tải dataset nếu chưa có.
        in_memory: Nạp ảnh vào RAM để tăng tốc đọc.
    """

    data_root: str = "data/raw"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    val_size: int = 5_000
    seed: int = 42
    augment: bool = True
    pin_memory: bool = True
    persistent_workers: bool = True
    download: bool = False
    in_memory: bool = True


def _safe_num_workers(requested: int) -> int:
    """Giới hạn số worker theo số CPU khả dụng.

    Args:
        requested: Số worker mong muốn từ config.

    Returns:
        Số worker thực tế sau khi giới hạn.
    """
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return max(0, min(requested, 2))
    return max(0, min(requested, cpu_count - 1))


def build_dataloaders(
    config: DataLoaderConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tạo ba DataLoader: train, validation, test.

    Tập train của CIFAR-10 được tách thành 45.000 train / 5.000 validation
    theo phân tầng lớp. Tập test giữ nguyên 10.000 mẫu chính thức.

    Args:
        config: Cấu hình data pipeline.

    Returns:
        Tuple `(train_loader, val_loader, test_loader)`.

    Raises:
        FileNotFoundError: Nếu thư mục dữ liệu không tồn tại và không cho phép tải.
    """
    root = resolve_path(config.data_root)
    if not root.exists() and not config.download:
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu '{root}'. "
            "Chạy scripts/download_data.py hoặc bật download=True."
        )

    num_workers = _safe_num_workers(config.num_workers)
    persistent = config.persistent_workers and num_workers > 0

    train_transform = build_transform(
        "train", image_size=config.image_size, augment=config.augment
    )
    eval_transform = build_transform("val", image_size=config.image_size)

    full_train = Cifar10Dataset(
        root=str(root),
        train=True,
        transform=train_transform,
        download=config.download,
        in_memory=config.in_memory,
    )
    train_subset, val_subset = split_train_val(
        full_train,
        val_size=config.val_size,
        seed=config.seed,
        train_transform=train_transform,
        val_transform=eval_transform,
    )

    test_dataset = Cifar10Dataset(
        root=str(root),
        train=False,
        transform=eval_transform,
        download=config.download,
        in_memory=config.in_memory,
    )

    common: dict = {
        "num_workers": num_workers,
        "pin_memory": config.pin_memory,
        "persistent_workers": persistent,
    }

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        **common,
    )

    LOGGER.info(
        "DataLoader: train=%d batch | val=%d batch | test=%d batch | workers=%d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
        num_workers,
    )

    return train_loader, val_loader, test_loader


def dataset_summary(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> dict:
    """Tóm tắt thông tin data pipeline để ghi vào run artifact.

    Args:
        train_loader: DataLoader tập train.
        val_loader: DataLoader tập validation.
        test_loader: DataLoader tập test.

    Returns:
        Dict chứa số batch, số mẫu và shape tensor đầu tiên của mỗi split.
    """
    summary: dict = {}
    for name, loader in (
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ):
        first_images, first_labels = next(iter(loader))
        summary[name] = {
            "num_batches": len(loader),
            "num_samples": len(loader.dataset),
            "batch_size": loader.batch_size,
            "image_shape": list(first_images.shape),
            "image_dtype": str(first_images.dtype),
            "label_shape": list(first_labels.shape),
            "label_dtype": str(first_labels.dtype),
        }
    return summary


def raw_data_path(config: DataLoaderConfig) -> Path:
    """Trả về đường dẫn tuyệt đối tới thư mục dữ liệu gốc.

    Args:
        config: Cấu hình data pipeline.

    Returns:
        Path tuyệt đối tới thư mục `data/raw`.
    """
    return resolve_path(config.data_root)
