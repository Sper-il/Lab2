"""Integration test: dựng DataLoader end-to-end từ config thật."""

from __future__ import annotations

import pytest
import torch

from src.data.dataloader import (
    DataLoaderConfig,
    build_dataloaders,
    dataset_summary,
)
from src.utils.config import load_config, resolve_path

DATA_ROOT = resolve_path("data/raw")
HAS_DATA = (DATA_ROOT / "cifar-10-batches-py").exists()
requires_data = pytest.mark.skipif(
    not HAS_DATA, reason="CIFAR-10 chưa được tải — chạy scripts/download_data.py"
)

# Cấu hình nhẹ cho test: ảnh nhỏ để test chạy nhanh
LIGHT_CONFIG = DataLoaderConfig(
    data_root="data/raw",
    image_size=32,
    batch_size=128,
    num_workers=0,
    val_size=1_000,
    download=False,
)


@requires_data
def test_build_dataloaders_returns_three_loaders() -> None:
    """Phải trả về đúng 3 DataLoader: train, val, test."""
    train_loader, val_loader, test_loader = build_dataloaders(LIGHT_CONFIG)

    assert len(train_loader.dataset) == 49_000
    assert len(val_loader.dataset) == 1_000
    assert len(test_loader.dataset) == 10_000


@requires_data
def test_batch_shapes() -> None:
    """Shape của batch phải là [B, C, H, W] và label [B]."""
    train_loader, _, _ = build_dataloaders(LIGHT_CONFIG)
    images, labels = next(iter(train_loader))

    assert images.dim() == 4
    assert images.shape[1] == 3
    assert labels.shape == (images.shape[0],)
    assert images.dtype == torch.float32


@requires_data
def test_train_loader_shuffles() -> None:
    """Tập train phải được shuffle (thứ tự batch khác nhau giữa 2 epoch)."""
    train_loader, _, _ = build_dataloaders(LIGHT_CONFIG)

    first = [labels.tolist() for _, labels in train_loader]
    second = [labels.tolist() for _, labels in train_loader]

    assert first != second, "DataLoader train không shuffle"


@requires_data
def test_val_loader_is_deterministic() -> None:
    """Tập val phải có thứ tự cố định giữa các lần chạy."""
    _, val_loader, _ = build_dataloaders(LIGHT_CONFIG)

    first = [labels.tolist() for _, labels in val_loader]
    second = [labels.tolist() for _, labels in val_loader]

    assert first == second, "DataLoader validation không deterministic"


@requires_data
def test_val_transform_differs_from_train_transform() -> None:
    """Val không được dùng augmentation của train."""
    train_loader, val_loader, _ = build_dataloaders(LIGHT_CONFIG)

    train_transforms = [type(t).__name__ for t in train_loader.dataset.transform.transforms]
    val_transforms = [type(t).__name__ for t in val_loader.dataset.transform.transforms]

    assert "RandomCrop" in train_transforms
    assert "RandomCrop" not in val_transforms


@requires_data
def test_dataset_summary_structure() -> None:
    """`dataset_summary` phải trả về đủ thông tin cho cả 3 split."""
    loaders = build_dataloaders(LIGHT_CONFIG)
    summary = dataset_summary(*loaders)

    assert set(summary) == {"train", "val", "test"}
    for info in summary.values():
        assert info["num_samples"] > 0
        assert info["num_batches"] > 0
        assert len(info["image_shape"]) == 4


@requires_data
def test_val_size_honoured() -> None:
    """Số mẫu validation phải đúng bằng config."""
    config = DataLoaderConfig(
        data_root="data/raw",
        image_size=32,
        batch_size=64,
        num_workers=0,
        val_size=2_500,
    )
    _, val_loader, _ = build_dataloaders(config)

    assert len(val_loader.dataset) == 2_500
    assert len(val_loader.dataset) + 47_500 == 50_000


def test_config_yaml_parses() -> None:
    """File config thật phải parse được và có đủ trường."""
    config = load_config("configs/data_config.yaml")

    assert "data" in config
    data = config["data"]
    for key in ("data_root", "image_size", "batch_size", "val_size", "seed"):
        assert key in data, f"Thiếu trường '{key}' trong config"
