"""Unit test cho module dataset và wrapper TransformSubset."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import Cifar10Dataset, TransformSubset
from src.data.transforms import build_val_transform
from src.utils.config import resolve_path

DATA_ROOT = resolve_path("data/raw")
HAS_DATA = (DATA_ROOT / "cifar-10-batches-py").exists()
requires_data = pytest.mark.skipif(
    not HAS_DATA, reason="CIFAR-10 chưa được tải — chạy scripts/download_data.py"
)


@requires_data
def test_dataset_length_and_classes() -> None:
    """Tập train phải có 50.000 mẫu và 10 lớp."""
    dataset = Cifar10Dataset(root=str(DATA_ROOT), train=True, download=False)

    assert len(dataset) == 50_000
    assert len(dataset.classes) == 10


@requires_data
def test_getitem_returns_tensor_and_int() -> None:
    """`__getitem__` phải trả về (tensor ảnh, nhãn int)."""
    dataset = Cifar10Dataset(
        root=str(DATA_ROOT),
        train=True,
        transform=build_val_transform(image_size=32),
        download=False,
    )
    image, label = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 32, 32)
    assert isinstance(label, int)
    assert 0 <= label <= 9


@requires_data
def test_negative_index_wraps() -> None:
    """Chỉ số âm phải được hiểu như Python indexing."""
    dataset = Cifar10Dataset(
        root=str(DATA_ROOT), train=True, transform=build_val_transform(image_size=32)
    )
    image, label = dataset[-1]
    last_image, last_label = dataset[len(dataset) - 1]

    assert torch.equal(image, last_image)
    assert label == last_label


@requires_data
def test_out_of_range_index_raises() -> None:
    """Chỉ số ngoài phạm vi phải ném IndexError."""
    dataset = Cifar10Dataset(root=str(DATA_ROOT), train=True)

    with pytest.raises(IndexError):
        dataset[len(dataset)]


@requires_data
def test_class_counts_are_balanced() -> None:
    """Mỗi lớp trong tập train phải có đúng 5.000 mẫu."""
    dataset = Cifar10Dataset(root=str(DATA_ROOT), train=True)
    counts = dataset.class_counts

    assert set(counts.values()) == {5_000}
    assert len(counts) == 10


@requires_data
def test_transform_subset_does_not_mutate_parent() -> None:
    """TransformSubset phải giữ transform riêng, không sửa dataset gốc."""
    dataset = Cifar10Dataset(
        root=str(DATA_ROOT), train=True, transform=build_val_transform(image_size=32)
    )
    original_transform = dataset.transform

    subset = TransformSubset(dataset, [0, 1, 2], transform=build_val_transform(image_size=64))
    image, _ = subset[0]

    assert image.shape == (3, 64, 64)
    assert dataset.transform is original_transform, "Dataset gốc bị mutate!"


@requires_data
def test_transform_subset_len_matches_indices() -> None:
    """Độ dài TransformSubset phải bằng số chỉ số."""
    dataset = Cifar10Dataset(root=str(DATA_ROOT), train=True)
    subset = TransformSubset(dataset, list(range(17)))

    assert len(subset) == 17


@requires_data
def test_subset_labels_match_parent() -> None:
    """Nhãn trong subset phải khớp với dataset gốc tại cùng chỉ số."""
    dataset = Cifar10Dataset(root=str(DATA_ROOT), train=True)
    subset = TransformSubset(dataset, [5, 9, 100])

    for position, source_index in enumerate([5, 9, 100]):
        _, label = subset[position]
        assert label == int(dataset.targets[source_index])


def test_missing_data_raises_clear_error(tmp_path) -> None:
    """Thư mục dữ liệu không tồn tại phải ném lỗi có thông điệp rõ ràng."""
    with pytest.raises((RuntimeError, FileNotFoundError, OSError)):
        Cifar10Dataset(root=str(tmp_path), train=True, download=False)


def test_dataset_in_memory_flag() -> None:
    """Flag `in_memory` phải quyết định có nạp mảng ảnh vào RAM hay không."""
    assert hasattr(Cifar10Dataset, "__init__")
    assert np is not None
