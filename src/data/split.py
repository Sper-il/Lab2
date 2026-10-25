"""Tách tập train CIFAR-10 thành train/validation có phân tầng theo lớp."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

from src.data.dataset import Cifar10Dataset, TransformSubset

LOGGER = logging.getLogger(__name__)

DEFAULT_SEED: int = 42


def stratified_indices(
    targets: Sequence[int],
    val_size: int,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Chia chỉ số train/val theo từng lớp để giữ đúng tỉ lệ phân bố.

    Với mỗi lớp, chọn ngẫu nhiên `val_size / n_classes` mẫu làm validation.
    Nhờ vậy tập validation có cùng tỉ lệ lớp với tập train.

    Args:
        targets: Nhãn của toàn bộ tập train.
        val_size: Số mẫu validation mong muốn.
        seed: Seed để tái lập kết quả.

    Returns:
        Tuple `(train_indices, val_indices)` dạng `numpy.ndarray` int64.

    Raises:
        ValueError: Nếu `val_size` không hợp lệ so với số lớp hiện có.
    """
    targets_array = np.asarray(targets)
    unique_classes = np.unique(targets_array)

    if val_size <= 0:
        raise ValueError(f"val_size phải > 0, nhận {val_size}")
    if val_size >= len(targets_array):
        raise ValueError(
            f"val_size ({val_size}) phải nhỏ hơn số mẫu train ({len(targets_array)})"
        )

    per_class = val_size // len(unique_classes)
    remainder = val_size - per_class * len(unique_classes)
    if per_class < 1:
        raise ValueError(
            f"val_size={val_size} quá nhỏ cho {len(unique_classes)} lớp "
            "(cần ít nhất 1 mẫu mỗi lớp)"
        )

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for position, class_index in enumerate(unique_classes):
        class_indices = np.flatnonzero(targets_array == class_index)
        class_indices = class_indices[rng.permutation(len(class_indices))]

        # Phần dư của phép chia được phân bố cho các lớp đầu tiên
        take = per_class + (1 if position < remainder else 0)
        if take >= len(class_indices):
            raise ValueError(
                f"Lớp {class_index} chỉ có {len(class_indices)} mẫu, "
                f"không đủ để lấy {take} mẫu validation"
            )

        val_parts.append(class_indices[:take])
        train_parts.append(class_indices[take:])

    train_indices = np.sort(np.concatenate(train_parts))
    val_indices = np.sort(np.concatenate(val_parts))

    LOGGER.info(
        "Split train/val: %d train / %d val (seed=%d, %d lớp)",
        len(train_indices),
        len(val_indices),
        seed,
        len(unique_classes),
    )

    return train_indices, val_indices


def split_train_val(
    dataset: Cifar10Dataset,
    val_size: int = 5_000,
    seed: int = DEFAULT_SEED,
    train_transform: Callable | None = None,
    val_transform: Callable | None = None,
) -> tuple[TransformSubset, TransformSubset]:
    """Tạo hai split train/validation với transform độc lập.

    Args:
        dataset: Dataset train CIFAR-10 chứa toàn bộ 50.000 mẫu.
        val_size: Số mẫu validation mong muốn.
        seed: Seed để tái lập kết quả.
        train_transform: Transform cho tập train (có augmentation).
        val_transform: Transform cho tập validation (deterministic).

    Returns:
        Tuple `(train_subset, val_subset)`.
    """
    train_indices, val_indices = stratified_indices(dataset.targets, val_size, seed)
    return (
        TransformSubset(dataset, train_indices.tolist(), train_transform),
        TransformSubset(dataset, val_indices.tolist(), val_transform),
    )
