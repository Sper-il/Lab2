"""Unit test cho hàm chia split phân tầng."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.split import stratified_indices


def _balanced_targets(n_per_class: int = 100, n_classes: int = 10) -> np.ndarray:
    """Tạo nhãn cân bằng cho test.

    Args:
        n_per_class: Số mẫu mỗi lớp.
        n_classes: Số lớp.

    Returns:
        Mảng nhãn int64.
    """
    return np.repeat(np.arange(n_classes), n_per_class)


def test_split_sizes() -> None:
    """Tổng số mẫu train + val phải bằng tổng ban đầu."""
    targets = _balanced_targets()
    train_idx, val_idx = stratified_indices(targets, val_size=200, seed=42)

    assert len(train_idx) + len(val_idx) == len(targets)
    assert len(val_idx) == 200
    assert len(train_idx) == len(targets) - 200


def test_no_index_overlap() -> None:
    """Chỉ số train và val không được trùng nhau."""
    targets = _balanced_targets()
    train_idx, val_idx = stratified_indices(targets, val_size=200, seed=42)

    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0


def test_stratified_class_distribution() -> None:
    """Mỗi lớp phải có đúng số mẫu val tương ứng (phân tầng)."""
    targets = _balanced_targets(n_per_class=100)
    _, val_idx = stratified_indices(targets, val_size=200, seed=42)

    _, counts = np.unique(targets[val_idx], return_counts=True)
    assert counts.tolist() == [20] * 10


def test_reproducible_with_same_seed() -> None:
    """Cùng seed phải cho cùng kết quả."""
    targets = _balanced_targets()
    first_train, first_val = stratified_indices(targets, val_size=200, seed=42)
    second_train, second_val = stratified_indices(targets, val_size=200, seed=42)

    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_val, second_val)


def test_different_seed_gives_different_split() -> None:
    """Seed khác phải cho split khác."""
    targets = _balanced_targets()
    _, first_val = stratified_indices(targets, val_size=200, seed=42)
    _, second_val = stratified_indices(targets, val_size=200, seed=7)

    assert not np.array_equal(first_val, second_val)


def test_remainder_distribution() -> None:
    """Phần dư của phép chia được phân bố cho các lớp đầu."""
    targets = _balanced_targets(n_per_class=100)
    _, val_idx = stratified_indices(targets, val_size=25, seed=42)

    _, counts = np.unique(targets[val_idx], return_counts=True)
    # 25 / 10 = 2 remainder 5 -> 5 lớp đầu có 3 mẫu, 5 lớp sau có 2 mẫu
    assert counts.tolist() == [3, 3, 3, 3, 3, 2, 2, 2, 2, 2]


def test_rejects_val_size_zero() -> None:
    """val_size <= 0 phải ném ValueError."""
    with pytest.raises(ValueError):
        stratified_indices(_balanced_targets(), val_size=0)


def test_rejects_val_size_too_large() -> None:
    """val_size >= số mẫu phải ném ValueError."""
    with pytest.raises(ValueError):
        stratified_indices(_balanced_targets(), val_size=1000)


def test_rejects_val_size_smaller_than_class_count() -> None:
    """val_size < số lớp phải ném ValueError."""
    with pytest.raises(ValueError):
        stratified_indices(_balanced_targets(), val_size=5)
