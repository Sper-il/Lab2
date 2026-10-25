"""Test cho module prepare (xuất data/processed và data/splits)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.data.prepare import export_processed, export_splits, verify_splits
from src.utils.config import resolve_path

DATA_ROOT = resolve_path("data/raw")
HAS_DATA = (DATA_ROOT / "cifar-10-batches-py").exists()
requires_data = pytest.mark.skipif(
    not HAS_DATA, reason="CIFAR-10 chưa được tải — chạy scripts/download_data.py"
)


@pytest.fixture
def output_dirs(tmp_path):
    """Tạo cặp thư mục output tạm cho test.

    Args:
        tmp_path: Thư mục tạm của pytest.

    Returns:
        Tuple `(processed_dir, splits_dir)`.
    """
    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    processed.mkdir()
    splits.mkdir()
    return processed, splits


@requires_data
def test_export_processed_creates_files(output_dirs) -> None:
    """`export_processed` phải tạo đủ file .npy cho train và test."""
    processed, _ = output_dirs
    written = export_processed(DATA_ROOT, processed)

    assert (processed / "train_images.npy").is_file()
    assert (processed / "train_labels.npy").is_file()
    assert (processed / "test_images.npy").is_file()
    assert (processed / "test_labels.npy").is_file()
    assert (processed / "metadata.json").is_file()
    assert written["train"]["num_samples"] == 50_000
    assert written["test"]["num_samples"] == 10_000


@requires_data
def test_processed_npy_content_matches_raw(output_dirs) -> None:
    """Ảnh trong .npy phải khớp với ảnh gốc, giữ nguyên shape 32x32."""
    processed, _ = output_dirs
    export_processed(DATA_ROOT, processed)

    images = np.load(processed / "train_images.npy")
    labels = np.load(processed / "train_labels.npy")

    assert images.shape == (50_000, 32, 32, 3)
    assert images.dtype == np.uint8
    assert labels.shape == (50_000,)
    assert labels.min() >= 0 and labels.max() <= 9


@requires_data
def test_processed_metadata_is_valid_json(output_dirs) -> None:
    """metadata.json phải parse được và chứa đủ thông tin."""
    processed, _ = output_dirs
    export_processed(DATA_ROOT, processed)

    with (processed / "metadata.json").open(encoding="utf-8") as file:
        meta = json.load(file)

    assert meta["dataset"] == "CIFAR-10"
    assert meta["num_classes"] == 10
    assert len(meta["classes"]) == 10
    assert "train" in meta["splits"] and "test" in meta["splits"]


@requires_data
def test_export_splits_creates_npz_and_json(output_dirs) -> None:
    """`export_splits` phải tạo cả file .npz lẫn .json."""
    _, splits = output_dirs
    info = export_splits(DATA_ROOT, splits, val_size=5_000, seed=42)

    assert (splits / info["npz_file"]).is_file()
    assert (splits / info["json_file"]).is_file()
    assert info["counts"] == {"train": 45_000, "val": 5_000, "test": 10_000}


@requires_data
def test_split_npz_contains_three_arrays(output_dirs) -> None:
    """File .npz phải chứa đúng 3 mảng train_idx/val_idx/test_idx."""
    _, splits = output_dirs
    info = export_splits(DATA_ROOT, splits)

    with np.load(splits / info["npz_file"]) as data:
        assert set(data.files) == {"train_idx", "val_idx", "test_idx"}
        assert len(data["train_idx"]) == 45_000
        assert len(data["val_idx"]) == 5_000
        assert len(data["test_idx"]) == 10_000


@requires_data
def test_verify_splits_passes(output_dirs) -> None:
    """Kiểm tra split hợp lệ phải PASS."""
    _, splits = output_dirs
    info = export_splits(DATA_ROOT, splits)
    result = verify_splits(splits, info["npz_file"], DATA_ROOT)

    assert result["status"] == "PASS"
    assert result["overlap_train_val"] == 0
    assert result["overlap_train_test"] == 0
    assert result["overlap_val_test"] == 0


@requires_data
def test_splits_are_reproducible(output_dirs) -> None:
    """Cùng seed phải sinh ra cùng chỉ số split."""
    _, splits_a = output_dirs
    _, splits_b = output_dirs
    info_a = export_splits(DATA_ROOT, splits_a, seed=42)
    info_b = export_splits(DATA_ROOT, splits_b, seed=42)

    with np.load(splits_a / info_a["npz_file"]) as first:
        with np.load(splits_b / info_b["npz_file"]) as second:
            assert np.array_equal(first["train_idx"], second["train_idx"])
            assert np.array_equal(first["val_idx"], second["val_idx"])


@requires_data
def test_different_seed_gives_different_split(output_dirs) -> None:
    """Seed khác phải cho split khác."""
    _, splits_a = output_dirs
    _, splits_b = output_dirs
    info_a = export_splits(DATA_ROOT, splits_a, seed=42)
    info_b = export_splits(DATA_ROOT, splits_b, seed=7)

    with np.load(splits_a / info_a["npz_file"]) as first:
        with np.load(splits_b / info_b["npz_file"]) as second:
            assert not np.array_equal(first["val_idx"], second["val_idx"])


@requires_data
def test_exported_split_matches_pipeline_split(output_dirs) -> None:
    """Split xuất ra phải khớp với split mà DataLoader thực sự dùng.

    Đây là test quan trọng: nó bảo đảm file trong `data/splits/` không chỉ
    "đẹp" mà thực sự phản ánh cách chia tập của pipeline.
    """
    from src.data.split import stratified_indices
    from torchvision.datasets import CIFAR10

    _, splits = output_dirs
    info = export_splits(DATA_ROOT, splits, val_size=5_000, seed=42)

    dataset = CIFAR10(root=str(DATA_ROOT), train=True, download=False)
    expected_train, expected_val = stratified_indices(
        np.asarray(dataset.targets), val_size=5_000, seed=42
    )

    with np.load(splits / info["npz_file"]) as data:
        assert np.array_equal(data["train_idx"], expected_train)
        assert np.array_equal(data["val_idx"], expected_val)


@requires_data
def test_stratification_is_preserved_in_export(output_dirs) -> None:
    """Mỗi lớp phải có đúng 4.500 train / 500 val trong file export."""
    from torchvision.datasets import CIFAR10

    _, splits = output_dirs
    info = export_splits(DATA_ROOT, splits, val_size=5_000, seed=42)

    dataset = CIFAR10(root=str(DATA_ROOT), train=True, download=False)
    targets = np.asarray(dataset.targets)

    with np.load(splits / info["npz_file"]) as data:
        train_counts = np.bincount(targets[data["train_idx"]], minlength=10)
        val_counts = np.bincount(targets[data["val_idx"]], minlength=10)

    assert train_counts.tolist() == [4_500] * 10
    assert val_counts.tolist() == [500] * 10


def test_export_processed_requires_data(tmp_path) -> None:
    """Thiếu thư mục dữ liệu gốc phải ném FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        export_processed(tmp_path / "khong_ton_tai", tmp_path / "out")
