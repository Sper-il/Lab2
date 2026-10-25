"""Chuẩn bị dữ liệu đã xử lý và file split cho CIFAR-10.

Sinh ra hai thư mục theo convention của project:

- ``data/processed/`` — ảnh + nhãn ở dạng ``.npy`` (uint8) kèm metadata.
  Nhanh hơn file pickle của torchvision khi nạp lại, và tách bạch phần
  "dữ liệu đã xử lý sạch" với phần "dữ liệu gốc".
- ``data/splits/`` — chỉ số train/val/test dưới dạng ``.npz`` + ``.json``,
  để mọi experiment dùng chung một cách chia tách duy nhất.

Chạy:
    .\\.venv\\Scripts\\python.exe -m src.data.prepare
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

from src.data.split import stratified_indices
from src.utils.config import project_root, resolve_path
from src.utils.runs import setup_logging

LOGGER = logging.getLogger("src.data.prepare")

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_SPLITS_DIR = "data/splits"
DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 5_000


def _write_npy_compressed(path: Path, array: np.ndarray) -> int:
    """Ghi mảng numpy ra file `.npy` (nén nhẹ nếu có thể).

    Args:
        path: Đường dẫn file output.
        array: Mảng cần ghi.

    Returns:
        Kích thước file sau khi ghi (byte).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array, allow_pickle=False)
    return path.stat().st_size


def export_processed(
    raw_dir: Path,
    processed_dir: Path,
) -> dict:
    """Xuất ảnh + nhãn CIFAR-10 sang `data/processed/` dạng `.npy`.

    Giữ nguyên ảnh gốc 32×32 uint8 — **không** resize về 224 ở bước này.
    Resize phải diễn ra trong transform lúc train vì augmentation (RandomCrop)
    cần thao tác trên ảnh gốc 32×32.

    Args:
        raw_dir: Thư mục chứa CIFAR-10 gốc.
        processed_dir: Thư mục output cho dữ liệu đã xử lý.

    Returns:
        Dict thông tin các file đã ghi.

    Raises:
        FileNotFoundError: Nếu thư mục dữ liệu gốc không tồn tại.
    """
    if not (raw_dir / "cifar-10-batches-py").exists():
        raise FileNotFoundError(
            f"Không tìm thấy CIFAR-10 tại {raw_dir}. "
            "Chạy scripts/download_data.py trước."
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    written: dict = {}
    for split, is_train in (("train", True), ("test", False)):
        dataset = CIFAR10(root=str(raw_dir), train=is_train, download=False)

        images = np.asarray(dataset.data, dtype=np.uint8)
        labels = np.asarray(dataset.targets, dtype=np.int64)

        images_path = processed_dir / f"{split}_images.npy"
        labels_path = processed_dir / f"{split}_labels.npy"

        images_bytes = _write_npy_compressed(images_path, images)
        labels_bytes = _write_npy_compressed(labels_path, labels)

        written[split] = {
            "images_file": images_path.name,
            "labels_file": labels_path.name,
            "num_samples": int(images.shape[0]),
            "image_shape": list(images.shape),
            "images_bytes": images_bytes,
            "labels_bytes": labels_bytes,
        }

        LOGGER.info(
            "%s: %d mẫu -> %s (%.1f MB) + %s",
            split,
            images.shape[0],
            images_path.name,
            images_bytes / 1024**2,
            labels_path.name,
        )

    meta = {
        "dataset": "CIFAR-10",
        "classes": list(dataset.classes),
        "num_classes": 10,
        "source": "torchvision.datasets.CIFAR10",
        "raw_dir": str(raw_dir),
        "note": (
            "Anh goc 32x32 uint8. Khong resize o 224 tai buoc nay - "
            "RandomCrop can thao tac tren anh goc 32x32."
        ),
        "splits": written,
    }
    meta_path = processed_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)
    written["metadata_file"] = meta_path.name

    LOGGER.info("Metadata -> %s", meta_path)
    return written


def export_splits(
    raw_dir: Path,
    splits_dir: Path,
    val_size: int = DEFAULT_VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Xuất chỉ số train/val/test sang `data/splits/`.

    Tập test giữ nguyên 10.000 mẫu chính thức của CIFAR-10.

    Args:
        raw_dir: Thư mục chứa CIFAR-10 gốc.
        splits_dir: Thư mục output cho file split.
        val_size: Số mẫu validation lấy từ tập train.
        seed: Seed cho thuật toán chia phân tầng.

    Returns:
        Dict thông tin các file đã ghi.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CIFAR10(root=str(raw_dir), train=True, download=False)
    targets = np.asarray(train_dataset.targets, dtype=np.int64)

    train_indices, val_indices = stratified_indices(targets, val_size, seed)
    test_indices = np.arange(len(CIFAR10(root=str(raw_dir), train=False).targets))

    npz_path = splits_dir / f"cifar10_split_seed{seed}.npz"
    np.savez_compressed(
        npz_path,
        train_idx=train_indices,
        val_idx=val_indices,
        test_idx=test_indices,
    )

    # Bản JSON để đọc bằng mắt và diff được trong git
    summary = {
        "dataset": "CIFAR-10",
        "strategy": "stratified",
        "seed": seed,
        "val_size": val_size,
        "num_classes": 10,
        "files": {
            "train": "train_idx",
            "val": "val_idx",
            "test": "test_idx",
        },
        "counts": {
            "train": int(len(train_indices)),
            "val": int(len(val_indices)),
            "test": int(len(test_indices)),
        },
        "class_names": list(train_dataset.classes),
    }
    json_path = splits_dir / f"cifar10_split_seed{seed}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    LOGGER.info(
        "Split: train=%d val=%d test=%d (seed=%d)",
        len(train_indices),
        len(val_indices),
        len(test_indices),
        seed,
    )
    LOGGER.info("  -> %s", npz_path)
    LOGGER.info("  -> %s", json_path)

    return {
        "npz_file": npz_path.name,
        "json_file": json_path.name,
        "counts": summary["counts"],
        "seed": seed,
        "val_size": val_size,
    }


def verify_splits(splits_dir: Path, npz_name: str, raw_dir: Path) -> dict:
    """Kiểm tra tính nhất quán của file split vừa ghi.

    Lưu ý quan trọng: `train_idx` và `test_idx` thuộc **hai không gian chỉ số
    khác nhau** — `train_idx` trỏ vào tập train (0..49.999), còn `test_idx` trỏ
    vào tập test (0..9.999). Vì vậy không thể so sánh overlap trực tiếp giữa
    hai mảng. Thay vào đó ta so sánh **nội dung ảnh thật** (hash) giữa các
    split, phát hiện được cả rò rỉ ảnh lẫn chỉ số sai.

    Args:
        splits_dir: Thư mục chứa file split.
        npz_name: Tên file `.npz`.
        raw_dir: Thư mục chứa CIFAR-10 gốc để đối chiếu nội dung ảnh.

    Returns:
        Dict kết quả kiểm tra.

    Raises:
        AssertionError: Nếu có bất kỳ bất quy tắc nào bị vi phạm.
    """
    with np.load(splits_dir / npz_name) as data:
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        test_idx = data["test_idx"]

    assert len(train_idx) + len(val_idx) == 50_000, "train + val != 50000"
    assert len(test_idx) == 10_000, "test != 10000"
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0, "train/val overlap"
    assert train_idx.min() >= 0 and train_idx.max() < 50_000, "train idx out of range"
    assert test_idx.min() >= 0 and test_idx.max() < 10_000, "test idx out of range"

    # Đối chiếu nội dung ảnh: hash từng ảnh ở mỗi split rồi so sánh tập hash
    train_set = CIFAR10(root=str(raw_dir), train=True, download=False)
    test_set = CIFAR10(root=str(raw_dir), train=False, download=False)

    train_images = np.asarray(train_set.data, dtype=np.uint8)
    test_images = np.asarray(test_set.data, dtype=np.uint8)

    train_hashes = {train_images[i].tobytes() for i in train_idx.tolist()}
    val_hashes = {train_images[i].tobytes() for i in val_idx.tolist()}
    test_hashes = {test_images[i].tobytes() for i in test_idx.tolist()}

    assert len(train_hashes) == len(train_idx), "train co anh trung lap"
    assert len(val_hashes) == len(val_idx), "val co anh trung lap"
    assert len(test_hashes) == len(test_idx), "test co anh trung lap"

    train_val_overlap = len(train_hashes & val_hashes)
    train_test_overlap = len(train_hashes & test_hashes)
    val_test_overlap = len(val_hashes & test_hashes)

    assert train_val_overlap == 0, f"train/val trung {train_val_overlap} anh"
    assert train_test_overlap == 0, f"train/test trung {train_test_overlap} anh"
    assert val_test_overlap == 0, f"val/test trung {val_test_overlap} anh"

    LOGGER.info(
        "Kiem tra split: PASS (khong trung anh giua cac split, dung so luong)"
    )

    return {
        "train": int(len(train_idx)),
        "val": int(len(val_idx)),
        "test": int(len(test_idx)),
        "overlap_train_val": train_val_overlap,
        "overlap_train_test": train_test_overlap,
        "overlap_val_test": val_test_overlap,
        "status": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Danh sách argument.

    Returns:
        Exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Chuan bi data/processed va data/splits cho CIFAR-10"
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    setup_logging()

    root = project_root()
    raw_dir = resolve_path(args.raw_dir)
    processed_dir = resolve_path(args.processed_dir)
    splits_dir = resolve_path(args.splits_dir)

    LOGGER.info("=" * 70)
    LOGGER.info("PREPARE DATA - xuat data/processed va data/splits")
    LOGGER.info("=" * 70)

    try:
        export_processed(raw_dir, processed_dir)
        split_info = export_splits(raw_dir, splits_dir, args.val_size, args.seed)
        verification = verify_splits(splits_dir, split_info["npz_file"], raw_dir)

        LOGGER.info("=" * 70)
        LOGGER.info("PREPARE DATA - DONE")
        LOGGER.info("  processed: %s", processed_dir.relative_to(root))
        LOGGER.info("  splits:    %s", splits_dir.relative_to(root))
        LOGGER.info("  counts:    %s", verification)
        LOGGER.info("=" * 70)
        return 0

    except Exception as exc:  # noqa: BLE001 - CLI cần báo lỗi rõ ràng
        LOGGER.error("Prepare data that bai: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
