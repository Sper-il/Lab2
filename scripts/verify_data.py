"""Kiểm tra tính toàn vẹn của CIFAR-10 sau khi tải.

Chạy: .\\.venv\\Scripts\\python.exe scripts\\verify_data.py
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections import Counter
from pathlib import Path

from torchvision.datasets import CIFAR10

LOGGER = logging.getLogger("verify_data")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# MD5 chính thức của CIFAR-10 do thư viện Toronto cung cấp
ARCHIVE_NAME = "cifar-10-python.tar.gz"
EXPECTED_MD5 = "c58f30108f718f92721af3b95e74349a"


def _md5(path: Path) -> str:
    """Tính MD5 của file theo chunk để không nạp hết vào RAM.

    Args:
        path: Đường dẫn file.

    Returns:
        Chuỗi hex 32 ký tự.
    """
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_archive() -> bool:
    """Kiểm tra MD5 của archive CIFAR-10 nếu archive còn tồn tại.

    Returns:
        `True` nếu hợp lệ hoặc không có archive để kiểm tra, `False` nếu sai.
    """
    archive = RAW_DIR / ARCHIVE_NAME
    if not archive.exists():
        LOGGER.info(
            "Không có %s trong %s — bỏ qua kiểm tra MD5.", ARCHIVE_NAME, RAW_DIR
        )
        return True

    actual = _md5(archive)
    if actual != EXPECTED_MD5:
        LOGGER.error(
            "MD5 archive SAI — file tải về bị hỏng hoặc không đúng CIFAR-10.\n"
            "  expected: %s\n  actual:   %s\n"
            "  Hãy tải lại từ https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
            EXPECTED_MD5,
            actual,
        )
        return False

    size_mb = archive.stat().st_size / 1024 / 1024
    LOGGER.info("MD5 archive OK: %s (%.1f MB)", actual, size_mb)
    return True


def main() -> int:
    """Load CIFAR-10, kiểm tra MD5 archive, số lượng mẫu, phân bố lớp, shape ảnh."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not (RAW_DIR / "cifar-10-batches-py").exists():
        LOGGER.error("Thiếu thư mục cifar-10-batches-py trong %s", RAW_DIR)
        return 1

    if not _verify_archive():
        return 1

    train_set = CIFAR10(root=str(RAW_DIR), train=True, download=False)
    test_set = CIFAR10(root=str(RAW_DIR), train=False, download=False)

    LOGGER.info("Train: %d mẫu | Test: %d mẫu", len(train_set), len(test_set))

    img, label = train_set[0]
    LOGGER.info("Shape ảnh đầu tiên: %s | label: %d", tuple(img.size), label)

    train_counts = Counter(train_set.targets)
    test_counts = Counter(test_set.targets)
    LOGGER.info("Phân bố lớp train: %s", dict(sorted(train_counts.items())))
    LOGGER.info("Phân bố lớp test:  %s", dict(sorted(test_counts.items())))
    LOGGER.info("Số lớp: %d | cân bằng: %s", len(train_counts),
                len(set(train_counts.values())) == 1)

    expected_train = {idx: 5000 for idx in range(10)}
    expected_test = {idx: 1000 for idx in range(10)}
    assert dict(train_counts) == expected_train, "Phân bố lớp train không đúng 5000/lớp"
    assert dict(test_counts) == expected_test, "Phân bố lớp test không đúng 1000/lớp"
    assert len(train_set) == 50_000 and len(test_set) == 10_000, "Số lượng mẫu sai"
    assert img.size == (32, 32), f"Shape ảnh sai: {img.size}"

    LOGGER.info("Tất cả kiểm tra PASS — dữ liệu sẵn sàng dùng.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
