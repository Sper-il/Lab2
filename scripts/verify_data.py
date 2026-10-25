"""Kiểm tra tính toàn vẹn của CIFAR-10 sau khi tải.

Chạy: .\\.venv\\Scripts\\python.exe scripts\\verify_data.py
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

from torchvision.datasets import CIFAR10

LOGGER = logging.getLogger("verify_data")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    """Load CIFAR-10, kiểm tra số lượng mẫu, phân bố lớp và shape ảnh."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if not (RAW_DIR / "cifar-10-batches-py").exists():
        LOGGER.error("Thiếu thư mục cifar-10-batches-py trong %s", RAW_DIR)
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
