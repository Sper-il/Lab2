"""Tải CIFAR-10 về data/raw/ bằng torchvision.

Chạy: .\\.venv\\Scripts\\python.exe scripts\\download_data.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from torchvision.datasets import CIFAR10

LOGGER = logging.getLogger("download_data")

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    """Tải và giải nén CIFAR-10, sau đó kiểm tra số lượng mẫu."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Đang tải CIFAR-10 vào %s", RAW_DIR)

    start = time.perf_counter()
    try:
        train_set = CIFAR10(root=str(RAW_DIR), train=True, download=True)
        test_set = CIFAR10(root=str(RAW_DIR), train=False, download=True)
    except Exception as exc:  # noqa: BLE001 - báo lỗi rõ ràng cho người dùng
        LOGGER.error("Không tải được CIFAR-10: %s", exc)
        LOGGER.error(
            "Tải thủ công từ https://www.cs.toronto.edu/~kriz/cifar.html "
            "rồi giải nén thư mục 'cifar-10-batches-py' vào %s",
            RAW_DIR,
        )
        return 1

    elapsed = time.perf_counter() - start
    LOGGER.info("Tải xong trong %.1f giây", elapsed)
    LOGGER.info("Train: %d mẫu, %d lớp", len(train_set), len(train_set.classes))
    LOGGER.info("Test:  %d mẫu", len(test_set))
    LOGGER.info("Các lớp: %s", ", ".join(train_set.classes))

    archive = RAW_DIR / "cifar-10-python.tar.gz"
    if archive.exists():
        size_mb = archive.stat().st_size / 1024 / 1024
        LOGGER.info("Archive giữ lại tại %s (%.1f MB)", archive, size_mb)

    return 0


if __name__ == "__main__":
    sys.exit(main())
