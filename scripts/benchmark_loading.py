"""Đo throughput đọc ảnh để quyết định có nên cache ảnh đã resize xuống disk.

Chạy: .\\.venv\\Scripts\\python.exe scripts\\benchmark_loading.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataloader import DataLoaderConfig, build_dataloaders  # noqa: E402
from src.utils.runs import setup_logging  # noqa: E402

LOGGER = logging.getLogger("benchmark")


def main() -> int:
    """Đo số ảnh/giây khi đọc + transform on-the-fly."""
    setup_logging()
    LOGGER.info("=" * 64)
    LOGGER.info("BENCHMARK - data loading throughput")
    LOGGER.info("=" * 64)

    for workers in (0, 4):
        config = DataLoaderConfig(
            data_root="data/raw",
            image_size=224,
            batch_size=64,
            num_workers=workers,
            val_size=5_000,
        )
        train_loader, _, _ = build_dataloaders(config)

        # Bỏ qua 5 batch đầu (warm-up: worker spawn, cache)
        iterator = iter(train_loader)
        for _ in range(5):
            next(iterator)

        n_batches = 20
        start = time.perf_counter()
        count = 0
        for _ in range(n_batches):
            images, _ = next(iterator)
            count += images.shape[0]
        elapsed = time.perf_counter() - start

        rate = count / elapsed
        LOGGER.info(
            "workers=%d | %d anh / %.2fs = %.1f anh/s | 1 epoch 45k = %.1f phut",
            workers,
            count,
            elapsed,
            rate,
            45_000 / rate / 60,
        )

    # Ước lượng kích thước cache uint8 ở 224x224
    size = 50_000 * 224 * 224 * 3
    LOGGER.info("Cache 224x224 uint8 cho 50k anh: %.1f GB", size / 1024**3)

    LOGGER.info("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
