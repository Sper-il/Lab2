"""CLI cho module data: dựng DataLoader và xuất báo cáo data pipeline.

Chạy:
    .\\.venv\\Scripts\\python.exe -m src.data --config configs/data_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from src.data.dataloader import DataLoaderConfig, build_dataloaders, dataset_summary
from src.utils.config import load_config
from src.utils.runs import (
    create_run_dir,
    save_config_snapshot,
    setup_logging,
    write_json,
)

LOGGER = logging.getLogger("src.data")


def _build_loader_config(data_section: dict[str, Any]) -> DataLoaderConfig:
    """Chuyển section `data` trong YAML thành `DataLoaderConfig`.

    Args:
        data_section: Nội dung section `data` của config.

    Returns:
        `DataLoaderConfig` đã validate qua dataclass.

    Raises:
        ValueError: Nếu config có trường không hợp lệ.
    """
    known_fields = set(DataLoaderConfig.__dataclass_fields__)
    unknown = set(data_section) - known_fields
    if unknown:
        raise ValueError(
            f"Trường không hợp lệ trong config section 'data': {sorted(unknown)}"
        )
    return DataLoaderConfig(**data_section)


def _validate_split(summary: dict[str, Any], val_size: int) -> None:
    """Kiểm tra tính nhất quán của các split.

    Args:
        summary: Kết quả từ `dataset_summary`.
        val_size: Số mẫu validation kỳ vọng.

    Raises:
        AssertionError: Nếu số mẫu không khớp kỳ vọng.
    """
    train_samples = summary["train"]["num_samples"]
    val_samples = summary["val"]["num_samples"]
    test_samples = summary["test"]["num_samples"]

    assert train_samples + val_samples == 50_000, (
        f"train + val = {train_samples + val_samples}, kỳ vọng 50000"
    )
    assert val_samples == val_size, f"val = {val_samples}, kỳ vọng {val_size}"
    assert test_samples == 10_000, f"test = {test_samples}, kỳ vọng 10000"


def main(argv: list[str] | None = None) -> int:
    """Entry point của CLI data module.

    Args:
        argv: Danh sách argument (mặc định lấy từ `sys.argv`).

    Returns:
        Exit code (0 = thành công).
    """
    parser = argparse.ArgumentParser(description="Dựng data pipeline cho CIFAR-10")
    parser.add_argument(
        "--config",
        default="configs/data_config.yaml",
        help="Đường dẫn file config YAML",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Ghi artifact vào thư mục run này thay vì tạo thư mục mới",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra config, không tạo run directory",
    )
    args = parser.parse_args(argv)

    run_dir: Path | None = None
    if not args.dry_run:
        run_dir = Path(args.run_dir) if args.run_dir else create_run_dir()
        setup_logging(run_dir)
    else:
        setup_logging()

    LOGGER.info("=" * 70)
    LOGGER.info("MODULE M1 — DATA PIPELINE (CIFAR-10)")
    LOGGER.info("=" * 70)

    try:
        base_config = load_config(args.config)
        data_section = base_config.get("data", {})
        loader_config = _build_loader_config(data_section)

        LOGGER.info("Config: %s", args.config)
        LOGGER.info("Data root: %s", loader_config.data_root)
        LOGGER.info(
            "image_size=%d | batch_size=%d | val_size=%d | seed=%d | augment=%s",
            loader_config.image_size,
            loader_config.batch_size,
            loader_config.val_size,
            loader_config.seed,
            loader_config.augment,
        )

        train_loader, val_loader, test_loader = build_dataloaders(loader_config)

        summary = dataset_summary(train_loader, val_loader, test_loader)
        _validate_split(summary, loader_config.val_size)

        for name, info in summary.items():
            LOGGER.info(
                "%-5s | %6d mẫu | %4d batch | image %s %s | label %s",
                name,
                info["num_samples"],
                info["num_batches"],
                info["image_shape"],
                info["image_dtype"],
                info["label_shape"],
            )

        if run_dir is not None:
            save_config_snapshot(base_config, run_dir / "config.yaml")
            report = {
                "module": "M1_data_pipeline",
                "config_path": args.config,
                "loader_config": {
                    "data_root": loader_config.data_root,
                    "image_size": loader_config.image_size,
                    "batch_size": loader_config.batch_size,
                    "num_workers_requested": loader_config.num_workers,
                    "num_workers_effective": train_loader.num_workers,
                    "val_size": loader_config.val_size,
                    "seed": loader_config.seed,
                    "augment": loader_config.augment,
                    "in_memory": loader_config.in_memory,
                },
                "splits": summary,
                "validation": {
                    "total_train_plus_val": summary["train"]["num_samples"]
                    + summary["val"]["num_samples"],
                    "expected_total_train_plus_val": 50_000,
                    "test_samples": summary["test"]["num_samples"],
                    "status": "PASS",
                },
            }
            write_json(report, run_dir / "metrics.json")
            LOGGER.info("Đã ghi artifact: %s", run_dir)
            LOGGER.info("  - config.yaml")
            LOGGER.info("  - metrics.json")
            LOGGER.info("  - logs.txt")
        else:
            LOGGER.info("Dry-run: không ghi artifact.")

        LOGGER.info("=" * 70)
        LOGGER.info("M1 DATA PIPELINE - DONE")
        LOGGER.info("=" * 70)
        return 0

    except Exception as exc:  # noqa: BLE001 - CLI cần báo lỗi rõ ràng
        LOGGER.error("M1 thất bại: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
