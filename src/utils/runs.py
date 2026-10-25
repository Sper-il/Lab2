"""Quản lý run directory, logging và ghi artifact cho mỗi lần chạy."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.utils.config import project_root

LOGGER = logging.getLogger(__name__)

RUNS_DIRNAME = Path("results") / "runs"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"


def create_run_dir(root: Path | None = None, timestamp: str | None = None) -> Path:
    """Tạo thư mục run mới theo convention `results/runs/<timestamp>/`.

    Args:
        root: Thư mục gốc project, mặc định lấy từ `project_root()`.
        timestamp: Chuỗi timestamp tùy chọn, mặc định lấy thời điểm hiện tại.

    Returns:
        Path tới thư mục run đã tạo.
    """
    base = (root or project_root()) / RUNS_DIRNAME
    base.mkdir(parents=True, exist_ok=True)

    stamp = timestamp or datetime.now().strftime(TIMESTAMP_FORMAT)
    run_dir = base / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}-{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True)
    return run_dir


def setup_logging(run_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Cấu hình logging ghi ra console và (tuỳ chọn) vào `logs.txt`.

    Console Windows mặc định dùng encoding cp1252, không encode được tiếng Việt
    có dấu. Vì vậy stdout được ép về UTF-8 với `errors="replace"` để log tiếng
    Việt không làm sập tiến trình.

    Args:
        run_dir: Thư mục run để ghi log file. `None` nghĩa là chỉ log console.
        level: Mức log tối thiểu.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - stream đặc biệt
                pass

    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stdout)]

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            run_dir / "logs.txt", encoding="utf-8", errors="replace"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    for handler in handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))
        root_logger.addHandler(handler)


def save_config_snapshot(config: dict[str, Any], path: Path) -> Path:
    """Ghi snapshot config YAML vào run directory.

    Args:
        config: Config cần lưu.
        path: Đường dẫn file output.

    Returns:
        Path tới file đã ghi.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    return path


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder hỗ trợ các kiểu số của numpy."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def write_json(data: dict[str, Any], path: Path) -> Path:
    """Ghi dict ra file JSON, hỗ trợ kiểu số của numpy.

    Args:
        data: Dữ liệu cần ghi.
        path: Đường dẫn file output.

    Returns:
        Path tới file đã ghi.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
    return path
