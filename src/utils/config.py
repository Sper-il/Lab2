"""Nạp và kiểm tra config YAML, xác định đường dẫn gốc của project."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILENAME = "configs"


def project_root() -> Path:
    """Trả về thư mục gốc của project (chứa `src/`, `configs/`, `results/`).

    Returns:
        Path tuyệt đối tới thư mục gốc project.
    """
    return Path(__file__).resolve().parent.parent.parent


def resolve_path(path: str | Path) -> Path:
    """Chuyển đường dẫn tương đối thành đường dẫn tuyệt đối dựa trên project root.

    Args:
        path: Đường dẫn tương đối hoặc tuyệt đối.

    Returns:
        Path tuyệt đối.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def load_config(path: str | Path) -> dict[str, Any]:
    """Nạp file YAML config và trả về mapping.

    Args:
        path: Đường dẫn tới file YAML config.

    Returns:
        Nội dung config dạng dict.

    Raises:
        FileNotFoundError: Nếu file không tồn tại.
        ValueError: Nếu nội dung YAML không phải mapping.
    """
    config_path = resolve_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config phải là YAML mapping, nhận được: {type(config).__name__}")

    return config


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Gộp đệ quy hai dict, giá trị trong `override` thắng.

    Args:
        base: Dict gốc.
        override: Dict ghi đè.

    Returns:
        Dict mới, không mutate input.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
