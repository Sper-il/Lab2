"""Tien ich TensorBoard cho M3 (phuc vu de bai E3)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

LOGGER = logging.getLogger(__name__)


def create_writer(log_dir: str | Path) -> SummaryWriter:
    """Tao SummaryWriter trong thu muc chi dinh."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(path))


def log_scalars(writer: SummaryWriter, step: int, scalars: dict[str, float]) -> None:
    """Ghi cac scalar (train/loss, val/accuracy, lr, ...) len TensorBoard."""
    for tag, value in scalars.items():
        writer.add_scalar(tag, float(value), step)


def log_weight_histograms(writer: SummaryWriter, model: nn.Module, step: int) -> None:
    """Ghi histogram trong so de phat hien exploding/vanishing."""
    for name, param in model.named_parameters():
        if param.requires_grad and param.data.numel():
            writer.add_histogram(f"model/{name}", param.data, step)


def log_grad_norms(writer: SummaryWriter, model: nn.Module, step: int) -> None:
    """Ghi do lon gradient trung binh moi epoch."""
    total_sq = 0.0
    count = 0
    for param in model.parameters():
        if param.grad is not None:
            total_sq += float(param.grad.data.norm(2).item()) ** 2
            count += 1
    if count:
        writer.add_scalar("model/grad_norm", total_sq**0.5, step)


def log_images(
    writer: SummaryWriter,
    images: torch.Tensor,
    step: int,
    tag: str = "samples/train_images",
    max_images: int = 8,
) -> None:
    """Ghi anh mau (da denormalize ve [0,1]) len TensorBoard."""
    from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    batch = images.detach().cpu()[:max_images].clone()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    batch = (batch * std + mean).clamp(0, 1)
    writer.add_images(tag, batch, step)


def log_graph(
    writer: SummaryWriter, model: nn.Module, input_shape: tuple[int, ...] = (1, 3, 224, 224)
) -> None:
    """Ghi model graph (yeu cau O2 mo rong). Tat mac dinh vi cham."""
    try:
        model.eval()
        with torch.no_grad():
            writer.add_graph(model, torch.zeros(*input_shape))
    except Exception as exc:  # pragma: no cover - tuy moi truong
        LOGGER.warning("Khong ghi duoc model graph: %s", exc)


def tb_config() -> dict[str, Any]:
    """Cac tag TensorBoard chuan cua project (de test kiem tra)."""
    return {
        "scalars": ["train/loss", "train/accuracy", "val/loss", "val/accuracy", "lr"],
        "histograms": ["model/param_histograms", "model/grad_norms"],
        "images": ["samples/train_images"],
    }
