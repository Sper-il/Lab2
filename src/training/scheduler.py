"""Dung learning-rate scheduler cho M3 (phuc vu de bai E2)."""

from __future__ import annotations

import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "none",
    epochs: int = 5,
    step_size: int = 2,
    gamma: float = 0.1,
) -> lr_scheduler.LRScheduler | None:
    """Dung scheduler theo ten.

    Args:
        optimizer: Optimizer can gan scheduler.
        name: "none" | "cosine" | "step" | "plateau".
        epochs: Tong so epoch (dung cho cosine).
        step_size: Chu ky giam lr (dung cho step).
        gamma: He so giam lr.

    Returns:
        Scheduler, hoac None neu name="none".

    Raises:
        ValueError: Neu ten scheduler khong ho tro.
    """
    key = name.strip().lower()
    if key == "none":
        return None
    if key == "cosine":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if key == "step":
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if key == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=gamma)
    raise ValueError(
        f"Scheduler khong ho tro: {name!r}. Chon: none | cosine | step | plateau"
    )
