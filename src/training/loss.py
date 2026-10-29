"""Ham loss cho M3: CrossEntropy + LabelSmoothing + Focal."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss cho classification (gamma=2 mac dinh).

    Args:
        gamma: He so tap trung vao mau kho.
        weight: Trong so moi lop (optional).
    """

    def __init__(self, gamma: float = 2.0, weight=None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_criterion(name: str = "cross_entropy", label_smoothing: float = 0.0) -> nn.Module:
    """Dung loss theo ten (phuc vu de bai O9).

    Args:
        name: "cross_entropy" | "label_smoothing" | "focal".
        label_smoothing: He so smoothing khi name="label_smoothing".

    Returns:
        Module loss san sang dung trong training loop.

    Raises:
        ValueError: Neu ten loss khong ho tro.
    """
    key = name.strip().lower()
    if key == "cross_entropy":
        return nn.CrossEntropyLoss()
    if key == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing or 0.1)
    if key == "focal":
        return FocalLoss()
    raise ValueError(
        f"Loss khong ho tro: {name!r}. Chon: cross_entropy | label_smoothing | focal"
    )
