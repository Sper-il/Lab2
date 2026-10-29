"""Dung optimizer cho M3: chi nhan params trainable (phuc vu de bai O9)."""

from __future__ import annotations

import torch.nn as nn
import torch.optim as optim


def trainable_params(model: nn.Module):
    """Lay cac param co `requires_grad=True` (backbone freeze bi loai)."""
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """Dung optimizer theo ten.

    Args:
        model: Model can train.
        name: "adam" | "adamw" | "sgd".
        lr: Learning rate.
        weight_decay: He so weight decay.
        momentum: Momentum (chi dung cho sgd).

    Returns:
        Optimizer san sang dung.

    Raises:
        ValueError: Neu ten optimizer khong ho tro, hoac model khong co
            param trainable nao (VD: quen mo head sau khi freeze).
    """
    params = trainable_params(model)
    if not params:
        raise ValueError(
            "Model khong co param trainable nao. Kiem tra lai freeze/head."
        )
    key = name.strip().lower()
    if key == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(
        f"Optimizer khong ho tro: {name!r}. Chon: adam | adamw | sgd"
    )
