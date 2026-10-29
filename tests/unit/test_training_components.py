"""Test M3: loss/optimizer/scheduler/tensorboard (nhanh, khong can data that)."""

import torch
import torch.nn as nn
import pytest

from src.training.loss import build_criterion
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.tensorboard_utils import (
    create_writer,
    log_grad_norms,
    log_images,
    log_scalars,
    log_weight_histograms,
    tb_config,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _model(freeze=True):
    m = TinyNet()
    if freeze:
        for name, p in m.named_parameters():
            p.requires_grad = name.startswith("fc")
    return m


def test_cross_entropy_loss_runs():
    crit = build_criterion("cross_entropy")
    loss = crit(torch.randn(4, 10), torch.randint(0, 10, (4,)))
    assert float(loss.item()) > 0


def test_label_smoothing_runs():
    crit = build_criterion("label_smoothing", 0.1)
    loss = crit(torch.randn(4, 10), torch.randint(0, 10, (4,)))
    assert float(loss.item()) > 0


def test_focal_loss_runs():
    crit = build_criterion("focal")
    loss = crit(torch.randn(4, 10), torch.randint(0, 10, (4,)))
    assert float(loss.item()) > 0


def test_unknown_loss_raises():
    with pytest.raises(ValueError):
        build_criterion("nope")


def test_optimizer_only_trainable():
    m = _model()
    opt = build_optimizer(m, name="sgd", lr=0.01)
    n_params = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_params == sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_optimizer_lr_and_variants():
    m = _model()
    assert build_optimizer(m, name="adam", lr=1e-3).param_groups[0]["lr"] == pytest.approx(1e-3)
    assert build_optimizer(m, name="adamw", lr=1e-3) is not None
    with pytest.raises(ValueError):
        build_optimizer(m, name="nope")


def test_scheduler_cosine_decays():
    m = _model()
    opt = build_optimizer(m, lr=0.1)
    sch = build_scheduler(opt, name="cosine", epochs=4)
    lrs = []
    for _ in range(4):
        lrs.append(opt.param_groups[0]["lr"])
        sch.step()
    assert lrs[0] > lrs[-1] > 0


def test_scheduler_none_and_errors():
    m = _model()
    opt = build_optimizer(m)
    assert build_scheduler(opt, name="none") is None
    assert build_scheduler(opt, name="step") is not None
    assert build_scheduler(opt, name="plateau") is not None
    with pytest.raises(ValueError):
        build_scheduler(opt, name="nope")


def test_tensorboard_tags_and_writes(tmp_path):
    tags = tb_config()
    assert "train/loss" in tags["scalars"] and "val/accuracy" in tags["scalars"]
    w = create_writer(tmp_path / "tb")
    log_scalars(w, 0, {"train/loss": 1.0, "val/accuracy": 0.5, "lr": 1e-3})
    m = _model()
    out = m(torch.randn(2, 8))
    out.sum().backward()
    log_grad_norms(w, m, 0)
    log_weight_histograms(w, m, 0)
    log_images(w, torch.rand(2, 3, 8, 8), 0)
    w.flush()
    w.close()
    assert any((tmp_path / "tb").glob("events*"))
