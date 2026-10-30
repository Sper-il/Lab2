"""Vong lap training cho M3 (phuc vu de bai O10).

1 epoch (bam sat slide 8-9 Chuong 3):
    optimizer.zero_grad() -> forward -> loss -> backward -> step
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.training.loss import build_criterion
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.tensorboard_utils import (
    log_grad_norms,
    log_images,
    log_scalars,
    log_weight_histograms,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Cau hinh training (moi hyperparameter chinh duoc tu YAML/CLI - E2)."""

    model: str = "resnet18"
    strategy: str = "fine_tune"
    freeze_until: int = 1
    freeze_head_full: bool = False
    weights: str | None = "DEFAULT"
    num_classes: int = 10
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "none"
    loss: str = "cross_entropy"
    label_smoothing: float = 0.0
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    image_size: int = 224
    early_stopping_patience: int = 0
    log_images: bool = True
    log_histograms: bool = True
    log_graph: bool = False

    def resolved_device(self) -> str:
        if self.device.strip().lower() == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device


class Trainer:
    """Chua toan bo vong lap train/val + checkpoint + TensorBoard (E3, O10)."""

    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        run_dir: Path | None = None,
        writer: SummaryWriter | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir) if run_dir is not None else Path("results/runs")
        self.device = config.resolved_device()
        self.model.to(self.device)

        self.criterion = build_criterion(config.loss, config.label_smoothing)
        self.optimizer = build_optimizer(
            model,
            name=config.optimizer,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
        )
        self.scheduler = build_scheduler(
            self.optimizer, name=config.scheduler, epochs=config.epochs
        )
        self.writer = writer
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = -1
        self._no_improve = 0
        self.stopped_early = False

    # -- core loop ------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        logged_images = False
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device).long().reshape(-1)

            self.optimizer.zero_grad()  # 1
            outputs = self.model(images)  # 2 forward
            loss = self.criterion(outputs, labels)  # 3 loss
            loss.backward()  # 4 backward
            self.optimizer.step()  # 5 step

            batch = labels.size(0)
            total_loss += float(loss.item()) * batch
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += batch

            if (
                self.writer is not None
                and self.config.log_images
                and not logged_images
            ):
                try:
                    log_images(self.writer, images.detach().cpu(), epoch)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Khong log duoc images: %s", exc)
                logged_images = True

        return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device).long().reshape(-1)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            batch = labels.size(0)
            total_loss += float(loss.item()) * batch
            correct += int((outputs.argmax(1) == labels).sum().item())
            total += batch
        return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}

    def save_checkpoint(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.config.__dict__,
                "best_val_acc": self.best_val_acc,
                "best_epoch": self.best_epoch,
            },
            path,
        )
        return path

    def train(self) -> dict[str, Any]:
        ckpt_dir = self.run_dir / "checkpoints"
        for epoch in range(self.config.epochs):
            start = time.perf_counter()
            train_m = self.train_one_epoch(epoch)
            val_m = self.validate()
            elapsed = time.perf_counter() - start
            lr_now = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_m["loss"])
            self.history["train_acc"].append(train_m["acc"])
            self.history["val_loss"].append(val_m["loss"])
            self.history["val_acc"].append(val_m["acc"])
            self.history["lr"].append(lr_now)

            if self.writer is not None:
                log_scalars(
                    self.writer,
                    epoch,
                    {
                        "train/loss": train_m["loss"],
                        "train/accuracy": train_m["acc"],
                        "val/loss": val_m["loss"],
                        "val/accuracy": val_m["acc"],
                        "lr": lr_now,
                        "epoch/time_seconds": elapsed,
                    },
                )
                try:
                    log_grad_norms(self.writer, self.model, epoch)
                    if self.config.log_histograms and (epoch == 0 or epoch == self.config.epochs - 1):
                        log_weight_histograms(self.writer, self.model, epoch)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Khong log duoc histogram/grad: %s", exc)

            # scheduler step (plateau dung val acc, con lai step moi epoch)
            if self.scheduler is not None:
                if "ReduceLROnPlateau" in type(self.scheduler).__name__:
                    self.scheduler.step(val_m["acc"])
                else:
                    self.scheduler.step()

            improved = val_m["acc"] > self.best_val_acc
            if improved:
                self.best_val_acc = val_m["acc"]
                self.best_epoch = epoch
                self._no_improve = 0
                self.save_checkpoint(ckpt_dir / "best.pt")
            else:
                self._no_improve += 1

            LOGGER.info(
                "epoch %d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.2e | %.1fs%s",
                epoch + 1,
                self.config.epochs,
                train_m["loss"],
                train_m["acc"],
                val_m["loss"],
                val_m["acc"],
                lr_now,
                elapsed,
                " *best*" if improved else "",
            )

            if (
                self.config.early_stopping_patience > 0
                and self._no_improve >= self.config.early_stopping_patience
            ):
                LOGGER.info("Early stopping o epoch %d.", epoch + 1)
                self.stopped_early = True
                break

        # luon co checkpoint ke ca val acc = 0
        best_path = ckpt_dir / "best.pt"
        if not best_path.exists():
            self.save_checkpoint(best_path)
        self.save_checkpoint(ckpt_dir / "last.pt")
        return {
            "best_val_accuracy": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "epochs_run": len(self.history["train_loss"]),
            "stopped_early": self.stopped_early,
            "history": self.history,
        }
