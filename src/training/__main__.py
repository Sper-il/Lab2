"""CLI train M3: python -m src.training --config configs/train_config.yaml."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.dataloader import DataLoaderConfig, build_dataloaders, dataset_summary
from src.models.factory import ModelConfig, build_model, count_parameters
from src.training.tensorboard_utils import create_writer, log_graph
from src.training.trainer import TrainConfig, Trainer
from src.utils.config import load_config, project_root, resolve_path
from src.utils.runs import create_run_dir, save_config_snapshot, setup_logging, write_json


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train model pre-trained CIFAR-10 (M3)")
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--models", nargs="*", default=None,
                   help="Ghi de 1 hoac nhieu model: resnet18 vgg11 densenet121")
    p.add_argument("--lr", nargs="*", type=float, default=None)
    p.add_argument("--batch-size", nargs="*", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--strategy", default=None,
                   choices=["freeze_backbone", "fine_tune", "from_scratch"])
    p.add_argument("--optimizer", default=None, choices=["adam", "adamw", "sgd"])
    p.add_argument("--scheduler", default=None,
                   choices=["none", "cosine", "step", "plateau"])
    p.add_argument("--no-tensorboard", action="store_true")
    return p.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_single(raw_train: dict, raw_tb: dict, run_parent: Path, tag: str) -> dict:
    cfg = TrainConfig(**raw_train)
    _set_seed(cfg.seed)
    torch.set_num_threads(max(1, ( __import__("os").cpu_count() or 8) - 4))

    run_dir = run_parent / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir)

    data_cfg = DataLoaderConfig(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        image_size=cfg.image_size,
        seed=cfg.seed,
    )
    train_loader, val_loader, _ = build_dataloaders(data_cfg)

    model = build_model(
        ModelConfig(
            name=cfg.model,
            num_classes=cfg.num_classes,
            strategy=cfg.strategy,
            weights=cfg.weights,
            freeze_until=cfg.freeze_until,
            freeze_head_full=cfg.freeze_head_full,
        )
    )
    params = count_parameters(model)

    writer = None
    if raw_tb.get("enabled", True):
        writer = create_writer(run_dir / "tb")
        if raw_tb.get("log_graph", False):
            log_graph(writer, model, (1, 3, cfg.image_size, cfg.image_size))

    trainer = Trainer(cfg, model, train_loader, val_loader,
                      run_dir=run_dir, writer=writer)
    summary = trainer.train()
    if writer is not None:
        writer.flush()
        writer.close()

    save_config_snapshot({"train": cfg.__dict__, "tensorboard": raw_tb},
                         run_dir / "config.yaml")
    write_json(
        {
            "model": cfg.model,
            "strategy": cfg.strategy,
            "params": params,
            "data": dataset_summary(train_loader, val_loader, val_loader),
            "train_summary": {
                "epochs": cfg.epochs,
                "epochs_run": summary["epochs_run"],
                "best_val_accuracy": summary["best_val_accuracy"],
                "best_epoch": summary["best_epoch"],
                "stopped_early": summary["stopped_early"],
            },
            "history": summary["history"],
        },
        run_dir / "metrics.json",
    )
    return {"run_dir": str(run_dir), **summary}


def main(argv=None) -> int:
    args = _parse_args(argv)
    raw = load_config(args.config)
    raw_train = dict(raw.get("train", raw))
    raw_tb = dict(raw.get("tensorboard", {}))
    if args.no_tensorboard:
        raw_tb["enabled"] = False

    # grid E2: models x lr x batch_size
    models = args.models or [raw_train.get("model", "resnet18")]
    lrs = args.lr or [raw_train.get("lr", 1e-3)]
    bss = args.batch_size or [raw_train.get("batch_size", 32)]

    combos = []
    for m in models:
        for lr in lrs:
            for bs in bss:
                c = dict(raw_train)
                c["model"] = m
                c["lr"] = lr
                c["batch_size"] = bs
                if args.epochs is not None:
                    c["epochs"] = args.epochs
                if args.image_size is not None:
                    c["image_size"] = args.image_size
                if args.strategy is not None:
                    c["strategy"] = args.strategy
                if args.optimizer is not None:
                    c["optimizer"] = args.optimizer
                if args.scheduler is not None:
                    c["scheduler"] = args.scheduler
                combos.append(c)

    parent = create_run_dir(root=project_root())
    print(f"Run parent: {parent} | so cau hinh: {len(combos)}")
    results = []
    for i, c in enumerate(combos):
        tag = f"{c['model']}-lr{c['lr']}-bs{c['batch_size']}-{i}"
        print(f"[{i+1}/{len(combos)}] {tag}")
        results.append(_train_single(c, raw_tb, parent, tag))

    write_json({"runs": results}, parent / "sweep_summary.json")
    print(f"Sweep summary: {parent / 'sweep_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
