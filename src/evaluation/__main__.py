"""CLI danh gia M4: python -m src.evaluation --config configs/eval_config.yaml --checkpoint <best.pt>."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from src.data.dataloader import DataLoaderConfig, build_dataloaders
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualize import plot_confusion_matrix, plot_per_class_f1
from src.models.factory import ModelConfig, build_model
from src.utils.config import load_config, resolve_path
from src.utils.runs import setup_logging, write_json

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Danh gia model tren test set (M4)")
    p.add_argument("--config", default="configs/eval_config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    raw = load_config(args.config)
    cfg = dict(raw.get("evaluation", raw))

    ckpt_path = resolve_path(args.checkpoint or cfg.get("checkpoint", ""))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Khong tim thay checkpoint: {ckpt_path}")

    model_name = cfg.get("model", "resnet18")
    strategy = cfg.get("strategy", "freeze_backbone")
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 4))
    image_size = int(cfg.get("image_size", 224))
    device = cfg.get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent.parent / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    train_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = build_model(
        ModelConfig(
            name=train_cfg.get("model", model_name),
            num_classes=train_cfg.get("num_classes", 10),
            strategy="from_scratch",  # giu nguyen requires_grad, chi load weights
            weights=None,
        )
    )
    model.load_state_dict(state)
    model.to(device)

    _, _, test_loader = build_dataloaders(
        DataLoaderConfig(
            batch_size=batch_size, num_workers=num_workers, image_size=image_size
        )
    )

    result = evaluate_model(model, test_loader, device=device, n_classes=10)
    schema = result.to_json_dict()

    metrics = {
        "model": train_cfg.get("model", model_name),
        "strategy": train_cfg.get("strategy", strategy),
        "checkpoint": str(ckpt_path),
        "test": schema,
        "train_summary": {
            "epochs": train_cfg.get("epochs", 0),
            "best_val_accuracy": ckpt.get("best_val_acc", 0.0)
            if isinstance(ckpt, dict) else 0.0,
            "best_epoch": ckpt.get("best_epoch", -1)
            if isinstance(ckpt, dict) else -1,
        },
    }
    write_json(metrics, out_dir / "metrics.json")

    if cfg.get("save_plots", True):
        plot_confusion_matrix(
            schema["confusion_matrix"], class_names=CIFAR10_CLASSES,
            out_path=out_dir / "confusion_matrix.png",
        )
        plot_per_class_f1(schema["per_class"], out_path=out_dir / "per_class_f1.png")

    print(f"Test accuracy: {schema['accuracy']:.4f} | macro-F1: {schema['macro_f1']:.4f}")
    print(f"Artifact: {out_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
