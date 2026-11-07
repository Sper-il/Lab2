"""Demo 1 lệnh chạy được: suy luận checkpoint tốt nhất trên ảnh test thật.

Chạy:
    python -m src.demo --checkpoint models/checkpoints/vgg11-best.pt --num-samples 8
    python -m src.demo --export-samples   (xuất data/processed/demo/ 8 ảnh mẫu)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from src.data.dataset import Cifar10Dataset
from src.data.transforms import build_val_transform
from src.models.factory import ModelConfig, build_model
from src.utils.config import load_config, project_root, resolve_path
from src.utils.runs import write_json

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo suy luận checkpoint tốt nhất (M5 demo)")
    p.add_argument("--config", default="configs/eval_config.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--export-samples", action="store_true",
                   help="Xuất data/processed/demo/ rồi thoát")
    return p.parse_args(argv)


def export_demo_samples(num: int = 8) -> Path:
    """Xuất N ảnh test đầu tiên ra data/processed/demo/ (.png + labels.json)."""
    from PIL import Image

    import numpy as np

    root = project_root()
    processed = root / "data" / "processed"
    test_images = np.load(processed / "test_images.npy")  # [10000,32,32,3] uint8
    test_labels = np.load(processed / "test_labels.npy")
    demo_dir = processed / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    labels = {}
    for i in range(min(num, len(test_images))):
        Image.fromarray(test_images[i]).save(demo_dir / f"demo_{i:02d}.png")
        labels[f"demo_{i:02d}.png"] = {
            "label": int(test_labels[i]),
            "class": CIFAR10_CLASSES[int(test_labels[i])],
        }
    (demo_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    return demo_dir


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if reconfig is not None:
            try:
                reconfig(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    args = _parse_args(argv)
    if args.export_samples:
        out = export_demo_samples(args.num_samples)
        print(f"Demo samples: {out}")
        return 0

    raw = load_config(args.config) if Path(args.config).is_file() or resolve_path(args.config).is_file() else {}
    cfg = dict(raw.get("evaluation", raw)) if isinstance(raw, dict) else {}
    ckpt_path = resolve_path(args.checkpoint or cfg.get(
        "checkpoint",
        "results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/checkpoints/best.pt"))
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    train_cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    model = build_model(ModelConfig(
        name=train_cfg.get("model", cfg.get("model", "vgg11")),
        num_classes=train_cfg.get("num_classes", 10),
        strategy="from_scratch", weights=None))
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    data_root = resolve_path("data/raw")
    ds = Cifar10Dataset(root=str(data_root), train=False,
                        transform=build_val_transform(image_size=args.image_size),
                        download=False, in_memory=True)
    n = min(args.num_samples, len(ds))
    results = []
    with torch.inference_mode():
        for i in range(n):
            img, true = ds[i]
            batch = img.unsqueeze(0).to(device)
            start = time.perf_counter()
            logits = model(batch)
            if batch.is_cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            probs = torch.softmax(logits, dim=1)[0].cpu()
            top3 = torch.topk(probs, k=3)
            preds = [{"class": CIFAR10_CLASSES[int(c)],
                       "label": int(c),
                       "prob": float(p)} for c, p in zip(top3.indices, top3.values)]
            ok = preds[0]["label"] == int(true)
            print(f"[{i}] thật={CIFAR10_CLASSES[int(true)]} → "
                  f"dự đoán={preds[0]['class']} ({preds[0]['prob']:.3f}) "
                  f"{'✓' if ok else '✗'} | {elapsed_ms:.1f} ms")
            results.append({"index": i, "true_label": int(true),
                            "true_class": CIFAR10_CLASSES[int(true)],
                            "top3": preds, "correct": ok,
                            "latency_ms": elapsed_ms})

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else project_root() / "results" / "demo" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"checkpoint": str(ckpt_path), "device": device,
                "image_size": args.image_size, "num_samples": n,
                "accuracy": sum(r["correct"] for r in results) / max(1, n),
                "mean_latency_ms": sum(r["latency_ms"] for r in results) / max(1, n),
                "predictions": results}, out_dir / "predictions.json")
    write_json({"checkpoint": str(ckpt_path), "image_size": args.image_size}, out_dir / "config.json")
    rows = "".join(
        f"<tr><td>{r['index']}</td><td>{r['true_class']}</td>"
        f"<td>{r['top3'][0]['class']} ({r['top3'][0]['prob']:.3f})</td>"
        f"<td>{'✓' if r['correct'] else '✗'}</td><td>{r['latency_ms']:.1f}</td></tr>"
        for r in results)
    (out_dir / "index.html").write_text(
        f"<!DOCTYPE html><html lang='vi'><head><meta charset='utf-8'/>"
        f"<title>Demo {stamp}</title><style>body{{font-family:Arial;margin:24px}}"
        f"td,th{{border:1px solid #999;padding:4px 10px}}table{{border-collapse:collapse}}</style>"
        f"</head><body><h1>Demo checkpoint {ckpt_path.name} — {n} ảnh test</h1>"
        f"<table><tr><th>#</th><th>Nhãn thật</th><th>Dự đoán top-1</th><th>Đúng?</th>"
        f"<th>Latency (ms)</th></tr>{rows}</table>"
        f"<p>Chi tiết: predictions.json. Checkpoint: {ckpt_path}</p></body></html>",
        encoding="utf-8")
    print(f"Demo artifact: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
