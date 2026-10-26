"""CLI kham pha model: in cau truc, dem parameters, kiem tra output shape.

Chay:
    python -m src.models --config configs/model_config.yaml
    python -m src.models --model vgg11 --strategy fine_tune --print-model
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.models.factory import (
    ModelConfig,
    build_model,
    count_parameters,
    describe_model,
)
from src.utils.config import load_config
from src.utils.runs import create_run_dir, setup_logging, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kham pha model pre-trained")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--model", default=None, help="Ghi de ten model trong config")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["freeze_backbone", "fine_tune", "from_scratch"],
    )
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument(
        "--print-model", action="store_true", help="In them toan bo cau truc"
    )
    args = parser.parse_args(argv)

    run_dir = create_run_dir()
    setup_logging(run_dir)

    raw = load_config(args.config)
    section = raw.get("model", raw)

    config = ModelConfig(
        name=args.model or section.get("name", "resnet18"),
        num_classes=args.num_classes or int(section.get("num_classes", 10)),
        strategy=args.strategy or section.get("strategy", "freeze_backbone"),
        weights=section.get("weights", "DEFAULT"),
        freeze_until=int(section.get("freeze_until", 1)),
        freeze_head_full=bool(section.get("freeze_head_full", False)),
    )

    model = build_model(config)
    stats = count_parameters(model)

    print(describe_model(model, full=args.print_model))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)

    expected = (1, config.num_classes)
    print(f"Output shape: {tuple(out.shape)} (ky vong {expected})")

    write_json(
        {
            "config": config.__dict__,
            "params": stats,
            "output_shape": list(out.shape),
        },
        run_dir / "model_info.json",
    )
    print(f"Artifact: {run_dir / 'model_info.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
