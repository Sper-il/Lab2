"""Integration test: model + DataLoader that chay end-to-end (weights=None)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.data.dataloader import DataLoaderConfig, build_dataloaders
from src.models.factory import (
    ModelConfig,
    build_model,
    count_parameters,
    describe_model,
)
from src.utils.config import resolve_path

DATA_ROOT = resolve_path("data/raw")
HAS_DATA = (DATA_ROOT / "cifar-10-batches-py").exists()
requires_data = pytest.mark.skipif(
    not HAS_DATA, reason="CIFAR-10 chua duoc tai — chay scripts/download_data.py"
)

LIGHT_CONFIG = DataLoaderConfig(
    data_root="data/raw",
    image_size=64,
    batch_size=16,
    num_workers=0,
    val_size=100,
    download=False,
)


def test_end_to_end_batch_to_logits() -> None:
    """Batch that tu DataLoader -> logits (B, 10)."""
    if not HAS_DATA:
        pytest.skip("CIFAR-10 chua duoc tai")
    train_loader, _, _ = build_dataloaders(LIGHT_CONFIG)
    images, _ = next(iter(train_loader))

    model = build_model(ModelConfig(name="resnet18", weights=None))
    model.eval()
    with torch.no_grad():
        out = model(images)
    assert out.shape == (images.shape[0], 10)


def test_all_models_output_shape() -> None:
    """Ca 3 model deu cho output (1, 10) voi input (1, 3, 224, 224)."""
    dummy = torch.randn(1, 3, 224, 224)
    for name in ("resnet18", "vgg11", "densenet121"):
        model = build_model(ModelConfig(name=name, weights=None))
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        assert tuple(out.shape) == (1, 10), name


def test_frozen_backbone_gradients_none() -> None:
    """Sau backward, .grad cua backbone van None (da freeze)."""
    model = build_model(
        ModelConfig(name="resnet18", strategy="freeze_backbone", weights=None)
    )
    model.train()
    out = model(torch.randn(4, 3, 224, 224))
    loss = out.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if name.startswith("fc."):
            continue
        assert param.grad is None, f"{name} co grad mac du da freeze"


def test_head_gradients_not_none() -> None:
    """Sau backward, .grad cua head phai khac None."""
    model = build_model(
        ModelConfig(name="resnet18", strategy="freeze_backbone", weights=None)
    )
    model.train()
    out = model(torch.randn(4, 3, 224, 224))
    out.sum().backward()

    head_grads = [p.grad for n, p in model.named_parameters() if n.startswith("fc.")]
    assert head_grads, "khong tim thay param head"
    assert all(g is not None for g in head_grads)


def test_checkpoint_save_load(tmp_path) -> None:
    """Save -> load -> cung output."""
    model = build_model(ModelConfig(name="resnet18", weights=None))
    ckpt = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt)

    reloaded = build_model(ModelConfig(name="resnet18", weights=None))
    reloaded.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))

    dummy = torch.randn(1, 3, 224, 224)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(model(dummy), reloaded(dummy))


def test_describe_model_contains_key_info() -> None:
    """Chuoi mo ta phai chua ten lop + so params."""
    model = build_model(ModelConfig(name="vgg11", weights=None))
    text = describe_model(model, full=False)
    assert "VGG" in text
    assert "params" in text
    stats = count_parameters(model)
    assert stats["params_total"] > 100_000_000
