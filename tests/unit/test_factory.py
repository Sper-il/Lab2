"""Unit test cho model factory (khong can weights ImageNet, khong can data that)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.factory import (
    ModelConfig,
    build_model,
    count_parameters,
    describe_model,
    freeze_backbone,
    replace_head,
    resolve_weights,
    unfreeze_last,
)


def _build_raw(name: str) -> nn.Module:
    """Dung model goc khong weights de test nhanh (< 3s moi test)."""
    from src.models.registry import resolve_spec

    spec = resolve_spec(name)
    return spec.builder(weights=None)


def test_build_model_returns_cnn() -> None:
    """build_model phai tra ve nn.Module."""
    model = build_model(ModelConfig(name="resnet18", weights=None))
    assert isinstance(model, nn.Module)


def test_output_shape_is_num_classes() -> None:
    """Forward (1,3,224,224) phai cho (1,10)."""
    model = build_model(ModelConfig(name="resnet18", weights=None))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert tuple(out.shape) == (1, 10)


def test_resnet_head_replaced() -> None:
    """Head resnet18 phai la Linear(512, 10)."""
    model = build_model(ModelConfig(name="resnet18", weights=None))
    assert isinstance(model.fc, nn.Linear)
    assert model.fc.out_features == 10


def test_vgg_head_replaced_keeps_layers() -> None:
    """Head vgg11 van la Sequential 7 phan tu, cuoi la Linear(4096, 10)."""
    model = build_model(ModelConfig(name="vgg11", weights=None))
    assert isinstance(model.classifier, nn.Sequential)
    assert len(model.classifier) == 7
    assert isinstance(model.classifier[6], nn.Linear)
    assert model.classifier[6].out_features == 10


def test_densenet_head_replaced() -> None:
    """Head densenet121 phai la Linear(1024, 10)."""
    model = build_model(ModelConfig(name="densenet121", weights=None))
    assert isinstance(model.classifier, nn.Linear)
    assert model.classifier.out_features == 10


def test_freeze_only_head_trainable() -> None:
    """freeze_backbone: % trainable phai < 1% cho ca 3 model."""
    for name in ("resnet18", "vgg11", "densenet121"):
        model = build_model(
            ModelConfig(name=name, strategy="freeze_backbone", weights=None)
        )
        pct = count_parameters(model)["params_trainable_pct"]
        assert pct < 1.0, f"{name}: {pct:.3f}% >= 1%"


def test_freeze_backbone_no_grad() -> None:
    """Backbone phai co requires_grad=False, head phai True."""
    model = build_model(
        ModelConfig(name="resnet18", strategy="freeze_backbone", weights=None)
    )
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad, name
        else:
            assert not param.requires_grad, name


def test_vgg_freeze_head_full_flag() -> None:
    """freeze_head_full=True tren vgg11 phai mo > 90% (bat loi cau hinh)."""
    model = build_model(
        ModelConfig(
            name="vgg11",
            strategy="freeze_backbone",
            weights=None,
            freeze_head_full=True,
        )
    )
    pct = count_parameters(model)["params_trainable_pct"]
    assert pct > 90.0, f"ky vong > 90%, nhan {pct:.3f}%"


def test_finetune_unfreezes_more_params() -> None:
    """fine_tune phai mo nhieu params hon freeze_backbone."""
    frozen = build_model(
        ModelConfig(name="resnet18", strategy="freeze_backbone", weights=None)
    )
    tuned = build_model(
        ModelConfig(
            name="resnet18", strategy="fine_tune", freeze_until=1, weights=None
        )
    )
    frozen_n = count_parameters(frozen)["params_trainable"]
    tuned_n = count_parameters(tuned)["params_trainable"]
    assert tuned_n > frozen_n


def test_finetune_block_has_grad() -> None:
    """Block cuoi phai co requires_grad=True sau fine_tune."""
    model = build_model(
        ModelConfig(
            name="resnet18", strategy="fine_tune", freeze_until=1, weights=None
        )
    )
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert any(n.startswith("layer4.") for n in trainable_names)
    assert not any(n.startswith("layer1.") for n in trainable_names)


def test_count_parameters_consistent() -> None:
    """total phai = trainable + frozen."""
    model = _build_raw("resnet18")
    replace_head(model, "resnet18", 10)
    freeze_backbone(model, "resnet18")
    stats = count_parameters(model)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    assert stats["params_total"] == stats["params_trainable"] + frozen


def test_unknown_model_raises() -> None:
    """Ten model sai phai nem ValueError."""
    with pytest.raises(ValueError, match="khong ho tro"):
        ModelConfig(name="alexnet")


def test_invalid_strategy_raises() -> None:
    """Strategy sai phai nem ValueError."""
    with pytest.raises(ValueError, match="khong hop le"):
        ModelConfig(name="resnet18", strategy="distill")


def test_invalid_n_blocks_raises() -> None:
    """n_blocks vuot qua so block phai nem ValueError."""
    model = _build_raw("resnet18")
    with pytest.raises(ValueError, match="chi co"):
        unfreeze_last(model, "resnet18", n_blocks=99)


def test_resolve_weights_default_is_passthrough() -> None:
    """weights='DEFAULT' duoc truyen thang cho torchvision."""
    assert resolve_weights("resnet18", "DEFAULT") == "DEFAULT"
    assert resolve_weights("resnet18", None) is None


def test_resolve_weights_invalid_raises() -> None:
    """Ten weights sai phai nem ValueError."""
    with pytest.raises(ValueError, match="khong ho tro"):
        resolve_weights("resnet18", "V2")


def test_describe_model_runs() -> None:
    """describe_model phai tra ve chuoi co ten lop va so params."""
    model = build_model(ModelConfig(name="resnet18", weights=None))
    text = describe_model(model, full=False)
    assert "ResNet" in text
    assert "Total parameters" in text
