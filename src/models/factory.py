"""Factory tao model pre-trained da thay head cho CIFAR-10."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torchvision.models as tv_models

from src.models.registry import SUPPORTED_MODELS, get_module, resolve_spec

# Enum weights tuong ung tung backbone (torchvision 0.29 chi co 1 gia tri moi model)
_WEIGHT_ENUMS = {
    "resnet18": "ResNet18_Weights",
    "vgg11": "VGG11_Weights",
    "densenet121": "DenseNet121_Weights",
}


def _head_modules(model: nn.Module, head_attr: str, last_only: bool) -> list[nn.Module]:
    """Tra ve cac module thuoc head can giu trainable.

    Voi VGG11, `classifier` chua 103M params (2 tang Linear lon). Neu
    unfreeze toan bo thi 92.8% model van trainable -> mat y nghia "freeze
    backbone". Vi vay mac dinh chi lay lop Linear cuoi cung.

    Args:
        model: Model da thay head.
        head_attr: Ten attribute chua head.
        last_only: Neu True chi lay lop Linear cuoi.

    Returns:
        Danh sach module.
    """
    head = getattr(model, head_attr)
    if not isinstance(head, nn.Sequential):
        return [head]

    linear_indices = [
        idx for idx, module in enumerate(head) if isinstance(module, nn.Linear)
    ]
    if not linear_indices:
        return [head]

    last_idx = linear_indices[-1]
    if last_only:
        return [head[last_idx]]
    return [head[idx] for idx in range(last_idx + 1)]


@dataclass
class ModelConfig:
    """Cau hinh xay dung model.

    Attributes:
        name: Ten backbone, vd: "resnet18".
        num_classes: So lop dau ra (CIFAR-10 = 10).
        strategy: "freeze_backbone" | "fine_tune" | "from_scratch".
        weights: "DEFAULT" | "IMAGENET1K_V1" | None (khong dung pre-trained).
        freeze_until: So block cuoi duoc mo gradient khi strategy="fine_tune".
        freeze_head_full: Voi head Sequential (VGG11), True = train ca cac
            Linear phia truoc lop cuoi; False = chi lop Linear cuoi.
    """

    name: str = "resnet18"
    num_classes: int = 10
    strategy: str = "fine_tune"
    weights: str | None = "DEFAULT"
    freeze_until: int = 1
    freeze_head_full: bool = False

    def __post_init__(self) -> None:
        if self.name.strip().lower() not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model khong ho tro: {self.name!r}. "
                f"Cac model ho tro: {list(SUPPORTED_MODELS)}"
            )
        if self.strategy not in {"freeze_backbone", "fine_tune", "from_scratch"}:
            raise ValueError(
                f"Strategy khong hop le: {self.strategy!r}. "
                "Chon: freeze_backbone | fine_tune | from_scratch"
            )
        if self.num_classes < 2:
            raise ValueError(f"num_classes phai >= 2, nhan {self.num_classes}")
        if self.freeze_until < 1:
            raise ValueError(f"freeze_until phai >= 1, nhan {self.freeze_until}")


def resolve_weights(name: str, weights: str | None):
    """Chuyen ten weights thanh enum cua torchvision.

    Args:
        name: Ten backbone.
        weights: "DEFAULT" | "IMAGENET1K_V1" | None.

    Returns:
        Enum weights, chuoi "DEFAULT", hoac None.

    Raises:
        ValueError: Neu ten weights khong hop le.
    """
    if weights is None:
        return None
    if weights == "DEFAULT":
        return "DEFAULT"

    key = name.strip().lower()
    if key not in _WEIGHT_ENUMS:
        raise ValueError(f"Model khong ho tro: {name!r}")
    if weights != "IMAGENET1K_V1":
        raise ValueError(
            f"Weights khong ho tro: {weights!r}. "
            "torchvision 0.29 chi co 'DEFAULT' | 'IMAGENET1K_V1'"
        )

    enum_cls = getattr(tv_models, _WEIGHT_ENUMS[key])
    return getattr(enum_cls, "IMAGENET1K_V1")


def replace_head(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """Thay lop phan loai cuoi cho dung `num_classes` lop.

    Xu ly 2 dang head khac nhau:
    - "linear": thang 1 lop Linear (resnet18, densenet121)
    - "sequential": Sequential nhieu lop — thay lop Linear cuoi cung,
      giu nguyen cac lop Linear/ReLU/Dropout phia truoc (vgg11)

    Args:
        model: Model goc.
        name: Ten backbone (de tra spec).
        num_classes: So lop moi.

    Returns:
        Model da thay head (sua tai cho, tra ve de tien dung chuoi).
    """
    spec = resolve_spec(name)

    if spec.head_type == "linear":
        old = getattr(model, spec.head_attr)
        setattr(model, spec.head_attr, nn.Linear(old.in_features, num_classes))
        return model

    classifier = getattr(model, spec.head_attr)
    last_linear_idx = None
    for idx in range(len(classifier) - 1, -1, -1):
        if isinstance(classifier[idx], nn.Linear):
            last_linear_idx = idx
            break
    if last_linear_idx is None:
        raise ValueError(f"Khong tim thay lop Linear nao trong {spec.head_attr}")

    old = classifier[last_linear_idx]
    new_layers = list(classifier)[:last_linear_idx] + [
        nn.Linear(old.in_features, num_classes)
    ]
    setattr(model, spec.head_attr, nn.Sequential(*new_layers))
    return model


def freeze_backbone(model: nn.Module, name: str, head_full: bool = False) -> nn.Module:
    """Dong bang toan bo backbone, chi giu head trainable.

    Args:
        model: Model da thay head.
        name: Ten backbone.
        head_full: Train toan bo head hay chi lop Linear cuoi.

    Returns:
        Model voi `requires_grad` da cap nhat.
    """
    spec = resolve_spec(name)

    for param in model.parameters():
        param.requires_grad_(False)

    for module in _head_modules(model, spec.head_attr, last_only=not head_full):
        for param in module.parameters():
            param.requires_grad_(True)

    return model


def unfreeze_last(
    model: nn.Module, name: str, n_blocks: int = 1, head_full: bool = False
) -> nn.Module:
    """Mo gradient cho n block cuoi cua backbone (fine-tuning).

    Args:
        model: Model da thay head.
        name: Ten backbone.
        n_blocks: So block cuoi duoc mo (theo `block_units`).
        head_full: Train toan bo head hay chi lop Linear cuoi.

    Returns:
        Model voi backbone cuoi cung + head deu trainable.

    Raises:
        ValueError: Neu `n_blocks` vuot qua so block co san.
    """
    spec = resolve_spec(name)
    if n_blocks < 1:
        raise ValueError(f"n_blocks phai >= 1, nhan {n_blocks}")
    if n_blocks > len(spec.block_units):
        raise ValueError(
            f"{name} chi co {len(spec.block_units)} block, nhan n_blocks={n_blocks}"
        )

    for param in model.parameters():
        param.requires_grad_(False)

    for module in _head_modules(model, spec.head_attr, last_only=not head_full):
        for param in module.parameters():
            param.requires_grad_(True)

    for path in spec.block_units[-n_blocks:]:
        for param in get_module(model, path).parameters():
            param.requires_grad_(True)

    for path in spec.extra_trainable:
        for param in get_module(model, path).parameters():
            param.requires_grad_(True)

    return model


def count_parameters(model: nn.Module) -> dict[str, float]:
    """Dem tong so tham so va so tham so dang train.

    Args:
        model: Model can dem.

    Returns:
        Dict voi `params_total`, `params_trainable`, `params_trainable_pct`.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": float(total),
        "params_trainable": float(trainable),
        "params_trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }


def describe_model(model: nn.Module, full: bool = True) -> str:
    """Sinh mo ta cau truc model dang text (phuc vu de bai buoc 3).

    Args:
        model: Model can mo ta.
        full: Neu True, in them toan bo cau truc (`str(model)`).

    Returns:
        Chuoi text mo ta tong quan + bang layer + so parameters.
    """
    stats = count_parameters(model)
    lines = [
        "=" * 70,
        "MODEL ARCHITECTURE SUMMARY",
        "=" * 70,
        f"Class: {model.__class__.__name__}",
        f"Total parameters:     {stats['params_total']:>12,.0f}",
        f"Trainable parameters: {stats['params_trainable']:>12,.0f} "
        f"({stats['params_trainable_pct']:.3f}%)",
        "-" * 70,
        "TOP-LEVEL LAYERS:",
    ]
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        lines.append(
            f"  {name:<20} {module.__class__.__name__:<25} "
            f"{n_params:>12,.0f} params"
        )
    if full:
        lines.append("-" * 70)
        lines.append("FULL STRUCTURE:")
        lines.append(str(model))
    lines.append("=" * 70)
    return "\n".join(lines)


def build_model(config: ModelConfig) -> nn.Module:
    """Ham chinh: dung model pre-trained da adapt cho CIFAR-10.

    Args:
        config: Cau hinh model.

    Returns:
        `nn.Module` san sang train, da thay head va ap chien luoc freeze.
    """
    name = config.name.strip().lower()
    spec = resolve_spec(name)

    weights = resolve_weights(name, config.weights)
    model = spec.builder(weights=weights)

    replace_head(model, name, config.num_classes)

    if config.strategy == "freeze_backbone":
        freeze_backbone(model, name, head_full=config.freeze_head_full)
    elif config.strategy == "fine_tune":
        unfreeze_last(
            model, name, config.freeze_until, head_full=config.freeze_head_full
        )
    # from_scratch: giu nguyen trang thai (tat ca trainable)

    return model
