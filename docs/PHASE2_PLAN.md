# Phase 2 Plan - CIFAR-10 with Pre-trained PyTorch Architectures

**Ngày tạo:** 2026-10-25
**Cập nhật:** 2026-10-25 (chia lại M2/M3/M4 cho 3 người, align với đề bài `đề.txt`)
**Phase:** 2 - Model Development, Training & Evaluation
**Số lượng người:** 3 người
**Dự kiến hoàn thành:** Sau Phase 1 (M1 — Data pipeline ✅)
**Dataset:** CIFAR-10 (45k train / 5k val / 10k test, 32×32 RGB, 10 classes)
**Runtime:** CPU-only (torch 2.14.1+cpu), 12 logical cores

---

## 📋 YÊU CẦU TỪ ĐỀ BÀI (`đề.txt`)

### 1. Learn (Nghiên cứu)
- Loading pre-trained models từ `torchvision.models`
- Model architecture exploration (in cấu trúc, đếm parameters)
- Transfer learning: freeze / unfreeze layers
- Hyperparameter tuning
- TensorBoard monitoring
- Model evaluation

### 2. Implement
- ✅ Load CIFAR-10 dataset *(Phase 1 — M1 đã xong)*
- ✅ Apply transforms *(Phase 1 — M1 đã xong)*
- ✅ Create DataLoader batches *(Phase 1 — M1 đã xong)*
- ⏳ **Load a Pre-trained Model** *(Người 1)*
- ⏳ **Explore the Model Architecture** *(Người 1)*
- ⏳ **Adapt the Model for a Specific Task** *(Người 1)*
- ⏳ **Write the training loop** *(Người 2)*
- ⏳ **Use TensorBoard to monitor** *(Người 2)*
- ⏳ **Evaluate the Model** *(Người 3)*

### 3. Tasks
- ⏳ **Experiment with different pre-trained models (VGG, DenseNet, etc.)** *(Người 1)*
- ⏳ **Adjust the hyperparameters (learning rate, batch size, etc.)** *(Người 2)*
- ⏳ **Visualize loss / monitoring** *(Người 2)*

### 4. Deliver
- ✅ Python code *(đang build)*
- ⏳ **Brief report** *(3 reports + 1 final report)*
- ⏳ **Loss graphs / TensorBoard logs** *(Người 2)*
- ⏳ **Evaluation report + confusion matrix** *(Người 3)*

### 5. Use
- `torchvision.models` (ResNet18, VGG11, DenseNet121)
- `torch.utils.tensorboard` / `tensorboard`
- `torch.utils.data.DataLoader`
- PyTorch docs/tutorials

---

## Tổng quan Workflow Phase 2

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Load + Adapt       │───→│  Training Loop +    │───→│  Evaluation +        │
│  Pre-trained Model  │    │  TensorBoard +      │    │  Metrics + Plots     │
│  Freeze / Unfreeze  │    │  Hyperparameter Tune│    │                      │
│   (Người 1)         │    │   (Người 2)         │    │   (Người 3)         │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
        │                          │                          ▲
        │    best model +          │    trained model +      │
        │    best strategy         │    history +            │
        │                          │    tb logs              │
        └──────────────────────────┴──────────────────────────┘
```

### Phân chia theo đề bài

| Nhóm | Người | Phụ trách | Files |
|------|-------|-----------|-------|
| **Model** | Người 1 | Load pre-trained, explore architecture, thay head, freeze/unfreeze, so sánh các backbone | `src/models/` |
| **Training** | Người 2 | Training loop (forward, loss, backward, optimize) + TensorBoard + hyperparameter sweep | `src/training/` |
| **Evaluation** | Người 3 | Evaluate trên test set + metrics + confusion matrix + biểu đồ | `src/evaluation/` |

---

## ⚠️ YẾU TỐ QUAN TRỌNG: THỜI GIAN TRAIN TRÊN CPU

Đã **đo thật** trên máy này (12 logical cores, `torch.set_num_threads(8)`, ResNet18 freeze):

| Input size | 1 epoch | 5 epoch | Ghi chú |
|-----------|---------|---------|---------|
| 224×224 | **22–37 phút** | ~2–3 giờ | Chuẩn ImageNet — kết quả báo cáo chính |
| 128×128 | 18 phút | ~1.5 giờ | |
| 64×64 | **5 phút** | ~24 phút | Dùng cho sweep rộng |

Fine-tune toàn bộ mạng chậm hơn ~3× so với freeze.

**Hệ quả trực tiếp:** VGG11 ở 224×224 mất **~144 phút/epoch** (≈12 giờ cho 5 epoch) → không khả thi trên CPU.

### Chiến lược 2 tầng (quyết định bắt buộc)

| Tier | `image_size` | Dùng cho | Thời gian ResNet18 freeze |
|------|--------------|----------|---------------------------|
| **A — Báo cáo chính** | 224 | Kết quả cuối cùng, model tốt nhất | ~3 giờ / 5 epoch |
| **B — Sweep rộng** | 64 | So sánh 3 backbone + sweep siêu tham số | ~24 phút / 5 epoch |

> ⚠️ **Bắt buộc:** báo cáo phải ghi rõ điều kiện của từng bảng. **Không trộn** kết quả Tier A (224) với Tier B (64) trong cùng một bảng so sánh — accuracy ở 64×64 thấp hơn do mất thông tin chi tiết.

---

## PHÂN CÔNG CÔNG VIỆC

---

## 👤 NGƯỜI 1: Model Factory — Load, Explore, Adapt

### Nhiệm vụ (theo đề)
1. **Load a Pre-trained Model** — nạp ResNet18 / VGG11 / DenseNet121 từ `torchvision.models` với `pretrained=True`
2. **Explore the Model Architecture** — in cấu trúc, đếm layers & parameters
3. **Adapt the Model for a Specific Task** — thay classification layer cuối về 10 lớp
4. **Freeze earlier layers** (transfer learning) / **Unfreeze final layers** (fine-tuning)

### File cần tạo/sửa

| File | Chức năng | Trạng thái |
|------|-----------|------------|
| `src/models/registry.py` | **Chính** — Bảng tra cứu backbone + head spec | Tạo mới |
| `src/models/factory.py` | **Chính** — `build_model()`, thay head, freeze/unfreeze | Tạo mới |
| `src/models/__init__.py` | Export public API | Sửa thêm |
| `src/models/__main__.py` | CLI khám phá — in cấu trúc, đếm params | Tạo mới |
| `configs/model_config.yaml` | Cấu hình model + chiến lược | Tạo mới |
| `tests/unit/test_factory.py` | Unit test cho factory | Tạo mới |
| `tests/integration/test_model_pipeline.py` | Integration test model ↔ dataloader | Tạo mới |

### Chi tiết implementation

#### 1.1. Registry — bảng tra cứu backbone

```python
# src/models/registry.py
"""Bảng tra cứu các backbone pre-trained hỗ trợ trong project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
import torchvision.models as tv_models


@dataclass(frozen=True)
class BackboneSpec:
    """Mô tả cấu trúc một backbone để thay head và unfreeze cho đúng.

    Attributes:
        builder: Hàm dựng model từ torchvision.
        head_attr: Tên attribute chứa classification head ("fc" | "classifier").
        head_type: "linear" (1 lớp Linear) | "sequential" (Sequential nhiều lớp).
        head_is_sequential: Head có phải Sequential hay không.
        block_units: Đường dẫn các "đơn vị" (block) theo thứ tự nông → sâu.
            Dùng để unfreeze n block cuối.
        extra_trainable: Đường dẫn module luôn trainable (VD: `features.norm5`).
    """

    builder: Callable[..., nn.Module]
    head_attr: str
    head_type: str
    block_units: tuple[str, ...]
    extra_trainable: tuple[str, ...] = ()


# --- ResNet18 ---
# Cấu trúc: conv1 → bn1 → relu → maxpool → layer1..layer4 → avgpool → fc
# Head gốc: fc = Linear(512, 1000)
# Block cuối: layer4 (2 BasicBlock, 8.39M params)
_RESNET18 = BackboneSpec(
    builder=tv_models.resnet18,
    head_attr="fc",
    head_type="linear",
    block_units=("layer1", "layer2", "layer3", "layer4"),
)

# --- VGG11 ---
# Cấu trúc: features (8 conv block) → avgpool → classifier (Sequential 7 phần tử)
# Head gốc: Linear(25088,4096) → ReLU → Dropout → Linear(4096,4096) → ReLU
#            → Dropout → Linear(4096,1000)
# Block cuối: 3 conv cuối của `features` (features.16, features.18)
_VGG11 = BackboneSpec(
    builder=tv_models.vgg11,
    head_attr="classifier",
    head_type="sequential",
    block_units=(
        "features.0", "features.3", "features.6", "features.8",
        "features.11", "features.13", "features.16", "features.18",
    ),
)

# --- DenseNet121 ---
# Cấu trúc: conv0 → norm0 → relu0 → pool0 → denseblock1..4 (+ transition1..3) → norm5
# Head gốc: classifier = Linear(1024, 1000)
# Block cuối: denseblock4 + transition3
_DENSENET121 = BackboneSpec(
    builder=tv_models.densenet121,
    head_attr="classifier",
    head_type="linear",
    block_units=(
        "features.denseblock1",
        "features.denseblock2",
        "features.denseblock3",
        "features.denseblock4",
    ),
    extra_trainable=("features.transition3", "features.norm5"),
)

BACKBONE_REGISTRY: dict[str, BackboneSpec] = {
    "resnet18": _RESNET18,
    "vgg11": _VGG11,
    "densenet121": _DENSENET121,
}

SUPPORTED_MODELS: tuple[str, ...] = tuple(BACKBONE_REGISTRY.keys())


def resolve_spec(name: str) -> BackboneSpec:
    """Tra cứu spec của backbone theo tên.

    Args:
        name: Tên backbone, vd: "resnet18".

    Returns:
        `BackboneSpec` tương ứng.

    Raises:
        ValueError: Nếu tên không có trong registry.
    """
    key = name.strip().lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Model không hỗ trợ: {name!r}. "
            f"Các model hỗ trợ: {list(SUPPORTED_MODELS)}"
        )
    return BACKBONE_REGISTRY[key]


def get_module(model: nn.Module, path: str) -> nn.Module:
    """Lấy module con theo đường dẫn, hỗ trợ chỉ số số.

    Args:
        model: Model gốc.
        path: Đường dẫn, vd: "features.18", "classifier.0".

    Returns:
        Module tương ứng.

    Raises:
        AttributeError: Nếu đường dẫn không tồn tại.
    """
    obj: object = model
    for part in path.split("."):
        if isinstance(obj, (list, tuple, nn.Sequential)):
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj  # type: ignore[return-value]
```

> ⚠️ **Đã verify bằng introspection thật** (không đoán) — xem bảng ở phần "Search Space" bên dưới.

#### 1.2. Model Factory — file chính

```python
# src/models/factory.py
"""Factory tạo model pre-trained đã thay head cho CIFAR-10."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.registry import (
    SUPPORTED_MODELS,
    get_module,
    resolve_spec,
)

# Chi lay lop Linear CUOI CUNG cua head (khong lay cao phan "classifier" cua VGG).
# Ly do: "classifier" cua VGG11 chua 103M params — neu unfreeze toan bo thi
# 92.8% model van trainable, vo nghia "freeze backbone".


def _head_modules(model: nn.Module, head_attr: str, last_only: bool) -> list[nn.Module]:
    """Trả về các module thuộc head cần giữ trainable.

    Args:
        model: Model đã thay head.
        head_attr: Tên attribute chứa head.
        last_only: Nếu True chỉ lấy lớp Linear cuối cùng (VGG11),
            ngược lại lấy toàn bộ head.

    Returns:
        Danh sách module.
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
    """Cấu hình xây dựng model.

    Attributes:
        name: Tên backbone, vd: "resnet18".
        num_classes: Số lớp đầu ra (CIFAR-10 = 10).
        strategy: "freeze_backbone" | "fine_tune" | "from_scratch".
        weights: "DEFAULT" | "IMAGENET1K_V1" | None (không dùng pre-trained).
        freeze_until: Số block cuối được mở gradient khi strategy="fine_tune".
        freeze_head_full: Với head Sequential (VGG11), True = train cả các
            Linear phía trước lớp cuối; False = chỉ lớp Linear cuối.
    """

    name: str = "resnet18"
    num_classes: int = 10
    strategy: str = "freeze_backbone"
    weights: str | None = "DEFAULT"
    freeze_until: int = 1
    freeze_head_full: bool = False

    def __post_init__(self) -> None:
        if self.name.strip().lower() not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model không hỗ trợ: {self.name!r}. "
                f"Các model hỗ trợ: {list(SUPPORTED_MODELS)}"
            )
        if self.strategy not in {"freeze_backbone", "fine_tune", "from_scratch"}:
            raise ValueError(
                f"Strategy không hợp lệ: {self.strategy!r}. "
                "Chọn: freeze_backbone | fine_tune | from_scratch"
            )
        if self.num_classes < 2:
            raise ValueError(f"num_classes phải >= 2, nhận {self.num_classes}")
        if self.freeze_until < 1:
            raise ValueError(f"freeze_until phải >= 1, nhận {self.freeze_until}")


def resolve_weights(name: str, weights: str | None):
    """Chuyển tên weights thành enum của torchvision.

    Args:
        name: Tên backbone.
        weights: "DEFAULT" | "IMAGENET1K_V1" | None.

    Returns:
        Enum weights hoặc None.

    Raises:
        ValueError: Nếu tên weights không hợp lệ.
    """
    if weights is None:
        return None
    if weights == "DEFAULT":
        return "DEFAULT"

    enum_names = {
        "resnet18": "ResNet18_Weights",
        "vgg11": "VGG11_Weights",
        "densenet121": "DenseNet121_Weights",
    }
    key = name.strip().lower()
    if key not in enum_names:
        raise ValueError(f"Model không hỗ trợ: {name!r}")
    if weights != "IMAGENET1K_V1":
        raise ValueError(
            f"Weights không hỗ trợ: {weights!r}. "
            "torchvision 0.29 chỉ có 'DEFAULT' | 'IMAGENET1K_V1'"
        )

    enum_cls = getattr(tv_models, enum_names[key])
    return getattr(enum_cls, "IMAGENET1K_V1")


def replace_head(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    """Thay lớp phân loại cuối cho đúng `num_classes` lớp.

    Xử lý 2 dạng head khác nhau:
    - "linear": thẳng 1 lớp Linear (resnet18, densenet121)
    - "sequential": Sequential nhiều lớp — thay lớp Linear cuối cùng,
      giữ nguyên các lớp Linear/ReLU/Dropout phía trước (vgg11)

    Args:
        model: Model gốc.
        name: Tên backbone (để tra spec).
        num_classes: Số lớp mới.

    Returns:
        Model đã thay head (sửa tại chỗ, trả về để tiện dùng chuỗi).
    """
    spec = resolve_spec(name)

    if spec.head_type == "linear":
        old = getattr(model, spec.head_attr)
        setattr(model, spec.head_attr, nn.Linear(old.in_features, num_classes))
        return model

    # sequential: thay lớp Linear cuối cùng, giữ phần còn lại
    classifier = getattr(model, spec.head_attr)
    last_linear_idx = None
    for idx in range(len(classifier) - 1, -1, -1):
        if isinstance(classifier[idx], nn.Linear):
            last_linear_idx = idx
            break
    if last_linear_idx is None:
        raise ValueError(f"Không tìm thấy lớp Linear nào trong {spec.head_attr}")

    old = classifier[last_linear_idx]
    new_layers = list(classifier)[:last_linear_idx] + [
        nn.Linear(old.in_features, num_classes)
    ]
    setattr(model, spec.head_attr, nn.Sequential(*new_layers))
    return model


def freeze_backbone(model: nn.Module, name: str, head_full: bool = False) -> nn.Module:
    """Đóng băng toàn bộ backbone, chỉ giữ head trainable.

    ⚠️ Với VGG11, `head_full=False` (mặc định) chỉ giữ lớp Linear cuối
    trainable. Nếu bật `head_full=True`, classifier của VGG11 chứa
    103M params → 92.8% model vẫn trainable, mất ý nghĩa "freeze backbone".

    Args:
        model: Model đã thay head.
        name: Tên backbone.
        head_full: Train toàn bộ head hay chỉ lớp Linear cuối.

    Returns:
        Model với `requires_grad` đã cập nhật.
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
    """Mở gradient cho n block cuối của backbone (fine-tuning).

    Args:
        model: Model đã thay head.
        name: Tên backbone.
        n_blocks: Số block cuối được mở (theo `block_units`).
        head_full: Train toàn bộ head hay chỉ lớp Linear cuối.

    Returns:
        Model với backbone cuối cùng + head đều trainable.

    Raises:
        ValueError: Nếu `n_blocks` vượt quá số block có sẵn.
    """
    spec = resolve_spec(name)
    if n_blocks < 1:
        raise ValueError(f"n_blocks phải >= 1, nhận {n_blocks}")
    if n_blocks > len(spec.block_units):
        raise ValueError(
            f"{name} chỉ có {len(spec.block_units)} block, nhận n_blocks={n_blocks}"
        )

    for param in model.parameters():
        param.requires_grad_(False)

    # Mở head
    for module in _head_modules(model, spec.head_attr, last_only=not head_full):
        for param in module.parameters():
            param.requires_grad_(True)

    # Mở n block cuối + các module đi kèm (VD: transition3, norm5 của DenseNet)
    for path in spec.block_units[-n_blocks:]:
        for param in get_module(model, path).parameters():
            param.requires_grad_(True)

    for path in spec.extra_trainable:
        for param in get_module(model, path).parameters():
            param.requires_grad_(True)

    return model


def count_parameters(model: nn.Module) -> dict[str, float]:
    """Đếm tổng số tham số và số tham số đang train.

    Args:
        model: Model cần đếm.

    Returns:
        Dict với `params_total`, `params_trainable`, `params_trainable_pct`.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "params_total": float(total),
        "params_trainable": float(trainable),
        "params_trainable_pct": (100.0 * trainable / total) if total else 0.0,
    }


def describe_model(model: nn.Module) -> str:
    """Sinh mô tả cấu trúc model dạng text (phục vụ đề bài bước 3).

    Args:
        model: Model cần mô tả.

    Returns:
        Chuỗi text mô tả tổng quan + bảng layer + số parameters.
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
    lines.append("-" * 70)
    lines.append("FULL STRUCTURE:")
    lines.append(str(model))
    lines.append("=" * 70)
    return "\n".join(lines)


def build_model(config: ModelConfig) -> nn.Module:
    """Hàm chính: dựng model pre-trained đã adapt cho CIFAR-10.

    Args:
        config: Cấu hình model.

    Returns:
        `nn.Module` sẵn sàng train, đã thay head và áp chiến lược freeze.
    """
    name = config.name.strip().lower()
    spec = resolve_spec(name)

    # 1. Dựng backbone kèm weights pre-trained
    weights = resolve_weights(name, config.weights)
    model = spec.builder(weights=weights) if weights is not None else spec.builder(weights=None)

    # 2. Thay head cho đúng số lớp
    replace_head(model, name, config.num_classes)

    # 3. Áp chiến lược train
    if config.strategy == "freeze_backbone":
        freeze_backbone(model, name, head_full=config.freeze_head_full)
    elif config.strategy == "fine_tune":
        unfreeze_last(
            model, name, config.freeze_until, head_full=config.freeze_head_full
        )
    # from_scratch: giữ nguyên trạng thái (tất cả trainable)

    return model
```

> ⚠️ **Lưu ý:** file trên dùng `tv_models` trong `resolve_weights` — cần `import torchvision.models as tv_models` ở đầu file.

#### 1.3. CLI khám phá — in cấu trúc (đề bài bước 3)

```python
# src/models/__main__.py
"""CLI khám phá model: in cấu trúc, đếm parameters, kiểm tra output shape."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.models.factory import ModelConfig, build_model, count_parameters, describe_model
from src.utils.config import load_config
from src.utils.runs import create_run_dir, setup_logging, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Khám phá model pre-trained")
    parser.add_argument("--config", default="configs/model_config.yaml")
    parser.add_argument("--model", default=None, help="Ghi đè tên model trong config")
    parser.add_argument("--strategy", default=None, choices=["freeze_backbone", "fine_tune", "from_scratch"])
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--print-model", action="store_true", help="In toàn bộ cấu trúc")
    args = parser.parse_args(argv)

    raw = load_config(args.config)
    section = raw.get("model", raw)

    config = ModelConfig(
        name=args.model or section.get("name", "resnet18"),
        num_classes=args.num_classes or int(section.get("num_classes", 10)),
        strategy=args.strategy or section.get("strategy", "freeze_backbone"),
        weights=section.get("weights", "DEFAULT"),
        freeze_until=int(section.get("freeze_until", 1)),
        dropout=float(section.get("dropout", 0.0)),
    )

    run_dir = create_run_dir()
    setup_logging(run_dir)

    model = build_model(config)
    stats = count_parameters(model)

    print(describe_model(model) if args.print_model else describe_model(model))

    # Kiểm tra forward shape
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        out = model(dummy)
    print(f"Output shape: {tuple(out.shape)} (kỳ vọng (1, {config.num_classes}))")

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
    raise SystemExit(main())
```

### Search Space / So sánh model (đề bài: "Experiment with different pre-trained models")

```yaml
# configs/model_config.yaml
model:
  # Backbone: resnet18 | vgg11 | densenet121
  name: resnet18

  # CIFAR-10 có 10 lớp
  num_classes: 10

  # DEFAULT = dùng weights pre-trained ImageNet
  # IMAGENET1K_V1 = chỉ định rõ weights
  # null = from scratch (không dùng pre-trained, để đối chứng)
  weights: DEFAULT

  # freeze_backbone: chỉ train head (transfer learning)
  # fine_tune: mở gradient n block cuối
  # from_scratch: train toàn bộ từ đầu
  strategy: freeze_backbone

  # Số block cuối mở gradient khi strategy = fine_tune
  # resnet18: layer1..layer4 (4 stage) | vgg11: 8 conv block | densenet121: 4 denseblock
  freeze_until: 1

  # Với head Sequential (VGG11): true = train cả Linear phía trước lớp cuối
  # false = chỉ lớp Linear cuối (KHUYẾN NGHỊ — xem bảng bên dưới)
  freeze_head_full: false
```

#### Số liệu đã đo thật (chạy code thật, không ước lượng)

**Cấu trúc head sau khi thay về 10 lớp:**

| Backbone | Head gốc | Head sau khi thay | Tổng params |
|----------|----------|-------------------|-------------|
| `resnet18` | `fc` = `Linear(512,1000)` | `Linear(512,10)` | 11,181,642 |
| `vgg11` | `classifier` = `Sequential[7]` | `Sequential[7]`, cuối = `Linear(4096,10)` | 128,807,306 |
| `densenet121` | `classifier` = `Linear(1024,1000)` | `Linear(1024,10)` | 6,964,106 |

**Tỉ lệ params trainable theo chiến lược:**

| Backbone | `freeze_backbone` | `fine_tune` (1 block) | `fine_tune` (3 block) |
|----------|-------------------|----------------------|----------------------|
| `resnet18` | 5,130 / 11.18M = **0.046%** | 8.40M = **75.1%** | 11.02M = **98.6%** |
| `vgg11` (`freeze_head_full: false`) | 40,970 / 128.8M = **0.032%** | 121.9M = **94.7%** | 126.7M = **98.3%** |
| `vgg11` (`freeze_head_full: true`) | 119.6M / 128.8M = **92.8%** ⚠️ | 94.7% | 98.3% |
| `densenet121` | 10,250 / 6.96M = **0.147%** | 2.17M = **31.1%** | 5.93M = **85.1%** |

> ⚠️ **Phát hiện quan trọng — VGG11:** `classifier` của VGG11 chứa **103M params** (2 tầng `Linear(25088→4096)` và `Linear(4096→4096)`). Nếu "freeze backbone" mà unfreeze toàn bộ `classifier` thì **92.8% model vẫn trainable** → mất hoàn toàn ý nghĩa của transfer learning, và mỗi epoch chậm gần như full fine-tune.
>
> → **Bắt buộc dùng `freeze_head_full: false`** (chỉ train `Linear(4096, 10)` = 40,970 params) cho VGG11. Đây là lý do plan dùng `freeze_head_full` làm cờ config thay vì hardcode.

> ⚠️ **Đã verify bằng introspection + chạy code thật** (không đoán):
> - `vgg11.classifier` là `Sequential` **7 phần tử** — `Linear(25088,4096)` → ReLU → Dropout → `Linear(4096,4096)` → ReLU → Dropout → `Linear(4096,1000)`. Code giả định mọi model có `model.fc` sẽ **crash ngay** khi chạy VGG11.
> - `resnet18` có 4 stage: `layer1..layer4`. `layer4` = 2 BasicBlock, 8.39M params.
> - `vgg11.features` có 8 Conv2d (chỉ số `0,3,6,8,11,13,16,18`).
> - `densenet121.features` có `denseblock1..4` xen kẽ `transition1..3`, kết thúc bằng `norm5`.
> - Cả 3 model sau khi thay head đều cho output shape `(1, 10)` với input `(1, 3, 224, 224)`.

### Requirements (theo đề)

✅ **Load a Pre-trained Model**:
- `weights="DEFAULT"` nạp weights ImageNet cho cả 3 backbone
- Bảng tra cứu `BACKBONE_REGISTRY` dễ mở rộng thêm model

✅ **Explore the Model Architecture**:
- `describe_model()` in tổng quan, bảng top-level layer, số params từng layer, full structure
- `count_parameters()` trả về total / trainable / tỉ lệ %

✅ **Adapt the Model for a Specific Task**:
- `replace_head()` thay lớp cuối về 10 lớp — xử lý cả `Linear` lẫn `Sequential`

✅ **Freeze earlier layers**:
- `freeze_backbone()` → chỉ head trainable
- ⚠️ Với VGG11 phải đặt `freeze_head_full: false` — nếu không sẽ 92.8% model vẫn trainable

✅ **Unfreeze final layers**:
- `unfreeze_last(model, name, n_blocks)` → mở block cuối + head theo `block_units` đã introspect
- BatchNorm trong block được mở luôn (nếu không, BN chỉ dùng running stats)

### Test
- Forward pass với input `(1, 3, 224, 224)` → verify output `(1, 10)` cho cả 3 model
- `count_parameters`: verify `total == trainable + frozen`
- Freeze: verify mọi param backbone có `requires_grad == False`
- Freeze VGG11: verify `params_trainable_pct < 1%` (bắt lỗi `freeze_head_full` sai)
- Fine-tune: verify `params_trainable > params_trainable(freeze)` và block cuối có gradient
- VGG11: verify head vẫn có 7 phần tử và phần tử cuối là `Linear(4096, 10)`
- `n_blocks` vượt số block của model → raise `ValueError`
- Tên model sai / strategy sai → raise `ValueError` với message rõ ràng

### Output cho Người 2
- **Tên model + strategy** tốt nhất (VD: `resnet18` + `freeze_backbone`)
- **`configs/model_config.yaml`** hoàn chỉnh để Người 2 dùng
- **Bảng so sánh params** của 3 backbone (đưa vào báo cáo)

### Report cần viết (lưu vào `docs/reports/nguoi1_models_report.md`)

```markdown
# Model & Architecture Report

## 1. Pre-trained Models Implemented
| Model | Source | Total Params | Trainable (freeze) | Trainable (fine-tune) |
|-------|--------|--------------|--------------------|----------------------|
| resnet18 | torchvision.models | 11,181,642 | 5,130 (0.046%) | 8,398,858 (75.1%) |
| vgg11 | torchvision.models | 128,807,306 | 40,970 (0.032%) | 121,946,634 (94.7%) |
| densenet121 | torchvision.models | 6,964,106 | 10,250 (0.147%) | 2,168,330 (31.1%) |

## 2. Architecture Exploration (đề bài bước 3)
- Cấu trúc từng model (output `describe_model`)
- Head gốc vs head sau khi thay
- Số lớp / số parameters

## 3. Adaptation Strategy
- Freeze backbone: kết quả, % params trainable
- Fine-tune n block cuối: kết quả, % params trainable
- So sánh 2 chiến lược
- ⚠️ Vấn đề VGG11: classifier 103M params → phải dùng `freeze_head_full: false`

## 4. Selected Model
- Model: ...
- Strategy: ...
- Lý do chọn (accuracy vs thời gian train)

## 5. Files Created
- List all files
```

---

## 👤 NGƯỜI 2: Training Loop + TensorBoard + Hyperparameters

### Nhiệm vụ (theo đề)
1. **Define the loss function and optimizer**
2. **Write the training loop** (forward, loss, backward, optimize)
3. **Use TensorBoard to monitor the training process**
4. **Adjust the hyperparameters** (learning rate, batch size, etc.)

### File cần tạo/sửa

| File | Chức năng | Trạng thái |
|------|-----------|------------|
| `src/training/loss.py` | `build_criterion()` — CrossEntropy / label smoothing | Tạo mới |
| `src/training/optimizer.py` | `build_optimizer()` — Adam / AdamW / SGD | Tạo mới |
| `src/training/scheduler.py` | `build_scheduler()` — cosine / step / plateau | Tạo mới |
| `src/training/trainer.py` | **Chính** — Training loop + validate + checkpoint | Tạo mới |
| `src/training/tensorboard_utils.py` | **Chính** — Ghi scalar/histogram/image lên TensorBoard | Tạo mới |
| `src/training/__init__.py` | Export training modules | Sửa thêm |
| `src/training/__main__.py` | CLI train + override hyperparameter + multi-model | Tạo mới |
| `configs/train_config.yaml` | Cấu hình train + hyperparameter | Tạo mới |
| `tests/unit/test_optimizer.py` | Unit test loss/optimizer/scheduler | Tạo mới |
| `tests/unit/test_tensorboard_utils.py` | Unit test TensorBoard writer | Tạo mới |
| `tests/integration/test_training_loop.py` | Integration test train end-to-end | Tạo mới |

### Chi tiết implementation

#### 2.1. Loss & Optimizer

```python
# src/training/loss.py
"""Các hàm loss dùng trong training."""

from __future__ import annotations

import torch.nn as nn


def build_criterion(name: str = "cross_entropy", label_smoothing: float = 0.0) -> nn.Module:
    """Tạo loss function theo tên.

    Args:
        name: "cross_entropy" | "label_smoothing".
        label_smoothing: Giá trị smoothing (chỉ dùng với label_smoothing).

    Returns:
        `nn.Module` loss.

    Raises:
        ValueError: Nếu tên không hợp lệ.
    """
    key = name.strip().lower()
    if key == "cross_entropy":
        return nn.CrossEntropyLoss()
    if key == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    raise ValueError(
        f"Loss không hỗ trợ: {name!r}. Chọn: cross_entropy | label_smoothing"
    )


# src/training/optimizer.py
"""Factory tạo optimizer — chỉ truyền các tham số đang train."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    name: str = "adam",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    """Tạo optimizer chỉ trên các tham số có `requires_grad=True`.

    ⚠️ Quan trọng: nếu truyền toàn bộ `model.parameters()`, optimizer sẽ theo
    dõi cả trọng số đã đóng băng → lãng phí state và gây hiểu nhầm về
    số lượng tham số thực sự được cập nhật.

    Args:
        model: Model đã áp chiến lược freeze.
        name: "adam" | "adamw" | "sgd".
        lr: Learning rate.
        weight_decay: Weight decay.
        momentum: Momentum (chỉ dùng cho SGD).

    Returns:
        `torch.optim.Optimizer`.

    Raises:
        ValueError: Nếu tên optimizer không hợp lệ.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("Model không có tham số nào trainable!")

    key = name.strip().lower()
    if key == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    raise ValueError(
        f"Optimizer không hỗ trợ: {name!r}. Chọn: adam | adamw | sgd"
    )
```

#### 2.2. Scheduler

```python
# src/training/scheduler.py
"""Các learning rate scheduler."""

from __future__ import annotations

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "none",
    epochs: int = 5,
    step_size: int = 2,
    gamma: float = 0.1,
):
    """Tạo LR scheduler theo tên.

    Args:
        optimizer: Optimizer đã tạo.
        name: "none" | "cosine" | "step" | "plateau".
        epochs: Tổng số epoch (dùng cho cosine).
        step_size: Bước nhảy cho StepLR.
        gamma: Hệ số giảm cho StepLR.

    Returns:
        Scheduler hoặc None nếu name="none".

    Raises:
        ValueError: Nếu tên không hợp lệ.
    """
    key = name.strip().lower()
    if key == "none":
        return None
    if key == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if key == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if key == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=gamma, patience=1)
    raise ValueError(
        f"Scheduler không hỗ trợ: {name!r}. Chọn: none | cosine | step | plateau"
    )
```

#### 2.3. Training Loop (forward → loss → backward → optimize)

```python
# src/training/trainer.py
"""Training loop cho CIFAR-10 với pre-trained backbone."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    """Cấu hình training.

    Attributes:
        model: Tên backbone.
        strategy: "freeze_backbone" | "fine_tune".
        freeze_until: Số block cuối mở khi fine_tune.
        epochs: Số epoch.
        batch_size: Kích thước batch.
        lr: Learning rate.
        optimizer: "adam" | "adamw" | "sgd".
        weight_decay: Weight decay.
        momentum: Momentum cho SGD.
        scheduler: "none" | "cosine" | "step" | "plateau".
        num_workers: Số worker DataLoader.
        seed: Random seed.
        device: "auto" | "cpu" | "cuda".
        image_size: Kích thước ảnh (224 = Tier A, 64 = Tier B).
        early_stopping_patience: Số epoch chờ trước khi dừng (0 = tắt).
    """

    model: str = "resnet18"
    strategy: str = "freeze_backbone"
    freeze_until: int = 1
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3
    optimizer: str = "adam"
    weight_decay: float = 1e-4
    momentum: float = 0.9
    scheduler: str = "none"
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"
    image_size: int = 224
    early_stopping_patience: int = 0
    history: dict[str, list[float]] = field(default_factory=dict)


class Trainer:
    """Thực hiện training loop: forward → loss → backward → optimize.

    Attributes:
        model: Model đã adapt.
        train_loader: DataLoader tập train.
        val_loader: DataLoader tập validation.
        criterion: Loss function.
        optimizer: Optimizer (chỉ trên params trainable).
        device: Thiết bị chạy.
        history: Lịch sử metric mỗi epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Any = None,
        run_dir: Path | None = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.run_dir = run_dir
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "epoch_seconds": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> tuple[float, float]:
        """Train 1 epoch — implements forward, loss, backward, optimize.

        Returns:
            Tuple `(epoch_loss, epoch_accuracy)` (accuracy theo %).
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in self.train_loader:
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward
            output = self.model(data)
            # Loss
            loss = self.criterion(output, target)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # Optimize
            self.optimizer.step()

            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        return total_loss / total, 100.0 * correct / total

    @torch.no_grad()
    def validate(self, loader: DataLoader | None = None) -> tuple[float, float]:
        """Đánh giá trên validation set.

        Args:
            loader: Loader cần đánh giá, mặc định `val_loader`.

        Returns:
            Tuple `(loss, accuracy)`.
        """
        loader = loader or self.val_loader
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in loader:
            data = data.to(self.device)
            target = target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        return total_loss / total, 100.0 * correct / total

    def save_checkpoint(self, path: Path, extra: dict | None = None) -> Path:
        """Lưu checkpoint (model + optimizer + history).

        Args:
            path: Đường dẫn file.
            extra: Dict metadata bổ sung (metrics, config...).

        Returns:
            Path tới file đã lưu.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 0,
        tb_writer: Any = None,
    ) -> dict[str, list[float]]:
        """Chạy full training loop qua các epoch.

        Args:
            epochs: Số epoch.
            early_stopping_patience: Số epoch không cải thiện thì dừng (0 = tắt).
            tb_writer: TensorBoard writer (tuỳ chọn).

        Returns:
            History dict.
        """
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            started = time.perf_counter()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            elapsed = time.perf_counter() - started

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_seconds"].append(elapsed)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"{elapsed:.1f}s"
            )

            if tb_writer is not None:
                tb_writer.log_epoch(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    lr=self.optimizer.param_groups[0]["lr"],
                    seconds=elapsed,
                )

            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Checkpoint theo val accuracy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                patience_counter = 0
                if self.run_dir is not None:
                    self.save_checkpoint(
                        self.run_dir / "checkpoints" / "best.pt",
                        extra={"epoch": epoch, "val_acc": val_acc},
                    )
            else:
                patience_counter += 1
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"Early stopping tại epoch {epoch} (val acc không cải thiện)")
                    break

        return self.history
```

#### 2.4. TensorBoard — monitor training process (đề bài bước 3)

```python
# src/training/tensorboard_utils.py
"""Ghi log lên TensorBoard để monitor quá trình train."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Wrapper quanh `SummaryWriter` với tag chuẩn hoá.

    Attributes:
        writer: `SummaryWriter` gốc.
        run_dir: Thư mục run.
    """

    def __init__(self, run_dir: Path, enabled: bool = True):
        self.run_dir = Path(run_dir)
        self.enabled = enabled
        self.writer: SummaryWriter | None = None
        if enabled:
            tb_dir = self.run_dir / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        seconds: float,
    ) -> None:
        """Ghi scalar của 1 epoch.

        Tags:
            train/loss, val/loss, train/accuracy, val/accuracy, lr, epoch/time_seconds
        """
        if self.writer is None:
            return
        self.writer.add_scalar("train/loss", train_loss, epoch)
        self.writer.add_scalar("val/loss", val_loss, epoch)
        self.writer.add_scalar("train/accuracy", train_acc, epoch)
        self.writer.add_scalar("val/accuracy", val_acc, epoch)
        self.writer.add_scalar("lr", lr, epoch)
        self.writer.add_scalar("epoch/time_seconds", seconds, epoch)

    def log_weight_histograms(self, model: torch.nn.Module, epoch: int) -> None:
        """Ghi histogram trọng số các layer chính (phát hiện exploding/vanishing).

        Args:
            model: Model cần ghi.
            epoch: Epoch hiện tại.
        """
        if self.writer is None:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(
                    f"model/grad/{name}", param.grad, epoch
                )

    def log_grad_norms(self, model: torch.nn.Module, epoch: int) -> None:
        """Ghi độ lớn gradient mỗi layer."""
        if self.writer is None:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.norm().item()
                self.writer.add_scalar(f"grad_norm/{name}", norm, epoch)

    def log_images(self, images: torch.Tensor, epoch: int) -> None:
        """Ghi ảnh mẫu (đã denormalize) lên TensorBoard.

        Args:
            images: Tensor `(B, 3, H, W)` đã normalize theo ImageNet.
            epoch: Epoch hiện tại.
        """
        if self.writer is None:
            return
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        unnorm = images[:16].detach().cpu() * std + mean
        self.writer.add_images("samples/train_images", unnorm, epoch)

    def close(self) -> None:
        """Đóng writer."""
        if self.writer is not None:
            self.writer.close()
```

**Bảng tag TensorBoard** (theo đề — monitor training process):

| Tag | Loại | Ý nghĩa |
|-----|------|---------|
| `train/loss` | scalar | Loss tập train — giảm = học được |
| `val/loss` | scalar | Loss tập validation — **tăng = overfit** |
| `train/accuracy` | scalar | Accuracy train |
| `val/accuracy` | scalar | Accuracy validation — **chính là metric chọn checkpoint** |
| `lr` | scalar | Learning rate hiện tại |
| `epoch/time_seconds` | scalar | Thời gian mỗi epoch |
| `grad_norm/<layer>` | scalar | Độ lớn gradient — phát hiện gradient explosion |
| `model/grad/<layer>` | histogram | Phân bố gradient |
| `samples/train_images` | image | Ảnh mẫu sau transform (đã denormalize) |

#### 2.5. CLI train + override hyperparameter

```python
# src/training/__main__.py
"""CLI training: chạy 1 model, nhiều model, hoặc sweep siêu tham số."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.data.dataloader import DataLoaderConfig, build_dataloaders
from src.models.factory import ModelConfig, build_model, count_parameters
from src.training.loss import build_criterion
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.training.tensorboard_utils import TensorBoardLogger
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.runs import create_run_dir, setup_logging, write_json


def set_seed(seed: int) -> None:
    """Đặt seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(spec: str) -> torch.device:
    """Chọn thiết bị theo config."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 classifier")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--models", nargs="+", default=None, help="Danh sách model cần chạy")
    parser.add_argument("--strategy", default=None, choices=["freeze_backbone", "fine_tune"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, nargs="+", default=None)
    parser.add_argument("--image-size", type=int, default=None, help="224 = Tier A, 64 = Tier B")
    parser.add_argument("--optimizer", default=None, choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler", default=None, choices=["none", "cosine", "step", "plateau"])
    args = parser.parse_args(argv)

    raw = load_config(args.config)
    section = raw.get("train", raw)
    tb_section = raw.get("tensorboard", {})
    data_section = raw.get("data", {})

    models = args.models or [section.get("model", "resnet18")]
    lrs = args.lr or [section.get("lr", 1e-3)]
    batch_sizes = args.batch_size or [section.get("batch_size", 32)]

    set_seed(int(section.get("seed", 42)))
    device = resolve_device(section.get("device", "auto"))
    all_results = []

    for model_name in models:
        for lr in lrs:
            for batch_size in batch_sizes:
                cfg = TrainConfig(
                    model=model_name,
                    strategy=args.strategy or section.get("strategy", "freeze_backbone"),
                    freeze_until=int(section.get("freeze_until", 1)),
                    epochs=args.epochs or int(section.get("epochs", 5)),
                    batch_size=batch_size,
                    lr=lr,
                    optimizer=args.optimizer or section.get("optimizer", "adam"),
                    weight_decay=float(section.get("weight_decay", 1e-4)),
                    momentum=float(section.get("momentum", 0.9)),
                    scheduler=args.scheduler or section.get("scheduler", "none"),
                    num_workers=int(section.get("num_workers", 4)),
                    seed=int(section.get("seed", 42)),
                    device=section.get("device", "auto"),
                    image_size=args.image_size or int(section.get("image_size", 224)),
                    early_stopping_patience=int(section.get("early_stopping_patience", 0)),
                )

                # --- Data ---
                data_cfg = DataLoaderConfig(
                    data_root=data_section.get("data_root", "data/raw"),
                    image_size=cfg.image_size,
                    batch_size=cfg.batch_size,
                    num_workers=cfg.num_workers,
                    seed=cfg.seed,
                    augment=bool(data_section.get("augment", True)),
                    in_memory=bool(data_section.get("in_memory", True)),
                )
                train_loader, val_loader, test_loader = build_dataloaders(data_cfg)

                # --- Model ---
                run_dir = create_run_dir()
                setup_logging(run_dir)
                model = build_model(
                    ModelConfig(
                        name=cfg.model,
                        num_classes=10,
                        strategy=cfg.strategy,
                        weights=section.get("weights", "DEFAULT"),
                        freeze_until=cfg.freeze_until,
                    )
                )

                # --- Loss / Optimizer / Scheduler ---
                criterion = build_criterion(
                    section.get("criterion", "cross_entropy"),
                    float(section.get("label_smoothing", 0.0)),
                )
                optimizer = build_optimizer(
                    model, cfg.optimizer, cfg.lr, cfg.weight_decay, cfg.momentum
                )
                scheduler = build_scheduler(optimizer, cfg.scheduler, cfg.epochs)

                # --- Train ---
                trainer = Trainer(
                    model, train_loader, val_loader, criterion,
                    optimizer, device, scheduler, run_dir,
                )
                tb = TensorBoardLogger(run_dir, enabled=tb_section.get("enabled", True))
                history = trainer.train(
                    epochs=cfg.epochs,
                    early_stopping_patience=cfg.early_stopping_patience,
                    tb_writer=tb,
                )
                tb.close()

                result = {
                    "model": cfg.model,
                    "strategy": cfg.strategy,
                    "lr": cfg.lr,
                    "batch_size": cfg.batch_size,
                    "image_size": cfg.image_size,
                    "epochs": cfg.epochs,
                    "best_val_acc": trainer.best_val_acc,
                    "best_epoch": trainer.best_epoch,
                    "params": count_parameters(model),
                    "run_dir": str(run_dir),
                }
                write_json(result, run_dir / "metrics.json")
                all_results.append(result)

    # Tổng hợp
    summary_dir = create_run_dir()
    write_json(all_results, summary_dir / "sweep_results.json")
    print("\n=== TỔNG KẾT ===")
    for r in all_results:
        print(
            f"{r['model']:<14} {r['strategy']:<18} lr={r['lr']:<8} "
            f"bs={r['batch_size']:<4} img={r['image_size']:<4} "
            f"best_val_acc={r['best_val_acc']:.2f}%"
        )
    print(f"\nTổng kết: {summary_dir / 'sweep_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Hyperparameter Search Space (đề bài: "Adjust the hyperparameters")

```yaml
# configs/train_config.yaml
train:
  # Backbone (được override bằng --models)
  model: resnet18
  strategy: freeze_backbone
  freeze_until: 1

  # Hyperparameters — đề bài yêu cầu "adjust hyperparameters"
  epochs: 5
  batch_size: 32
  lr: 1.0e-3
  optimizer: adam
  weight_decay: 1.0e-4
  momentum: 0.9
  scheduler: none
  criterion: cross_entropy
  label_smoothing: 0.0

  # Tier A = 224 (báo cáo chính) | Tier B = 64 (sweep rộng)
  image_size: 224

  num_workers: 4
  seed: 42
  device: auto
  early_stopping_patience: 0

tensorboard:
  enabled: true
  log_images: true
  log_histograms: true

# Search space dùng cho sweep (ghi rõ trong báo cáo)
search_space:
  model: [resnet18, vgg11, densenet121]
  learning_rate: [1.0e-4, 1.0e-3, 1.0e-2]
  batch_size: [32, 64]
  optimizer: [adam, adamw, sgd]
  scheduler: [none, cosine]
```

### Requirements (theo đề)

✅ **Define the loss function and optimizer**:
- `build_criterion()`: CrossEntropy (mặc định) + label smoothing
- `build_optimizer()`: Adam / AdamW / SGD, **chỉ truyền params trainable**
- `build_scheduler()`: cosine / step / plateau

✅ **Write the training loop** — đủ 4 bước:
- **Forward**: `output = model(data)`
- **Loss**: `loss = criterion(output, target)`
- **Backward**: `loss.backward()`
- **Optimize**: `optimizer.step()` (sau `optimizer.zero_grad()`)

✅ **Use TensorBoard**:
- Scalar: loss / accuracy (train + val) / lr / thời gian epoch
- Histogram: gradient theo từng layer
- Image: ảnh mẫu đã denormalize
- Checkpoint theo val accuracy + early stopping

✅ **Adjust the hyperparameters**:
- Toàn bộ hyperparameter nằm trong `configs/train_config.yaml`
- Override được từ CLI: `--lr`, `--batch-size`, `--optimizer`, `--scheduler`, `--epochs`
- Quét nhiều tổ hợp: `--lr 1e-4 1e-3 --batch-size 32 64`

### Test
- Train 1 epoch với model nhỏ + `FakeData` → verify loss giảm hoặc params đổi
- Verify `.grad` của head khác `None`, `.grad` của backbone đã freeze là `None`
- Verify `best.pt` được tạo khi val acc tăng
- Verify load checkpoint → cùng output
- Verify `metrics.json` có `best_val_acc`
- Verify TensorBoard tạo file event, không raise khi log scalar/histogram/image
- Verify early stopping kích hoạt đúng patience

> **Test nhanh:** dùng `FakeData` + model nhỏ, `epochs=1`, `batch_size=4` → mỗi test < 5 giây.

### Input từ Người 1
- Tên model + strategy tốt nhất (VD: `resnet18` + `freeze_backbone`)
- `configs/model_config.yaml` hoàn chỉnh
- Bảng params để so sánh

### Output cho Người 3
- **Checkpoint**: `results/runs/<ts>/checkpoints/best.pt`
- **Training history**: `results/runs/<ts>/metrics.json` (train/val loss + acc mỗi epoch)
- **TensorBoard logs**: `results/runs/<ts>/tb/`
- **Kết quả sweep**: `results/runs/<ts>/sweep_results.json`

### Report cần viết (lưu vào `docs/reports/nguoi2_training_report.md`)

```markdown
# Training Report

## 1. Training Configuration
- Model: ... | Strategy: ...
- Image size (Tier): ... (224 = Tier A / 64 = Tier B)
- Batch size: ... | Learning rate: ... | Optimizer: ... | Scheduler: ...
- Loss: ... | Epochs: ...

## 2. Training Progress
| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Time (s) |
|-------|------------|----------|-----------|---------|----------|

## 3. TensorBoard Monitoring
- Đường dẫn log: results/runs/<ts>/tb/
- Tags đã ghi: ...
- Nhận xét về overfit (gap train/val)

## 4. Hyperparameter Sweep
| Model | LR | Batch | Optimizer | Image Size | Best Val Acc |
|-------|-----|-------|-----------|------------|--------------|

## 5. Best Model
- Model + strategy: ...
- Epoch tốt nhất: ...
- Best val accuracy: ...%
- Checkpoint: results/runs/<ts>/checkpoints/best.pt

## 6. Training Time
- Tổng: ... | Mỗi epoch: ... | Ảnh hưởng của image_size: ...

## 7. Files Created
- List all files
```

---

## 👤 NGƯỜI 3: Evaluation & Metrics

### Nhiệm vụ (theo đề)
1. **Evaluate the Model** — đánh giá hiệu năng trên test set
2. Báo cáo metrics đầy đủ (accuracy, precision, recall, F1, confusion matrix)
3. Trực quan hoá kết quả

### File cần tạo/sửa

| File | Chức năng | Trạng thái |
|------|-----------|------------|
| `src/evaluation/metrics.py` | **Chính** — accuracy, macro-F1, per-class, confusion matrix | Tạo mới |
| `src/evaluation/evaluator.py` | **Chính** — chạy inference trên test set | Tạo mới |
| `src/evaluation/visualize.py` | **Chính** — biểu đồ: confusion matrix, F1 bar, ảnh đoán sai | Tạo mới |
| `src/evaluation/__init__.py` | Export evaluation modules | Sửa thêm |
| `src/evaluation/__main__.py` | CLI đánh giá | Tạo mới |
| `configs/eval_config.yaml` | Cấu hình đánh giá | Tạo mới |
| `tests/unit/test_metrics.py` | Unit test các công thức metrics | Tạo mới |
| `tests/integration/test_evaluation.py` | Integration test đánh giá end-to-end | Tạo mới |

### Chi tiết implementation

#### 3.1. Metrics

```python
# src/evaluation/metrics.py
"""Các chỉ số đánh giá — tự tính bằng torch thuần."""

from __future__ import annotations

import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Độ chính xác toàn cục.

    Args:
        pred: Dự đoán `(N,)`.
        target: Nhãn thật `(N,)`.

    Returns:
        Accuracy trong khoảng [0, 1].
    """
    if pred.numel() == 0:
        raise ValueError("Input rỗng — không thể tính accuracy")
    return (pred == target).float().mean().item()


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Ma trận nhầm lẫn.

    Args:
        pred: Dự đoán `(N,)`.
        target: Nhãn thật `(N,)`.
        num_classes: Số lớp.

    Returns:
        Tensor `(num_classes, num_classes)`, ô `[i][j]` = số mẫu thật là `i`,
        bị dự đoán sai thành `j`.

    Raises:
        ValueError: Nếu nhãn nằm ngoài `[0, num_classes)`.
    """
    if pred.numel() == 0:
        raise ValueError("Input rỗng — không thể tính confusion matrix")
    if target.min().item() < 0 or target.max().item() >= num_classes:
        raise ValueError(
            f"Nhãn phải nằm trong [0, {num_classes}), "
            f"nhận min={target.min().item()}, max={target.max().item()}"
        )

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(target.cpu().tolist(), pred.cpu().tolist()):
        cm[t, p] += 1
    return cm


def per_class_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> dict[str, dict[str, float]]:
    """Precision / Recall / F1 cho từng lớp.

    Args:
        pred: Dự đoán `(N,)`.
        target: Nhãn thật `(N,)`.
        num_classes: Số lớp.

    Returns:
        Dict `{class_index: {precision, recall, f1, support}}`.
    """
    cm = confusion_matrix(pred, target, num_classes)
    result: dict[str, dict[str, float]] = {}

    for cls in range(num_classes):
        tp = cm[cls, cls].item()
        fp = cm[:, cls].sum().item() - tp
        fn = cm[cls, :].sum().item() - tp
        support = cm[cls, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        result[str(cls)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support),
        }
    return result


def macro_f1(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """F1 trung bình macro — metric chính của bài.

    Args:
        pred: Dự đoán `(N,)`.
        target: Nhãn thật `(N,)`.
        num_classes: Số lớp.

    Returns:
        Macro F1 trong khoảng [0, 1].
    """
    per_class = per_class_metrics(pred, target, num_classes)
    return sum(v["f1"] for v in per_class.values()) / num_classes
```

> **Quyết định:** tự tính bằng `torch` thuần thay vì thêm `scikit-learn`. Thêm dependency chỉ để tính F1 là không đáng, và giữ `requirements.txt` gọn. (Khác với plan mẫu FashionMNIST dùng `sklearn` — project này đã cài `torch` + `torchvision` + `tensorboard` + `matplotlib` + `numpy` + `pyyaml`.)

#### 3.2. Evaluator

```python
# src/evaluation/evaluator.py
"""Chạy inference trên test set và tính metrics."""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    accuracy,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


class ModelEvaluator:
    """Đánh giá model đã train trên test set.

    Attributes:
        model: Model cần đánh giá.
        test_loader: DataLoader tập test.
        device: Thiết bị chạy.
        class_names: Tên các lớp.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device | str = "cpu",
        class_names: list[str] | None = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = torch.device(device)
        self.class_names = class_names or CIFAR10_CLASSES

    @torch.no_grad()
    def predict(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dự đoán trên toàn bộ test set.

        Returns:
            Tuple `(preds, labels, probs)` — mỗi phần tử tensor `(N,)`.
        """
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_probs: list[torch.Tensor] = []

        for data, target in self.test_loader:
            data = data.to(self.device)
            output = self.model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = output.max(1)

            all_preds.append(predicted.cpu())
            all_labels.append(target.cpu())
            all_probs.append(probs.cpu())

        return (
            torch.cat(all_preds),
            torch.cat(all_labels),
            torch.cat(all_probs),
        )

    def evaluate(self) -> dict:
        """Đánh giá đầy đủ.

        Returns:
            Dict metrics gồm accuracy, macro_f1, per_class, confusion_matrix,
            inference time.
        """
        started = time.perf_counter()
        preds, labels, _ = self.predict()
        elapsed = time.perf_counter() - started
        n = labels.numel()

        num_classes = len(self.class_names)
        per_class = per_class_metrics(preds, labels, num_classes)
        # Đổi key từ index sang tên lớp cho dễ đọc
        named = {
            self.class_names[int(k)]: v for k, v in per_class.items()
        }

        return {
            "accuracy": accuracy(preds, labels),
            "macro_f1": macro_f1(preds, labels, num_classes),
            "per_class": named,
            "confusion_matrix": confusion_matrix(preds, labels, num_classes),
            "inference_time_ms_per_sample": (elapsed / n * 1000) if n else 0.0,
            "num_samples": n,
        }
```

#### 3.3. Visualize

```python
# src/evaluation/visualize.py
"""Biểu đồ kết quả đánh giá."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # headless — không mở cửa sổ
import matplotlib.pyplot as plt  # noqa: E402


def plot_confusion_matrix(
    cm: torch.Tensor,
    class_names: list[str],
    save_path: Path | None = None,
    normalize: bool = True,
) -> Path | None:
    """Vẽ heatmap ma trận nhầm lẫn.

    Args:
        cm: Ma trận `(C, C)`.
        class_names: Tên lớp.
        save_path: Đường dẫn lưu ảnh.
        normalize: Chuẩn hoá theo hàng (tỉ lệ trong mỗi lớp thật).

    Returns:
        Path file đã lưu nếu có `save_path`.
    """
    matrix = cm.float().numpy()
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)")

    thresh = matrix.max() / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:.2f}",
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return Path(save_path)
    plt.close(fig)
    return None


def plot_per_class_f1(
    per_class: dict[str, dict[str, float]],
    save_path: Path | None = None,
) -> Path | None:
    """Vẽ bar chart F1 của từng lớp.

    Args:
        per_class: Dict `{class_name: {precision, recall, f1, support}}`.
        save_path: Đường dẫn lưu ảnh.

    Returns:
        Path file đã lưu nếu có `save_path`.
    """
    names = list(per_class.keys())
    f1s = [per_class[n]["f1"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if f >= 0.8 else "#e67e22" if f >= 0.5 else "#e74c3c" for f in f1s]
    ax.bar(names, f1s, color=colors)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    for idx, f1 in enumerate(f1s):
        ax.text(idx, f1 + 0.02, f"{f1:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return Path(save_path)
    plt.close(fig)
    return None


def plot_misclassified(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: list[str],
    save_path: Path | None = None,
    max_samples: int = 12,
) -> Path | None:
    """Vẽ grid các ảnh model đoán sai.

    Args:
        images: Tensor `(N, 3, H, W)` đã normalize.
        true_labels: Nhãn thật `(N,)`.
        pred_labels: Nhãn dự đoán `(N,)`.
        class_names: Tên lớp.
        save_path: Đường dẫn lưu ảnh.
        max_samples: Số ảnh tối đa.

    Returns:
        Path file đã lưu nếu có `save_path`.
    """
    wrong = (true_labels != pred_labels).nonzero(as_tuple=True)[0][:max_samples]
    if wrong.numel() == 0:
        return None

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    n = len(wrong)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
    axes = np.array(axes).flatten()

    for ax_idx, idx in enumerate(wrong):
        img = (images[idx].detach().cpu() * std + mean).clamp(0, 1)
        ax = axes[ax_idx]
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(
            f"True: {class_names[true_labels[idx]]}\n"
            f"Pred: {class_names[pred_labels[idx]]}",
            fontsize=8, color="red",
        )
        ax.axis("off")

    for ax_idx in range(n, len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle("Misclassified Examples", fontsize=12)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return Path(save_path)
    plt.close(fig)
    return None


def plot_train_curves(
    history: dict[str, list[float]],
    save_path: Path | None = None,
) -> Path | None:
    """Vẽ lại đường loss / accuracy từ history.

    Args:
        history: History từ Trainer.
        save_path: Đường dẫn lưu ảnh.

    Returns:
        Path file đã lưu nếu có `save_path`.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(history["val_loss"], label="Val Loss", marker="s")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc", marker="o")
    ax2.plot(history["val_acc"], label="Val Acc", marker="s")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return Path(save_path)
    plt.close(fig)
    return None
```

#### 3.4. CLI đánh giá

```python
# src/evaluation/__main__.py
"""CLI đánh giá model trên test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.data.dataloader import DataLoaderConfig, build_dataloaders
from src.evaluation.evaluator import CIFAR10_CLASSES, ModelEvaluator
from src.evaluation.visualize import (
    plot_confusion_matrix,
    plot_misclassified,
    plot_per_class_f1,
    plot_train_curves,
)
from src.models.factory import ModelConfig, build_model
from src.utils.config import load_config
from src.utils.runs import create_run_dir, setup_logging, write_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đánh giá model trên test set")
    parser.add_argument("--config", default="configs/eval_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Đường dẫn best.pt")
    parser.add_argument("--model", default=None)
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args(argv)

    raw = load_config(args.config)
    section = raw.get("evaluation", raw)
    data_section = raw.get("data", {})

    model_name = args.model or section.get("model", "resnet18")
    strategy = args.strategy or section.get("strategy", "freeze_backbone")
    image_size = args.image_size or int(section.get("image_size", 224))
    ckpt_path = Path(args.checkpoint or section.get("checkpoint", ""))

    # --- Data ---
    data_cfg = DataLoaderConfig(
        data_root=data_section.get("data_root", "data/raw"),
        image_size=image_size,
        batch_size=int(section.get("batch_size", 64)),
        num_workers=int(section.get("num_workers", 4)),
        seed=42,
        augment=False,
        in_memory=True,
    )
    _, _, test_loader = build_dataloaders(data_cfg)

    # --- Model ---
    model = build_model(
        ModelConfig(name=model_name, num_classes=10, strategy=strategy, weights="DEFAULT")
    )
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # --- Evaluate ---
    run_dir = create_run_dir()
    setup_logging(run_dir)
    evaluator = ModelEvaluator(model, test_loader, device="cpu", class_names=CIFAR10_CLASSES)
    metrics = evaluator.evaluate()

    print(f"Test Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1      : {metrics['macro_f1'] * 100:.2f}%")
    print(f"Inference     : {metrics['inference_time_ms_per_sample']:.2f} ms/sample")

    write_json(
        {
            "model": model_name,
            "strategy": strategy,
            "image_size": image_size,
            "checkpoint": str(ckpt_path),
            "test": metrics,
            "train_summary": {
                "best_val_accuracy": checkpoint.get("best_val_acc"),
                "best_epoch": checkpoint.get("best_epoch"),
            },
        },
        run_dir / "metrics.json",
    )

    # --- Plots ---
    plots_dir = run_dir / "plots"
    plot_confusion_matrix(metrics["confusion_matrix"], CIFAR10_CLASSES, plots_dir / "confusion_matrix.png")
    plot_per_class_f1(metrics["per_class"], plots_dir / "per_class_f1.png")
    if checkpoint.get("history"):
        plot_train_curves(checkpoint["history"], plots_dir / "train_curves.png")

    print(f"Artifact: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### Requirements (theo đề)

✅ **Evaluate the Model** trên test dataset:
- Accuracy tổng thể
- Macro-F1 (metric chính — dataset cân bằng 1000 mẫu/lớp)
- Per-class precision / recall / F1 / support
- Confusion matrix (chuẩn hoá theo hàng)
- Inference time ms/sample

✅ **Sinh báo cáo + biểu đồ**:
- `metrics.json` theo schema thống nhất cho mọi run → M5 tổng hợp được
- Heatmap confusion matrix
- Bar chart F1 mỗi lớp
- Grid ảnh đoán sai
- Đường loss/accuracy vẽ lại từ history

### Test
- `accuracy` với dự đoán đúng hết → 1.0; sai hết → 0.0
- `macro_f1` đối chiếu với giá trị tính tay
- `confusion_matrix`: tổng mọi ô = số mẫu; tổng mỗi hàng = support của lớp đó
- Nhãn ngoài `[0, 10)` → `ValueError`
- Input rỗng → `ValueError`
- Evaluate end-to-end với model giả → `metrics.json` đủ key
- Load lại checkpoint → cùng accuracy

### Input từ Người 2
- **Checkpoint**: `results/runs/<ts>/checkpoints/best.pt`
- **Training history**: `results/runs/<ts>/metrics.json`
- **Thông tin config** để dựng lại đúng model (tên, strategy, image_size)

### Output (Final Deliverables theo đề)
- ✅ **Python code**
- ⏳ **Brief report** (3 reports + 1 final report)
- ⏳ **Evaluation report** (`docs/reports/nguoi3_evaluation_report.md`)
- ⏳ **Confusion matrix + biểu đồ** (`results/runs/<ts>/plots/`)

### Report cần viết (lưu vào `docs/reports/nguoi3_evaluation_report.md`)

```markdown
# Evaluation Report

## 1. Test Set Configuration
- Checkpoint: ... | Model: ... | Strategy: ...
- Image size (Tier): ... (224 = Tier A / 64 = Tier B)
- Số mẫu test: 10.000

## 2. Overall Metrics
| Metric | Value |
|--------|-------|
| Accuracy | ...% |
| Macro-F1 | ...% |
| Inference time | ... ms/sample |

## 3. Per-Class Metrics
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|

## 4. Confusion Matrix
- File: results/runs/<ts>/plots/confusion_matrix.png
- Cặp lớp dễ nhầm nhất: ...
- Nhận xét: ...

## 5. Error Analysis
- File: results/runs/<ts>/plots/misclassified.png
- Mô tả: ...

## 6. Comparison With Baseline
| Run | Model | Strategy | Image Size | Accuracy | Macro-F1 |
|-----|-------|----------|------------|----------|----------|

## 7. Files Created
- List all files
```

---

## 📋 FILES SUMMARY

### Theo người phụ trách

| Người | Module (theo đề) | File chính | File phụ |
|-------|------------------|------------|----------|
| 1 | Load + Adapt Pre-trained Model | `factory.py`, `registry.py` | `__main__.py`, `model_config.yaml`, `test_factory.py`, `test_model_pipeline.py` |
| 2 | Training Loop + TensorBoard + Hyperparams | `trainer.py`, `tensorboard_utils.py` | `loss.py`, `optimizer.py`, `scheduler.py`, `__main__.py`, `train_config.yaml` |
| 3 | Evaluate + Metrics + Plots | `metrics.py`, `evaluator.py`, `visualize.py` | `__main__.py`, `eval_config.yaml`, `test_metrics.py`, `test_evaluation.py` |

### Mapping files với yêu cầu đề

| Yêu cầu đề | Người | File |
|------------|-------|------|
| Load CIFAR-10 dataset | (Phase 1) | `src/data/dataset.py` ✅ |
| Apply transforms | (Phase 1) | `src/data/transforms.py` ✅ |
| Use DataLoader to create batches | (Phase 1) | `src/data/dataloader.py` ✅ |
| **Load a Pre-trained Model** | **Người 1** | `src/models/factory.py`, `src/models/registry.py` |
| **Explore the Model Architecture** | **Người 1** | `src/models/factory.py` (`describe_model`), `src/models/__main__.py` |
| **Adapt the Model for a Specific Task** | **Người 1** | `src/models/factory.py` (`replace_head`) |
| **Freeze earlier layers** | **Người 1** | `src/models/factory.py` (`freeze_backbone`) |
| **Unfreeze final layers** | **Người 1** | `src/models/factory.py` (`unfreeze_last`) |
| **Experiment with different pre-trained models** | **Người 1** | `configs/model_config.yaml` + CLI `--models` |
| **Define the loss function and optimizer** | **Người 2** | `src/training/loss.py`, `src/training/optimizer.py` |
| **Write the training loop** | **Người 2** | `src/training/trainer.py` |
| **Use TensorBoard to monitor** | **Người 2** | `src/training/tensorboard_utils.py` |
| **Adjust the hyperparameters** | **Người 2** | `configs/train_config.yaml` + CLI override |
| **Evaluate the Model** | **Người 3** | `src/evaluation/evaluator.py`, `src/evaluation/metrics.py` |

---

## 📁 CẤU TRÚC THƯ MỤC

```
src/
├── data/                              # Phase 1 ✅
│   ├── __init__.py
│   ├── dataset.py                     # Cifar10Dataset ✅
│   ├── transforms.py                  # build_transform() ✅
│   ├── dataloader.py                  # build_dataloaders() ✅
│   ├── split.py                       # split_train_val() ✅
│   ├── prepare.py                     # export processed/splits ✅
│   └── __main__.py                    # CLI data ✅
│
├── models/                            # 👤 Người 1
│   ├── __init__.py
│   ├── registry.py                    # BACKBONE_REGISTRY, HEAD_SPECS
│   ├── factory.py                     # build_model, replace_head, freeze/unfreeze
│   └── __main__.py                    # CLI khám phá architecture
│
├── training/                          # 👤 Người 2
│   ├── __init__.py
│   ├── loss.py                        # build_criterion
│   ├── optimizer.py                   # build_optimizer
│   ├── scheduler.py                   # build_scheduler
│   ├── trainer.py                     # Trainer class - training loop
│   ├── tensorboard_utils.py           # TensorBoardLogger
│   └── __main__.py                    # CLI train + sweep
│
├── evaluation/                        # 👤 Người 3
│   ├── __init__.py
│   ├── metrics.py                     # accuracy, macro_f1, per_class, confusion_matrix
│   ├── evaluator.py                   # ModelEvaluator
│   ├── visualize.py                   # confusion matrix, F1 bar, misclassified
│   └── __main__.py                    # CLI evaluate
│
└── utils/                             # ✅ (Phase 1)
    ├── config.py                      # load_config, deep_merge, project_root
    └── runs.py                        # create_run_dir, setup_logging, write_json

configs/
├── data_config.yaml                   # ✅ (Phase 1)
├── model_config.yaml                  # 👤 Người 1
├── train_config.yaml                  # 👤 Người 2
└── eval_config.yaml                   # 👤 Người 3

docs/
├── data-flow.md                       # ✅ (Phase 1)
├── architecture.md                    # Tạo ở M2
└── reports/
    ├── nguoi1_models_report.md        # 👤 Người 1
    ├── nguoi2_training_report.md      # 👤 Người 2
    └── nguoi3_evaluation_report.md    # 👤 Người 3

tests/
├── unit/
│   ├── test_dataset.py                # ✅
│   ├── test_transforms.py             # ✅
│   ├── test_split.py                  # ✅
│   ├── test_prepare.py                # ✅
│   ├── test_factory.py                # 👤 Người 1
│   ├── test_optimizer.py              # 👤 Người 2
│   └── test_metrics.py                # 👤 Người 3
└── integration/
    ├── test_dataloader_pipeline.py    # ✅
    ├── test_model_pipeline.py         # 👤 Người 1
    ├── test_training_loop.py          # 👤 Người 2
    └── test_evaluation.py             # 👤 Người 3

results/runs/
└── <timestamp>/                       # Mỗi run 1 folder
    ├── config.yaml                    # Snapshot config
    ├── logs.txt                       # Log console
    ├── metrics.json                   # Metrics (schema thống nhất)
    ├── model_info.json                # Tham số model (M2)
    ├── sweep_results.json             # Tổng hợp sweep (M3)
    ├── tb/                            # TensorBoard event files
    ├── checkpoints/
    │   └── best.pt                    # Checkpoint theo val accuracy
    └── plots/
        ├── confusion_matrix.png       # 👤 Người 3
        ├── per_class_f1.png           # 👤 Người 3
        ├── misclassified.png          # 👤 Người 3
        └── train_curves.png           # 👤 Người 3
```

---

## ✅ CHECKLIST HOÀN THÀNH (theo đề)

### 👤 Người 1: Model Factory
- [ ] `src/models/registry.py` tạo xong (3 backbone + block_units đã introspect)
- [ ] `src/models/factory.py` tạo xong
- [ ] `src/models/__main__.py` tạo xong
- [ ] `configs/model_config.yaml` tạo xong
- [ ] Forward pass test passed với 3 architectures → output `(1, 10)`
- [ ] `describe_model()` in cấu trúc + đếm params (đề bài bước 3)
- [ ] `replace_head()` test passed cho cả `Linear` và `Sequential` (VGG11)
- [ ] Freeze test passed — verify % params trainable < 1% cho cả 3 model
- [ ] **VGG11: `freeze_head_full: false` — verify chỉ 40,970 params trainable**
- [ ] Fine-tune test passed — verify block cuối có gradient
- [ ] `docs/reports/nguoi1_models_report.md` viết xong

### 👤 Người 2: Training Loop + TensorBoard
- [ ] `src/training/loss.py` tạo xong
- [ ] `src/training/optimizer.py` tạo xong (chỉ params trainable)
- [ ] `src/training/scheduler.py` tạo xong
- [ ] `src/training/trainer.py` tạo xong (forward, loss, backward, optimize)
- [ ] `src/training/tensorboard_utils.py` tạo xong
- [ ] `src/training/__main__.py` tạo xong (override hyperparameter + multi-model)
- [ ] `configs/train_config.yaml` tạo xong
- [ ] Training loop test passed — loss giảm
- [ ] **TensorBoard log đầy đủ** (`results/runs/<ts>/tb/`)
- [ ] **Checkpoint lưu theo val accuracy** (`results/runs/<ts>/checkpoints/best.pt`)
- [ ] Hyperparameter sweep chạy được (Tier B, 64×64)
- [ ] `docs/reports/nguoi2_training_report.md` viết xong

### 👤 Người 3: Evaluation + Metrics
- [ ] `src/evaluation/metrics.py` tạo xong
- [ ] `src/evaluation/evaluator.py` tạo xong
- [ ] `src/evaluation/visualize.py` tạo xong
- [ ] `src/evaluation/__main__.py` tạo xong
- [ ] `configs/eval_config.yaml` tạo xong
- [ ] Test accuracy + macro-F1 đúng (đối chiếu giá trị tính tay)
- [ ] Confusion matrix tổng = số mẫu test
- [ ] **Biểu đồ sinh đủ** (`plots/confusion_matrix.png`, `plots/per_class_f1.png`)
- [ ] **Ảnh đoán sai** (`plots/misclassified.png`)
- [ ] `docs/reports/nguoi3_evaluation_report.md` viết xong

### 📦 Final Deliverables (theo đề)
- [ ] Python code hoàn chỉnh
- [ ] 3 reports từ 3 người
- [ ] TensorBoard logs (`results/runs/<ts>/tb/`)
- [ ] Biểu đồ đánh giá (`results/runs/<ts>/plots/`)
- [ ] `docs/PHASE2_FINAL_REPORT.md` (tổng hợp)

---

## 🎯 BỔ SUNG

### Optional Enhancements
1. **Data Augmentation** — đã có sẵn trong M1 (`RandomCrop` + `HorizontalFlip`); có thể thêm `ColorJitter`/`RandomAffine` ở Tier A
2. **GradCAM** — Người 3 có thể thêm visualization cho quyết định của model
3. **Mixed precision / gradient accumulation** — không cần trên CPU
4. **Learning rate finder** — Người 2 có thể thêm để chọn `lr` tự động
5. **Unit Tests** — mỗi người tự viết test theo checklist

### Integration Test
- Chạy full pipeline: Load data → Build model → Train → Evaluate → Save/Load
- Tạo `docs/reports/PHASE2_INTEGRATION_REPORT.md`

### Final Report
Tạo `docs/PHASE2_FINAL_REPORT.md`:
- Executive summary
- So sánh 3 backbone × 2 chiến lược (freeze / fine-tune)
- Bảng kết quả cuối cùng (kèm điều kiện Tier A / Tier B)
- Lessons learned
- Recommendations

---

## 📞 WORKFLOW GIỮA 3 NGƯỜI

```
Người 1 (Model)                Người 2 (Training)              Người 3 (Evaluation)
─────────────────              ─────────────────                ─────────────────
1. Registry 3 backbone
2. replace_head (Linear +
   Sequential — VGG11)
3. freeze_backbone
4. unfreeze_last
5. describe_model ─────────→  6. build_optimizer (chỉ
6. So sánh params                params trainable)
7. Chọn model + strategy ───→  7. Training loop (forward,
   + model_config.yaml             loss, backward, optimize)
                                8. TensorBoard logger
                                9. Hyperparameter sweep
                                  (Tier B, 64×64)
                               10. Train Tier A (224) ───→  11. Load best.pt
                                  best model                      12. Evaluate trên
                               13. Save checkpoint ─────────→     test set
                                  best.pt                         13. Metrics +
                               14. History + tb logs ─────→     confusion matrix
                                                                 14. Plots
                                  ←───────────────────────────────── 15. Final report
```

### Giao tiếp giữa các người

| Từ | Đến | Giao |
|----|-----|------|
| Người 1 | Người 2 | Tên model + strategy tốt nhất, `model_config.yaml`, bảng params |
| Người 2 | Người 3 | Checkpoint `best.pt`, training history, image_size đã dùng (Tier A/B), tb logs |
| Người 3 | Tất cả | Test accuracy, macro-F1, confusion matrix, đề xuất model cuối |

---

## ⚠️ LƯU Ý QUAN TRỌNG CHO CẢ 3 NGƯỜI

1. **Không sửa `src/data/`** — M1 đã ổn định, 47 test pass. M2–M4 chỉ **đọc** từ đó.
2. **Không hardcode hyperparameter** — mọi giá trị vào YAML (đề bài yêu cầu "adjust hyperparameters" → phải chỉnh được từ file/lệnh).
3. **Ghi rõ Tier trong mọi bảng** — kết quả 224×224 và 64×64 **không** so sánh trực tiếp được.
4. **Mọi run ghi `metrics.json` theo cùng schema** → M5 tổng hợp tự động được.
5. **Test không cần data thật** — dùng `FakeData` để test nhanh; chỉ chạy full data ở bước verify.
6. **Mỗi người commit riêng** — 1 module = 1 commit, Conventional Commits.
7. **Cập nhật `progress_status.md` + `README.md`** sau khi xong phần mình.

---

**Ngày tạo:** 2026-10-25
**Cập nhật lần cuối:** 2026-10-25 (align với `đề.txt`, 3 người, CIFAR-10, có đo thời gian train thật)
**Phiên bản:** 3.0
**Trạng thái:** Ready for Phase 2 (3 người, aligned với đề bài)
