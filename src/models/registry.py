"""Bang tra cuu cac backbone pre-trained ho tro trong project.

Cau truc backbone o day duoc introspect truc tiep tu torchvision 0.29
(khong doan), vi moi model co cach to chuc head khac nhau:

- resnet18     : fc = Linear(512, 1000)              | stage: layer1..layer4
- vgg11        : classifier = Sequential[7]           | 8 Conv2d trong features
- densenet121  : classifier = Linear(1024, 1000)     | denseblock1..4 + transition
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn
import torchvision.models as tv_models


@dataclass(frozen=True)
class BackboneSpec:
    """Mo ta cau truc mot backbone de thay head va unfreeze cho dung.

    Attributes:
        builder: Ham dung model tu torchvision.
        head_attr: Ten attribute chua classification head ("fc" | "classifier").
        head_type: "linear" (1 lop Linear) | "sequential" (Sequential nhieu lop).
        block_units: Duong dan cac "don vi" (block) theo thu tu nong -> sau.
            Dung de unfreeze n block cuoi.
        extra_trainable: Duong dan module luon trainable khi fine-tune
            (VD: `features.transition3`, `features.norm5` cua DenseNet).
    """

    builder: Callable[..., nn.Module]
    head_attr: str
    head_type: str
    block_units: tuple[str, ...]
    extra_trainable: tuple[str, ...] = ()


# --- ResNet18 ---
# conv1 -> bn1 -> relu -> maxpool -> layer1..layer4 -> avgpool -> fc
_RESNET18 = BackboneSpec(
    builder=tv_models.resnet18,
    head_attr="fc",
    head_type="linear",
    block_units=("layer1", "layer2", "layer3", "layer4"),
)

# --- VGG11 ---
# features (8 conv block) -> avgpool -> classifier (Sequential 7 phan tu)
_VGG11 = BackboneSpec(
    builder=tv_models.vgg11,
    head_attr="classifier",
    head_type="sequential",
    block_units=(
        "features.0",
        "features.3",
        "features.6",
        "features.8",
        "features.11",
        "features.13",
        "features.16",
        "features.18",
    ),
)

# --- DenseNet121 ---
# conv0 -> norm0 -> relu0 -> pool0 -> denseblock1..4 (+ transition1..3) -> norm5
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
    """Tra cuu spec cua backbone theo ten.

    Args:
        name: Ten backbone, vd: "resnet18".

    Returns:
        `BackboneSpec` tuong ung.

    Raises:
        ValueError: Neu ten khong co trong registry.
    """
    key = name.strip().lower()
    if key not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Model khong ho tro: {name!r}. Cac model ho tro: {list(SUPPORTED_MODELS)}"
        )
    return BACKBONE_REGISTRY[key]


def get_module(model: nn.Module, path: str) -> nn.Module:
    """Lay module con theo duong dan, ho tro chi so so.

    Args:
        model: Model goc.
        path: Duong dan, vd: "features.18", "classifier.0".

    Returns:
        Module tuong ung.

    Raises:
        AttributeError: Neu duong dan khong ton tai.
    """
    obj: object = model
    for part in path.split("."):
        if isinstance(obj, nn.Module) and hasattr(obj, part):
            # Uu tien attribute (VD: features.denseblock4 cua DenseNet,
            # layer4 cua ResNet) — dung ca khi parent la Sequential
            # nhung key la ten chuoi thay vi chi so.
            obj = getattr(obj, part)
        elif isinstance(obj, (list, tuple, nn.Sequential)) and part.isdigit():
            obj = obj[int(part)]
        elif isinstance(obj, nn.Module) and part in obj._modules:
            obj = obj._modules[part]
        else:
            obj = getattr(obj, part)  # raise AttributeError ro rang
    return obj  # type: ignore[return-value]
