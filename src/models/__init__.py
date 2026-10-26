"""Model factory: chon backbone pre-trained, thay head, freeze/unfreeze."""

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
from src.models.registry import (
    BACKBONE_REGISTRY,
    SUPPORTED_MODELS,
    BackboneSpec,
    get_module,
    resolve_spec,
)

__all__ = [
    "BACKBONE_REGISTRY",
    "SUPPORTED_MODELS",
    "BackboneSpec",
    "ModelConfig",
    "build_model",
    "count_parameters",
    "describe_model",
    "freeze_backbone",
    "get_module",
    "replace_head",
    "resolve_spec",
    "resolve_weights",
    "unfreeze_last",
]
