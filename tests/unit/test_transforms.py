"""Unit test cho module transforms."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.data.transforms import (
    CIFAR10_IMAGE_SIZE,
    IMAGENET_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transform,
    build_train_transform,
    build_val_transform,
)


@pytest.fixture
def sample_image() -> Image.Image:
    """Tạo một ảnh PIL 32×32 ngẫu nhiên để test."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(CIFAR10_IMAGE_SIZE, CIFAR10_IMAGE_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_train_transform_output_shape(sample_image: Image.Image) -> None:
    """Transform train phải trả về tensor CHW đúng kích thước."""
    transform = build_train_transform()
    output = transform(sample_image)

    assert output.shape == (3, IMAGENET_INPUT_SIZE, IMAGENET_INPUT_SIZE)
    assert output.dtype.is_floating_point


def test_val_transform_output_shape(sample_image: Image.Image) -> None:
    """Transform val/test phải trả về tensor CHW đúng kích thước."""
    transform = build_val_transform()
    output = transform(sample_image)

    assert output.shape == (3, IMAGENET_INPUT_SIZE, IMAGENET_INPUT_SIZE)


def test_transform_respects_custom_image_size(sample_image: Image.Image) -> None:
    """Kích thước output phải theo tham số `image_size`."""
    transform = build_val_transform(image_size=64)
    output = transform(sample_image)

    assert output.shape == (3, 64, 64)


def test_normalization_uses_imagenet_stats(sample_image: Image.Image) -> None:
    """Ảnh đã normalize phải có giá trị âm (vì std > 0 và mean > 0)."""
    transform = build_val_transform()
    output = transform(sample_image)

    assert output.min() < 0, "Ảnh chưa được normalize theo ImageNet stats"
    assert output.max() > 0


def test_train_transform_without_augment_is_deterministic(sample_image: Image.Image) -> None:
    """Tắt augmentation thì transform train phải cho kết quả giống nhau."""
    transform = build_train_transform(augment=False)
    first = transform(sample_image)
    second = transform(sample_image)

    assert torch_allclose(first, second)


def test_train_transform_with_augment_is_stochastic(sample_image: Image.Image) -> None:
    """Bật augmentation thì hai lần transform phải khác nhau."""
    transform = build_train_transform(augment=True)
    first = transform(sample_image)
    second = transform(sample_image)

    assert not torch_allclose(first, second), "Augmentation không có tác dụng"


def test_build_transform_dispatch(sample_image: Image.Image) -> None:
    """`build_transform` phải dispatch đúng theo tên split."""
    assert build_transform("train").transforms[0].__class__.__name__.startswith("RandomCrop")
    assert build_transform("val").transforms[0].__class__.__name__ == "Resize"
    assert build_transform("TEST").transforms[0].__class__.__name__ == "Resize"


def test_build_transform_rejects_invalid_split() -> None:
    """Tên split không hợp lệ phải ném ValueError."""
    with pytest.raises(ValueError, match="Split khong hop le|Split không hợp lệ"):
        build_transform("unknown_split")


def test_imagenet_constants() -> None:
    """Mean/std ImageNet phải là các giá trị chuẩn."""
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)


def torch_allclose(first, second) -> bool:
    """So sánh hai tensor với sai số cho phép.

    Args:
        first: Tensor thứ nhất.
        second: Tensor thứ hai.

    Returns:
        `True` nếu hai tensor gần nhau.
    """
    import torch

    return bool(torch.allclose(first, second))
