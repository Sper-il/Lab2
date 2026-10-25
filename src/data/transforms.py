"""Transforms tiền xử lý ảnh CIFAR-10 cho các backbone pre-trained ImageNet."""

from __future__ import annotations

from typing import Any

from torchvision import transforms

# Mean/std chuẩn của ImageNet — dùng chung cho mọi model trong `torchvision.models`
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Kích thước input mà các backbone pre-trained ImageNet kỳ vọng
IMAGENET_INPUT_SIZE: int = 224

# Kích thước ảnh gốc của CIFAR-10
CIFAR10_IMAGE_SIZE: int = 32


def build_train_transform(
    image_size: int = IMAGENET_INPUT_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    augment: bool = True,
) -> transforms.Compose:
    """Tạo pipeline transform cho tập train.

    Thứ tự biến đổi: augment trên ảnh 32×32 gốc → resize lên kích thước
    backbone mong đợi → chuyển sang tensor → normalize theo ImageNet.

    Args:
        image_size: Kích thước output (chiều dài cạnh), mặc định 224.
        mean: Mean normalize theo từng kênh.
        std: Std normalize theo từng kênh.
        augment: Bật/tắt augmentation (RandomCrop + HorizontalFlip).

    Returns:
        `transforms.Compose` áp dụng cho ảnh PIL trong tập train.
    """
    steps: list[Any] = []

    if augment:
        steps.extend(
            [
                transforms.RandomCrop(CIFAR10_IMAGE_SIZE, padding=4),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    steps.extend(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transforms.Compose(steps)


def build_val_transform(
    image_size: int = IMAGENET_INPUT_SIZE,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> transforms.Compose:
    """Tạo pipeline transform cho tập validation/test.

    Không augmentation: chỉ resize + tensor + normalize để kết quả đánh giá
    có tính xác định và reproducible.

    Args:
        image_size: Kích thước output (chiều dài cạnh), mặc định 224.
        mean: Mean normalize theo từng kênh.
        std: Std normalize theo từng kênh.

    Returns:
        `transforms.Compose` áp dụng cho ảnh PIL trong tập val/test.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_transform(
    split: str,
    image_size: int = IMAGENET_INPUT_SIZE,
    augment: bool = True,
) -> transforms.Compose:
    """Chọn transform theo tên split.

    Args:
        split: Tên split, không phân biệt hoa thường: `train`, `val`, `test`.
        image_size: Kích thước output.
        augment: Có dùng augmentation cho tập train hay không.

    Returns:
        `transforms.Compose` tương ứng.

    Raises:
        ValueError: Nếu tên split không hợp lệ.
    """
    normalized = split.strip().lower()

    if normalized == "train":
        return build_train_transform(image_size=image_size, augment=augment)
    if normalized in {"val", "valid", "validation", "test"}:
        return build_val_transform(image_size=image_size)

    raise ValueError(
        f"Split không hợp lệ: {split!r}. Chỉ chấp nhận 'train', 'val', 'test'."
    )
