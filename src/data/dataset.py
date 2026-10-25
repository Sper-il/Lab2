"""Dataset wrapper cho CIFAR-10 hỗ trợ chạy trên RAM để tăng tốc đọc ảnh."""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

LOGGER = logging.getLogger(__name__)

CIFAR10_MEAN_RGB: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_RGB: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


class Cifar10Dataset(Dataset):
    """Dataset CIFAR-10 trả về `(image_tensor, label)`.

    Mặc định nạp toàn bộ ảnh vào RAM dưới dạng `numpy.ndarray` (uint8). CIFAR-10
    chỉ khoảng 150 MB nên cách này nhanh hơn nhiều so với đọc file pickle từng
    lần trên Windows, nơi I/O thường là nút thắt cổ chai.

    Attributes:
        data: Mảng ảnh uint8 có shape `[N, 32, 32, 3]`.
        targets: Mảng nhãn int có shape `[N]`.
        classes: Danh sách tên lớp.
        transform: Pipeline transform áp dụng cho ảnh PIL.
    """

    def __init__(
        self,
        root: str,
        train: bool,
        transform: Callable | None = None,
        download: bool = False,
        in_memory: bool = True,
    ) -> None:
        """Khởi tạo dataset.

        Args:
            root: Thư mục gốc chứa dữ liệu CIFAR-10.
            train: `True` để lấy tập train, `False` để lấy tập test.
            transform: Pipeline transform áp dụng cho ảnh.
            download: Cho phép tải dataset nếu chưa có.
            in_memory: Nạp toàn bộ ảnh vào RAM để tăng tốc.

        Raises:
            RuntimeError: Nếu `download=True` nhưng không tải được dữ liệu.
        """
        try:
            base = CIFAR10(root=root, train=train, download=download)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Không tải được CIFAR-10 từ '{root}'. "
                "Hãy chạy scripts/download_data.py trước."
            ) from exc

        self.transform = transform
        self.classes = list(base.classes)
        self.targets = np.asarray(base.targets, dtype=np.int64)
        self._base = base

        if in_memory:
            self.data = np.asarray(base.data, dtype=np.uint8)
        else:
            self.data = None

    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset."""
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple:
        """Trả về `(image, label)` tại vị trí `index`.

        Args:
            index: Chỉ số mẫu.

        Returns:
            Tuple gồm tensor ảnh đã transform và nhãn dạng int.

        Raises:
            IndexError: Nếu chỉ số ngoài phạm vi.
        """
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(f"Chỉ số {index} ngoài phạm vi [0, {len(self) - 1}]")

        if self.data is not None:
            array = self.data[index]
        else:
            array = self._base.data[index]

        image = Image.fromarray(array)
        if self.transform is not None:
            image = self.transform(image)

        return image, int(self.targets[index])

    @property
    def class_counts(self) -> dict[int, int]:
        """Đếm số mẫu của từng lớp theo index.

        Returns:
            Dict `{class_index: count}`.
        """
        labels, counts = np.unique(self.targets, return_counts=True)
        return {int(label): int(count) for label, count in zip(labels, counts)}


class TransformSubset(Dataset):
    """Subset có transform riêng, không mutate dataset gốc.

    `torch.utils.data.Subset` chỉ giữ tham chiếu tới dataset gốc, nên gán
    `subset.dataset.transform` sẽ làm thay đổi transform của dataset gốc —
    tức là làm hỏng cả tập train. Wrapper này giữ transform riêng cho từng split.

    Attributes:
        dataset: Dataset gốc chứa toàn bộ mẫu.
        indices: Danh sách chỉ số thuộc split này.
        transform: Pipeline transform riêng của split.
    """

    def __init__(
        self,
        dataset: Cifar10Dataset,
        indices: Sequence[int],
        transform: Callable | None = None,
    ) -> None:
        """Khởi tạo subset với transform độc lập.

        Args:
            dataset: Dataset gốc.
            indices: Chỉ số các mẫu thuộc split.
            transform: Transform riêng cho split này.
        """
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        """Trả về số mẫu trong split."""
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple:
        """Trả về `(image, label)` tại vị trí `index` trong split.

        Args:
            index: Chỉ số tương đối trong split.

        Returns:
            Tuple gồm tensor ảnh đã transform và nhãn dạng int.
        """
        sample_index = self.indices[index]

        if self.dataset.data is not None:
            array = self.dataset.data[sample_index]
        else:
            array = self.dataset._base.data[sample_index]

        image = Image.fromarray(array)
        if self.transform is not None:
            image = self.transform(image)

        return image, int(self.dataset.targets[sample_index])
