# Data — CIFAR-10

> Dataset phân loại ảnh 10 lớp, ảnh RGB 32×32. Đây là dataset được nêu trong slide 24 của tài liệu Chương 3.

## Nguồn chính thức

- Trang chủ: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- File tải trực tiếp: [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) (~163 MB)

## Cách 1 — Tự động (khuyến nghị)

```powershell
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\verify_data.py
```

Script sẽ tải về `data/raw/cifar-10-python.tar.gz`, giải nén thành `data/raw/cifar-10-batches-py/`, rồi kiểm tra số lượng mẫu, phân bố lớp và shape ảnh.

## Cách 2 — Tải thủ công

1. Mở [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. Giải nén (7-Zip / Windows `tar -xzf cifar-10-python.tar.gz`) sao cho có thư mục `data/raw/cifar-10-batches-py/`
3. Chạy `.\.venv\Scripts\python.exe scripts\verify_data.py` để xác nhận dữ liệu đúng.

## Cấu trúc sau khi tải

```
data/raw/
├── cifar-10-python.tar.gz          # archive gốc (giữ lại để re-download nhanh)
└── cifar-10-batches-py/
    ├── data_batch_1 .. data_batch_5  # 5 batch train × 10.000 ảnh
    ├── test_batch                    # 10.000 ảnh test
    ├── batches.meta                  # tên 10 lớp
    └── readme.html
```

## Thông tin dataset

| Thuộc tính | Giá trị |
|------------|---------|
| Tổng số ảnh | 60.000 |
| Train / Test | 50.000 / 10.000 |
| Kích thước ảnh | 32×32 pixel, 3 kênh RGB |
| Số lớp | 10 (cân bằng tuyệt đối) |
| Mẫu train mỗi lớp | 5.000 |
| Mẫu test mỗi lớp | 1.000 |

**10 lớp:** `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## Ghi chú quan trọng về kích thước ảnh

CIFAR-10 chỉ có 32×32, trong khi các backbone pre-trained ImageNet (`resnet18`, `vgg11`, `densenet121`) kỳ vọng input 224×224. Pipeline của project sẽ **resize 32 → 224** (đồng thời dùng `RandomCrop(32, padding=4)` + `RandomHorizontalFlip` cho augmentation) rồi normalize theo mean/std của ImageNet.

Đây là đánh đổi bắt buộc của hướng dùng pre-trained ImageNet. Nếu giữ nguyên 32×32 thì phải train từ đầu (`weights=None`) — mất đúng ý nghĩa của phần transfer learning trong đề bài.

## Lưu ý

Thư mục `data/` đã nằm trong `.gitignore` — dataset không bị commit vào git. Người clone repo chạy `scripts/download_data.py` để tải lại.
