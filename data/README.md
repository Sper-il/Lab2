# Data — CIFAR-10

> Dataset phân loại ảnh 10 lớp, ảnh RGB 32×32. Đây là dataset được nêu trong slide 24 của tài liệu Chương 3.

## Nguồn chính thức

- Trang chủ: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
- File tải trực tiếp: [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) (~163 MB)

## Cách 1 — Tự động (khuyến nghị)

```powershell
# Bước 1 — tải + giải nén vào data/raw/
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\verify_data.py

# Bước 2 — xuất data/processed/ và data/splits/
.\.venv\Scripts\python.exe -m src.data.prepare
```

Script sẽ tải về `data/raw/cifar-10-python.tar.gz`, giải nén thành `data/raw/cifar-10-batches-py/`, rồi kiểm tra số lượng mẫu, phân bố lớp và shape ảnh. Bước 2 xuất tiếp ảnh dạng `.npy` và file split.

## Cách 2 — Tải thủ công

1. Mở [https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. Giải nén (7-Zip / Windows `tar -xzf cifar-10-python.tar.gz`) sao cho có thư mục `data/raw/cifar-10-batches-py/`
3. Chạy `.\.venv\Scripts\python.exe scripts\verify_data.py` để xác nhận dữ liệu đúng.

## Cấu trúc sau khi tải

```
data/
├── raw/                                # dữ liệu gốc — không bao giờ sửa
│   ├── cifar-10-python.tar.gz          # archive gốc (giữ lại để re-download nhanh)
│   └── cifar-10-batches-py/
│       ├── data_batch_1 .. data_batch_5  # 5 batch train × 10.000 ảnh
│       ├── test_batch                    # 10.000 ảnh test
│       ├── batches.meta                  # tên 10 lớp
│       └── readme.html
│
├── processed/                          # dữ liệu đã xử lý sạch
│   ├── train_images.npy                 # uint8 [50000, 32, 32, 3]  (146.5 MB)
│   ├── train_labels.npy                 # int64 [50000]              (390.8 KB)
│   ├── test_images.npy                  # uint8 [10000, 32, 32, 3]  ( 29.3 MB)
│   ├── test_labels.npy                  # int64 [10000]              ( 78.2 KB)
│   └── metadata.json                    # tên lớp + thống kê + nguồn gốc
│
└── splits/                             # chỉ số train/val/test
    ├── cifar10_split_seed42.npz         # train_idx / val_idx / test_idx
    └── cifar10_split_seed42.json        # bản tóm tắt đọc được
```

### Vì sao `processed/` giữ ảnh 32×32, không resize 224?

Resize phải xảy ra **trong transform lúc train**, không lúc prepare. Lý do:

- `RandomCrop(32, padding=4)` cần thao tác trên ảnh gốc 32×32. Nếu resize trước rồi mới crop, ta chỉ crop trên ảnh đã up-sample — không thu được thông tin nào, chỉ lãng phí 7 GB disk.
- Augmentation phải biến đổi **mỗi epoch** khác nhau để chống overfitting. Nếu ảnh đã resize vĩnh viễn trên disk thì augmentation bị giới hạn ở ảnh đã up-sample.

Benchmark thực tế trên máy này (CPU-only): `num_workers=4` đạt **403 ảnh/s → ~1.9 phút/epoch** cho 45.000 ảnh. Data loading không phải nút thắt, nên không cần cache ảnh 224×224 (7 GB).

### Vì sao `splits/` lưu cả `.npz` và `.json`?

- `.npz` — định dạng nén, nạp bằng `np.load` trong nanosecond, dùng trong code.
- `.json` — đọc được bằng mắt, diff được trong git, tiện cho báo cáo.

Nội dung `.npz` dùng **cùng thuật toán** `stratified_indices` với pipeline (`src/data/split.py`) nên file split khớp 100% với cách DataLoader thực sự chia tập — có test riêng kiểm tra điều này.

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

Thư mục `data/` đã nằm trong `.gitignore` — dataset không bị commit vào git. Người clone repo chạy `scripts/download_data.py` rồi `python -m src.data.prepare` để tạo lại toàn bộ.

Ngoại lệ: `data/README.md` được giữ lại (file này chứa hướng dẫn, không phải dataset).
