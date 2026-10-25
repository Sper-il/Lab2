# Data Flow — M1 Data Pipeline

> Mô tả chi tiết luồng dữ liệu từ file gốc CIFAR-10 đến batch sẵn sàng cho model.
> Module: **M1 — Data Pipeline**. Trạng thái: ✅ hoàn thành (run `20261025-150847`).

## 1. Sơ đồ tổng thể

```
data/raw/cifar-10-python.tar.gz
        │  (torchvision tải về + giải nén, scripts/download_data.py)
        ▼
data/raw/cifar-10-batches-py/
        │  data_batch_1..5 (50.000 ảnh) + test_batch (10.000 ảnh)
        ▼
┌─────────────────────────────────────────────────────────┐
│  BƯỚC 0 (offline, 1 lần): python -m src.data.prepare   │
│                                                         │
│  data/processed/                                       │
│    train_images.npy  uint8 [50000, 32, 32, 3]          │
│    train_labels.npy  int64 [50000]                     │
│    test_images.npy   uint8 [10000, 32, 32, 3]          │
│    test_labels.npy   int64 [10000]                     │
│    metadata.json     tên lớp + thống kê                │
│                                                         │
│  data/splits/                                          │
│    cifar10_split_seed42.npz  train_idx/val_idx/test_idx│
│    cifar10_split_seed42.json  bản tóm tắt              │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Cifar10Dataset(root="data/raw", train=True)            │
│  • base.data  → numpy uint8 [N, 32, 32, 3]  (~147 MB) │
│  • base.targets → numpy int64 [N]                      │
│  • in_memory=True: giữ ảnh trong RAM                    │
└─────────────────────────────────────────────────────────┘
        │
        ├──────────────► stratified_indices(targets, 5000, seed=42)
        │                        │
        │                        ├── train_indices (45.000)  → 4.500/lớp
        │                        └── val_indices   (5.000)   →  500/lớp
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  TransformSubset (wrapper tự viết)                      │
│  ├── train_subset: transform = build_train_transform()  │
│  └── val_subset:   transform = build_val_transform()    │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  DataLoader                                               │
│  train: batch 64, shuffle=True,  704 batch               │
│  val:   batch 64, shuffle=False,  79 batch               │
│  test:  batch 64, shuffle=False, 157 batch  (10.000 ảnh) │
└─────────────────────────────────────────────────────────┘
        │
        ▼
   images [B, 3, 224, 224] float32   labels [B] int64
```

## 1b. Bước 0 — `src/data/prepare.py` (chạy 1 lần, offline)

Convention `data/{raw, processed, splits}` của skill `ai-fusion-workflow` yêu cầu
cả ba thư mục có nội dung thật. Module này sinh ra hai thư mục còn lại:

```powershell
.\.venv\Scripts\python.exe -m src.data.prepare
```

| Output | Nội dung | Kích thước |
|--------|----------|------------|
| `data/processed/train_images.npy` | uint8 `[50000, 32, 32, 3]` | 146.5 MB |
| `data/processed/train_labels.npy` | int64 `[50000]` | 390.8 KB |
| `data/processed/test_images.npy` | uint8 `[10000, 32, 32, 3]` | 29.3 MB |
| `data/processed/test_labels.npy` | int64 `[10000]` | 78.2 KB |
| `data/processed/metadata.json` | tên lớp, thống kê, nguồn gốc | 1 KB |
| `data/splits/cifar10_split_seed42.npz` | `train_idx` / `val_idx` / `test_idx` | 89.8 KB |
| `data/splits/cifar10_split_seed42.json` | bản tóm tắt đọc được | 0.4 KB |

### Vì sao `processed/` giữ ảnh 32×32, không resize 224?

Resize phải diễn ra **trong transform lúc train**, không lúc prepare:

- `RandomCrop(32, padding=4)` cần thao tác trên ảnh gốc 32×32. Resize trước rồi
  mới crop chỉ crop trên ảnh đã up-sample — không thu được thông tin nào.
- Augmentation phải khác nhau **mỗi epoch** để chống overfitting. Ảnh resize
  vĩnh viễn trên disk làm mất tính chất này.

### Benchmark quyết định có cache ảnh 224×224 không

`scripts/benchmark_loading.py` đo trên máy này (CPU-only):

| `num_workers` | Throughput | 1 epoch (45k ảnh) |
|----------------|------------|-------------------|
| 0 | 149.6 ảnh/s | ~5.0 phút |
| 4 | 403.5 ảnh/s | ~1.9 phút |

Cache 224×224 uint8 cho 50k ảnh tốn **7.0 GB**. Với throughput đã đạt 403 ảnh/s
(data loading không phải nút thắt), cache 7 GB **không đáng** → giữ ảnh 32×32.

### Verify split bằng hash nội dung ảnh

`train_idx` (0..49.999) và `test_idx` (0..9.999) thuộc **hai không gian chỉ số
khác nhau** → không thể so sánh overlap trực tiếp. `verify_splits` thay vào đó
hash **nội dung ảnh** (`image.tobytes()`) rồi so sánh tập hash:

| Phép so sánh | Kết quả |
|--------------|---------|
| `train_idx ∩ val_idx` (cùng không gian) | 0 |
| hash(train) ∩ hash(val) | 0 |
| hash(train) ∩ hash(test) | 0 |
| hash(val) ∩ hash(test) | 0 |

Test `test_exported_split_matches_pipeline_split` xác nhận file `.npz` trong
`data/splits/` **khớp 100%** với output của `stratified_indices` mà pipeline
thực sự dùng.

## 2. Transform chi tiết

### Tập train (`build_train_transform`)

| Bước | Transform | Ghi chú |
|------|-----------|---------|
| 1 | `RandomCrop(32, padding=4)` | Augmentation trên ảnh **gốc 32×32**, padding 4 pixel → crop 32×32 ngẫu nhiên |
| 2 | `RandomHorizontalFlip(p=0.5)` | Lật ngang, xác suất 50% |
| 3 | `Resize((224, 224), antialias=True)` | Up-sample 32 → 224 để khớp backbone ImageNet |
| 4 | `ToTensor()` | PIL uint8 → tensor float32 `[0, 1]`, shape CHW |
| 5 | `Normalize(mean, std)` | mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)` |

> **Thứ tự có chủ đích:** augment *trước* khi resize. Nếu resize trước rồi mới crop, ta chỉ crop trên ảnh đã up-sample — thông tin chi tiết bị nhân bản chứ không thu được thông tin mới.

### Tập val / test (`build_val_transform`)

| Bước | Transform |
|------|-----------|
| 1 | `Resize((224, 224), antialias=True)` |
| 2 | `ToTensor()` |
| 3 | `Normalize(mean, std)` |

Không augmentation → kết quả đánh giá **deterministic**, có thể tái lập.

### Về mean/std

Dùng **mean/std của ImageNet** (không phải của CIFAR-10) vì backbone là model pre-trained trên ImageNet. Dùng sai thống kê sẽ làm lệch phân phối đầu vào so với lúc pre-train, phá vỡ hiệu quả của weight đã học.

Thống kê CIFAR-10 (`0.4914/0.4822/0.4465`, `0.2470/0.2435/0.2616`) được lưu trong `src/data/dataset.py` làm `CIFAR10_MEAN_RGB` / `CIFAR10_STD_RGB` để tham khảo, không dùng trong pipeline chính.

## 3. Thuật toán chia tập phân tầng

`stratified_indices(targets, val_size=5000, seed=42)` trong `src/data/split.py`:

1. Lấy `unique_classes` = 10 lớp.
2. `per_class = 5000 // 10 = 500`, `remainder = 0`.
3. Với mỗi lớp: lấy chỉ số của lớp đó, shuffle bằng `np.random.default_rng(42)`.
4. 500 mẫu đầu → validation; phần còn lại (4.500) → train.
5. Sort chỉ số trước khi trả về để giữ thứ tự deterministic.

**Kết quả:** mỗi lớp có đúng 4.500 train / 500 val / 1.000 test.

### Vì sao phân tầng thay vì `random_split`?

`torch.utils.data.random_split` chọn ngẫu nhiên hoàn toàn → tỉ lệ lớp trong validation có thể lệch khỏi train (với 5.000 mẫu và 10 lớp, sai số vài chục mẫu/lớp là chuyện bình thường). Val lệch phân bố → val accuracy không còn phản ánh đúng năng lực model, quyết định "best checkpoint" bị nhiễu.

## 4. Vì sao tự viết `TransformSubset` thay vì dùng `torch.utils.data.Subset`?

`Subset` giữ **tham chiếu** tới dataset gốc:

```python
train_subset, val_subset = split_train_val(dataset, ...)
val_subset.dataset.transform = eval_transform   # ← SAI
```

Vì `train_subset.dataset` và `val_subset.dataset` là **cùng một object**, lệnh gán trên đổi transform của cả tập train → tập train mất augmentation, và nếu thứ tự gán bị đảo thì ngược lại tập val lại có augmentation.

`TransformSubset` (`src/data/dataset.py`) giữ `indices` + `transform` riêng, truy cập ảnh qua `dataset.data[sample_index]` mà không mutate gì:

```python
class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
```

Có test riêng khẳng định `dataset.transform` không đổi: `tests/unit/test_dataset.py::test_transform_subset_does_not_mutate_parent`.

## 5. Tối ưu hiệu năng

| Kỹ thuật | Chi tiết |
|-----------|---------|
| `in_memory=True` | Toàn bộ 50.000 ảnh uint8 (~147 MB) nằm trong RAM. CIFAR-10 đủ nhỏ để làm vậy, nhanh hơn nhiều so với đọc pickle từng lần trên Windows |
| `num_workers` tự giới hạn | `min(config, cpu_count - 1)`, và 0 nếu máy chỉ có ≤ 2 core — tránh spawn process thừa |
| `Resize(antialias=True)` | Áp dụng đúng thuật toán lọc khi downscale; ở đây là upsample nên ảnh hưởng nhỏ nhưng nhất quán với khuyến nghị torchvision |
| `pin_memory` | Mặc định `false` (CPU-only). Bật sẵn trong config khi chuyển sang GPU |

## 6. Artifact sinh ra

Mỗi lần chạy `python -m src.data` tạo `results/runs/<YYYYMMDD-HHMMSS>/`:

| File | Nội dung |
|------|----------|
| `config.yaml` | Snapshot config đã dùng (để tái lập run) |
| `metrics.json` | Số mẫu / số batch / shape tensor / trạng thái validation |
| `logs.txt` | Log đầy đủ kèm timestamp |
| `plots/class_distribution.png` | Biểu đồ cột phân bố lớp của 3 split |
| `plots/sample_images.png` | Lưới 12 ảnh mẫu sau transform (đã denormalize để nhìn được) |

## 7. Cấu hình

Mọi tham số nằm trong `configs/data_config.yaml`:

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `data_root` | `data/raw` | Thư mục chứa CIFAR-10 |
| `image_size` | `224` | Kích thước output (khớp backbone ImageNet) |
| `batch_size` | `64` | Số ảnh mỗi batch |
| `num_workers` | `4` | Số process DataLoader (tự giới hạn theo CPU) |
| `val_size` | `5000` | Số mẫu validation lấy từ train |
| `seed` | `42` | Seed cho split + shuffle |
| `augment` | `true` | Bật augmentation cho tập train |
| `in_memory` | `true` | Nạp ảnh vào RAM |
| `download` | `false` | Cho phép tải tự động nếu thiếu dữ liệu |

## 8. Ranh giới module (cho M2–M5)

M1 **cung cấp**:

- `build_dataloaders(DataLoaderConfig) -> (train_loader, val_loader, test_loader)`
- `dataset_summary(...) -> dict` để ghi vào artifact

M1 **không làm**:

- Không đụng tới model, optimizer, loss, training loop (thuộc M2–M3)
- Không tự tạo test split mới — dùng nguyên 10.000 ảnh test chính thức
- Không lưu ảnh đã xử lý xuống disk (không cần, RAM đủ)
