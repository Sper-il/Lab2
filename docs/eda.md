# Phân tích dữ liệu khám phá (EDA) — CIFAR-10

> Mọi con số đều đo thật từ dữ liệu và pipeline trong repo, không ước lượng.
> Nguồn chuẩn: `data/processed/metadata.json`, `data/splits/cifar10_split_seed42.json`,
> `results/runs/20261025-150847/metrics.json`, `results/runs/20261025-150847/plots/`,
> notebook `notebooks/20261026_eda_cifar10_demo.ipynb`.

## 1. Tổng quan dataset

| Thuộc tính | Giá trị |
|------------|---------|
| Tên dataset | CIFAR-10 (slide 24 tài liệu Chương 3) |
| Tổng số ảnh | 60.000 ảnh màu RGB 32×32 |
| Tập gốc | 50.000 train + 10.000 test |
| Số lớp | 10 lớp, cân bằng tuyệt đối |
| Dung lượng sau xử lý | `train_images.npy` 146,5 MB + `test_images.npy` 29,3 MB (uint8) |
| File gốc | `cifar-10-python.tar.gz` ~163 MB, MD5 `c58f30108f718f92721af3b95e74349a` khớp — PASS |

**10 lớp:** `airplane` (máy bay), `automobile` (ô tô), `bird` (chim), `cat` (mèo),
`deer` (hươu), `dog` (chó), `frog` (ếch), `horse` (ngựa), `ship` (tàu), `truck` (xe tải).

## 2. Phân bố lớp — dữ liệu sạch, không lệch

Đếm bằng `np.bincount` trên nhãn thật (`notebooks/20261026_eda_cifar10_demo.ipynb`, cell 2):

| Lớp | Train gốc | Test |
|-----|-----------|------|
| Mỗi lớp | 5.000 | 1.000 |
| Tổng | 50.000 | 10.000 |

Biểu đồ kiểm chứng: `results/runs/20261025-150847/plots/class_distribution.png` —
10 cột cao bằng nhau ở cả 3 tập. Kết luận: **không lệch lớp**,
không cần cân bằng lại bằng `WeightedSampler` hay trọng số loss.

## 3. Chia tập train/val/test

Hàm `stratified_indices(targets, val_size=5000, seed=42)` trong `src/data/split.py`
chia phân tầng theo từng lớp:

| Tập | Số mẫu | Mỗi lớp | Cách lấy |
|-----|--------|---------|----------|
| Train | 45.000 | 4.500 | phần còn lại sau khi tách val |
| Val | 5.000 | 500 | 500 mẫu đầu mỗi lớp sau shuffle seed 42 |
| Test | 10.000 | 1.000 | giữ nguyên tập test chính thức, không đụng tới |

Kiểm tra:

- `train + val = 50.000` đúng bằng tập train gốc — `validation.status = PASS`
  (`results/runs/20261025-150847/metrics.json`).
- Trùng lặp: `train/val = 0`, `train/test = 0`, `val/test = 0`
  (so bằng hash nội dung ảnh `image.tobytes()`, vì `train_idx` và `test_idx`
  thuộc hai không gian chỉ số khác nhau nên không so trực tiếp được).
  Test `test_exported_split_matches_pipeline_split` xác nhận file
  `data/splits/cifar10_split_seed42.npz` khớp 100% với pipeline.
- Tái lập được: cùng `seed=42` cho chỉ số split giống hệt.

Vì sao không dùng `random_split`? Chia ngẫu nhiên hoàn toàn dễ làm val lệch tỉ lệ
lớp vài chục mẫu mỗi lớp, khiến val accuracy nhiễu và chọn sai best checkpoint.
Chia phân tầng giữ đúng tỉ lệ 10 lớp ở mọi tập.

## 4. Thống kê pixel — ảnh lành mạnh, chỉ mờ tự nhiên

Đo trên 256 ảnh đầu (`notebooks/20261026_eda_cifar10_demo.ipynb`, cell 3):

```text
min 0.0   max 255.0   mean 119.3   std 62.9
pixel == 0 (đen tuyệt đối):    0.30%
pixel == 255 (trắng tuyệt đối): 0.72%
```

- Histogram phân bố đều, đỉnh ở khoảng 96–127 chiếm 18,3% — không bị nén hay làm nhạt.
- Trung bình từng kênh: R 124,2 / G 121,7 / B 112,0 — cân đối, không lệch màu.
- Kết luận: **dữ liệu lành mạnh, không hỏng**.

### Vì sao ảnh nhìn mờ?

Đây là đặc tính tự nhiên của CIFAR-10, không phải lỗi tải hay giải nén.
Ảnh gốc chỉ 32×32 nên mỗi pixel gộp nhiều chi tiết — không thể thấy rõ lông mèo
hay chữ trên xe. Resize lên 224×224 (demo M1) hay 64×64 (train chốt) chỉ nội suy,
không tạo thêm thông tin nên ảnh vẫn mờ. Minh họa:
`results/runs/20261025-150847/plots/sample_images.png` (lưới 12 ảnh mẫu đã
denormalize để nhìn được).

## 5. Kích thước ảnh — 32×32 gốc, resize lúc train

| Giai đoạn | Ảnh lưu trên disk | Ảnh đưa vào model |
|-----------|-------------------|-------------------|
| `data/processed/` | giữ nguyên 32×32 | — |
| Demo M1 (`configs/data_config.yaml`) | — | resize 32 → 224, batch `[B, 3, 224, 224]` |
| Train/eval chốt (`configs/train_config.yaml`, GPU Colab T4 + AMP) | — | resize 32 → 64, batch `[B, 3, 64, 64]` |

Vì sao không resize sẵn trên disk? `RandomCrop(32, padding=4)` phải thao tác trên
ảnh gốc 32×32, và augmentation phải khác nhau mỗi epoch. Cache ảnh 224×224 tốn
7,0 GB mà data loading đã đạt 403 ảnh/s (`num_workers=4`, ~1,9 phút/epoch) nên
không đáng — xem `docs/data-flow.md` §1b.

## 6. Transform — augment trước, resize sau

`src/data/transforms.py`, normalize theo mean/std ImageNet
(`0.485/0.456/0.406`, `0.229/0.224/0.225`) vì backbone pre-trained trên ImageNet:

**Train** (`build_train_transform`): `RandomCrop(32, padding=4)` →
`RandomHorizontalFlip(p=0.5)` → `Resize` (224 demo M1 / 64 train chốt) →
`ToTensor()` → `Normalize`.

**Val/test** (`build_val_transform`): chỉ `Resize` → `ToTensor()` → `Normalize`,
không augmentation nên đánh giá deterministic, tái lập được.

Thứ tự augment trên ảnh 32×32 gốc rồi mới resize là chủ đích — crop trên ảnh đã
up-sample chỉ crop vùng đã nội suy, không thêm thông tin.

## 7. DataLoader — số batch đo thật

Run demo M1 (`results/runs/20261025-150847/metrics.json`, ảnh 224, batch 64):

| Tập | Mẫu | Batch | Shape |
|-----|-----|-------|-------|
| Train | 45.000 | 704 | `[64, 3, 224, 224]` float32 |
| Val | 5.000 | 79 | `[64, 3, 224, 224]` float32 |
| Test | 10.000 | 157 | `[64, 3, 224, 224]` float32 |

Train chốt ảnh 64 (xem `metrics.json` trong `results/runs/20261031-*/`):

| Batch size | Train | Val | Test | Shape |
|------------|-------|-----|------|-------|
| 32 (ResNet18, VGG11 chốt) | 1.407 | 157 | 313 | `[32, 3, 64, 64]` |
| 64 (DenseNet121 chốt) | 704 | 79 | 157 | `[64, 3, 64, 64]` |

`shuffle=True` cho train (chống học vẹt thứ tự), `shuffle=False` cho val/test.
`num_workers` tự giới hạn theo CPU, `in_memory=True` giữ ~147 MB ảnh trong RAM
cho nhanh trên Windows.

## 8. Tính toàn vẹn — MD5 và verify tự động

`scripts/verify_data.py` kiểm tra sau khi tải:

- MD5 archive khớp `c58f30108f718f92721af3b95e74349a` — PASS.
- Số mẫu, phân bố lớp, shape ảnh đúng như §2–§3.

## 9. Nhận xét ảnh hưởng tới model

Ngay từ EDA đã thấy 2 cặp dễ nhầm: mèo với chó, chim với hươu — ảnh 32×32 mờ,
lông và dáng giống nhau. Kết quả test 20 epoch xác nhận đúng dự đoán này
(`results/TEST_RESULTS.md`, model tốt nhất VGG11 87,74%):

- F1 thấp nhất: mèo 0,76; cao nhất: tàu 0,93, ô tô 0,93.
- Nhầm nhiều nhất: mèo → chó 109 tấm, chó → mèo 106 tấm.
- Cộng toàn bộ confusion matrix đúng 10.000 tấm.

Hình: `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/eval/confusion_matrix.png`
(đường chéo là đoán đúng, 2 ô mèo–chó sáng nhất ngoài đường chéo),
`.../eval/per_class_f1.png` (F1 từng lớp).

## 10. Cách tái tạo

```powershell
# Tải + verify + xuất processed/splits
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\verify_data.py
.\.venv\Scripts\python.exe -m src.data.prepare

# Chạy pipeline M1 + vẽ lại biểu đồ
.\.venv\Scripts\python.exe -m src.data --config configs/data_config.yaml
.\.venv\Scripts\python.exe scripts\plot_data_report.py --run-dir results\runs\<timestamp>

# Mở notebook EDA
.\.venv\Scripts\python.exe -m notebook
# notebooks/20261026_eda_cifar10_demo.ipynb
```

Tài liệu liên quan: `data/README.md` (hướng dẫn tải), `docs/data-flow.md` (luồng M1
chi tiết), `docs/reports/final_report.md` §2 (tóm tắt dữ liệu chốt).
