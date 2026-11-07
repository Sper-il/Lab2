# Data (chuẩn pipeline raw → processed → splits)

> Chuẩn dữ liệu của project theo `modules/data.md` của skill. Tài liệu chi tiết:
> luồng đầy đủ ở `docs/data-flow.md`, phân tích khám phá ở `docs/eda.md`, hướng dẫn tải ở `data/README.md`.

## 1. Nguồn và tính toàn vẹn

- Dataset: CIFAR-10 (slide 24 tài liệu Chương 3) — 50.000 train + 10.000 test, ảnh 32×32 RGB, 10 lớp cân bằng tuyệt đối.
- File gốc: `cifar-10-python.tar.gz` (~163 MB), MD5 `c58f30108f718f92721af3b95e74349a` — **PASS** (`scripts/verify_data.py`).
- Tải: `scripts/download_data.py` (tự động qua `torchvision.datasets`, fallback link thủ công trong `data/README.md`).

## 2. Ba tầng dữ liệu

| Tầng | Nội dung | Kích thước | Ghi chú |
|------|----------|-----------|---------|
| `data/raw/` | archive + `cifar-10-batches-py/` | ~170 MB | Không bao giờ sửa; gitignore |
| `data/processed/` | `train_images.npy` uint8 `[50000,32,32,3]` (146.5 MB), `test_images.npy` `[10000,32,32,3]` (29.3 MB), labels + `metadata.json` | ~176 MB | **Giữ ảnh gốc 32×32** — resize diễn ra trong transform lúc train |
| `data/splits/` | `cifar10_split_seed42.npz` + `.json` | ~90 KB | `train_idx` 45k / `val_idx` 5k / `test_idx` 10k; `.npz` dùng trong code, `.json` để đọc/diff |

Sinh bằng: `.\.venv\Scripts\python.exe -m src.data.prepare` (chạy 1 lần, offline).

## 3. Chống leakage

- Val 5.000 mẫu tách từ train bằng `stratified_indices(seed=42)` — mỗi lớp đúng 4.500/500/1.000.
- Test giữ nguyên 10.000 ảnh chính thức, chỉ chấm 1 lần cuối.
- Verify overlap bằng **hash nội dung ảnh** (không so chỉ số vì `train_idx` và `test_idx` thuộc hai không gian khác nhau): train/val = 0, train/test = 0, val/test = 0.
- Test `test_exported_split_matches_pipeline_split` khẳng định file `.npz` khớp 100% pipeline.

## 4. Transform và DataLoader

- Train: `RandomCrop(32,padding=4)` + `Flip` **trên ảnh gốc 32** → `Resize` 64 (chốt) / 224 (demo M1) → `ToTensor` → `Normalize(ImageNet)`.
- Val/test: chỉ `Resize` → `Normalize` (deterministic).
- `in_memory=True` (~147 MB RAM), `num_workers` tự giới hạn theo CPU, `pin_memory` bật sẵn cho GPU.
- Batch chốt ảnh 64: bs 32 → 1407/157/313 batch; demo M1 ảnh 224 bs 64 → 704/79/157 batch.

## 5. Chất lượng dữ liệu (đo thật, `docs/eda.md`)

- Phân bố lớp: 10 cột cao bằng nhau ở cả 3 split — không lệch, không cần `WeightedSampler`.
- Pixel (256 ảnh đầu): mean 119.3, std 62.9; 0.30% đen tuyệt đối, 0.72% trắng tuyệt đối — lành mạnh.
- Ảnh "mờ" là đặc tính tự nhiên của CIFAR-10 (gốc 32×32), không phải dữ liệu hỏng.
