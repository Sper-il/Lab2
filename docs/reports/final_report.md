# Báo cáo tổng hợp — Practice 2 (M1+M2, tính đến 2026-10-26)

> Báo cáo tổng hợp toàn project: bài toán, dữ liệu, model, kết quả đã đo, và việc còn lại.
> Quy tắc: **mọi con số đều có nguồn** (file artifact hoặc lệnh đã chạy). Chỗ nào chưa có số thật thì ghi rõ "chưa đo", không bịa.

## 1. Bài toán & phạm vi

- **Đề bài** (`đề.txt`): Practice 2 — nạp và fine-tune model pre-trained (`torchvision.models`), tinh chỉnh hyperparameter, theo dõi bằng TensorBoard.
- **Bài toán cụ thể:** phân loại ảnh CIFAR-10 (10 lớp) bằng backbone ImageNet, so sánh transfer learning (freeze) vs fine-tuning (unfreeze lớp cuối).
- **Môi trường:** Windows, Python 3.13.2, PyTorch 2.14.1+cpu / torchvision 0.29.1+cpu / tensorboard 2.21.0, **CPU-only** (không có GPU NVIDIA).
- **Phạm vi báo cáo này:** M1 (data pipeline ✅) + M2 (model factory ✅). M3 (training) + M4 (evaluation) + M5 (sweep) chưa làm — xem §6.

## 2. Dữ liệu (M1 ✅)

| Hạng mục | Kết quả đo | Nguồn |
|----------|-----------|-------|
| Dataset | CIFAR-10: 50.000 train + 10.000 test, 32×32 RGB, 10 lớp cân bằng (5.000/lớp train, 1.000/lớp test) | `data/processed/metadata.json`, run `20261025-150847` |
| Split | 45.000 train (4.500/lớp) / 5.000 val (500/lớp) / 10.000 test — stratified, seed=42, train+val = 50.000 ✓ | `data/splits/cifar10_split_seed42.json`, `metrics.json` |
| Overlap | train/val = 0, train/test = 0, val/test = 0 (hash nội dung ảnh) | `tests/unit/test_prepare.py` (11 test) |
| Batch | `[64, 3, 224, 224]` float32 + label `[64]` int64; train 704 / val 79 / test 157 batch | `metrics.json` (`validation.status = PASS`) |
| Transform | Train: RandomCrop(32, pad=4) + Flip → Resize 224 → Normalize ImageNet; Val: Resize 224 → Normalize | `src/data/transforms.py` |
| Toàn vẹn file | MD5 archive `c58f30108f718f92721af3b95e74349a` khớp — PASS | `scripts/verify_data.py` |
| Chất lượng ảnh | mean 119.3, std 62.9 (256 ảnh đầu); 0.30% pixel đen, 0.72% trắng → lành mạnh | `data/README.md` |
| Test | 47/47 pass (36 pipeline + 11 prepare) | `pytest -q`, 97s |

Quyết định quan trọng: giữ ảnh gốc 32×32 trong `data/processed/` (không cache 224 — tốn 7GB mà data loading chỉ ~1.9 phút/epoch, không phải nút thắt).

## 3. Model (M2 ✅)

| Model | Total params | Trainable (freeze) | Fine-tune 1 block | Output | Run |
|-------|-------------|--------------------|--------------------|--------|-----|
| resnet18 | 11,181,642 | 5,130 (0.046%) | 8,398,858 (75.1%) | `(1, 10)` ✓ | `20261026-115633` |
| vgg11 | 128,807,306 | 40,970 (0.032%) | 121,946,634 (94.7%) | `(1, 10)` ✓ | `20261026-120218` |
| densenet121 | 6,964,106 | 10,250 (0.147%) | 2,168,330 (31.1%) | `(1, 10)` ✓ | `20261026-121817` |

Nguồn: `count_parameters()` sau `build_model()` với weights ImageNet thật (`results/runs/*/model_info.json`).

**Điểm kỹ thuật đáng chú ý (đã verify bằng chạy thật):**

1. VGG11 `classifier` là `Sequential[7]`, không phải 1 `Linear` — code giả định mọi model có `.fc` sẽ crash. `replace_head()` xử lý 2 dạng.
2. VGG11 `classifier` chứa 103M params — unfreeze cả sẽ khiến 92.8% model trainable, mất ý nghĩa freeze. Mặc định `freeze_head_full: false` (chỉ `Linear(4096,10)` cuối) + test bắt lỗi.
3. `unfreeze_last` dùng `block_units` đã introspect (resnet: `layer1..4`; vgg: 8 conv; densenet: 4 denseblock + `transition3`/`norm5`); mở luôn BatchNorm trong block.

**Model mặc định:** `resnet18` + `freeze_backbone` (`configs/model_config.yaml`) — nhẹ nhất mà train thực tế được trên CPU.

**Test:** 23/23 pass (17 unit `weights=None` + 6 integration; CLI verify weights thật cả 3 model).

## 4. Đánh giá theo đề bài (14 yêu cầu trong `đề.txt`)

| # | Yêu cầu | Trạng thái | Bằng chứng |
|---|---------|-----------|------------|
| E1 | Thử nhiều pre-trained model | ✅ | 3 backbone + report M2 |
| E2 | Tinh chỉnh hyperparameter | ⏳ | M3/M5 — `configs/*.yaml` + `src/training/` |
| E3 | TensorBoard | ⏳ | M3 — `src/training/trainer.py` |
| O1–O5 | Load / explore / adapt / freeze / unfreeze | ✅ | `src/models/factory.py`, 23 test |
| O6–O8 | Data / transforms / DataLoader | ✅ | `src/data/`, 47 test |
| O9–O10 | Loss+optimizer / training loop | ⏳ | M3 |
| O11 | Đánh giá test set | ⏳ | M4 |

Chi tiết từng milestone: `docs/reports/m2_models_report.md` (M2). Demo chạy được: `notebooks/20261026_eda_cifar10_demo.ipynb`.

## 5. Hạn chế đã biết

- **Chưa có số accuracy nào.** Mọi con số "0.74/0.73" từng xuất hiện trong docs cũ là bịa — đã xoá. Số thật chỉ có sau M3/M4.
- **CPU-only nên train chậm** (đã đo thật): ResNet18 freeze 224×224 ≈ 37 phút/epoch; VGG11 ≈ 144 phút/epoch (~12 giờ/5 epoch — không khả thi). Plan M3 dùng 2 tầng: Tier A 224×224 (báo cáo chính) + Tier B 64×64 (~5 phút/epoch, sweep rộng).
- **Ảnh 32×32 upsample lên 224×224** — mất chi tiết, nhưng bắt buộc để dùng weights ImageNet.
- **Mục tiêu accuracy là tự đặt** (freeze ≥ 0.70, fine-tune ≥ 0.80) — đề bài không cho ngưỡng.

## 6. Việc còn lại (M3–M5)

| Module | Nội dung | Ước lượng |
|--------|----------|-----------|
| M3 | `src/training/` (loop, scheduler, checkpoint) + TensorBoard (scalar/histogram/graph/image) + `configs/train_config.yaml` | Plan `docs/plan-m2-m4.md` §2 |
| M4 | `src/evaluation/metrics.py` (acc, macro-F1, confusion matrix) + `configs/eval_config.yaml` | Plan §M4 |
| M5 | Sweep lr/batch/optimizer + Tier B 64×64 + bảng so sánh, điền mục 8 Results trong `README.md` | Sau M3/M4 |

## Phụ lục — file & commit

- Commit: `f130862` (scaffold) → `d0321eb` + `28aa1ff` (M1) → `a1a6b71` (align đề) → `75455dd` (M2).
- Code: `src/data/` (7 file), `src/models/` (4 file), `src/utils/` (2 file), `scripts/` (4 file).
- Test: 70 passed (47 M1 + 23 M2).
- Audit: `docs/AUDIT_LOG.md`. Demo: `notebooks/20261026_eda_cifar10_demo.ipynb`.
