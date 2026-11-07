# Báo cáo tổng hợp — Practice 2 (M1–M5, chốt 2026-11-01)

> Báo cáo tổng hợp toàn project: bài toán, dữ liệu, model, kết quả đã đo.
> Quy tắc: **mọi con số đều có nguồn** (file artifact hoặc lệnh đã chạy).

## 1. Bài toán & phạm vi

- **Đề bài** (`đề.txt`): Practice 2 — nạp và fine-tune model pre-trained (`torchvision.models`), tinh chỉnh hyperparameter, theo dõi bằng TensorBoard.
- **Bài toán cụ thể:** phân loại ảnh CIFAR-10 (10 lớp) bằng backbone ImageNet, fine-tune mở block cuối bằng Adam.
- **Môi trường:** Windows + Python 3.13, PyTorch 2.14.1 / torchvision 0.29.1 / tensorboard 2.21.0. Pipeline M1/demo/test chạy local CPU; train nặng (sweep 30 cấu hình 5 epoch + retrain 20 epoch) chạy trên **GPU Colab T4 có AMP** (ảnh 64×64).
- **Phạm vi báo cáo này:** M1 (data pipeline ✅) + M2 (model factory ✅) + M3 (training loop + TensorBoard ✅ — sweep 30 cấu hình + retrain 20 epoch) + M4 (evaluation ✅ — 3 eval chốt) + M5 (sweep + báo cáo ✅).

## 2. Dữ liệu (M1 ✅)

| Hạng mục | Kết quả đo | Nguồn |
|----------|-----------|-------|
| Dataset | CIFAR-10: 50.000 train + 10.000 test, 32×32 RGB, 10 lớp cân bằng (5.000/lớp train, 1.000/lớp test) | `data/processed/metadata.json`, run `20261025-150847` |
| Split | 45.000 train (4.500/lớp) / 5.000 val (500/lớp) / 10.000 test — stratified, seed=42, train+val = 50.000 ✓ | `data/splits/cifar10_split_seed42.json`, `metrics.json` |
| Overlap | train/val = 0, train/test = 0, val/test = 0 (hash nội dung ảnh) | `tests/unit/test_prepare.py` (11 test) |
| Batch M1 demo (224) | `[64, 3, 224, 224]` float32 + label `[64]` int64; train 704 / val 79 / test 157 batch | `metrics.json` (`validation.status = PASS`) |
| Batch train chốt (64) | bs 32: train 1407 / val 157 / test 313 batch; bs 64: 704 / 79 / 157 batch, shape `[B, 3, 64, 64]` | `results/runs/20261031-*/<cfg>/metrics.json` |
| Transform | Train: RandomCrop(32, pad=4) + Flip → Resize 64 (train chốt) / 224 (demo M1) → Normalize ImageNet; Val/Test: Resize → Normalize | `src/data/transforms.py` |
| Toàn vẹn file | MD5 archive `c58f30108f718f92721af3b95e74349a` khớp — PASS | `scripts/verify_data.py` |
| Chất lượng ảnh | mean 119.3, std 62.9 (256 ảnh đầu); 0.30% pixel đen, 0.72% trắng → lành mạnh | `data/README.md` |
| Test | 47/47 pass (36 pipeline + 11 prepare) | `pytest -q` |

Quyết định quan trọng: giữ ảnh gốc 32×32 trong `data/processed/` (không cache resize — tốn disk mà data loading không phải nút thắt).

## 3. Model (M2 ✅)

| Model | Total params | Trainable (freeze) | Fine-tune 1 block (params trainable chốt 20 epoch) | Output | Run |
|-------|-------------|--------------------|-----------------------------------------------------|--------|-----|
| resnet18 | 11,181,642 | 5,130 (0.046%) | 8,398,858 (75.1%) | `(1, 10)` ✓ | CLI verify M2 (run `20261026-115633` đã dọn, số liệu giữ nguyên) |
| vgg11 | 128,807,306 | 40,970 (0.032%) | 2,400,778 (1.86%) | `(1, 10)` ✓ | CLI verify M2 (run `20261026-120218` đã dọn, số liệu giữ nguyên) |
| densenet121 | 6,964,106 | 10,250 (0.147%) | 2,696,714 (38.7%) | `(1, 10)` ✓ | CLI verify M2 (run `20261026-121817` đã dọn, số liệu giữ nguyên) |

Nguồn: `count_parameters()` sau `build_model()` với weights ImageNet thật. Cột fine-tune chốt lấy từ `metrics.json` 3 run 20 epoch (VGG11/DenseNet121 mở ít hơn bảng M2 cũ vì `freeze_head_full: false` + `freeze_until: 1`).

**Điểm kỹ thuật đáng chú ý (đã verify bằng chạy thật):**

1. VGG11 `classifier` là `Sequential[7]`, không phải 1 `Linear` — code giả định mọi model có `.fc` sẽ crash. `replace_head()` xử lý 2 dạng.
2. VGG11 `classifier` chứa 103M params — unfreeze cả sẽ khiến 92.8% model trainable, mất ý nghĩa freeze. Mặc định `freeze_head_full: false` (chỉ `Linear(4096,10)` cuối) + test bắt lỗi.
3. `unfreeze_last` dùng `block_units` đã introspect (resnet: `layer1..4`; vgg: 8 conv; densenet: 4 denseblock + `transition3`/`norm5`); mở luôn BatchNorm trong block.

**Chiến lược chốt:** `fine_tune` (`configs/model_config.yaml` mặc định `fine_tune`) — toàn bộ kết quả báo cáo đều fine-tune bằng Adam.

**Test:** 24/24 pass (18 unit + 6 integration; CLI verify weights thật cả 3 model).

## 4. Đánh giá theo đề bài (14 yêu cầu trong `đề.txt`)

| # | Yêu cầu | Trạng thái | Bằng chứng |
|---|---------|-----------|------------|
| E1 | Thử nhiều pre-trained model | ✅ | 3 backbone + report M2 |
| E2 | Tinh chỉnh hyperparameter | ✅ | Sweep 30 cấu hình 5 epoch (lr × bs) + retrain 20 epoch (`colab_sweep.ipynb`, `colab_best3_retrain.ipynb`, `OTM_RESULTS.md`, `sweep-FT-lr-bs.log`); CLI `--lr/--batch-size/--optimizer/--scheduler` |
| E3 | TensorBoard | ✅ | `tb/` mỗi run (tag `train/*`, `val/*`, `lr`, `model/grad_norm`, `samples/train_images`); vd `tensorboard --logdir results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/tb` |
| O1–O5 | Load / explore / adapt / freeze / unfreeze | ✅ | `src/models/factory.py`, 24 test |
| O6–O8 | Data / transforms / DataLoader | ✅ | `src/data/`, 47 test |
| O9–O10 | Loss+optimizer / training loop | ✅ | `loss.py`/`optimizer.py`/`trainer.py` (đúng 6 bước slide 8–9 Ch.3: device→zero_grad→forward→loss→backward→step), 14 test |
| O11 | Đánh giá test set | ✅ | 3 eval chốt 20 epoch (R18 0.8626 / VGG11 0.8774 ⭐ / DenseNet121 0.8715), 18 test |

Chi tiết từng milestone: `docs/reports/m2_models_report.md` (M2), `m3_training_report.md` (M3), `m4_evaluation_report.md` (M4). Demo: `notebooks/20261026_eda_cifar10_demo.ipynb` + demo 1 lệnh `src/demo` (`docs/demo.md`, chạy thật VGG11 8 ảnh test: đúng 6/8, ~50 ms/ảnh CPU → `results/demo/20261107-094717/`).

## 5. Training (M3 ✅) & Evaluation (M4 ✅) — số chốt 20 epoch GPU T4

- M3: `Trainer` (zero_grad→forward→loss→backward→step), `best.pt` theo val acc, TensorBoard scalar/histogram/grad/image; CLI sweep models×lr×bs; 14 test pass. Sweep 30 cấu hình 5 epoch (lr 0.0001/0.0005/0.001/0.003/0.005 × bs 32/64, Adam, ảnh 64, AMP): ngon nhất ResNet18 lr 0.0001 bs 32 (0.8456), VGG11 lr 0.0001 bs 32 (0.8538), DenseNet121 lr 0.0005 bs 32/64 (0.8618). Chi tiết: `docs/reports/m3_training_report.md`, `results/OTM_RESULTS.md`.
- Retrain 20 epoch: R18 (lr 0.0001, bs 32) best val **0.8634 @ ep 17**; VGG11 (lr 0.0001, bs 32) best val **0.8786 @ ep 20**; DenseNet121 (lr 0.0005, bs 64) best val **0.8800 @ ep 12**. Runs: `20261031-120033/-121323/-122534`.
- M4: `EvalResult` + `evaluate_model` + plots `out_path` + CLI eval; 18 test pass. Eval test 10.000 ảnh: R18 **0.8626**, VGG11 **0.8774** ⭐ (macro-F1 0.8765), DenseNet121 **0.8715** (macro-F1 0.8709). Chi tiết: `docs/reports/m4_evaluation_report.md`, `results/TEST_RESULTS.md`. Báo cáo HTML offline mỗi run: `report.html` (sinh bằng `src/evaluation/visualize_html.py`, mở bằng double-click).

## 6. Hạn chế đã biết

- **Train chốt trên GPU Colab T4** (ResNet18/VGG11 ~27–35 giây/epoch, DenseNet121 ~48–86 giây/epoch, ảnh 64, AMP). Local CPU chỉ dùng cho M1/demo/test (ResNet18 freeze 224 ~37 phút/epoch) nên không dùng ảnh 224 cho kết quả chốt.
- **Ảnh 32×32 upsample lên 64×64** (train/eval chốt) hoặc 224×224 (demo M1) — mất chi tiết, nhưng bắt buộc để dùng weights ImageNet (slide 20: transfer learning cho dataset nhỏ).
- **Mục tiêu accuracy là tự đặt** (freeze ≥ 0.70, fine-tune ≥ 0.80) — đề bài không cho ngưỡng. Cả 3 model chốt đều vượt 0.86 ✅.

## 7. Việc còn lại

| Module | Nội dung | Trạng thái |
|--------|----------|-----------|
| M3 train | Sweep 30 cấu hình 5 epoch + retrain 3 cấu hình 20 epoch → `best.pt` + `metrics.json` + `tb/` | ✅ xong |
| M4 eval | 3 eval chốt + PNG (`confusion_matrix.png`, `per_class_f1.png`) | ✅ xong |
| M5 | Tổng hợp README §8 + `TEST_RESULTS.md` + `OTM_RESULTS.md` | ✅ xong |

## Phụ lục — file & commit

- Code: `src/data/` (7 file), `src/models/` (4 file), `src/training/` (6 file), `src/evaluation/` (4 file + 2 test nhóm), `src/demo/` (demo 1 lệnh), `src/utils/` (2 file), `scripts/` (4 file).
- Test: 102 passed (47 M1 + 24 M2 + 13 M3 + 18 M4) — gồm `test_finetune_all_models` bắt bug B4 (DenseNet `get_module`); verify lại 2026-11-07: **102/102 pass, ~137s**.
- Notebook: `20261026_eda_cifar10_demo.ipynb` (M1/M2), `20261029_train_eval_demo.ipynb` (M3/M4 demo), `colab_sweep.ipynb` (sweep 30 cấu hình 5 epoch), `colab_best3_retrain.ipynb` (retrain 20 epoch + chấm test).
- Docs chuẩn skill (bổ sung 2026-11-07): `architecture.md`, `api.md`, `description.md`, `data.md`, `demo.md`, `CHANGELOG.md` + `visualize_html.py` + `report.html` 3 run chốt + `src/demo` + `data/processed/demo/` (8 ảnh mẫu).
- Audit: `docs/AUDIT_LOG.md`. Reports: `m2_models_report.md`, `m3_training_report.md`, `m4_evaluation_report.md`.
