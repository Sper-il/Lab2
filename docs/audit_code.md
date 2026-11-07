# Audit code — kiểm định mã nguồn (M1–M4 + demo)

> Kiểm định tĩnh + động toàn bộ mã nguồn theo yêu cầu đề bài (`đề.txt`) và skill `ai-fusion-workflow`.
> Mọi nhận xét đều đối chiếu code thật trong `src/`, config trong `configs/`, test trong `tests/` + `src/evaluation/test_*.py`.
> Nhật ký sửa lỗi chi tiết: `docs/AUDIT_LOG.md`. Kiến trúc model: `docs/model.md`.

**Ngày audit:** 2026-11-07
**Người audit:** AI agent (Cursor) + user review
**Phạm vi:** `src/data/*`, `src/models/*`, `src/training/*`, `src/evaluation/*`, `src/demo/*`, `src/utils/*`, `configs/*.yaml`, `scripts/*`
**Phương pháp:** đọc code từng file + chạy test thật (`pytest -q` → **102 passed**) + chạy CLI thật (`src.data`, `src.models`, `src.demo`, `visualize_html`) + đối chiếu từng yêu cầu E1–E3/O1–O11 trong `đề.txt`.

## 1. Tổng quan codebase

| Module | File | Chức năng | Trạng thái |
|--------|------|-----------|------------|
| M1 data | `src/data/dataset.py`, `transforms.py`, `split.py`, `dataloader.py`, `prepare.py`, `__main__.py` | `Cifar10Dataset` in-memory, transform train/val, chia stratified 45k/5k, 3 `DataLoader`, xuất `data/processed/` + `data/splits/` | ✅ |
| M2 model | `src/models/registry.py`, `factory.py`, `__main__.py` | `BACKBONE_REGISTRY` 3 backbone, `build_model`, `replace_head` (2 dạng head), `freeze_backbone`/`unfreeze_last`, `count_parameters`/`describe_model`, CLI khám phá | ✅ |
| M3 training | `src/training/trainer.py`, `loss.py`, `optimizer.py`, `scheduler.py`, `tensorboard_utils.py`, `__main__.py` | `TrainConfig` + `Trainer` (train/val/checkpoint/TensorBoard), CE/label-smoothing/focal, Adam/AdamW/SGD, none/cosine/step/plateau, sweep models×lr×bs | ✅ |
| M4 evaluation | `src/evaluation/metrics.py`, `visualize.py`, `visualize_html.py`, `__main__.py` | `EvalResult` + `evaluate_model`, confusion matrix/per-class F1/PR/misclassified/train curves, `report.html` offline, CLI eval | ✅ |
| M5 demo | `src/demo/__main__.py` | Export 8 ảnh mẫu + suy luận checkpoint + `predictions.json` + `index.html` | ✅ |
| Utils | `src/utils/config.py`, `runs.py` | Nạp YAML, `project_root`/`resolve_path`, `create_run_dir`, `setup_logging` (ép UTF-8), `write_json` (hỗ trợ numpy) | ✅ |

## 2. Audit tĩnh theo file

### 2.1 `src/data/*` (M1)

- `transforms.py`: augment (RandomCrop 32 padding 4 + Flip 0.5) **trước** resize — đúng quyết định đã ghi (crop trên ảnh gốc giữ chi tiết). Val/test không augment → đánh giá xác định. Normalize ImageNet đúng mean/std chuẩn.
- `dataset.py`: `Cifar10Dataset` nạp RAM (~150 MB) + `TransformSubset` giữ transform riêng — đã sửa bug `Subset.dataset.transform` làm hỏng transform train (AUDIT_LOG 2026-10-25).
- `split.py`: `stratified_indices` seed 42, 45k/5k, kiểm tra overlap bằng **hash nội dung ảnh** (không so chỉ số hai không gian khác nhau) — đã sửa cảnh báo giả (AUDIT_LOG 2026-10-25).
- `dataloader.py`: `_safe_num_workers` giới hạn theo CPU (`cpu_count - 1`, ≤2 core → ≤2 worker); `pin_memory=True`, `persistent_workers` chỉ khi có worker; `download=False` mặc định → không tải lén; thiếu data báo `FileNotFoundError` rõ ràng kèm hướng dẫn.
- `prepare.py`: giữ ảnh gốc 32×32 trong `processed/` (không cache 224 → tiết kiệm 7 GB, đã benchmark 403 ảnh/s); xuất cả `.npz` (code) + `.json` (đọc/diff).
- `scripts/verify_data.py`: kiểm tra MD5 archive theo chunk (`c58f30108f718f92721af3b95e74349a`) + số mẫu/lớp/shape — đã bổ sung sau khi phát hiện claim sai (AUDIT_LOG 2026-10-25).

### 2.2 `src/models/*` (M2)

- `registry.py`: `BackboneSpec` introspect từ torchvision 0.29 (không đoán): `resnet18` (`fc`, 4 stage), `vgg11` (`classifier` Sequential 7 phần tử, 8 conv), `densenet121` (`classifier` Linear, 4 denseblock + `extra_trainable`). `get_module` ưu tiên `getattr` trước index — đã sửa crash `features.denseblock4` của DenseNet (bug B4, AUDIT_LOG 2026-10-29).
- `factory.py`:
  - `ModelConfig.__post_init__` validate `name`, `strategy`, `num_classes >= 2`, `freeze_until >= 1` → fail sớm, thông báo rõ.
  - `resolve_weights`: `None`/`DEFAULT`/`IMAGENET1K_V1`, sai tên báo `ValueError` liệt kê lựa chọn.
  - `replace_head`: xử lý 2 dạng head (linear thay trực tiếp; sequential thay Linear cuối, giữ ReLU/Dropout) → code giả định mọi model có `fc` sẽ crash đã bị loại.
  - `freeze_backbone`/`unfreeze_last`: tắt hết `requires_grad` rồi chỉ mở head + block cuối + `extra_trainable`; VGG11 mặc định `freeze_head_full: false` (chỉ Linear cuối 40.970 params thay vì cả `classifier` 103M → 92,8% trainable) + test bắt lỗi cấu hình sai.
  - `build_model`: `from_scratch` giữ nguyên trainable (không freeze nhầm).
- `__main__.py`: CLI in cấu trúc + đếm params + kiểm tra output `(1, 10)` bằng dummy 224 trước khi train → bắt mismatch kênh/số lớp sớm.

### 2.3 `src/training/*` (M3)

- `trainer.py`: đúng 6 bước slide 8–9 Chương 3 (zero_grad → forward → loss → backward → step); `train_one_epoch` set `train()`, `validate()` + eval dùng `@torch.no_grad()`/`inference_mode()`; AMP chỉ khi `cuda` (`GradScaler("cuda")`, `autocast float16`); scheduler plateau dùng `val_acc` còn lại step mỗi epoch; best checkpoint theo val acc + luôn có `last.pt`; early stopping theo `patience`.
- `loss.py`: `CrossEntropyLoss` chốt + `label_smoothing` + `FocalLoss(gamma=2.0)`; tên sai báo `ValueError`.
- `optimizer.py`: chỉ nhận params `requires_grad=True` → optimizer không theo dõi trọng số freeze; model không có param trainable báo lỗi rõ (bắt lỗi quên mở head).
- `scheduler.py`: none/cosine/step/plateau, sai tên báo `ValueError`.
- `tensorboard_utils.py`: scalar (loss/acc/lr), histogram trọng số, grad norm, ảnh mẫu (denormalize về [0,1]), graph (try/except + tắt mặc định vì chậm).
- `__main__.py`: sweep models×lr×bs, seed cố định, `torch.set_num_threads(cpu-4)`, snapshot config + `metrics.json` mỗi combo + `sweep_summary.json`; bug truyền val 2 lần đã sửa (B1).

### 2.4 `src/evaluation/*` (M4)

- `metrics.py`: `_prepare_tensors` ép tensor long 1D + kiểm tra cùng độ dài; `accuracy` từ chối data rỗng; `confusion_matrix` kiểm tra nhãn ngoài `[0, n_classes)`; `per_class_metrics` tránh chia 0; key per-class dạng string để JSON-serializable; `evaluate_model` kiểm tra logits `(B, 10)`, dùng `inference_mode`, đồng bộ CUDA trước khi đo giờ, từ chối loader rỗng.
- `visualize.py`: kiểm tra ma trận vuông, khớp tên lớp, PR curve bỏ lớp thiếu mẫu dương/âm, `misclassified` từ chối `max_images <= 0` và tập không có ảnh sai, `train_curves` từ chối history rỗng/không có key hợp lệ; đã sửa warning categorical khi vẽ F1 (B2).
- `visualize_html.py`: render `report.html` offline duy nhất (ảnh nhúng base64, số khớp `metrics.json`).
- `__main__.py`: load checkpoint (`state_dict` + `config` train), build lại model bằng `from_scratch` + `weights=None` rồi `load_state_dict` (không tải lại ImageNet); eval đúng `image_size` checkpoint (64); xuất `metrics.json` + 2 PNG.

### 2.5 `src/utils/*` + `src/demo/*`

- `config.py`: `load_config` báo `FileNotFoundError`/`ValueError` rõ; `deep_merge` không mutate input; `resolve_path` neo theo `project_root`.
- `runs.py`: `create_run_dir` không ghi đè run cũ (thêm suffix); `setup_logging` ép UTF-8 cho stdout/stderr + file UTF-8 (đã sửa `UnicodeEncodeError` cp1252 trên Windows); `_NumpyEncoder` xử lý `np.integer`/`floating`/`ndarray`/`Path`.
- `demo/__main__.py`: ép UTF-8 cho console (đã sửa crash in tiếng Việt); checkpoint thiếu báo `FileNotFoundError`; đo latency từng mẫu + đồng bộ CUDA; xuất `predictions.json` + `index.html`.

## 3. An toàn và tài nguyên

| Khía cạnh | Hiện trạng |
|-----------|-----------|
| Exception handling | Mọi input ngoài (config, weights, strategy, split, checkpoint, loader rỗng, nhãn sai miền) đều có `ValueError`/`FileNotFoundError` kèm lựa chọn hợp lệ; khối TensorBoard/graph bọc try/except chỉ cảnh báo |
| Thiết bị | `device: auto` → cuda nếu có else cpu; AMP chỉ bật khi cuda; `pin_memory` + `non_blocking` khi chuyển batch |
| Bộ nhớ | In-memory ~150 MB CIFAR-10; không cache 224 (tiết kiệm ~7 GB); `*.pt` (~590 MB, VGG11 ~515 MB) và `tb/` local-only qua `.gitignore`; chỉ push JSON/YAML/PNG/`report.html` |
| Encoding Windows | `setup_logging` + `demo` ép UTF-8 (`errors="replace"`) cho stdout/stderr/file → log tiếng Việt không sập tiến trình |
| Seed/tái tạo | Seed 42 cho split/shuffle/train; mỗi run lưu `config.yaml` + `metrics.json` + `logs.txt`; checkpoint lưu `config` + `best_val_acc`/`best_epoch` |

## 4. Test coverage (102 passed, ~137s local CPU)

| Nhóm | File | Số test | Nội dung chính |
|------|------|---------|----------------|
| M1 unit | `tests/unit/test_transforms.py`, `test_split.py`, `test_dataset.py`, `test_prepare.py` | 36 + 11 prepare | transform/augment, stratified split, dataset RAM, hash chống leakage |
| M1 integration | `tests/integration/test_dataloader_pipeline.py` | — | 3 loader end-to-end (tổng M1: 47) |
| M2 | `tests/unit/test_factory.py` (18, gồm `test_finetune_all_models` bắt bug B4) + `tests/integration/test_model_pipeline.py` (6) | 24 | registry/freeze/unfreeze/params/output `(1,10)` cả 3 model |
| M3 | `tests/unit/test_training_components.py` (9) + `tests/integration/test_training_loop.py` (4) | 13 | loss/optimizer/scheduler/train 1 epoch/checkpoint |
| M4 | `src/evaluation/test_metrics.py` (13) + `src/evaluation/test_evaluation.py` (5) | 18 | acc/F1/CM/per-class/report, end-to-end/schema JSON/plots/reload checkpoint |

Lệnh: `.\.venv\Scripts\python.exe -m pytest -q`

## 5. Đối chiếu đề bài (`đề.txt`)

| # | Yêu cầu | Hiện thực | Kết quả |
|---|---------|-----------|---------|
| E1 | Thử nhiều pre-trained (VGG, DenseNet...) | `factory.py` 3 backbone + sweep 30 cấu hình + retrain 20 epoch | ✅ R18 0,8626 / VGG11 0,8774 ⭐ / DenseNet121 0,8715 test |
| E2 | Tinh chỉnh hyperparameter | `configs/*.yaml` + CLI sweep lr×bs + notebook Colab | ✅ 30 cấu hình 5 epoch → 3 cấu hình 20 epoch |
| E3 | TensorBoard | `tensorboard_utils.py` + `tb/` mỗi run | ✅ scalar/histogram/grad/image (+ graph tùy chọn) |
| O1 | Load pre-trained | `weights="DEFAULT"` ImageNet | ✅ cả 3 model |
| O2 | Khám phá kiến trúc | `describe_model` + `count_parameters` + CLI `--print-model` | ✅ output `(1,10)` |
| O3 | Thay lớp cuối (10 lớp) | `replace_head` 2 dạng head | ✅ |
| O4 | Freeze (transfer learning) | `freeze_backbone` | ✅ (VGG11 chỉ Linear cuối) |
| O5 | Unfreeze vài lớp cuối (fine-tune) | `unfreeze_last` + `block_units` | ✅ |
| O6 | Chuẩn bị data | `src/data/` 45k/5k/10k | ✅ |
| O7 | `torchvision.transforms` | `transforms.py` | ✅ |
| O8 | `DataLoader` | `dataloader.py` 3 loader | ✅ |
| O9 | Loss + optimizer | CE + Adam (chốt) | ✅ |
| O10 | Training loop | `trainer.py` 6 bước | ✅ |
| O11 | Eval test set | `evaluate_model` + CM + `metrics.json` | ✅ test 10.000 ảnh |

Bước 1 Environment Setup: ✅ `requirements.txt` pin torch 2.14.1/torchvision 0.29.1/tensorboard 2.21.0 (bản CPU local; train chốt GPU Colab T4 + AMP).

## 6. Phát hiện và xử lý

Bug logic đã sửa trước khi chốt (chi tiết `docs/AUDIT_LOG.md`): `ImportError` registry (A1), `unfreeze_last` sai block ResNet (A2), biến chưa định nghĩa `resolve_weights` (A3), VGG11 unfreeze cả `classifier` 103M (A4), `dataset_summary` truyền nhầm loader (B1), warning vẽ F1 (B2), eval nhầm `image_size` 224 (B3), crash `get_module` với DenseNet (B4), `UnicodeEncodeError` cp1252 ở `demo` (F5). Docs số bịa/shape sai/link chết (D1–D8, C1–C10, E1–E3) đã sửa và đồng bộ theo 3 run chốt `20261031-*`.

Rủi ro còn lại: VGG11 `best.pt` ~515 MB vượt giới hạn GitHub → giữ local-only (đã chặn `.gitignore`); sweep/eval đã khóa test (chỉ eval cuối 1 lần), mọi thử nghiệm trên val.

## 7. Kết luận

- Code M1–M4 + demo chạy end-to-end trên môi trường sạch, test **102/102 pass**, số liệu báo cáo tái tạo từ `results/runs/20261031-*/<cfg>/metrics.json` + `eval/metrics.json`.
- Không có secrets, không commit dataset/checkpoint nặng, log rõ ràng, docs đã đồng bộ đề bài.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.models --model vgg11 --print-model
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --models vgg11 --lr 0.0001 --batch-size 32 --epochs 20
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml --checkpoint results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/checkpoints/best.pt
.\.venv\Scripts\python.exe -m src.demo --checkpoint models/checkpoints/vgg11-best.pt --num-samples 8
```
