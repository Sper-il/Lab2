# Practice 2 — Hands-on with Pre-trained Neural Network Architectures

> Single source of truth cho **đang làm gì, tại sao, và đã làm được gì**.
> Cập nhật lần cuối: 2026-11-01 09:30 (chốt retrain 20 epoch GPU Colab T4)

## Project Context (Phase 1 — DEFINE)

| Trường | Giá trị |
|--------|---------|
| Tên đầy đủ | Practice 2 — Hands-on with Pre-trained Neural Network Architectures |
| Slug | `lab2` |
| Bài toán | Phân loại ảnh CIFAR-10 (10 lớp) bằng model pre-trained ImageNet từ `torchvision.models`, so sánh transfer learning (freeze) và fine-tuning (unfreeze lớp cuối) |
| Người dùng mục tiêu | Sinh viên môn AI/ML — Viện CNTT&ĐT, TSU TP.HCM |
| Input | Ảnh RGB 32×32 → resize 64×64 trong các run train chốt (override `image_size: 64` để train nhanh trên GPU), hoặc 224×224 chuẩn ImageNet khi chạy pipeline M1/demo (`configs/data_config.yaml`), batch `[B, 3, 64, 64]` hoặc `[B, 3, 224, 224]`, `float32`, normalize theo ImageNet mean/std |
| Output | Logits `[B, 10]` → nhãn lớp + confidence |
| Tính năng cốt lõi | Must: nạp model pre-trained (ResNet18/VGG11/DenseNet121), thay head, freeze/unfreeze, training loop, validation, TensorBoard, test evaluation. Should: config YAML, bảng so sánh, augmentation, metrics.json. Could: scheduler, gradient clipping, ONNX |
| Tech construction | Python 3.13, PyTorch 2.14.1 (CPU), torchvision 0.29.1, TensorBoard 2.21.0, PyYAML, pytest |
| Tiêu chí đánh giá | Test acc (freeze) ≥ 0.70; Test acc (fine-tune) ≥ 0.80; macro-F1 ≥ 0.70; val−train gap ≤ 0.15; 1 epoch ≤ 15 phút (CPU) |

### Trả lời bộ câu hỏi bắt buộc

**What?**
- Task: image classification (10 classes).
- Data: CIFAR-10, 50.000 train + 10.000 test ảnh 32×32 RGB, 10 lớp cân bằng (5.000 ảnh/lớp train).
- Output: class label + softmax probability.
- Split: 45.000 train / 5.000 val (lấy từ train, stratified, seed=42) / 10.000 test.

**Why?**
- Dataset nhỏ (50k ảnh) → train từ đầu dễ overfit; dùng backbone ImageNet làm feature extractor là cách chuẩn và đúng nội dung Chương 3 (slide 20–21).
- CIFAR-10 là dataset được nêu trong slide 24 của tài liệu.
- Chọn ResNet18 (nhẹ, baseline chuẩn), VGG11 (đơn giản, dễ học), DenseNet121 (skip connection, mạnh hơn) để "Experiment with different pre-trained models".

**Where / When / Which?**
- Where: train nặng trên **GPU Colab T4 có AMP** (sweep 30 cấu hình 5 epoch + retrain 20 epoch); local CPU-only dùng cho pipeline M1/demo/test.
- When: chạy offline theo batch experiment, mỗi run sinh `results/runs/<timestamp>/`.
- Which: PyTorch (framework chính của Chương 3), torchvision (models + datasets + transforms), TensorBoard (theo slide 13–15).

**How?**
- Pipeline: `CIFAR10 → transforms → DataLoader → backbone (freeze/unfreeze) → head(10) → CrossEntropyLoss → Adam → train loop → val → TensorBoard → checkpoint → test eval`.
- Hyperparameter chốt: fine-tune Adam, ảnh 64, 20 epoch — ResNet18 lr 0.0001 bs 32, VGG11 lr 0.0001 bs 32, DenseNet121 lr 0.0005 bs 64 (chọn từ sweep 30 cấu hình 5 epoch).
- Augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize(ImageNet).
- Metric: accuracy (val để chọn best), accuracy + macro-F1 + confusion matrix (test).

### Đối chiếu với đề bài (`đề.txt`)

> Bảng này là hợp đồng giữa đề bài và hiện thực. Mỗi mục được kiểm tra lại khi đóng milestone. Nguồn chi tiết: mục "2b" trong `README.md`.

| # | Yêu cầu đề bài | Hiện thực | Module | Trạng thái |
|---|-----------------|-----------|--------|-----------|
| E1 | Experiment with different pre-trained models (VGG, DenseNet, etc.) | `src/models/factory.py` | M2 | ✅ |
| E2 | Adjust the hyperparameters (lr, batch size, etc.) | Sweep 30 cấu hình fine-tune Adam 5 epoch + retrain 3 cấu hình tốt nhất 20 epoch | `configs/*.yaml` + `src/training/` + `notebooks/colab_sweep.ipynb` + `notebooks/colab_best3_retrain.ipynb` | M3, M5 | ✅ |
| E3 | Use TensorBoard to monitor the training process | `src/training/trainer.py` + `tb/` mỗi run, tag scalar/histogram/grad/image | M3 | ✅ |
| O1 | Load a Pre-trained Model (`weights=`/`pretrained=True`) | `src/models/factory.py` | M2 | ✅ |
| O2 | Explore the Model Architecture — print model, identify layers & params | `src/models/factory.py` | M2 | ✅ |
| O3 | Adapt the Model — replace final classification layer (10 classes) | `src/models/factory.py` | M2 | ✅ |
| O4 | Freeze earlier layers (transfer learning) | `src/models/factory.py` | M2 | ✅ |
| O5 | Unfreeze a few of the final layers (fine-tuning) | `src/models/factory.py` | M2 | ✅ |
| O6 | Prepare the Data — load & prepare dataset | `src/data/` | **M1** | ✅ |
| O7 | Use `torchvision.transforms` to preprocess | `src/data/transforms.py` | **M1** | ✅ |
| O8 | Use `torch.utils.data.DataLoader` to create batches | `src/data/dataloader.py` | **M1** | ✅ |
| O9 | Define the loss function and optimizer | `src/training/loss.py` + `optimizer.py` (CE/label-smoothing/focal; Adam/AdamW/SGD) | M3 | ✅ |
| O10 | Write the training loop | `src/training/trainer.py` (đúng 6 bước slide 8–9 Ch.3) | M3 | ✅ |
| O11 | Evaluate the Model on a test dataset | `src/evaluation/` (3 eval chốt 20 epoch: R18 0.8626 / VGG11 0.8774 ⭐ / DenseNet121 0.8715) | M4 | ✅ |

Bước 1 (Environment Setup) — ✅ xong: `requirements.txt` pin torch 2.14.1 / torchvision 0.29.1 / tensorboard 2.21.0, bản CPU.

## Architecture Snapshot (Phase 2 — DESIGN)

- Pattern: transfer learning pipeline (config-driven, CLI theo module).
- Modules: `src/data` (dataset/transforms/split) ✅ M1, `src/models` (factory + head replacement + freeze logic) ✅ M2, `src/training` (trainer + scheduler + checkpoint + sweep 30 cấu hình + retrain 20 epoch) ✅ M3, `src/evaluation` (metrics + confusion matrix + visualize, 3 eval chốt 20 epoch) ✅ M4.
- Data flow: `raw (torchvision download) → transforms → batches → model → logits → loss → weights → metrics.json`.
- Risks: xem bảng dưới.

## Active Module: M1 — Data Pipeline (dataset/transforms/split)

- **Describe** — dựng pipeline data cho CIFAR-10: tải/verify dataset, chia train/val phân tầng, transform train/val, DataLoader. Input: `data/raw/cifar-10-python.tar.gz`. Output: 3 DataLoader (train 45k / val 5k / test 10k), ảnh float32 đã normalize ImageNet — `[B,3,224,224]` với pipeline demo M1, `[B,3,64,64]` với các run train chốt (`image_size: 64`).
- **Goal / purpose** — nền tảng dữ liệu reproducible cho M2–M5: mọi experiment sau dùng chung split & transform, seed cố định.
- **Pipeline** — `data/raw → Cifar10Dataset (in-memory) → stratified_indices(seed=42) → TransformSubset(train_transform|val_transform) → DataLoader → [B,3,64,64] (train chốt) / [B,3,224,224] (demo M1)`.
- **How / experiment plan**
  - [x] `src/data/transforms.py` — RandomCrop(32,pad=4)+Flip (train) / Resize+ToTensor+Normalize (val), ImageNet mean/std
  - [x] `src/data/dataset.py` — `Cifar10Dataset` nạp ảnh vào RAM (uint8), `TransformSubset` giữ transform riêng
  - [x] `src/data/split.py` — `stratified_indices` chia phân tầng 45k/5k, seed=42, kiểm tra overlap
  - [x] `src/data/dataloader.py` — `DataLoaderConfig` + `build_dataloaders`, giới hạn `num_workers` theo CPU
  - [x] `src/data/__main__.py` — CLI xuất artifact `config.yaml` + `metrics.json` + `logs.txt`
  - [x] `configs/data_config.yaml` — toàn bộ hyperparameter data pipeline
  - [x] `src/utils/config.py` + `src/utils/runs.py` — nạp YAML, tạo run dir, ghi JSON/artifact
  - [x] Tests — `pytest -q` → **47 passed** (36 test data pipeline + 11 test `prepare.py`)
  - [x] Run — `python -m src.data` → `results/runs/20261025-150847/`
  - [x] Biểu đồ — `scripts/plot_data_report.py` → `plots/class_distribution.png`, `plots/sample_images.png`
  - [x] `src/data/prepare.py` — xuất `data/processed/` (`.npy`) + `data/splits/` (`.npz`/`.json`), verify bằng hash nội dung ảnh
  - [x] `scripts/benchmark_loading.py` — đo throughput để quyết định có cache ảnh 224×224 (kết luận: không, tốn 7 GB mà không cải thiện đáng kể)
  - [x] Tests bổ sung — `tests/unit/test_prepare.py` → 11 test (tổng 47)
- **Status** — Describe ✅ / Generate ✅ / Review ✅ / Run ✅ / Test ✅ / Refactor ✅ / Commit ✅
- **Last run** — `results/runs/20261025-150847/` (metrics.json: validation status PASS)
- **Commit** — `f130862` (M0 scaffold), `d0321eb` (M1 data pipeline), `28aa1ff` (prepare step: `data/processed` + `data/splits`), `a1a6b71` (docs align với đề bài + MD5 check)
- **DoD checklist** — xem Definition of Done ở cuối file này

## Module Log

| # | Module | Ngày | Status | Commit | Ghi chú |
|---|--------|------|--------|--------|---------|
| 0 | M0 — Environment & Scaffold | 2026-10-25 | ✅ | — | Python 3.13.2, torch 2.14.1+cpu, torchvision 0.29.1+cpu, tensorboard 2.21.0; CPU-only |
| 0b | M0b — Dataset CIFAR-10 | 2026-10-25 | ✅ | — | Tải + giải nén + verify: 50k train / 10k test, 10 lớp cân bằng, ảnh 32×32 |
| 1 | M1 — Data pipeline (dataset/transforms/split) | 2026-10-25 | ✅ | `d0321eb` + `28aa1ff` | 45k/5k/10k stratified seed=42; ảnh `[B,3,224,224]` (demo M1) / `[B,3,64,64]` (train chốt); 47 test pass; run `20261025-150847` |
| 1b | M1b — Docs align với `đề.txt` | 2026-10-25 | ✅ | `a1a6b71` | Ma trận truy vết 14 yêu cầu đề; sửa shape, số liệu bịa, path không tồn tại; thêm MD5 check cho `verify_data.py` |
| 2 | M2 — Model factory (backbone/head/freeze) | 2026-10-26 | ✅ | `75455dd` (+`81fa425` fix B4) | `registry.py` + `factory.py` + CLI khám phá; 24 test pass (18 unit + 6 integration); verify weights thật cả 3 model; report `docs/reports/m2_models_report.md` |
| 2b | M2b — Docs bổ sung (notebook/audit/final report) | 2026-10-26 | ✅ | (commit này) | `notebooks/20261026_eda_cifar10_demo.ipynb` + `docs/AUDIT_LOG.md` + `docs/reports/final_report.md`; sửa D1–D5 trong audit |
| 3 | M3 — Training loop + TensorBoard | 2026-10-31 | ✅ | `2e98e31` | `trainer.py` (zero_grad→forward→loss→backward→step), loss/opt/scheduler/tb_utils, CLI sweep models×lr×bs; 14 test pass; sweep 30 cấu hình 5 epoch GPU T4 (`OTM_RESULTS.md`) + retrain 3 cấu hình tốt nhất 20 epoch: R18 val 0.8634, VGG11 val 0.8786, DenseNet121 val 0.8800 |
| 4 | M4 — Evaluation & metrics | 2026-10-31 | ✅ | `2e98e31` | `EvalResult` + `evaluate_model` + `out_path` plots + CLI eval + `eval_config.yaml`; 18 test pass; 3 eval chốt 20 epoch test 10.000 ảnh: R18 0.8626 / VGG11 0.8774 ⭐ / DenseNet121 0.8715 (`TEST_RESULTS.md` + `eval/*.png`) |
| 5 | M5 — Experiment sweep + báo cáo | 2026-10-31 | ✅ | `2e98e31` | Sweep 30 cấu hình xong (DenseNet121 0.8618 tốt nhất sweep 5 epoch) + retrain 20 epoch xong (VGG11 test **0.8774** tốt nhất); bảng so sánh README §8 + `TEST_RESULTS.md` |
| 6 | M6 — Docs chuẩn skill còn thiếu | 2026-11-07 | ✅ | (commit này) | Bổ sung `architecture.md` + `api.md` + `description.md` + `data.md` + `demo.md` + `CHANGELOG.md` + `model.md` + `evaluation.md` + `audit_code.md` theo folder convention skill; `visualize_html.py` + `report.html` offline 3 run chốt; `src/demo` demo 1 lệnh (8/8 ảnh mẫu, VGG11 đúng 6/8, ~50 ms/ảnh CPU) |
| 7 | M6b — Demo 1 lệnh + HTML visualize | 2026-11-07 | ✅ | (commit này) | `src/demo/__main__.py` (export `data/processed/demo/` 8 PNG + `labels.json`, suy luận + `results/demo/<ts>/predictions.json + index.html`); `report.html` 3 run `20261031-*` (pass/fail, curves, tốt/xấu, latency) |

> **Kế hoạch chi tiết M2–M4:** `docs/plan-m2-m4.md` — danh sách file, chức năng từng hàm, code mẫu đã kiểm chứng, test, tiêu chí hoàn thành, và ước lượng thời gian train **đã đo thật** trên máy này.

## Decisions Log

| Ngày | Quyết định | Lựa chọn khác | Lý do |
|------|------------|---------------|-------|
| 2026-10-25 | Dùng Python 3.13.2 | 3.11 (launcher hỏng), 3.12 (không có) | Chỉ 3.13 cài được và tương thích torch 2.14.1+cpu cp313 |
| 2026-10-25 | Cài bản CPU của PyTorch (`+cpu`) | CUDA wheel (không có GPU) | `nvidia-smi` không tồn tại trên máy |
| 2026-10-25 | Dataset = CIFAR-10 | MNIST, FashionMNIST, STL-10 | Đúng ví dụ slide 24; tải tự động, 10 lớp |
| 2026-10-25 | Ảnh train chốt resize lên **64×64** (override `image_size: 64` trong `train_config.yaml`), ảnh demo M1 giữ **224×224** chuẩn ImageNet | Resize 224 cho mọi run | Backbone pre-trained kỳ vọng input 224 nhưng train GPU chốt dùng 64 cho nhanh (~27–86 giây/epoch AMP) mà vẫn đạt 86–88% test; demo M1 giữ 224 để đúng chuẩn |
| 2026-10-25 | Config YAML cho mọi experiment | Hardcode trong code | Reproducibility: mỗi run ghi lại `config.yaml` |
| 2026-10-25 | 5.000 ảnh val lấy từ train | Dùng test set làm val | Giữ test set sạch cho đánh giá cuối |
| 2026-10-25 | `requirements.txt` pin không kèm hậu tố `+cpu` | `torch==2.14.1+cpu` | Wheel PyPI chính thức trên Windows đã là CPU; `pip install -r requirements.txt` phải chạy được với index mặc định |
| 2026-10-25 | Tải CIFAR-10 bằng `torchvision.datasets` (`download=True`) | Tải thủ công từ cs.toronto.edu | Tự động, có kiểm tra MD5 tích hợp, giải nén đúng cấu trúc; fallback thủ công vẫn được ghi trong `data/README.md` |
| 2026-10-25 | Nạp toàn bộ ảnh CIFAR-10 vào RAM (`in_memory=True`) | Đọc pickle từng lần qua `torchvision.datasets` | CIFAR-10 chỉ ~150 MB; nạp RAM nhanh hơn nhiều trên Windows nơi I/O là nút thắt |
| 2026-10-25 | `TransformSubset` tự viết thay vì `torch.utils.data.Subset` | `Subset` + gán `subset.dataset.transform` | `Subset` giữ tham chiếu dataset gốc → gán `.transform` làm hỏng transform của cả tập train. `TransformSubset` giữ transform riêng cho từng split |
| 2026-10-25 | Chia train/val **phân tầng theo lớp** (`stratified_indices`) | `random_split` ngẫu nhiên | Bảo đảm val có cùng tỉ lệ 10 lớp với train → val accuracy phản ánh đúng đánh giá |
| 2026-10-25 | Augmentation trên ảnh 32×32 **trước**, resize 224 **sau** | Resize 224 trước rồi crop | Crop trên ảnh gốc 32×32 giữ được nội dung chi tiết; crop trên ảnh up-sample 224 chỉ crop vùng upscale |
| 2026-10-25 | Ép stdout/stderr về UTF-8 trong `setup_logging` | Bỏ tiếng Việt có dấu trong log | Console Windows dùng cp1252, log tiếng Việt gây `UnicodeEncodeError` làm sập tiến trình |
| 2026-10-25 | `num_workers` bị giới hạn theo `os.cpu_count() - 1` | Cứ để 4 như config | Tránh spawn quá nhiều process trên máy CPU yếu; 0 worker nếu máy chỉ có ≤2 core |
| 2026-10-25 | Xuất `data/processed/` (`.npy`) và `data/splits/` (`.npz`+`.json`) | Để hai thư mục rỗng như scaffold ban đầu | Convention `data/{raw,processed,splits}` của skill `ai-fusion-workflow` yêu cầu chúng có nội dung thật; `processed/` = dữ liệu đã xử lý sạch, `splits/` = file split |
| 2026-10-25 | `processed/` giữ ảnh gốc 32×32, **không** resize 224 | Cache ảnh 224×224 uint8 (7.0 GB) | Resize phải ở transform lúc train: `RandomCrop` cần thao tác trên ảnh gốc 32×32, và augmentation phải khác nhau mỗi epoch. Benchmark: workers=4 → 403 ảnh/s (~1.9 phút/epoch) → data loading không phải nút thắt, không cần cache 7 GB |
| 2026-10-25 | Verify split bằng **hash nội dung ảnh**, không so chỉ số | So sánh trực tiếp `set(train_idx) ∩ set(test_idx)` | `train_idx` (0..49.999) và `test_idx` (0..9.999) thuộc hai không gian chỉ số khác nhau → so chánh overlap trực tiếp là vô nghĩa và tạo cảnh báo sai. So hash nội dung ảnh mới thật sự phát hiện rò rỉ |
| 2026-10-25 | `data/splits/` lưu cả `.npz` (dùng trong code) và `.json` (đọc/diff được) | Chỉ `.npz` | `.npz` nạp nhanh bằng `np.load`; `.json` giúp con người kiểm tra và git diff |
| 2026-10-25 | Thêm bảng **"Đối chiếu với đề bài"** vào `README.md` + `progress_status.md` | Chỉ mô tả tự do | Đề bài có 3 bài tập (E1–E3) + 7 bước (O1–O11); bảng ánh xạ từng yêu cầu → file/module/trạng thái giúp kiểm tra "đã làm đúng đề chưa" mỗi milestone, tránh docs trôi khỏi đề |
| 2026-10-25 | Sửa shape trong docs thành `[B, 3, 224, 224]` (không phải `[B, 3, 32, 32]`) | Giữ nguyên như đã viết | Ảnh 32×32 được resize lên 224×224 **trong transform**; shape thật sau transform là 224×224. Docs ghi sai khiến người đọc hiểu sai kích thước đầu vào model |
| 2026-10-25 | Đánh dấu rõ lệnh/lệnh-cfg **chưa tồn tại** trong README thay vì để trông như đã có | Bỏ hẳn mục 6.6 | `src/models`, `src/training`, `src/evaluation` mới chỉ có `__init__.py`; `configs/train_config.yaml` chưa tạo. Ghi rõ trạng thái "thiết kế dự kiến" tránh gây hiểu nhầm là đã chạy được |
| 2026-10-25 | Bổ sung kiểm tra **MD5 archive** vào `scripts/verify_data.py` | Chỉ kiểm tra số mẫu / lớp / shape | `download_data.py` và docs đều claim "có kiểm tra MD5 tích hợp" nhưng script thực tế **không** kiểm tra → claim sai. Nay đã khớp: hash theo chunk, so với `c58f30108f718f92721af3b95e74349a`, exit 1 + hướng dẫn tải lại |
| 2026-10-25 | Ghi rõ 4 tiêu chí đánh giá là **mục tiêu tự đặt**, không phải ngưỡng từ đề bài | Để trông như yêu cầu của đề | `đề.txt` không đặt ngưỡng accuracy nào; ghi rõ nguồn gốc tránh hiểu nhầm là yêu cầu giáo viên |
| 2026-11-01 | Train chốt trên **GPU Colab T4 có AMP** (sweep 30 cấu hình 5 epoch + retrain 3 cấu hình 20 epoch, ảnh 64) | Chỉ train local CPU | Local CPU quá chậm (VGG11 224 ~144 phút/epoch); GPU T4 chỉ ~27–86 giây/epoch nên mới sweep/retrain được; local giữ lại cho M1/demo/test |
| 2026-10-25 | Tách `src/models/registry.py` (bảng tra cứu) khỏi `src/models/factory.py` (logic) | Gộp tất cả vào 1 file | `registry.py` là dữ liệu (đổi khi thêm model), `factory.py` là logic (ổn định). Thêm `mobilenet_v3_small` sau này chỉ sửa 1 file |
| 2026-10-25 | `unfreeze_last` dùng **số block** (`freeze_until: 1`) thay vì tên layer cụ thể | Hardcode tên layer cho từng model | 3 model có cấu trúc khác nhau (`layer4` vs `features[-3:]` vs `denseblock4`) — dùng số block khiến code chung, chỉ khác cách xác định "block cuối" |
| 2026-10-25 | Tự tính macro-F1 bằng `torch` thuần, **không** thêm `scikit-learn` | Dùng `sklearn.metrics` | Thêm dependency chỉ để tính F1 là không đáng; giữ `requirements.txt` gọn và tránh rủi ro cài đặt |
| 2026-10-25 | `build_optimizer` chỉ truyền params có `requires_grad=True` | Truyền toàn bộ `model.parameters()` | Tránh optimizer theo dõi trọng số đã freeze; phản ánh đúng "params đang train" khi báo cáo |
| 2026-10-25 | Với VGG11, `freeze_backbone` chỉ unfreeze **lớp Linear cuối** (`freeze_head_full: false`) | Unfreeze toàn bộ `classifier` | **Đã đo thật**: `classifier` của VGG11 chứa 103M params. Unfreeze cả sẽ khiến **92.8% model trainable** → mất ý nghĩa transfer learning và epoch chậm gần như full fine-tune. Chỉ Linear cuối = 40,970 params (0.032%) |
| 2026-10-25 | Dùng `block_units` (đường dẫn module đã introspect) để unfreeze n block cuối | Hardcode tên layer cho từng model | `resnet18` có 4 stage (`layer1..4`), `vgg11` có 8 conv block (chỉ số `0,3,6,8,11,13,16,18`), `densenet121` có 4 denseblock — khai báo tập trung trong `registry.py`, code unfreeze dùng chung cho cả 3 |
| 2026-10-25 | `unfreeze_last` mở luôn BatchNorm trong block được unfreeze | Chỉ mở Conv/Linear | BN đóng băng sẽ chỉ dùng running stats → không thích ứng được với phân bố CIFAR-10, làm giảm hiệu quả fine-tune |

## Risks & Mitigations

| Rủi ro | Khả năng | Tác động | Giảm thiểu | Phát hiện |
|--------|----------|----------|------------|-----------|
| Training quá chậm trên CPU (VGG11/DenseNet121) | Cao | Đã xảy ra → chuyển train chốt lên GPU Colab T4 | Sweep 30 cấu hình 5 epoch + retrain 20 epoch trên T4 có AMP (~27–86 giây/epoch); local chỉ chạy M1/demo/test | Log thời gian mỗi epoch + `sweep-FT-lr-bs.log` |
| Overfitting nặng do ảnh 32×32 upsample | Trung bình | Test acc thấp hơn val | Augmentation (crop/flip), weight decay, theo dõi gap val−train | Biểu đồ TensorBoard |
| Tải CIFAR-10 lỗi mạng | Trung bình | Không chạy được | `download=True` có retry; fallback sang `FakeData` cho smoke test | Exception khi load |
| Overfit từ quá nhiều epoch | Trung bình | Kết quả kém | Early stop theo val accuracy, lưu best checkpoint | Val acc giảm |

## Open Questions

- [x] Tên project + bài toán
- [x] Input/output shape
- [x] Tính năng cốt lõi (Must/Should/Could)
- [x] Tech stack + lý do
- [x] Metric đánh giá + target
- [x] Có được dùng `torch.compile` / AMP không — có, **đã dùng AMP trên GPU Colab T4** (`use_amp: true` trong `train_config.yaml` + `colab_best3_retrain.ipynb`); local CPU giữ đơn giản, không dùng `torch.compile`.

## Definition of Done (áp dụng mỗi module)

- [x] Đúng yêu cầu
- [x] Code chạy end-to-end
- [x] Tests pass — **102/102** (47 M1 + 24 M2 + 13 M3 + 18 M4; `pytest -q` ~137s local CPU)
- [x] Không có secrets; không push artifact nặng (`*.pt` ~590 MB local-only, `tb/` local-only — `metrics.json`/PNG/config/`report.html` đã push; `data/processed/demo/` 8 ảnh mẫu + `results/demo/` nhẹ được push)
- [x] Code đã self-review
- [x] Có thể giải thích được
- [x] Docs đã cập nhật (README, `docs/progress_status.md`, `docs/AI_Usage.md` + khung chuẩn đủ: `architecture.md`, `api.md`, `description.md`, `data.md`, `model.md`, `evaluation.md`, `audit_code.md`, `demo.md`, `CHANGELOG.md`)
- [x] **Docs bám sát `đề.txt`** — mục "Đối chiếu với đề bài" trong `README.md` và `docs/progress_status.md` được cập nhật mỗi milestone
- [x] Đã committed (Conventional Commit) — `f130862` (M0), `d0321eb`+`28aa1ff` (M1), `a1a6b71` (docs align đề), `75455dd` (M2), `390b21d` (M3/M4 code), `c78f84e` (sweep FT), `c8b6fff` (models/), `5b810f0` (đối chiếu đề+slide, sweep E2), `2e98e31` (retrain 3 cấu hình 20 epoch + chấm test Colab T4), + commit hiện tại (bổ sung docs chuẩn skill còn thiếu: architecture/api/description/data/demo/CHANGELOG + `visualize_html.py` + `report.html` 3 run + `src/demo` demo 1 lệnh)
