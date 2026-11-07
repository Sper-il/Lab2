# Practice 2 — Hands-on with Pre-trained Neural Network Architectures

> Bài thực hành Chương 3 (Training Deep Neural Networks): nạp và fine-tune các model pre-trained từ `torchvision.models` (VGG, DenseNet, ResNet...), tinh chỉnh hyperparameter, và theo dõi quá trình train bằng TensorBoard.

## 1. Project Overview

Project thực hiện **Practice 2** của Chương 3 — *Training Deep Neural Networks* (TSU, Viện CNTT&ĐT).

Mục tiêu chính:

1. Làm quen với việc nạp và sử dụng model pre-trained từ `torchvision.models` (`weights="DEFAULT"` = pre-train trên ImageNet).
2. Hiểu cách **adapt** model pre-trained cho bài toán cụ thể: thay lớp phân loại cuối, **freeze** các lớp trước (transfer learning) hoặc **unfreeze** vài lớp cuối (fine-tuning).
3. Tinh chỉnh hyperparameter (learning rate, batch size, optimizer, số epoch) và so sánh kết quả giữa nhiều kiến trúc.
4. Giám sát training bằng **TensorBoard** (`SummaryWriter`): loss/accuracy, histogram trọng số, model graph, ảnh mẫu.

Người học: sinh viên lớp AI/ML — Viện CNTT&ĐT, TPHCM.
Công cụ: PyTorch 2.x + torchvision + TensorBoard.

## 2. Problem Statement

**Bài toán:** Phân loại ảnh CIFAR-10 (10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) bằng model pre-trained ImageNet.

**Vấn đề cốt lõi:** CIFAR-10 chỉ có 50.000 ảnh 32×32 — rất ít so với ImageNet (hơn 1.2 triệu ảnh). Train CNN từ đầu sẽ overfit nặng. Transfer learning (giữ phần feature extractor của model pre-trained, chỉ thay head) giải quyết trực tiếp vấn đề này.

**Vì sao chọn CIFAR-10:**

- Là dataset chuẩn trong slide "Example (using the CIFAR-10 dataset)" của tài liệu Chương 3.
- Tải tự động qua `torchvision.datasets.CIFAR10` — không cần thu thập dữ liệu thủ công.
- Đủ nhỏ để train nhanh trên CPU, đủ lớn để đo được khác biệt giữa các chiến lược freeze/fine-tune.

**Input:** ảnh RGB 32×32 × 3 → resize lên **64×64** trong các run train chính (override `image_size: 64` trong `configs/train_config.yaml` để train nhanh), hoặc **224×224** chuẩn ImageNet khi chạy pipeline M1/demo (`configs/data_config.yaml`, một batch: `[B, 3, 224, 224]`, dtype `float32`, đã normalize theo ImageNet mean/std).
**Output:** vector logits `[B, 10]` → `argmax` ra nhãn lớp; probability qua softmax.

**Chia dữ liệu:** CIFAR-10 chuẩn — 50.000 ảnh train / 10.000 ảnh test. Ta lấy 5.000 ảnh từ tập train làm **validation** (stratified, seed cố định), còn lại 45.000 để train. Test set chỉ dùng để đánh giá cuối cùng.

**Metric:** accuracy trên validation (để chọn best checkpoint) và accuracy trên test (báo cáo cuối). Bổ sung: per-class precision/recall/F1, confusion matrix, thời gian suy luận.

## 2b. Đối chiếu với đề bài (Practice 2)

> Nguồn: `đề.txt`. Bảng này là hợp đồng giữa đề bài và hiện thực — dùng để kiểm tra "đã làm đúng chưa" ở mỗi milestone.
>
> **Kế hoạch triển khai M2–M4** (file nào, hàm nào, test nào, ước lượng thời gian): [`docs/plan-m2-m4.md`](docs/plan-m2-m4.md)

| # | Yêu cầu trong `đề.txt` | Nội dung cụ thể | Hiện thực | Module | Trạng thái |
|---|------------------------|-----------------|-----------|--------|-----------|
| E1 | Experiment with different pre-trained models (VGG, DenseNet, etc.) | So sánh ResNet18 / VGG11 / DenseNet121 | `src/models/factory.py` | M2 | ✅ |
| E2 | Adjust the hyperparameters (learning rate, batch size, etc.) | Sweep 30 cấu hình fine-tune Adam 5 epoch (lr 0.0001/0.0005/0.001/0.003/0.005 × batch 32/64) + retrain 3 cấu hình tốt nhất 20 epoch | `configs/*.yaml` + `src/training/` + `notebooks/colab_sweep.ipynb` + `notebooks/colab_best3_retrain.ipynb` | M3, M5 | ✅ |
| E3 | Use TensorBoard to monitor the training process | Scalar (loss/acc), histogram weight, graph, image | `src/training/trainer.py` | M3 | ✅ |
| O1 | Load a Pre-trained Model (`pretrained=True` / `weights=`) | `torchvision.models.*(weights="DEFAULT")` | `src/models/factory.py` | M2 | ✅ |
| O2 | Explore the Model Architecture — print model, identify layers & params | In cấu trúc + đếm `params_total` / `params_trainable` | `src/models/factory.py` | M2 | ✅ |
| O3 | Adapt the Model — replace final classification layer (10 classes) | Thay `fc` / `classifier` bằng `Linear(..., 10)` | `src/models/factory.py` | M2 | ✅ |
| O4 | Freeze earlier layers (transfer learning) | `freeze_backbone` — chỉ train head | `src/models/factory.py` | M2 | ✅ |
| O5 | Unfreeze a few of the final layers (fine-tuning) | `fine_tune` — unfreeze block cuối + norm + head | `src/models/factory.py` | M2 | ✅ |
| O6 | Prepare the Data — load & prepare dataset | CIFAR-10 45k/5k/10k, in-memory | `src/data/` | **M1** | ✅ |
| O7 | Use `torchvision.transforms` to preprocess | RandomCrop+Flip → Resize 64 (train chốt GPU) / 224 (pipeline M1 demo) → ToTensor → Normalize | `src/data/transforms.py` | **M1** | ✅ |
| O8 | Use `torch.utils.data.DataLoader` to create batches | 3 loader: 704/79/157 batch (224, M1 demo `[B,3,224,224]`); train chốt 64: 1407/157/313 batch (bs 32) hoặc 704/79/157 batch (bs 64) | `src/data/dataloader.py` | **M1** | ✅ |
| O9 | Define the loss function and optimizer | `CrossEntropyLoss` + Adam | `src/training/trainer.py` | M3 | ✅ |
| O10 | Write the training loop | zero_grad → forward → loss → backward → step | `src/training/trainer.py` | M3 | ✅ |
| O11 | Evaluate the Model on a test dataset | Test acc + macro-F1 + confusion matrix | `src/evaluation/metrics.py` | M4 | ✅ |

**Bước 1 của đề — Environment Setup:** đã hoàn thành (`requirements.txt` pin `torch==2.14.1` / `torchvision==0.29.1` / `tensorboard==2.21.0`; CPU-only do máy không có GPU NVIDIA).

**Phần Exercise còn lại (E1–E3) là trọng tâm M2–M5.**

## 3. Main Features

| Độ ưu tiên | Tính năng | Lý do |
|------------|-----------|-------|
| Must | Nạp model pre-trained từ `torchvision.models` (ResNet18, VGG11, DenseNet121) | Yêu cầu trực tiếp của Practice 2 — "Experiment with different pre-trained models" |
| Must | Thay head classifier theo số lớp (10) | Bắt buộc để model chạy được trên CIFAR-10 |
| Must | Transfer learning: freeze toàn bộ backbone, chỉ train head | Bài toán dataset nhỏ — cần điều chỉnh trọng số ít |
| Must | Fine-tuning: unfreeze N lớp cuối (block + norm + head) | Yêu cầu "unfreeze a few of the final layers" |
| Must | Training loop đầy đủ (zero_grad → forward → loss → backward → step) | Theo slide 8–9 của Chương 3 |
| Must | Validation mỗi epoch, lưu best checkpoint theo val accuracy | Phát hiện overfitting (slide 10–11) |
| Must | TensorBoard: scalar, histogram, graph, image | Yêu cầu "Use TensorBoard to monitor the training process" (slide 13–15) |
| Must | Evaluation trên test set + confusion matrix + báo cáo metrics JSON | "Evaluate the model's performance on a test dataset" |
| Should | Config YAML cho toàn bộ experiment (model, hyperparameter, đường dẫn) | Reproducibility — mỗi run ghi lại config đã dùng |
| Should | So sánh nhiều model/hyperparameter, xuất bảng so sánh | Trả lời "adjust hyperparameters" một cách có hệ thống |
| Should | Data augmentation (RandomCrop, HorizontalFlip) + Normalize ImageNet | Chống overfitting khi dataset nhỏ |
| Should | Lưu `metrics.json` + `config.yaml` vào `results/runs/<timestamp>/` | Theo convention của workflow |
| Could | Learning rate scheduler (CosineAnnealing / StepLR) | Tinh chỉnh lr theo thời gian |
| Could | Trực quan hoá activation/weight histogram | Phát hiện vanishing/exploding gradient |
| Could | Gradient clipping | Ổn định khi fine-tune với lr lớn |
| Could | Export checkpoint sang ONNX | Deploy nhanh hơn |

## 4. System Architecture

```
[CIFAR-10 /root]  ──►  [Transforms]  ──►  [DataLoader]
                        resize 32→64 (train/eval chốt GPU T4)      shuffle=True (train)
                        resize 32→224 (demo M1)                    shuffle=False (val/test)
                        normalize ImageNet   num_workers=N
                              │
                              ▼
                [torchvision.models backbone]
                 (ResNet18 / VGG11 / DenseNet121)
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
      [FREEZE backbone]              [UNFREEZE last block]
              │                               │
              └───────────────┬───────────────┘
                              ▼
                    [Replaced head: Linear → 10]
                              │
                              ▼
                [CrossEntropyLoss] + [Adam]  (chốt chỉ dùng Adam)
                              │
                              ▼
                   [Training loop + Validation]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
         [TensorBoard logs]     [best.pt checkpoint]
              results/runs/*/tb/    results/runs/*/checkpoints/
                    │
                    ▼
              [Test evaluation]
     accuracy / F1 / confusion matrix → results/runs/<ts>/metrics.json
```

Chi tiết luồng dữ liệu: xem `docs/data-flow.md` (M1) + `docs/plan-m2-m4.md` (M2–M4).

> Sơ đồ trên là pipeline **đã hiện thực đầy đủ M1–M4**: data (M1 ✅, run `20261025-150847`) → factory 3 backbone + freeze/unfreeze (M2 ✅) → training loop + TensorBoard + `best.pt` (M3 ✅, 3 run chốt 20 epoch `20261031-120033/-121323/-122534` trên GPU Colab T4) → test evaluation + `metrics.json` + PNG (M4 ✅).

## 5. Folder Structure

```
lab2/
├── data/                  # dữ liệu CIFAR-10 (gitignore, trừ data/README.md)
│   ├── raw/               # file gốc torchvision tải về + giải nén
│   ├── processed/         # ảnh + nhãn dạng .npy + metadata.json
│   └── splits/            # chỉ số train/val/test (.npz + .json)
│
├── src/
│   ├── data/              # dataset, transforms, split, dataloader  ✅ M1
│   │   ├── __main__.py    # CLI: python -m src.data
│   │   ├── dataset.py     # Cifar10Dataset + TransformSubset
│   │   ├── transforms.py  # train/val transform
│   │   ├── split.py       # stratified_indices (45k/5k, seed=42)
│   │   ├── prepare.py     # xuất data/processed/ + data/splits/
│   │   └── dataloader.py  # DataLoaderConfig + build_dataloaders
│   ├── utils/             # config YAML, run dir, logging, ghi artifact
│   │   ├── config.py
│   │   └── runs.py
│   ├── models/            # factory chọn backbone + thay head + freeze/unfreeze  ✅ M2
│   │   ├── __init__.py    # export public API
│   │   ├── __main__.py    # CLI: python -m src.models --model <name>
│   │   ├── registry.py    # BACKBONE_REGISTRY, block_units, resolve_spec
│   │   └── factory.py     # build_model, replace_head, freeze/unfreeze, count/describe
│   ├── training/          # training loop, scheduler, checkpoint  ✅ M3
│   │   ├── trainer.py     # TrainConfig + Trainer (train/val/checkpoint/TB)
│   │   ├── loss.py        # build_criterion (CE/label-smoothing/focal)
│   │   ├── optimizer.py   # build_optimizer (chi params trainable)
│   │   ├── scheduler.py   # build_scheduler (none/cosine/step/plateau)
│   │   ├── tensorboard_utils.py  # scalar/histogram/grad/image/graph
│   │   └── __main__.py    # CLI train + sweep models×lr×bs
│   └── evaluation/        # metrics, confusion matrix, visualize  ✅ M4
│       ├── metrics.py     # EvalResult + evaluate_model + report
│       ├── visualize.py   # plot CM / per-class F1 / PR / misclassified
│       ├── visualize_html.py  # render report.html offline từ metrics.json + PNG
│       └── __main__.py    # CLI eval: checkpoint → metrics.json + PNG
│   └── demo/              # demo 1 lệnh chạy được  ✅ M5 demo
│       └── __main__.py    # python -m src.demo --checkpoint <best.pt>
│
├── models/
│   ├── checkpoints/       # best.pt theo từng run (gitignore)
│   ├── trained/           # model cuối cùng đã đánh giá (gitignore)
│   └── configs/           # config kiến trúc (yaml)
│
├── notebooks/             # EDA + demo M1/M2 + sweep/retrain Colab: 20261026_eda_cifar10_demo.ipynb, 20261029_train_eval_demo.ipynb, colab_sweep.ipynb (sweep 30 cấu hình 5 epoch), colab_best3_retrain.ipynb (retrain 20 epoch + chấm test)
├── tests/
│   ├── unit/              # M1 (transforms/split/dataset/prepare) + M2 (factory) + M3 (training_components)
│   │   ├── test_transforms.py
│   │   ├── test_split.py
│   │   ├── test_dataset.py
│   │   ├── test_prepare.py
│   │   ├── test_factory.py        # 18 test M2 (gồm test_finetune_all_models bắt bug B4)
│   │   └── test_training_components.py  # 9 test M3
│   └── integration/       # M1 (dataloader) + M2 (model_pipeline) + M3 (training_loop)
│       ├── test_dataloader_pipeline.py
│       ├── test_model_pipeline.py # 6 test M2
│       └── test_training_loop.py  # 4 test M3
│
├── src/evaluation/ test nhóm (giữ nguyên theo yêu cầu):
│   ├── test_metrics.py      # 13 test M4
│   └── test_evaluation.py   # 5 test M4
│
├── docs/                  # architecture.md + data-flow.md + api.md + description.md + data.md + model.md + evaluation.md + audit_code.md + eda.md + plan-m2-m4.md + demo.md + CHANGELOG.md + reports/m2 + m3 + m4 + final_report.md, AUDIT_LOG.md
├── configs/               # data_config.yaml ✅ | model_config.yaml ✅ M2 | train_config.yaml ✅ M3 | eval_config.yaml ✅ M4
├── results/runs/<ts>/     # metrics.json, config.yaml, logs.txt, plots/, tb/, eval/, report.html (offline)
├── results/demo/<ts>/     # predictions.json + index.html (demo 1 lệnh)
│
├── .venv/                 # môi trường ảo Python 3.13 (gitignore)
├── scripts/               # tiện ích: tải & verify dataset, sinh biểu đồ, benchmark
│   ├── download_data.py
│   ├── verify_data.py
│   ├── plot_data_report.py
│   └── benchmark_loading.py
├── requirements.txt
├── README.md
├── AI_Usage.md
└── progress_status.md
```

## 6. Installation / Run / Test

### 6.1 Yêu cầu

- Windows 10/11, PowerShell
- Python 3.13 (đã cài sẵn: `C:\Program Files\Python313\python.exe`)
- ~6 GB disk cho `.venv` + dataset CIFAR-10 (~170 MB)
- **Máy local không có GPU NVIDIA** → cài bản **CPU** của PyTorch để chạy pipeline M1/demo và test. Train nặng (sweep 30 cấu hình + retrain 20 epoch) chạy trên **GPU Colab T4 có AMP**; `configs/train_config.yaml` để sẵn `image_size: 64`, `use_amp: true` để chạy nhanh trên GPU.

### 6.2 Tạo môi trường ảo (đã thực hiện sẵn)

```powershell
# Python 3.13
py -3.13 -m venv .venv

# Kích hoạt
.\.venv\Scripts\Activate.ps1
# hoặc dùng trực tiếp: .\.venv\Scripts\python.exe

# Nâng pip + cài dependency
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` (bản CPU — wheel PyPI chính thức trên Windows không kèm CUDA):

```text
torch==2.14.1
torchvision==0.29.1
tensorboard==2.21.0
numpy==2.5.4
pillow==13.0.0
PyYAML==6.0.3
matplotlib==3.11.2
pytest==9.1.1
```

> Nếu dùng máy có GPU NVIDIA, đổi sang index CUDA tương ứng:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### 6.3 Kiểm tra môi trường

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision, tensorboard; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

Kết quả mong đợi: `2.14.1+cpu 0.29.1+cpu False`.

### 6.4 Chuẩn bị data

```powershell
# Bước 1 — Tải CIFAR-10 về data/raw/ (tự động giải nén) + kiểm tra tính toàn vẹn
.\.venv\Scripts\python.exe scripts\download_data.py
.\.venv\Scripts\python.exe scripts\verify_data.py

# Bước 2 — Xuất data/processed/ (ảnh .npy) và data/splits/ (chỉ số split)
.\.venv\Scripts\python.exe -m src.data.prepare
```

Tải thủ công: [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) → giải nén vào `data/raw/`. Chi tiết ở `data/README.md`.

**Kết quả bước 2:**

```text
data/processed/
├── train_images.npy       146.5 MB   uint8 [50000, 32, 32, 3]
├── train_labels.npy       390.8 KB   int64 [50000]
├── test_images.npy         29.3 MB   uint8 [10000, 32, 32, 3]
├── test_labels.npy         78.2 KB   int64 [10000]
└── metadata.json                     tên lớp + thống kê

data/splits/
├── cifar10_split_seed42.npz           train_idx / val_idx / test_idx
└── cifar10_split_seed42.json          bản tóm tắt đọc được
```

Verify tự động: `train + val = 50.000`, `test = 10.000`, và **không có ảnh nào trùng giữa các split** (so sánh hash nội dung ảnh, không chỉ so chỉ số).

### 6.5 Chạy data pipeline (M1)

```powershell
# Dựng DataLoader + xuất artifact vào results/runs/<timestamp>/
.\.venv\Scripts\python.exe -m src.data --config configs/data_config.yaml

# Sinh biểu đồ kiểm chứng (phân bố lớp + ảnh mẫu)
.\.venv\Scripts\python.exe scripts\plot_data_report.py --run-dir results\runs\<timestamp>
```

Kết quả M1 (run `20261025-150847`):

```text
train |  45000 mẫu |  704 batch | image [64, 3, 224, 224] torch.float32 | label [64]
val   |   5000 mẫu |   79 batch | image [64, 3, 224, 224] torch.float32 | label [64]
test  |  10000 mẫu |  157 batch | image [64, 3, 224, 224] torch.float32 | label [64]
validation: total_train_plus_val=50000, test_samples=10000 → PASS
```

Artifact ghi ra: `results/runs/20261025-150847/` gồm `config.yaml`, `metrics.json`, `logs.txt`, `plots/class_distribution.png`, `plots/sample_images.png`.

### 6.6 Chạy training (M3 ✅) + đánh giá (M4 ✅)

> **Trạng thái:** M3 (training loop + TensorBoard) và M4 (evaluation + CLI) đã hiện thực **và chạy thật xong**. Mặc định train **ảnh 64×64** cho nhanh; muốn ảnh 224×224 chuẩn ImageNet thì thêm `--image-size 224`. Quy trình chốt:
> 1. Sweep 30 cấu hình fine-tune Adam 5 epoch trên GPU Colab T4 (lr 0.0001/0.0005/0.001/0.003/0.005 × batch 32/64) — chi tiết `results/OTM_RESULTS.md` + `results/sweep-FT-lr-bs.log` — chọn ra 3 cấu hình tốt nhất.
> 2. Retrain 3 cấu hình tốt nhất **20 epoch** trên GPU Colab T4 có AMP (`notebooks/colab_best3_retrain.ipynb`) rồi chấm test 10.000 ảnh — kết quả `results/TEST_RESULTS.md`: ResNet18 (lr 0.0001, bs 32) test **86,26%** + VGG11 (lr 0.0001, bs 32) test **87,74%** ⭐ + DenseNet121 (lr 0.0005, bs 64) test **87,15%**.
>
> **Kế hoạch chi tiết** (file nào, hàm nào, test nào, mất bao lâu): xem [`docs/plan-m2-m4.md`](docs/plan-m2-m4.md).

```powershell
# M2 — khám phá model (chạy được)
.\.venv\Scripts\python.exe -m src.models --config configs/model_config.yaml --model resnet18
.\.venv\Scripts\python.exe -m src.models --model vgg11 --print-model

# M3 — train (mặc định ảnh 64×64 cho nhanh; muốn ảnh 224 chuẩn ImageNet thì thêm --image-size 224)
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml

# So sánh nhiều model: ResNet18 / VGG11 / DenseNet121
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --models resnet18 vgg11 densenet121

# Đánh giá checkpoint trên test set
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml

# Báo cáo HTML offline (1 file duy nhất, mở bằng double-click)
.\.venv\Scripts\python.exe -m src.evaluation.visualize_html --run-dir results/runs/20261031-121323/vgg11-lr0.0001-bs32-0
# → results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/report.html (bảng pass/fail, curves, mẫu tốt/xấu, latency)

# Demo 1 lệnh (checkpoint tốt nhất VGG11, 8 ảnh test thật)
.\.venv\Scripts\python.exe -m src.demo --checkpoint models/checkpoints/vgg11-best.pt --num-samples 8
# → results/demo/<ts>/predictions.json + index.html (chi tiết: docs/demo.md)

# TensorBoard (đường dẫn đúng: tb/ nằm trong từng run config — ví dụ run VGG11 20 epoch tốt nhất)
.\.venv\Scripts\tensorboard.exe --logdir results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/tb
# mở http://localhost:6006/ — tags: train/loss, train/accuracy, val/loss, val/accuracy, lr, model/grad_norm, samples/train_images
```

> ⚠️ **Về thời gian chạy:** local CPU đã đo thật (12 logical cores, ResNet18, batch 32): freeze 224×224 ~37 phút/epoch, 64×64 ~5 phút/epoch; VGG11 224 ~144 phút/epoch; DenseNet121 224 ~124 phút/epoch — nên train chốt chuyển lên **GPU Colab T4**: ResNet18/VGG11 chỉ ~27–35 giây/epoch, DenseNet121 ~48–86 giây/epoch (sweep 5 epoch, ảnh 64, AMP). Vì vậy ảnh chốt là **64×64** cho cả sweep lẫn retrain 20 epoch; ảnh 224×224 chỉ dùng cho pipeline M1/demo. Chi tiết: `docs/plan-m2-m4.md` §2.4.

### 6.7 Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Kết quả hiện tại: **102 passed** (47 M1 + 24 M2 + 13 M3 unit/integration + 18 M4).

### 6.8 Module M2 đã xong — Model Factory ✅

- CLI khám phá: `.\.venv\Scripts\python.exe -m src.models --model resnet18|vgg11|densenet121 [--print-model]`
- Kết quả verify với weights ImageNet thật (output `(1, 10)` cả 3 model):

| Model | Total | Trainable (freeze) | Ghi chú |
|-------|-------|--------------------|---------|
| resnet18 | 11,181,642 | 5,130 (0.046%) | verify weights ImageNet thật, output `(1, 10)` |
| vgg11 | 128,807,306 | 40,970 (0.032%) | verify weights ImageNet thật, output `(1, 10)` |
| densenet121 | 6,964,106 | 10,250 (0.147%) | verify weights ImageNet thật, output `(1, 10)` |

- Chi tiết: `docs/reports/m2_models_report.md`

## 7. Example Input → Output

**Input** (train/eval chốt ảnh 64×64)

```python
# Một batch ảnh CIFAR-10 đã transform, shape [B, 3, 64, 64] (chốt GPU T4)
images, labels = next(iter(train_loader))
# images: torch.float32 [64, 3, 64, 64] | labels: torch.int64 [64]
# Demo M1 chuẩn ImageNet: [B, 3, 224, 224] (run 20261025-150847)

**Output** (số thật chốt 20 epoch: VGG11 fine-tune 64×64, `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/`)

> Test 10.000 mẫu: acc 0.8774, macro-F1 0.8765. Per-class F1 thấp nhất: cat 0.7629; cao nhất: ô tô 0.9343 / tàu 0.9347. Nhầm nhiều nhất: mèo → chó 109 tấm, chó → mèo 106 tấm.

```json
{
  "model": "vgg11",
  "strategy": "fine_tune",
  "epochs": 20,
  "best_val_accuracy": 0.8786,
  "best_epoch": 19,
  "test_accuracy": 0.8774,
  "macro_f1": 0.8765,
  "params_total": 128807306,
  "params_trainable": 2400778
}

> Số đo thật từ `eval/metrics.json` (test 10.000 mẫu, ảnh 64×64, Adam lr 0.0001 batch 32).
```

## 8. Results

Kết quả chốt: retrain 3 cấu hình tốt nhất **20 epoch** trên GPU Colab T4 (ảnh 64×64, Adam, fine-tune, AMP) rồi chấm test 10.000 ảnh — nguồn `results/TEST_RESULTS.md` + `results/runs/20261031-*/<cfg>/eval/metrics.json`. Sweep 30 cấu hình 5 epoch để chọn ra 3 cấu hình này: `results/OTM_RESULTS.md` + `results/sweep-FT-lr-bs.log`.

| Model | LR | Batch | Epoch | Val tốt nhất | Test Acc | Macro F1 | Run |
|-------|----|-------|-------|--------------|----------|---------|-----|
| ResNet18 fine-tune | 0.0001 | 32 | 20 | 0.8634 @ ep 17 | 0.8626 | 0.8626 | `20261031-120033` |
| **VGG11 fine-tune ⭐** | **0.0001** | **32** | **20** | **0.8786 @ ep 20** | **0.8774** | **0.8765** | `20261031-121323` |
| DenseNet121 fine-tune | 0.0005 | 64 | 20 | 0.8800 @ ep 12 | 0.8715 | 0.8709 | `20261031-122534` |

Ngon nhất test là VGG11 với 87,74% (F1 mèo thấp nhất 0,76; tàu cao nhất 0,93; nhầm nhiều nhất mèo → chó 109 tấm, chó → mèo 106 tấm).

Tiêu chí đạt (định nghĩa trước khi code):

> ⚠️ **Đề bài không đặt ngưỡng accuracy cụ thể.** Bốn tiêu chí dưới đây là **mục tiêu tự đặt** của project để có thang đo, được ghi lại trước khi code để tránh "chỉnh kết quả cho vừa". Giá trị tham chiếu: transfer learning trên CIFAR-10 với ResNet18 đạt ~85–90% (fine-tune, 224×224).

| Metric | Target | Nguồn |
|--------|--------|-------|
| Test accuracy (freeze backbone, ResNet18) | ≥ 0.70 | tự đặt |
| Test accuracy (fine-tune) | ≥ 0.80 | tự đặt |
| Overfitting (val acc − train acc) | ≤ 0.15 | tự đặt |
| Thời gian 1 epoch (CPU) | ≤ 15 phút | tự đặt |

## 9. Limitations

- **Train chính trên GPU Colab T4 (có AMP), ảnh 64×64**: sweep 30 cấu hình 5 epoch (~27–86 giây/epoch tùy model) rồi retrain 3 cấu hình tốt nhất 20 epoch. Local CPU-only (ResNet18 freeze 224 = 36.6 min/epoch; VGG11 = ~144 min/epoch) chỉ dùng cho pipeline M1/demo, không dùng cho kết quả chốt — vì vậy `configs/train_config.yaml` để `image_size: 64`, `use_amp: true`.
- **Ảnh 32×32 được upsample lên 64×64** (train/eval chốt) hoặc 224×224 (pipeline M1/demo) để khớp input backbone pre-trained — tốn bộ nhớ và làm mất một phần chi tiết; đây là hệ quả của việc dùng backbone ImageNet.

  > **Ghi chú về độ mờ ảnh:** không phải dữ liệu hỏng. Đo trên 256 ảnh đầu: `mean 119.3`, `std 62.9`, chỉ 0.30% pixel đen tuyệt đối và 0.72% trắng tuyệt đối, histogram phân bố đều quanh giữa → dữ liệu lành mạnh. Ảnh vẫn mờ vì CIFAR-10 gốc chỉ 32×32 và resize 224 chỉ nội suy chứ không tạo thêm chi tiết. Chi tiết đo đạc: `data/README.md`.
- **Sweep 5 epoch để chọn cấu hình, retrain chốt 20 epoch** để hội tụ hoàn toàn (kết quả §8). Config mặc định `epochs: 5` chỉ dùng cho sweep nhanh; retrain chốt override `--epochs 20` (notebook `colab_best3_retrain.ipynb`).
- BatchNorm statistics của backbone được đóng băng khi freeze → không tối ưu lại theo phân phối CIFAR.
- Chưa thử `torch.compile` hay distributed training.

## 10. Future Improvements

- Fine-tune toàn bộ backbone với lr nhỏ (1e-4 → 1e-5) khi có GPU.
- Thêm Mixup/CutMix + RandomErasing để giảm overfitting.
- Dùng `torchvision.models` weight = `IMAGENET1K_V1` vs `DEFAULT` để so sánh recipe.
- Thêm label smoothing, cosine annealing scheduler.
- Đo throughput/latency đầy đủ (preprocess + inference + postprocess) cho mục "Full pipeline".
- Thêm kiến trúc hiện đại (ConvNeXt, EfficientNet-V2) để so sánh công bằng.

---

Được duy trì với skill **ai-fusion-workflow**.
Xem `docs/AI_Usage.md` để biết cách AI được sử dụng trong project và `docs/progress_status.md` để xem tiến độ/quyết định.
