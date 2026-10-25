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

**Input:** ảnh RGB 32×32 × 3 → được resize lên **224×224** để khớp input của backbone pre-trained ImageNet (một batch: `[B, 3, 224, 224]`, dtype `float32`, đã normalize theo ImageNet mean/std).
**Output:** vector logits `[B, 10]` → `argmax` ra nhãn lớp; probability qua softmax.

**Chia dữ liệu:** CIFAR-10 chuẩn — 50.000 ảnh train / 10.000 ảnh test. Ta lấy 5.000 ảnh từ tập train làm **validation** (stratified, seed cố định), còn lại 45.000 để train. Test set chỉ dùng để đánh giá cuối cùng.

**Metric:** accuracy trên validation (để chọn best checkpoint) và accuracy trên test (báo cáo cuối). Bổ sung: per-class precision/recall/F1, confusion matrix, thời gian suy luận.

## 2b. Đối chiếu với đề bài (Practice 2)

> Nguồn: `đề.txt`. Bảng này là hợp đồng giữa đề bài và hiện thực — dùng để kiểm tra "đã làm đúng chưa" ở mỗi milestone.
>
> **Kế hoạch triển khai M2–M4** (file nào, hàm nào, test nào, ước lượng thời gian): [`docs/plan-m2-m4.md`](docs/plan-m2-m4.md)

| # | Yêu cầu trong `đề.txt` | Nội dung cụ thể | Hiện thực | Module | Trạng thái |
|---|------------------------|-----------------|-----------|--------|-----------|
| E1 | Experiment with different pre-trained models (VGG, DenseNet, etc.) | So sánh ResNet18 / VGG11 / DenseNet121 | `src/models/factory.py` | M2 | ⏳ |
| E2 | Adjust the hyperparameters (learning rate, batch size, etc.) | Sweep lr / batch size / optimizer / epoch | `configs/*.yaml` + `src/training/` | M3, M5 | ⏳ |
| E3 | Use TensorBoard to monitor the training process | Scalar (loss/acc), histogram weight, graph, image | `src/training/trainer.py` | M3 | ⏳ |
| O1 | Load a Pre-trained Model (`pretrained=True` / `weights=`) | `torchvision.models.*(weights="DEFAULT")` | `src/models/factory.py` | M2 | ⏳ |
| O2 | Explore the Model Architecture — print model, identify layers & params | In cấu trúc + đếm `params_total` / `params_trainable` | `src/models/factory.py` | M2 | ⏳ |
| O3 | Adapt the Model — replace final classification layer (10 classes) | Thay `fc` / `classifier` bằng `Linear(..., 10)` | `src/models/factory.py` | M2 | ⏳ |
| O4 | Freeze earlier layers (transfer learning) | `freeze_backbone` — chỉ train head | `src/models/factory.py` | M2 | ⏳ |
| O5 | Unfreeze a few of the final layers (fine-tuning) | `fine_tune` — unfreeze block cuối + norm + head | `src/models/factory.py` | M2 | ⏳ |
| O6 | Prepare the Data — load & prepare dataset | CIFAR-10 45k/5k/10k, in-memory | `src/data/` | **M1** | ✅ |
| O7 | Use `torchvision.transforms` to preprocess | RandomCrop+Flip → Resize 224 → ToTensor → Normalize | `src/data/transforms.py` | **M1** | ✅ |
| O8 | Use `torch.utils.data.DataLoader` to create batches | 3 loader: 704/79/157 batch, `[B,3,224,224]` | `src/data/dataloader.py` | **M1** | ✅ |
| O9 | Define the loss function and optimizer | `CrossEntropyLoss` + Adam | `src/training/trainer.py` | M3 | ⏳ |
| O10 | Write the training loop | zero_grad → forward → loss → backward → step | `src/training/trainer.py` | M3 | ⏳ |
| O11 | Evaluate the Model on a test dataset | Test acc + macro-F1 + confusion matrix | `src/evaluation/metrics.py` | M4 | ⏳ |

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
                        resize 32→224      shuffle=True
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
                [CrossEntropyLoss] + [Adam/SGD]
                              │
                              ▼
                   [Training loop + Validation]
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
         [TensorBoard logs]     [best.pt checkpoint]
              runs/exp/           models/checkpoints/
                    │
                    ▼
              [Test evaluation]
     accuracy / F1 / confusion matrix → results/runs/<ts>/metrics.json
```

Chi tiết luồng dữ liệu: xem `docs/data-flow.md` (đã viết cho M1).

> Sơ đồ trên mô tả **thiết kế đích** của toàn bộ pipeline. Phần đã hiện thực (M1) là khối `[CIFAR-10] → [Transforms] → [DataLoader]`; các khối còn lại thuộc M2–M4.

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
│   ├── models/            # factory chọn backbone + thay head + freeze/unfreeze  ⏳ M2
│   │   └── factory.py     # (chưa tạo)
│   ├── training/          # training loop, scheduler, checkpoint  ⏳ M3
│   │   └── trainer.py     # (chưa tạo)
│   └── evaluation/        # metrics, confusion matrix, visualize  ⏳ M4
│       └── metrics.py     # (chưa tạo)
│
├── models/
│   ├── checkpoints/       # best.pt theo từng run (gitignore)
│   ├── trained/           # model cuối cùng đã đánh giá (gitignore)
│   └── configs/           # config kiến trúc (yaml)
│
├── notebooks/             # EDA, thử nhanh (đặt tên theo ngày)
├── tests/
│   ├── unit/              # test transforms, split, dataset, prepare
│   │   ├── test_transforms.py
│   │   ├── test_split.py
│   │   ├── test_dataset.py
│   │   └── test_prepare.py
│   └── integration/       # test end-to-end DataLoader
│       └── test_dataloader_pipeline.py
│
├── docs/                  # data-flow.md (M1), plan-m2-m4.md — sẽ thêm architecture.md ở M2
├── configs/               # data_config.yaml ✅ | train_config.yaml, model_config.yaml, eval_config.yaml ⏳
├── results/runs/<ts>/     # metrics.json, config.yaml, logs.txt, plots/
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
- **Không có GPU NVIDIA** trên máy này → cài bản **CPU** của PyTorch. Toàn bộ hyperparameter trong `configs/` đã chọn để chạy được trên CPU.

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

### 6.6 Chạy training (⚠️ sẽ có sau khi build M2–M3)

> **Trạng thái:** các lệnh dưới đây là **thiết kế dự kiến**, chưa chạy được — `src/models/factory.py`, `src/training/trainer.py`, `src/evaluation/metrics.py` hiện mới chỉ là thư mục rỗng (`__init__.py`), và `configs/train_config.yaml` / `model_config.yaml` / `eval_config.yaml` chưa được tạo. Được cập nhật khi M2–M4 hoàn tất.
>
> **Kế hoạch chi tiết** (file nào, hàm nào, test nào, mất bao lâu): xem [`docs/plan-m2-m4.md`](docs/plan-m2-m4.md).

```powershell
# Train 1 model (config trong configs/train_config.yaml)
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml

# So sánh nhiều model: ResNet18 / VGG11 / DenseNet121
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --models resnet18 vgg11 densenet121

# Đánh giá checkpoint trên test set
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml

# TensorBoard
.\.venv\Scripts\tensorboard.exe --logdir=runs
# mở http://localhost:6006/
```

> ⚠️ **Về thời gian chạy trên CPU:** đã đo thật trên máy này (12 logical cores, ResNet18, batch 32):
>
> | Cấu hình | 1 epoch | 5 epoch |
> |----------|---------|---------|
> | ResNet18 freeze, 224×224 | ~37 phút | ~3 giờ |
> | ResNet18 freeze, 64×64 | ~5 phút | ~24 phút |
> | VGG11 freeze, 224×224 | ~144 phút | ~12 giờ |
> | DenseNet121 freeze, 224×224 | ~124 phút | ~10 giờ |
>
> Vì vậy plan chia làm 2 tầng: **Tier A (224×224)** cho kết quả báo cáo chính, **Tier B (64×64)** cho sweep rộng. Chi tiết: `docs/plan-m2-m4.md` §2.4.

### 6.7 Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 7. Example Input → Output

**Input**

```python
# Một batch ảnh CIFAR-10 đã transform, shape [B, 3, 224, 224]
images, labels = next(iter(train_loader))
# images: torch.float32 [64, 3, 224, 224] | labels: torch.int64 [64]
```

**Output**

> ⚠️ **Schema dự kiến** — chưa có số liệu thật cho tới khi hoàn thành M3–M4. Các giá trị số dưới đây là **placeholder minh hoạ định dạng**, không phải kết quả đo được. File `metrics.json` thực tế sẽ được sinh ra sau khi train.

```json
{
  "model": "resnet18",
  "strategy": "freeze_backbone",
  "epochs": 5,
  "best_val_accuracy": 0.0,
  "test_accuracy": 0.0,
  "macro_f1": 0.0,
  "params_total": 11689512,
  "params_trainable": 5130,
  "inference_time_ms_per_sample": 0.0
}
```

## 8. Results

Bảng so sánh được điền sau khi chạy; nguồn chuẩn: `results/runs/<timestamp>/metrics.json`.

| Model | Strategy | LR | Batch | Epoch | Val Acc | Test Acc | Macro F1 | Run |
|-------|----------|----|-------|-------|---------|----------|---------|-----|
| ResNet18 | freeze backbone | 1e-3 | 64 | 5 | TBD | TBD | TBD | — |
| ResNet18 | fine-tune block4 | 1e-4 | 64 | 5 | TBD | TBD | TBD | — |
| VGG11 | freeze backbone | 1e-3 | 64 | 5 | TBD | TBD | TBD | — |
| DenseNet121 | freeze backbone | 1e-3 | 64 | 5 | TBD | TBD | TBD | — |

Tiêu chí đạt (định nghĩa trước khi code):

> ⚠️ **Đề bài không đặt ngưỡng accuracy cụ thể.** Bốn tiêu chí dưới đây là **mục tiêu tự đặt** của project để có thang đo, được ghi lại trước khi code để tránh "chỉnh kết quả cho vừa". Giá trị tham chiếu: transfer learning trên CIFAR-10 với ResNet18 đạt ~85–90% (fine-tune, 224×224).

| Metric | Target | Nguồn |
|--------|--------|-------|
| Test accuracy (freeze backbone, ResNet18) | ≥ 0.70 | tự đặt |
| Test accuracy (fine-tune) | ≥ 0.80 | tự đặt |
| Overfitting (val acc − train acc) | ≤ 0.15 | tự đặt |
| Thời gian 1 epoch (CPU) | ≤ 15 phút | tự đặt |

## 9. Limitations

- **CPU-only**: ước lượng mỗi epoch VGG11/DenseNet121 trên CIFAR-10 mất **10–20 phút** với `num_workers=4` (⚠️ con số này **chưa đo thực tế** — sẽ cập nhật sau khi chạy M3); không thể chạy sweep rộng. Riêng phần data loading đã đo được ~1.9 phút/epoch, phần còn lại là forward/backward của backbone trên CPU.
- **Ảnh 32×32 được upsample lên 224×224** để khớp input của model pre-trained — tốn bộ nhớ và làm mất một phần chi tiết; đây là hệ quả của việc dùng backbone ImageNet.

  > **Ghi chú về độ mờ ảnh:** không phải dữ liệu hỏng. Đo trên 256 ảnh đầu: `mean 119.3`, `std 62.9`, chỉ 0.30% pixel đen tuyệt đối và 0.72% trắng tuyệt đối, histogram phân bố đều quanh giữa → dữ liệu lành mạnh. Ảnh vẫn mờ vì CIFAR-10 gốc chỉ 32×32 và resize 224 chỉ nội suy chứ không tạo thêm chi tiết. Chi tiết đo đạc: `data/README.md`.
- **Chỉ 5 epoch mặc định** để phù hợp giới hạn thời gian; kết quả chưa hội tụ hoàn toàn.
- BatchNorm statistics của backbone được đóng băng khi freeze → không tối ưu lại theo phân phối CIFAR.
- Chưa thử `torch.compile`, AMP, hay distributed training.

## 10. Future Improvements

- Fine-tune toàn bộ backbone với lr nhỏ (1e-4 → 1e-5) khi có GPU.
- Thêm Mixup/CutMix + RandomErasing để giảm overfitting.
- Dùng `torchvision.models` weight = `IMAGENET1K_V1` vs `DEFAULT` để so sánh recipe.
- Thêm label smoothing, cosine annealing scheduler.
- Đo throughput/latency đầy đủ (preprocess + inference + postprocess) cho mục "Full pipeline".
- Thêm kiến trúc hiện đại (ConvNeXt, EfficientNet-V2) để so sánh công bằng.

---

Được duy trì với skill **ai-fusion-workflow**.
Xem `AI_Usage.md` để biết cách AI được sử dụng trong project và `progress_status.md` để xem tiến độ/quyết định.
