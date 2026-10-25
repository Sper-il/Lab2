# Kế hoạch chi tiết — M2 / M3 / M4

> Nguồn: `đề.txt` (Practice 2), `progress_status.md` (Project Context, Decisions Log), `docs/data-flow.md` (M1).
> Lập ngày 2026-10-25. Mọi số liệu thời gian trong tài liệu này **đã đo thật** trên máy này (CPU, 12 logical cores).

---

## 0. Tóm tắt

| Module | Mở khoá yêu cầu đề | File chính | Ước lượng |
|--------|-------------------|-------------|-----------|
| **M2 — Model factory** | E1, O1, O2, O3, O4, O5 | `src/models/factory.py` + `configs/model_config.yaml` | 1–1.5 giờ |
| **M3 — Training + TensorBoard** | E2, E3, O9, O10 | `src/training/trainer.py` + `configs/train_config.yaml` | 3–4 giờ (+ thời gian train) |
| **M4 — Evaluation & metrics** | O11 | `src/evaluation/metrics.py` | 2–3 giờ |

**M2 là điểm mở khoá lớn nhất**: 6/14 yêu cầu đề nằm gọn trong một module, và M3/M4 đều phụ thuộc vào nó.

### Rủi ro lớn nhất: thời gian train trên CPU

Đã đo trên máy này (ResNet18, `batch=32`, `torch.set_num_threads(8)`):

| Chiến lược | Input 224×224 | Input 128×128 | Input 64×64 |
|------------|---------------|---------------|-------------|
| **freeze** (chỉ train head) | **36.6 min/epoch** | 18.2 min | 4.8 min |
| **fine-tune** (train toàn bộ) | **~107 min/epoch** | ~54 min | ~14 min |

Với 45.000 ảnh train, 5 epoch:

| Cấu hình | Một epoch | 5 epoch | Ghi chú |
|----------|-----------|---------|---------|
| ResNet18 freeze, 224 | 36.6 min | **~3 giờ** | Khả thi |
| ResNet18 freeze, 64 | 4.8 min | **~24 phút** | Khả thi — dùng cho sweep |
| VGG11 freeze, 224 | ~144 min | **~12 giờ** | ⚠️ quá dài |
| DenseNet121 freeze, 224 | ~124 min | **~10 giờ** | ⚠️ quá dài |

**Quyết định thiết kế (xem §2.4):** dùng **2 tầng cấu hình** —
- **Tier A (báo cáo chính)**: ảnh 224×224, đúng chuẩn ImageNet → kết quả "đẹp" nhất
- **Tier B (sweep rộng)**: ảnh 64×64 → chạy được cả 3 model × nhiều tổ hợp siêu tham số

---

## 1. M2 — Model factory

### 1.1 Yêu cầu đề bài được phủ

| # | Yêu cầu trong `đề.txt` | Câu chữ gốc |
|---|------------------------|-------------|
| E1 | Experiment with different pre-trained models | "Experiment with different pre-trained models (VGG, DenseNet, etc.)" |
| O1 | Load a Pre-trained Model | "`pretrained=True` loads weights pre-trained on the ImageNet dataset" |
| O2 | Explore the Model Architecture | "Print the model to examine its structure. Identify the layers and parameters" |
| O3 | Adapt the Model for a Specific Task | "the final layer of a pre-trained model needs to be replaced to match the number of classes" |
| O4 | Freeze earlier layers (transfer learning) | "You can freeze earlier layers to prevent them from updating weights" |
| O5 | Unfreeze final layers (fine-tuning) | "Or you can unfreeze a few of the final layers to fine-tune the model" |

### 1.2 Kiến trúc đã kiểm chứng

Đã introspect trực tiếp từ `torchvision.models` (không đoán):

| Model | Attr chứa head | Cấu trúc head | Tổng params | Block cuối để unfreeze |
|-------|-----------------|---------------|-------------|----------------------|
| `resnet18` | `fc` | `Linear(512, 1000)` | 11.69M | `layer4` (2× BasicBlock, 8.39M) |
| `vgg11` | `classifier` | `Sequential` 7 phần tử: `Linear(25088,4096)` → ReLU → Dropout → `Linear(4096,4096)` → ReLU → Dropout → `Linear(4096,1000)` | 132.86M | `features` block 8 (Conv 512→512) + `classifier` |
| `densenet121` | `classifier` | `Linear(1024, 1000)` | 7.98M | `features.denseblock4` + `features.transition3` |

> **Khác biệt quan trọng:** VGG11 có head là `Sequential` **nhiều tầng**, không phải một `Linear` đơn. Code thay head **không được giả định** mọi model đều có `fc`.

### 1.3 Files cần tạo

#### `src/models/factory.py` — file chính (~350 dòng)

| Hàm / class | Chức năng | Input | Output |
|-------------|-----------|-------|--------|
| `ModelConfig` (dataclass) | Cấu hình: `name`, `num_classes`, `strategy`, `weights`, `freeze_until`, `dropout` | — | — |
| `BACKBONES` (dict) | Registry: `resnet18` / `vgg11` / `densenet121` → hàm dựng model | — | `nn.Module` |
| `build_model(config) -> nn.Module` | Hàm chính: dựng backbone pre-trained → thay head → áp chiến lược freeze | `ModelConfig` | `nn.Module` |
| `replace_head(model, num_classes) -> nn.Module` | Thay lớp phân loại cuối bằng `Linear(in_features, num_classes)` | model, số lớp | model đã thay |
| `freeze_backbone(model) -> nn.Module` | Set `requires_grad=False` cho toàn bộ trừ head | model | model |
| `unfreeze_last(model, n_blocks) -> nn.Module` | Mở lại gradient cho block cuối (+ norm + head) | model, số block | model |
| `count_parameters(model) -> dict` | Đếm `params_total`, `params_trainable`, `params_trainable_pct` | model | dict |
| `describe_model(model) -> str` | In cấu trúc (cho O2: "Print the model") | model | chuỗi text |

**Chi tiết `replace_head` — phải xử lý 3 dạng head khác nhau:**

```text
resnet18      model.fc          = Linear(512, 1000)     → Linear(512, 10)
vgg11         model.classifier  = Sequential[...7]       → Sequential[...5, Linear(4096, 10)]
densenet121   model.classifier  = Linear(1024, 1000)    → Linear(1024, 10)
```

Với VGG: giữ lại `Linear(25088,4096)`, ReLU, Dropout, `Linear(4096,4096)`, ReLU, Dropout → thay `Linear(4096,1000)` bằng `Linear(4096,10)`.

**Chi tiết `unfreeze_last` — khác nhau theo model:**

| Model | "Block cuối" nghĩa là | Tham số mở lại |
|-------|----------------------|----------------|
| resnet18 | `layer4` + `fc` | ~8.39M + 5.130 |
| vgg11 | `features[-3:]` (3 conv cuối) + toàn bộ `classifier` | ~2.36M + 26.9M |
| densenet121 | `features.denseblock4` + `features.transition3` + `classifier` | ~3.10M + 1.025M |

> **Quyết định thiết kế:** `unfreeze_last` dùng **số block** (`freeze_until: 1`) thay vì tên layer cụ thể → code giống nhau cho cả 3 model, chỉ khác cách xác định "block cuối" (dùng `named_children()` đếm từ cuối).

#### `src/models/registry.py` — tách riêng (~80 dòng)

| Nội dung | Chức năng |
|----------|-----------|
| `BACKBONE_REGISTRY` | Map tên → hàm dựng (`resnet18` → `torchvision.models.resnet18`) |
| `HEAD_SPECS` | Map tên → (attr chứa head, kiểu head, số block cuối) |
| `SUPPORTED_MODELS` | Tuple danh sách model được hỗ trợ |
| `resolve_backbone(name)` | Tra cứu + báo lỗi rõ ràng nếu tên sai |

> **Tách riêng vì:** `registry.py` là **bảng tra cứu dữ liệu** (thay đổi khi thêm model), `factory.py` là **logic** (ổn định hơn). Giúp thêm `mobilenet_v3_small` sau này chỉ cần sửa 1 file.

#### `src/models/__init__.py` — export public API

```python
from src.models.factory import build_model, count_parameters, describe_model
from src.models.registry import SUPPORTED_MODELS
```

#### `src/models/__main__.py` — CLI khám phá (~120 dòng)

```powershell
.\.venv\Scripts\python.exe -m src.models --config configs/model_config.yaml
```

Chức năng:
- Dựng model theo config → in cấu trúc (phục vụ **O2**)
- In `params_total` / `params_trainable` / tỉ lệ %
- Kiểm tra output shape: input `(1,3,224,224)` → output `(1,10)`
- Xuất artifact `results/runs/<ts>/model_info.json`

#### `configs/model_config.yaml`

```yaml
model:
  name: resnet18
  num_classes: 10
  weights: DEFAULT          # hoặc IMAGENET1K_V1
  strategy: freeze_backbone # freeze_backbone | fine_tune
  freeze_until: 1           # số block cuối mở khi fine_tune
  dropout: 0.0
```

### 1.4 Tests cho M2

#### `tests/unit/test_factory.py` (~14 test)

| Test | Kiểm tra |
|------|---------|
| `test_build_model_returns_cnn` | Output là `nn.Module` |
| `test_output_shape_is_num_classes` | `(1,3,224,224)` → `(1,10)` |
| `test_resnet_head_replaced` | `model.fc.out_features == 10` |
| `test_vgg_head_replaced_keeps_layers` | `classifier` còn 7 phần tử, phần tử cuối là `Linear(4096,10)` |
| `test_densenet_head_replaced` | `model.classifier.out_features == 10` |
| `test_freeze_only_head_trainable` | `params_trainable` < 1% tổng |
| `test_freeze_backbone_no_grad` | `p.requires_grad == False` cho backbone |
| `test_finetune_unfreezes_more_params` | `params_trainable(fine_tune) > params_trainable(freeze)` |
| `test_finetune_block4_has_grad` | `layer4` params `requires_grad == True` |
| `test_count_parameters_consistent` | `total == trainable + frozen` |
| `test_unknown_model_raises` | Tên sai → `ValueError` rõ ràng |
| `test_invalid_strategy_raises` | Strategy sai → `ValueError` |
| `test_weights_none_works` | `weights=None` vẫn dựng được (cho test nhanh) |
| `test_num_classes_configurable` | `num_classes=5` → head 5 |

#### `tests/integration/test_model_pipeline.py` (~6 test)

| Test | Kiểm tra |
|------|---------|
| `test_end_to_end_batch_to_logits` | Batch thật từ DataLoader → logits `(64,10)` |
| `test_all_models_output_shape` | Cả 3 model cho shape đúng |
| `test_frozen_backbone_gradients_none` | Sau backward, `.grad` của backbone vẫn `None` |
| `test_head_gradients_not_none` | `.grad` của head khác `None` |
| `test_checkpoint_save_load` | Save → load → cùng output |
| `test_describe_model_contains_key_info` | Chuỗi mô tả chứa tên lớp + số params |

### 1.5 Tiêu chí hoàn thành M2

- [ ] `build_model` chạy được cho cả 3 model
- [ ] Head thay đúng, output shape `(B, 10)`
- [ ] Freeze → chỉ head trainable, đếm params khớp
- [ ] Fine-tune → mở đúng block cuối, params nhiều hơn freeze
- [ ] CLI in cấu trúc model (O2 thỏa mãn)
- [ ] ≥ 20 test pass
- [ ] Docs cập nhật, commit

---

## 2. M3 — Training loop + TensorBoard

### 2.1 Yêu cầu đề bài được phủ

| # | Yêu cầu trong `đề.txt` | Câu chữ gốc |
|---|------------------------|-------------|
| E2 | Adjust the hyperparameters | "Adjust the hyperparameters (learning rate, batch size, etc.)" |
| E3 | Use TensorBoard | "Use TensorBoard to monitor the training process" |
| O9 | Define the loss function and optimizer | "Define the loss function and optimizer" |
| O10 | Write the training loop | "Write the training loop to train the model on your dataset" |

### 2.2 Files cần tạo

#### `src/training/trainer.py` — file chính (~320 dòng)

| Hàm / class | Chức năng |
|-------------|-----------|
| `TrainConfig` (dataclass) | `model`, `epochs`, `batch_size`, `lr`, `optimizer`, `weight_decay`, `momentum`, `scheduler`, `num_workers`, `seed`, `device`, `strategy`, `freeze_until` |
| `Trainer` (class) | Chứa toàn bộ vòng lặp train/val |
| `Trainer.train()` | Vòng lặp chính qua các epoch |
| `Trainer.train_one_epoch()` | 1 epoch: `zero_grad → forward → loss → backward → step` (O10) |
| `Trainer.validate()` | Đánh giá trên val, `torch.no_grad()` |
| `Trainer.save_checkpoint()` | Lưu `best.pt` khi val acc tăng |
| `Trainer.log_scalars()` | Ghi TensorBoard |

**Cấu trúc 1 epoch (bám sát slide 8–9 Chương 3):**

```text
for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()              # 1
    outputs = model(images)           # 2  forward
    loss = criterion(outputs, labels) # 3  loss
    loss.backward()                   # 4  backward
    optimizer.step()                  # 5  step
```

#### `src/training/loss.py` (~40 dòng)

| Hàm | Chức năng |
|-----|-----------|
| `build_criterion(name)` | `cross_entropy` (mặc định) \| `label_smoothing` \| `focal` |
| `build_optimizer(model, name, lr, weight_decay, momentum)` | `adam` \| `adamw` \| `sgd` — chỉ truyền params có `requires_grad=True` |

> **Quan trọng:** optimizer **phải** chỉ nhận `params_trainable`. Nếu truyền toàn bộ params, Adam sẽ cập nhật trọng số đã freeze (dù `grad=None` → không sai số, nhưng vẫn lãng phí state và gây hiểu nhầm).

#### `src/training/scheduler.py` (~60 dòng)

| Hàm | Chức năng |
|-----|-----------|
| `build_scheduler(optimizer, name, epochs)` | `none` \| `cosine` \| `step` \| `plateau` |
| `CosineAnnealingLR` | lr giảm dần theo cosine |

#### `src/training/tensorboard_utils.py` (~120 dòng)

| Hàm | Chức năng |
|-----|-----------|
| `SummaryWriter` wrapper | Tạo writer trong `results/runs/<ts>/tb/` |
| `log_scalars()` | `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `lr` |
| `log_weight_histograms()` | `model.layer4.weight` histogram (phát hiện exploding/vanishing) |
| `log_grad_norms()` | Độ lớn gradient từng epoch |
| `log_images()` | Lưu ảnh mẫu đã denormalize (giống `plot_data_report.py`) |
| `log_graph()` | `torch.utils.tensorboard.add_graph` — yêu cầu O2 "print model" mở rộng |

#### `src/training/__main__.py` — CLI train (~150 dòng)

```powershell
# Train 1 model
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml

# So sánh nhiều model (E1)
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --models resnet18 vgg11 densenet121

# Sweep siêu tham số (E2)
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --lr 1e-3 1e-4 --batch-size 32 64
```

Chức năng:
- Parse config + override từ CLI (E2: siêu tham số chỉnh được bằng lệnh)
- Dựng model (gọi `src.models.build_model`)
- Train với TensorBoard
- Xuất `metrics.json` + `config.yaml` + `logs.txt` vào `results/runs/<ts>/`
- Lưu `best.pt` vào `results/runs/<ts>/checkpoints/`

#### `configs/train_config.yaml`

```yaml
train:
  model: resnet18
  strategy: freeze_backbone
  freeze_until: 1
  epochs: 5
  batch_size: 32
  lr: 1.0e-3
  optimizer: adam
  weight_decay: 1.0e-4
  momentum: 0.9
  scheduler: none
  num_workers: 4
  seed: 42
  device: auto
  image_size: 224
  early_stopping_patience: 0     # 0 = tắt

tensorboard:
  enabled: true
  log_dir: results/runs
  log_images: true
  log_histograms: true
  log_graph: false               # chậm, bật khi cần
```

### 2.3 Biểu đồ TensorBoard (cho E3)

| Tag | Loại | Ý nghĩa |
|-----|------|---------|
| `train/loss`, `val/loss` | scalar | Đường loss (phát hiện overfit khi val tăng) |
| `train/accuracy`, `val/accuracy` | scalar | So sánh 2 đường → thấy gap overfit |
| `lr` | scalar | Lịch lr giảm |
| `epoch/time_seconds` | scalar | Thời gian mỗi epoch |
| `model/param_histograms` | histogram | Phân bố trọng số |
| `model/grad_norms` | histogram | Độ lớn gradient |
| `samples/train_images` | image | Ảnh mẫu sau transform |

### 2.4 Chiến lược chạy (quan trọng vì CPU)

**2 tầng cấu hình** để cân bằng giữa "kết quả đúng chuẩn" và "thời gian CPU":

| Tier | `image_size` | Dùng khi | Thời gian ResNet18 freeze |
|------|--------------|----------|---------------------------|
| **A — Báo cáo chính** | 224 | Kết quả cuối cùng trong báo cáo | ~3 giờ / 5 epoch |
| **B — Sweep rộng** | 64 | So sánh model / siêu tham số | ~24 phút / 5 epoch |

Lưu ý khi xuống 64×64: điều chỉnh `Resize` trong transform. **Không** đổi `image_size` trong `data_config.yaml` (giữ nguyên cho M1) — thêm override ở `train_config.yaml`.

> ⚠️ **Khi dùng 64×64, kết quả accuracy sẽ thấp hơn 224×224.** Báo cáo phải nêu rõ điều này, không trộn hai bảng.

**Ước lượng tổng thời gian train (Tier A, 224×224, freeze, 5 epoch):**

| Model | 1 epoch | 5 epoch | 2 chiến lược (freeze + FT) |
|-------|---------|---------|---------------------------|
| ResNet18 | ~37 min | ~3 giờ | ~8 giờ (FT chậm hơn ~3×) |
| VGG11 | ~144 min | ~12 giờ | ⚠️ ~30 giờ — không khả thi |
| DenseNet121 | ~124 min | ~10 giờ | ⚠️ ~25 giờ — không khả thi |

**Khuyến nghị:** Tier A chỉ chạy **ResNet18** (freeze + fine-tune) để có kết quả "đẹp". VGG11 và DenseNet121 chạy ở **Tier B (64×64)** để so sánh kiến trúc — vẫn đáp ứng E1 "Experiment with different pre-trained models".

### 2.5 Tests cho M3

#### `tests/unit/test_scheduler.py` (~6 test)
- Cosine LR giảm đúng công thức
- StepLR giảm đúng mốc
- Scheduler không giảm khi `name=none`
- LR cuối > 0
- `build_scheduler` với tên sai → `ValueError`
- Scheduler step đúng số lần gọi

#### `tests/unit/test_loss_optimizer.py` (~6 test)
- `CrossEntropyLoss` nhận logits `(B,10)`
- Label smoothing hoạt động
- Optimizer chỉ nhận params trainable
- SGD momentum đúng
- LR đúng giá trị truyền vào
- `build_optimizer` tên sai → `ValueError`

#### `tests/unit/test_tensorboard_utils.py` (~5 test)
- `SummaryWriter` tạo file event trong thư mục đích
- `log_scalars` không raise
- `log_weight_histograms` không raise
- `log_images` không raise
- Tag đúng tên quy ước

#### `tests/integration/test_training_loop.py` (~7 test)

| Test | Kiểm tra |
|------|---------|
| `test_one_epoch_runs_and_updates_weights` | Loss giảm hoặc params đổi |
| `test_gradients_flow_to_head` | Head có `.grad` khác `None` |
| `test_frozen_backbone_no_gradients` | Backbone `.grad` là `None` |
| `test_checkpoint_saved_on_best_val` | File `best.pt` tồn tại |
| `test_checkpoint_reload_same_output` | Load lại cho output giống |
| `test_metrics_json_written` | `metrics.json` có `best_val_accuracy` |
| `test_early_stopping_triggers` | Patience = 1 → dừng sớm |

> **Test nhanh:** dùng `FakeData` + model nhỏ, `epochs=1`, `batch_size=4` → mỗi test < 5 giây.

### 2.6 Tiêu chí hoàn thành M3

- [ ] Training loop chạy end-to-end, loss giảm
- [ ] TensorBoard ghi đủ scalar (train/val loss, accuracy, lr)
- [ ] `best.pt` lưu theo val accuracy
- [ ] Siêu tham số override được từ CLI (E2)
- [ ] Nhiều model chạy được trong 1 lệnh (E1)
- [ ] ≥ 24 test pass
- [ ] Docs cập nhật, commit

---

## 3. M4 — Evaluation & metrics

### 3.1 Yêu cầu đề bài được phủ

| # | Yêu cầu trong `đề.txt` | Câu chữ gốc |
|---|------------------------|-------------|
| O11 | Evaluate the Model | "Evaluate the model's performance on a test dataset" |

### 3.2 Files cần tạo

#### `src/evaluation/metrics.py` — file chính (~220 dòng)

| Hàm | Chức năng | Input → Output |
|-----|-----------|---------------|
| `accuracy(pred, target)` | Độ chính xác | `(B,)` × `(B,)` → float |
| `per_class_metrics(...)` | precision/recall/F1 mỗi lớp | → dict `{class: {p, r, f1, support}}` |
| `macro_f1(...)` | F1 trung bình macro (metric chính) | → float |
| `confusion_matrix(pred, target, n_classes)` | Ma trận nhầm lẫn | → `(10,10)` ndarray |
| `classification_report(...)` | Bảng report đầy đủ (dùng `sklearn` nếu có, tự tính nếu không) | → str |
| `evaluate_model(model, loader, device)` | Chạy inference trên test, gom dự đoán | → dict metrics |

> **Không thêm `scikit-learn`**: PyTorch có sẵn `torchmetrics`? Không — và thêm dependency chỉ để tính F1 là không đáng. Tự tính bằng `torch` thuần (~30 dòng) giữ `requirements.txt` gọn.

#### `src/evaluation/visualize.py` (~150 dòng)

| Hàm | Chức năng | Output |
|-----|-----------|--------|
| `plot_confusion_matrix(cm, class_names)` | Vẽ ma trận nhầm lẫn heatmap | `plots/confusion_matrix.png` |
| `plot_per_class_f1(metrics)` | Bar chart F1 mỗi lớp | `plots/per_class_f1.png` |
| `plot_pr_curve(...)` | Đường PR cho từng lớp | `plots/pr_curve.png` |
| `plot_confusion_examples(...)` | Ảnh mẫu model đoán sai | `plots/misclassified.png` |
| `plot_train_curves(history)` | Vẽ lại đường loss/acc từ TensorBoard log | `plots/train_curves.png` |

#### `src/evaluation/__main__.py` — CLI đánh giá (~130 dòng)

```powershell
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml --checkpoint results/runs/<ts>/checkpoints/best.pt
```

Chức năng:
- Load checkpoint + config
- Chạy trên test set → `metrics.json`
- Sinh toàn bộ biểu đồ vào `plots/`
- In bảng so sánh với baseline

#### `configs/eval_config.yaml`

```yaml
evaluation:
  checkpoint: results/runs/20261025-150847/checkpoints/best.pt
  model: resnet18
  strategy: freeze_backbone
  batch_size: 64
  num_workers: 4
  device: auto
  save_plots: true
  top_k_errors: 12
```

### 3.3 Nội dung `metrics.json` (chuẩn hoá cho M5)

```json
{
  "model": "resnet18",
  "strategy": "freeze_backbone",
  "checkpoint": "results/runs/<ts>/checkpoints/best.pt",
  "test": {
    "accuracy": 0.0,
    "macro_f1": 0.0,
    "per_class": {
      "airplane": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1000}
    },
    "confusion_matrix": [[0]],
    "inference_time_ms_per_sample": 0.0
  },
  "train_summary": {
    "epochs": 5,
    "best_val_accuracy": 0.0,
    "best_epoch": 0
  }
}
```

> **Cùng schema cho mọi run** → M5 chỉ cần đọc `metrics.json` ra bảng so sánh.

### 3.4 Tests cho M4

#### `tests/unit/test_metrics.py` (~12 test)

| Test | Kiểm tra |
|------|---------|
| `test_accuracy_perfect` | Dự đoán đúng hết → 1.0 |
| `test_accuracy_wrong` | Dự đoán sai hết → 0.0 |
| `test_accuracy_half` | Đúng một nửa → 0.5 |
| `test_macro_f1_perfect` | → 1.0 |
| `test_macro_f1_hand_computed` | So với giá trị tính tay |
| `test_confusion_matrix_diagonal` | Dự đoán đúng → đường chéo = số lượng |
| `test_confusion_matrix_off_diagonal` | Sai lẫn lộn → đúng ô sai |
| `test_confusion_matrix_shape` | → `(10,10)` |
| `test_confusion_matrix_rows_sum` | Tổng mỗi hàng = số mẫu của lớp đó |
| `test_per_class_support_sums` | Tổng support = số mẫu |
| `test_metrics_empty_input_raises` | Input rỗng → `ValueError` |
| `test_label_out_of_range_raises` | Nhãn ≥ n_classes → lỗi rõ ràng |

#### `tests/integration/test_evaluation.py` (~5 test)

| Test | Kiểm tra |
|------|---------|
| `test_evaluate_model_end_to_end` | Model giả + loader giả → metrics dict đầy đủ |
| `test_metrics_json_schema` | Đủ key theo §3.3 |
| `test_plots_generated` | Tạo đủ file PNG |
| `test_checkpoint_reload_consistency` | Đánh giá lại checkpoint → cùng accuracy |
| `test_confusion_matrix_sum_equals_samples` | Tổng ma trận = số mẫu test |

### 3.5 Tiêu chí hoàn thành M4

- [ ] Test accuracy + macro-F1 + per-class metrics đúng (đối chiếu giá trị tính tay)
- [ ] Ma trận nhầm lẫn tổng = số mẫu test
- [ ] Biểu đồ sinh đủ và đọc được
- [ ] `metrics.json` theo schema thống nhất
- [ ] ≥ 17 test pass
- [ ] Docs cập nhật, commit

---

## 4. Thứ tự thực hiện đề xuất

```text
M2 (Model factory)          ~1.5 giờ code
  └─> verify: python -m src.models --config configs/model_config.yaml
      └─> commit: feat(models): implement M2 model factory

M3 (Training + TensorBoard) ~4 giờ code
  └─> verify: python -m src.training --config configs/train_config.yaml
      └─> commit: feat(training): implement M3 training loop with TensorBoard

M4 (Evaluation)             ~3 giờ code
  └─> verify: python -m src.evaluation --config configs/eval_config.yaml
      └─> commit: feat(evaluation): implement M4 evaluation and metrics

M5 (Sweep + báo cáo)        chạy train + tổng hợp bảng so sánh
  └─> commit: feat(experiments): add M5 experiment sweep and report
```

**Mỗi module theo vòng lặp chuẩn của skill:** Describe → Generate → Review → Run → Test → Refactor → Commit.

**Ưu tiên nếu thiếu thời gian:** M2 → M3 → M4 → M5. Bỏ M5 (sweep rộng) vẫn nộp được bài vì E1 chỉ cần so sánh 2–3 model, không cần sweep tổ hợp.

---

## 5. Rủi ro & kế hoạch ứng phó

| Rủi ro | Khả năng | Tác động | Giảm thiểu |
|--------|----------|----------|------------|
| Train trên CPU quá lâu (> 12 giờ cho VGG11) | **Cao** | Trượt deadline | Tier A chỉ chạy ResNet18; VGG/DenseNet dùng Tier B (64×64). Ghi rõ điều kiện trong báo cáo |
| Accuracy thấp hơn kỳ vọng | Trung bình | Không đạt tiêu chí tự đặt | Đo trước 1 epoch; nếu thấp thì so sánh với baseline đã biết (transfer learning CIFAR-10 ~85–90%) |
| Fine-tune overfit (val giảm) | Trung bình | Kết quả kém | Early stopping theo val accuracy, lưu best checkpoint, weight decay |
| `add_graph` rất chậm / lỗi với Sequential | Trung bình | Tắt mục 3 | `log_graph: false` mặc định; bật thủ công khi cần |
| RAM đầy khi train song song nhiều model | Thấp | Crash | Chạy tuần tự, không parallel |

---

## 6. Ảnh hưởng tới docs hiện tại

Khi M2–M4 hoàn thành, cần cập nhật:

| File | Nội dung cần sửa |
|------|-----------------|
| `README.md` | §2b (trạng thái E1–E3, O1–O11 → ✅), §4 (sơ đồ kiến trúc đầy đủ), §5 (cấu trúc thư mục), §6.6 (lệnh chạy thật), §8 (bảng kết quả thật), §9 (limitations cập nhật) |
| `progress_status.md` | Module Log (M2/M3/M4 ✅ + commit), Decisions Log (quyết định mới), Project Context (tiêu chí đánh giá), DoD |
| `AI_Usage.md` | Bảng artifact, Student Modifications, Validation Method |
| `docs/data-flow.md` | §8 "Ranh giới module (cho M2–M5)" → cập nhật sang "đã cung cấp cho M2–M4" |
| `docs/architecture.md` | **Tạo mới** — sơ đồ kiến trúc đầy đủ (README §4 đang trỏ tới file này) |

---

## 7. Nguyên tắc giữ nguyên khi làm M2–M4

1. **Không đổi `src/data/`** — M1 đã ổn định, 47 test pass. M2–M4 chỉ **đọc** từ đó.
2. **Mọi config vào YAML** — không hardcode hyperparameter trong code (E2 yêu cầu "adjust hyperparameters" → phải chỉnh được từ file/lệnh).
3. **Mọi run ghi `metrics.json` + `config.yaml`** cùng schema → M5 tổng hợp được.
4. **Mọi thay đổi docs phải khớp code** — đặc biệt shape và số liệu (đã có bài học từ đợt rà soát trước).
5. **Test không cần data thật** — dùng `FakeData` để test nhanh, chỉ chạy full data ở bước verify.
6. **Cập nhật ma trận truy vết đề bài** sau mỗi module (README §2b + progress_status.md).
