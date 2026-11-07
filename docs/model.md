# Model — chuẩn kiến trúc và version (M2)

> Tài liệu kiến trúc model của project, khởi tạo từ `modules/model.md` của skill `ai-fusion-workflow`.
> Mọi con số đều đo thật từ code và checkpoint, không ước lượng.
> Tài liệu liên quan: `docs/architecture.md` (tổng thể), `docs/api.md` (interfaces M2),
> `docs/reports/m2_models_report.md` (báo cáo chi tiết), `docs/CHANGELOG.md` (version).

## 1. Bài toán → họ model

- **Bài toán:** phân loại ảnh (image classification) 10 lớp CIFAR-10.
- **Họ kiến trúc:** CNN backbone pre-trained ImageNet + thay head tuyến tính — đúng yêu cầu Practice 2
  (`torchvision.models`, transfer learning / fine-tuning, Chương 3 slide 20–21).
- **Vì sao chọn họ này:** CIFAR-10 chỉ 50.000 ảnh 32×32 — train từ đầu dễ overfit;
  giữ feature extractor ImageNet và chỉ học head/block cuối là cách chuẩn cho dataset nhỏ.

## 2. Baseline

- **Baseline:** ResNet18 + `fine_tune` (`configs/model_config.yaml` mặc định).
- **Lý do:** nhẹ (11,18M params), train nhanh nhất GPU T4 (~27–35 giây/epoch, ảnh 64, AMP),
  đủ mạnh để đạt 86,26% test — làm mốc so sánh cho VGG11 và DenseNet121.
- **Metric baseline (chốt 20 epoch):** ResNet18 val 0,8634 @ ep 17 → test **0,8626**, macro-F1 0,8626
  (`results/runs/20261031-120033/resnet18-lr0.0001-bs32-0/`).

## 3. Backbone + head

| Backbone | Nguồn weights | Attr head | Dạng head | Head sau khi thay (10 lớp) |
|----------|---------------|-----------|-----------|----------------------------|
| `resnet18` | `torchvision` 0.29 `DEFAULT` (ImageNet) | `fc` | `Linear(512, 1000)` | `Linear(512, 10)` |
| `vgg11` | `torchvision` 0.29 `DEFAULT` (ImageNet) | `classifier` | `Sequential[7]` | giữ 6 lớp đầu, thay lớp cuối `Linear(4096, 10)` |
| `densenet121` | `torchvision` 0.29 `DEFAULT` (ImageNet) | `classifier` | `Linear(1024, 1000)` | `Linear(1024, 10)` |

- **Input:** `[B, 3, 64, 64]` (train/eval chốt, GPU T4 + AMP) hoặc `[B, 3, 224, 224]` (demo M1 chuẩn ImageNet).
- **Output:** logits `[B, 10]` → `argmax` ra nhãn + softmax probability.
- **Chiến lược chốt:** `fine_tune` (`freeze_until: 1`, `freeze_head_full: false`, Adam) —
  toàn bộ sweep 30 cấu hình + retrain 20 epoch đều fine-tune.
- **Block cuối mở khi fine-tune** (`block_units` đã introspect trong `src/models/registry.py`):
  ResNet `layer4`, VGG `features.16 + features.18`, DenseNet `denseblock4 + transition3/norm5`
  (BatchNorm trong block được mở luôn để thích ứng phân bố CIFAR-10).
- **Điểm đặc biệt VGG11:** `classifier` chứa 103M params — unfreeze cả sẽ khiến 92,8% model
  trainable, mất ý nghĩa freeze. Mặc định `freeze_head_full: false` (chỉ `Linear` cuối 40.970 params),
  có test `test_vgg_freeze_head_full_flag` bắt lỗi cấu hình sai.

## 4. Loss function

- **Chốt:** `CrossEntropyLoss` (`build_criterion("cross_entropy")` trong `src/training/loss.py`).
- **Có sẵn để thử:** `label_smoothing` (`CrossEntropyLoss(label_smoothing=0.1)`), `focal` (`FocalLoss(gamma=2.0)`).
- **Vì sao chốt CE:** CIFAR-10 cân bằng tuyệt đối (5.000 ảnh/lớp train) nên không cần class weight;
  sweep + retrain toàn bộ dùng CE + Adam đã đạt 86–88% test, không cần focal.
- **Optimizer chốt:** Adam (chỉ params `requires_grad=True`), lr 0,0001 (ResNet18/VGG11) / 0,0005 (DenseNet121).

## 5. Số tham số / dung lượng

Đo bằng `count_parameters()` sau `build_model()` (weights ImageNet thật):

| Model | Total params | Trainable freeze | Trainable fine-tune chốt 20 epoch | Checkpoint local |
|-------|-------------|------------------|-----------------------------------|------------------|
| ResNet18 | 11.181.642 | 5.130 (0,046%) | 8.398.858 (75,1%) | `results/runs/20261031-120033/.../checkpoints/best.pt` (~45 MB) |
| VGG11 ⭐ | 128.807.306 | 40.970 (0,032%) | 2.400.778 (1,86%) | `results/runs/20261031-121323/.../checkpoints/best.pt` (~515 MB) |
| DenseNet121 | 6.964.106 | 10.250 (0,147%) | 2.696.714 (38,7%) | `results/runs/20261031-122534/.../checkpoints/best.pt` (~28 MB) |

- Dung lượng vượt giới hạn GitHub (100 MB) với VGG11 → `*.pt` LOCAL ONLY (`.gitignore` chặn),
  chỉ push `models/trained/*.json` + `models/configs/*.yaml` + `metrics.json`/PNG/`report.html`.
- Weight tái tạo được: `python -m src.training` với cùng seed 42.
- Inference (test 10.000 ảnh): VGG11 ~0,33 ms/mẫu (GPU T4); demo CPU local ~50 ms/mẫu.

## 6. Config hóa

- Mọi siêu tham số kiến trúc nằm trong `configs/model_config.yaml` (`name`, `num_classes`,
  `weights`, `strategy`, `freeze_until`, `freeze_head_full`) — không hardcode trong code (phục vụ E2).
- Model card từng version: `models/configs/{resnet18,vgg11,densenet121}-cifar10.yaml`
  (`input_size: [3, 64, 64]`, `test_accuracy`, `macro_f1`, `run_ref` 20261031-*, `trainable_params_chot_20epoch`).
- Kết quả cuối: `models/trained/{resnet18,vgg11,densenet121}-final.json` (VGG11 `best_model: true`, test 0,8774).
- Thay đổi kiến trúc ghi vào `docs/CHANGELOG.md` (breaking change 2026-11-01: ảnh 224→64; 2026-10-25: `freeze_head_full`; 2026-10-30: fix B4).

## 7. Naming

- Quy ước skill: `{domain}_{architecture}_{version}.{ext}` — áp dụng: `models/checkpoints/{resnet18,vgg11,densenet121}-best.pt`
  (copy local từ 3 run `20261031-*`), run `results/runs/YYYYMMDD-HHMM/`.
- Dataset: `cifar10_split_seed42.{npz,json}` (`{purpose}_{date}.{ext}`).

## 8. Checklist review (M2)

- [x] Có baseline đơn giản chạy trước (ResNet18 fine-tune 86,26% test) rồi mới so VGG11/DenseNet.
- [x] Pretrained từ `torchvision.models` 0.29 (`weights="DEFAULT"` = ImageNet, giấy phép BSD cho nghiên cứu/học tập).
- [x] Input/output khớp `Interfaces/APIs`: input `[B,3,64,64]`/`[B,3,224,224]` → output `(B,10)` (verify CLI cả 3 model).
- [x] Loss xử lý imbalance: không cần weight (data cân bằng); có sẵn focal/label-smoothing nếu cần.
- [x] Mọi số (`num_classes`, `freeze_until`, `freeze_head_full`, `dropout`) nằm trong config.
- [x] Đã ước lượng params/VRAM/latency trước train dài (bảng §5 + `docs/plan-m2-m4.md` §2.4 đo thật CPU).
- [x] Quyết định chọn/loại ghi Decisions Log (`docs/progress_status.md`: backbone 3 model, `freeze_head_full`, `block_units`, GPU T4 + ảnh 64).

## 9. Rủi ro và cách xử lý trong code

| Rủi ro | Xử lý trong project |
|--------|---------------------|
| Model quá lớn so với data nhỏ → overfit | Bắt đầu từ ResNet18 nhẹ; weight decay 1e-4 + best checkpoint theo val + augmentation crop/flip |
| Quên `model.train()/eval()` + `torch.no_grad()` khi eval | `Trainer.train_one_epoch` set `train()`, `validate()` + `evaluate_model` dùng `@torch.no_grad()`/`inference_mode()` |
| Mismatch kênh/số lớp data vs model | `replace_head` xử lý 2 dạng head + verify 1 batch thật → output `(1,10)` trước khi train; `ModelConfig` validate `num_classes >= 2` |
| Không seed/version → không so sánh được | Seed 42 cố định split/shuffle/train; version qua `models/trained/*.json` + `CHANGELOG.md` + `run_ref` |

## 10. Tái tạo

```powershell
# Khám phá 1 backbone (in cấu trúc + đếm params + kiểm tra output)
.\.venv\Scripts\python.exe -m src.models --model vgg11 --print-model
.\.venv\Scripts\python.exe -m src.models --config configs/model_config.yaml --model resnet18

# Train lại (ví dụ VGG11 cấu hình chốt)
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --models vgg11 --lr 0.0001 --batch-size 32 --epochs 20
```
