# CHANGELOG

> Version model, dataset và các thay đổi phá vỡ tương thích (breaking change).
> Quy ước version model: `{mục đích}_{ngày}.{ext}`; run: `results/runs/YYYYMMDD-HHMM/`.

## Model

### 2026-11-01 — 3 model chốt retrain 20 epoch (GPU Colab T4, ảnh 64, Adam, fine-tune)

| Model | Train | Best val | Test acc | Macro-F1 | Checkpoint (local) |
|-------|-------|----------|----------|----------|--------------------|
| ResNet18 | lr 0.0001, bs 32 | 0.8634 @ ep 17 | 0.8626 | 0.8626 | `results/runs/20261031-120033/.../checkpoints/best.pt` |
| VGG11 ⭐ | lr 0.0001, bs 32 | 0.8786 @ ep 20 | **0.8774** | 0.8765 | `results/runs/20261031-121323/.../checkpoints/best.pt` |
| DenseNet121 | lr 0.0005, bs 64 | 0.8800 @ ep 12 | 0.8715 | 0.8709 | `results/runs/20261031-122534/.../checkpoints/best.pt` |

- Model card JSON (được push): `models/trained/{resnet18,vgg11,densenet121}-final.json`.
- Weight `*.pt` LOCAL ONLY (VGG11 ~515 MB > giới hạn GitHub 100 MB) — tái tạo bằng `python -m src.training` với cùng seed 42.
- Kiến trúc head: `models/configs/*-cifar10.yaml` (`input_size: [3,64,64]`).

### 2026-10-26 — M2 factory verify (weights ImageNet thật)

- ResNet18 11.181.642 params (freeze 5.130) / VGG11 128.807.306 (freeze 40.970) / DenseNet121 6.964.106 (freeze 10.250). Run CLI `20261026-*` đã dọn, số liệu giữ nguyên.

## Dataset

### 2026-10-25 — CIFAR-10 `seed42` (hiện hành)

- 45.000 train / 5.000 val / 10.000 test; stratified; overlap 3 cặp = 0 (hash ảnh).
- `data/processed/*.npy` (ảnh gốc 32×32) + `data/splits/cifar10_split_seed42.{npz,json}`.
- MD5 archive `c58f30108f718f92721af3b95e74349a` — PASS.

## Breaking changes

| Ngày | Thay đổi | Ảnh hưởng |
|------|----------|-----------|
| 2026-11-01 | Train chốt chuyển **ảnh 224 → 64 + AMP trên GPU Colab T4** | `train_config.yaml`: `image_size: 64`, `use_amp: true`; ảnh 224 chỉ còn demo M1 (`data_config.yaml` giữ nguyên) |
| 2026-10-25 | VGG11 `freeze_backbone` mặc định **chỉ Linear cuối** (`freeze_head_full: false`) | Unfreeze cả `classifier` (103M params, 92.8%) bị coi là cấu hình sai — có test bắt lỗi |
| 2026-10-30 | `get_module` ưu tiên `getattr` trước khi ép chỉ số (fix B4) | DenseNet `features.denseblock4` hết crash; thêm test `test_finetune_all_models` |
