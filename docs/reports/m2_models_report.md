# M2 — Model & Architecture Report

**Ngày:** 2026-10-26
**Phạm vi:** E1, O1–O5 (Load pre-trained, explore architecture, adapt head, freeze/unfreeze)

---

## 1. Pre-trained Models Implemented

| Model | Source | Total Params | Trainable (freeze) | Trainable (fine-tune 1 block, chốt `freeze_head_full: false`) |
|-------|--------|--------------|--------------------|------------------------------------------|
| resnet18 | `torchvision.models` 0.29 | 11,181,642 | 5,130 (0.046%) | 8,398,858 (75.1%) |
| vgg11 | `torchvision.models` 0.29 | 128,807,306 | 40,970 (0.032%) | 2,400,778 (1.86%) |
| densenet121 | `torchvision.models` 0.29 | 6,964,106 | 10,250 (0.147%) | 2,696,714 (38.7%) |

Số freeze đo bằng `count_parameters()` sau `build_model()` (run `20261026-*`). Số fine-tune chốt lấy từ 3 run retrain 20 epoch (`20261031-120033/-121323/-122534/metrics.json`) với `freeze_until: 1`, `freeze_head_full: false`.

## 2. Architecture Exploration (đề bài bước 3)

Lệnh: `python -m src.models --model <name> [--print-model]`

Kết quả ResNet18 (weights ImageNet thật; run CLI M2 `results/runs/20261026-115633/` đã dọn sau M2, số liệu giữ nguyên):

```text
Class: ResNet
Total parameters:       11,181,642
Trainable parameters:        5,130 (0.046%)
TOP-LEVEL LAYERS:
  conv1                Conv2d                           9,408 params
  bn1                  BatchNorm2d                        128 params
  layer1               Sequential                     147,968 params
  layer2               Sequential                     525,568 params
  layer3               Sequential                   2,099,712 params
  layer4               Sequential                   8,393,728 params
  fc                   Linear                           5,130 params
Output shape: (1, 10) (kỳ vọng (1, 10))
```

Cấu trúc head sau khi thay về 10 lớp:

| Backbone | Head gốc | Head sau khi thay |
|----------|----------|-------------------|
| `resnet18` | `fc = Linear(512, 1000)` | `Linear(512, 10)` |
| `vgg11` | `classifier = Sequential[7]` | `Sequential[7]`, phần tử cuối = `Linear(4096, 10)` |
| `densenet121` | `classifier = Linear(1024, 1000)` | `Linear(1024, 10)` |

Điểm khác biệt quan trọng (đã verify bằng introspection + chạy thật):
- `vgg11.classifier` là `Sequential` 7 phần tử, không phải 1 `Linear` — code giả định mọi model có `model.fc` sẽ crash ngay.
- `vgg11.features` có 8 Conv2d (chỉ số `0,3,6,8,11,13,16,18`).
- `densenet121.features` có `denseblock1..4` xen kẽ `transition1..3`, kết thúc bằng `norm5`.
- Cả 3 model sau khi thay head đều cho output `(1, 10)` với input `(1, 3, 224, 224)`.

## 3. Adaptation Strategy

### Freeze backbone (transfer learning)

`freeze_backbone()` đóng băng toàn bộ backbone, chỉ giữ head trainable.

⚠️ **Vấn đề VGG11:** `classifier` của VGG11 chứa **103M params** (2 tầng `Linear(25088→4096)` và `Linear(4096→4096)`). Nếu "freeze backbone" mà unfreeze toàn bộ `classifier` thì **92.8% model vẫn trainable** → mất hoàn toàn ý nghĩa transfer learning. Vì vậy mặc định `freeze_head_full: false` — chỉ lớp `Linear` cuối (40,970 params = 0.032%). Test `test_vgg_freeze_head_full_flag` bắt lỗi cấu hình này.

### Fine-tune (unfreeze n block cuối)

`unfreeze_last(model, name, n_blocks)` mở n block cuối theo `block_units` đã introspect trong `registry.py`, cộng head và `extra_trainable` (DenseNet: `transition3`, `norm5`). BatchNorm trong block được mở luôn (nếu không, BN chỉ dùng running stats → không thích ứng phân bố CIFAR-10).

## 4. Selected Model

- **Chiến lược chốt:** `fine_tune` (mở block cuối + norm + head, optimizer Adam) — toàn bộ kết quả báo cáo (sweep 30 cấu hình + retrain 20 epoch) đều fine-tune, không dùng freeze.
- **Config mặc định:** `resnet18` + `fine_tune` (`configs/model_config.yaml`).
- **Lý do giữ resnet18 làm mặc định:**
  - Nhẹ nhất trong 3 backbone (11.18M vs 128.8M của VGG11).
  - Train nhanh nhất trên GPU T4 (~27–35 giây/epoch, ảnh 64, AMP).
  - DenseNet121 nhẹ (6.96M) nhưng kiến trúc dense connection phức tạp hơn, để dành so sánh ở M3/M5.
  - Sweep/retrain override bằng CLI `--models` hoặc notebook Colab.

## 5. Test Results

| Suite | File | Kết quả |
|-------|------|---------|
| Unit | `tests/unit/test_factory.py` | 18 passed (31.8s, `weights=None` — không tải ImageNet; gồm `test_finetune_all_models` bắt bug B4) |
| Integration | `tests/integration/test_model_pipeline.py` | 6 passed (42.6s) |
| CLI verify (weights ImageNet thật) | `python -m src.models --model <name>` | PASS cả 3 — output `(1, 10)`, artifact `model_info.json` |

Tổng: **24/24 pass** (18 unit + 6 integration).

### CLI verify chi tiết (weights `DEFAULT` thật)

| Model | Run | Total | Trainable (freeze) | Output |
|-------|-----|-------|--------------------|--------|
| resnet18 | CLI verify M2 (run `20261026-115633` đã dọn) | 11,181,642 | 5,130 (0.046%) | `(1, 10)` ✓ |
| vgg11 | CLI verify M2 (run `20261026-120218` đã dọn) | 128,807,306 | 40,970 (0.032%) | `(1, 10)` ✓ |
| densenet121 | CLI verify M2 (run `20261026-121817` đã dọn) | 6,964,106 | 10,250 (0.147%) | `(1, 10)` ✓ |

VGG11 tải ~14 phút (file 500MB, rớt mạng 1 lần ở 57.7% rồi chạy lại thành công). DenseNet121 tải < 30s.

## 6. Files Created

- `src/models/registry.py` — `BACKBONE_REGISTRY`, `block_units`, `resolve_spec`, `get_module`
- `src/models/factory.py` — `ModelConfig`, `build_model`, `replace_head`, `freeze_backbone`, `unfreeze_last`, `count_parameters`, `describe_model`
- `src/models/__init__.py` — export public API
- `src/models/__main__.py` — CLI khám phá
- `configs/model_config.yaml` — config mặc định (resnet18 + fine-tune)
- `tests/unit/test_factory.py` — 18 test (gồm `test_finetune_all_models` bắt bug B4)
- `tests/integration/test_model_pipeline.py` — 6 test
