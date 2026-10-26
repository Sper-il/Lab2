# M2 — Model & Architecture Report

**Ngày:** 2026-10-26
**Phạm vi:** E1, O1–O5 (Load pre-trained, explore architecture, adapt head, freeze/unfreeze)

---

## 1. Pre-trained Models Implemented

| Model | Source | Total Params | Trainable (freeze) | Trainable (fine-tune 1 block) |
|-------|--------|--------------|--------------------|-------------------------------|
| resnet18 | `torchvision.models` 0.29 | 11,181,642 | 5,130 (0.046%) | 8,398,858 (75.1%) |
| vgg11 | `torchvision.models` 0.29 | 128,807,306 | 40,970 (0.032%) | 121,946,634 (94.7%) |
| densenet121 | `torchvision.models` 0.29 | 6,964,106 | 10,250 (0.147%) | 2,168,330 (31.1%) |

Số liệu đo bằng `count_parameters()` sau khi `build_model()` — không ước lượng.

## 2. Architecture Exploration (đề bài bước 3)

Lệnh: `python -m src.models --model <name> [--print-model]`

Kết quả ResNet18 (weights ImageNet thật, run `results/runs/20261026-115633/`):

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

- **Model:** `resnet18`
- **Strategy mặc định:** `freeze_backbone`
- **Lý do:**
  - Nhẹ nhất trong 3 backbone có thể train thực tế trên CPU (11.18M vs 128.8M của VGG11).
  - Freeze chỉ còn 5,130 params trainable → 1 epoch 224×224 mất 22–37 phút; VGG11 mất ~144 phút/epoch (~12 giờ cho 5 epoch — không khả thi).
  - DenseNet121 nhẹ (6.96M) nhưng fine-tune 1 block chỉ mở 31.1% — kiến trúc dense connection phức tạp hơn, để dành so sánh ở M3/M5.
  - Config mặc định nằm trong `configs/model_config.yaml`; M3 override bằng CLI `--models` khi sweep.

## 5. Test Results

| Suite | File | Kết quả |
|-------|------|---------|
| Unit | `tests/unit/test_factory.py` | 17 passed (31.8s, `weights=None` — không tải ImageNet) |
| Integration | `tests/integration/test_model_pipeline.py` | 6 passed (42.6s) |
| CLI verify (weights ImageNet thật) | `python -m src.models --model <name>` | PASS cả 3 — output `(1, 10)`, artifact `model_info.json` |

Tổng: **23/23 pass**.

### CLI verify chi tiết (weights `DEFAULT` thật)

| Model | Run | Total | Trainable (freeze) | Output |
|-------|-----|-------|--------------------|--------|
| resnet18 | `results/runs/20261026-115633/` | 11,181,642 | 5,130 (0.046%) | `(1, 10)` ✓ |
| vgg11 | `results/runs/20261026-120218/` | 128,807,306 | 40,970 (0.032%) | `(1, 10)` ✓ |
| densenet121 | `results/runs/20261026-121817/` | 6,964,106 | 10,250 (0.147%) | `(1, 10)` ✓ |

VGG11 tải ~14 phút (file 500MB, rớt mạng 1 lần ở 57.7% rồi chạy lại thành công). DenseNet121 tải < 30s.

## 6. Files Created

- `src/models/registry.py` — `BACKBONE_REGISTRY`, `block_units`, `resolve_spec`, `get_module`
- `src/models/factory.py` — `ModelConfig`, `build_model`, `replace_head`, `freeze_backbone`, `unfreeze_last`, `count_parameters`, `describe_model`
- `src/models/__init__.py` — export public API
- `src/models/__main__.py` — CLI khám phá
- `configs/model_config.yaml` — config mặc định (resnet18 + freeze)
- `tests/unit/test_factory.py` — 17 test
- `tests/integration/test_model_pipeline.py` — 6 test
