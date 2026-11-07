# Model — mô tả model cho project để làm

> Đặc tả thi công module model (M2): cần làm model nào, thay head ra sao, freeze/fine-tune thế nào, viết ở file nào, nghiệm thu bằng gì.
> Đọc trước khi code `src/models/`. Số liệu tham chiếu đã đo thật từ code và checkpoint, không ước lượng.
> Tài liệu liên quan: `docs/architecture.md` (vị trí module model trong hệ thống), `docs/api.md` (interface chi tiết),
> `docs/reports/m2_models_report.md` (kết quả đã làm xong), `docs/CHANGELOG.md` (version).

## 1. Mục đích của tài liệu này

- Nói rõ **cần làm gì** ở module model trước khi viết code: làm mấy backbone, mỗi backbone thay head thế nào, freeze ra sao.
- Làm căn cứ để viết `src/models/registry.py`, `src/models/factory.py`, `src/models/__main__.py` và `configs/model_config.yaml`.
- Làm căn cứ nghiệm thu: chạy lệnh nào, kiểm tra output nào, test nào phải pass.

## 2. Bài toán và ràng buộc để làm model

- **Bài toán cần làm:** phân loại ảnh CIFAR-10, 10 lớp (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
- **Input model phải nhận:** batch ảnh `[B, 3, 64, 64]` (train/eval chốt trên GPU T4) hoặc `[B, 3, 224, 224]` (demo M1 chuẩn ImageNet), `float32`, đã normalize theo ImageNet mean/std.
- **Output model phải trả:** logits `[B, 10]`, sau đó `argmax` ra nhãn và `softmax` lấy confidence.
- **Ràng buộc khi làm:**
  - Dùng backbone pre-trained ImageNet từ `torchvision.models` 0.29 (`weights="DEFAULT"`), không tự thiết kế CNN từ đầu.
  - Dataset nhỏ (50.000 ảnh 32×32) nên phải làm transfer learning/fine-tuning, không train full từ đầu làm phương án chính.
  - Mọi siêu tham số kiến trúc phải nằm trong `configs/model_config.yaml`, không hardcode trong code.
  - Seed cố định 42 để tái tạo.

## 3. Danh sách model cần làm

Cần làm đủ 3 backbone để so sánh theo yêu cầu E1 trong `đề.txt`:

| STT | Model cần làm | Vai trò trong project | Kết quả tham chiếu sau khi làm xong |
|-----|---------------|-----------------------|-------------------------------------|
| 1 | `resnet18` | Baseline chính, nhẹ nhất, làm trước để có mốc so sánh | 11.181.642 params; test 0,8626 (`results/runs/20261031-120033/`) |
| 2 | `vgg11` | Model so sánh, head dạng `Sequential` đặc biệt, cho accuracy cao nhất | 128.807.306 params; test 0,8774 ⭐ (`results/runs/20261031-121323/`) |
| 3 | `densenet121` | Model so sánh, kiến trúc dense connection, nhẹ nhưng mạnh | 6.964.106 params; test 0,8715 (`results/runs/20261031-122534/`) |

Thứ tự làm: `resnet18` trước, chạy thông rồi mới làm `vgg11` và `densenet121`.

## 4. Đặc tả chi tiết từng model cần làm

### 4.1. ResNet18 cần làm

- **Backbone gốc cần nạp:** `torchvision.models.resnet18(weights="DEFAULT")`, head gốc `fc = Linear(512, 1000)`.
- **Head cần thay:** `fc = Linear(512, 10)`, giữ nguyên toàn bộ phần còn lại (`conv1` → `bn1` → `layer1` → `layer2` → `layer3` → `layer4` → `avgpool`).
- **Block cuối cần mở khi fine-tune:** `layer4` (theo `block_units` trong `src/models/registry.py`).
- **Số params sau khi làm:** total 11.181.642; freeze chỉ train 5.130 (0,046%); fine-tune 1 block train 8.398.858 (75,1%).

### 4.2. VGG11 cần làm

- **Backbone gốc cần nạp:** `torchvision.models.vgg11(weights="DEFAULT")`, head gốc `classifier = Sequential[7]`, 8 Conv2d trong `features` (chỉ số `0, 3, 6, 8, 11, 13, 16, 18`).
- **Head cần thay:** giữ nguyên 6 lớp đầu của `classifier`, chỉ thay lớp `Linear` cuối thành `Linear(4096, 10)`.
- **Block cuối cần mở khi fine-tune:** `features.16` + `features.18` + `classifier.6`.
- **Lưu ý bắt buộc khi làm:** `classifier` chứa tới 103M params. Khi freeze backbone, mặc định chỉ mở lớp `Linear` cuối (`freeze_head_full: false` → 40.970 params, 0,032%). Nếu mở cả `classifier` thì 92,8% model vẫn trainable, mất ý nghĩa transfer learning — phải có test bắt lỗi cấu hình này.
- **Số params sau khi làm:** total 128.807.306; fine-tune 1 block train 2.400.778 (1,86%).

### 4.3. DenseNet121 cần làm

- **Backbone gốc cần nạp:** `torchvision.models.densenet121(weights="DEFAULT")`, head gốc `classifier = Linear(1024, 1000)`, 4 denseblock xen kẽ 3 transition, kết thúc bằng `norm5`.
- **Head cần thay:** `classifier = Linear(1024, 10)`.
- **Block cuối cần mở khi fine-tune:** `features.denseblock4` + `extra_trainable` là `features.transition3` và `features.norm5`.
- **Lưu ý bắt buộc khi làm:** đường dẫn `features.denseblock4` là key chuỗi, không phải chỉ số — hàm lấy module phải ưu tiên `getattr` trước, fallback index sau, nếu không sẽ crash (lỗi B4 đã gặp).
- **Số params sau khi làm:** total 6.964.106; freeze chỉ train 10.250 (0,147%); fine-tune 1 block train 2.696.714 (38,7%).

## 5. File và interface cần viết

| File cần viết | Nội dung cần làm |
|---------------|------------------|
| `src/models/registry.py` | Viết `BackboneSpec` (`builder`, `head_attr`, `head_type`, `block_units`, `extra_trainable`), `BACKBONE_REGISTRY` cho 3 model, hàm `resolve_spec(name)` và `get_module(model, path)` |
| `src/models/factory.py` | Viết `ModelConfig` (validate `name`, `strategy`, `num_classes >= 2`, `freeze_until >= 1`), `resolve_weights`, `replace_head` (xử lý cả 2 dạng head linear/sequential), `freeze_backbone`, `unfreeze_last`, `count_parameters`, `describe_model`, `build_model` |
| `src/models/__main__.py` | Viết CLI khám phá: nhận `--model/--strategy/--print-model`, in cấu trúc, đếm params, kiểm tra output `(1, 10)` bằng ảnh dummy `(1, 3, 224, 224)` rồi ghi `model_info.json` |
| `configs/model_config.yaml` | Khai báo `name: resnet18`, `num_classes: 10`, `weights: DEFAULT`, `strategy: fine_tune`, `freeze_until: 1`, `freeze_head_full: false` |
| `models/configs/*-cifar10.yaml` | Viết model card từng version sau khi làm xong (`input_size`, `test_accuracy`, `macro_f1`, `run_ref`, `trainable_params_chot_20epoch`) |
| `models/trained/*-final.json` | Ghi kết quả cuối từng model (VGG11 đánh dấu `best_model: true`) |

## 6. Chiến lược freeze/fine-tune cần làm

- **Chiến lược chốt cần làm:** `fine_tune` với `freeze_until: 1`, `freeze_head_full: false`, optimizer Adam.
- **`freeze_backbone` cần làm:** tắt hết `requires_grad`, chỉ mở head (VGG11 mặc định chỉ lớp `Linear` cuối).
- **`unfreeze_last(n_blocks)` cần làm:** tắt hết rồi mở head + n block cuối theo `block_units` + `extra_trainable` (DenseNet); mở luôn BatchNorm trong block để thích ứng phân bố CIFAR-10.
- **`from_scratch` cần làm:** giữ nguyên toàn bộ trainable, dùng để đối chứng.

## 7. Loss và optimizer đi kèm để train được model

- **Loss cần làm trong `src/training/loss.py`:** chốt `CrossEntropyLoss`; làm thêm `label_smoothing` và `FocalLoss(gamma=2.0)` để thử nghiệm. CIFAR-10 cân bằng nên chốt không dùng class weight.
- **Optimizer chốt:** Adam, chỉ truyền params có `requires_grad=True`, lr 0,0001 (ResNet18/VGG11) / 0,0005 (DenseNet121), weight decay 1e-4.

## 8. Tiêu chí nghiệm thu sau khi làm

- [ ] Chạy `python -m src.models --model <resnet18|vgg11|densenet121> --print-model` đều ra output `(1, 10)`.
- [ ] `count_parameters()` ra đúng bảng §4 (ResNet18 11.181.642 / VGG11 128.807.306 / DenseNet121 6.964.106).
- [ ] Test pass: `tests/unit/test_factory.py` (18 test) + `tests/integration/test_model_pipeline.py` (6 test).
- [ ] Train thử 20 epoch ảnh 64 trên GPU T4 đạt test: ResNet18 ≥ 0,86 / VGG11 ≥ 0,87 / DenseNet121 ≥ 0,87.
- [ ] Mọi quyết định chọn/loại model ghi vào Decisions Log trong `docs/progress_status.md`.

## 9. Thứ tự làm

1. Viết `registry.py` cho `resnet18` trước, verify bằng introspect + chạy thử.
2. Viết `factory.py` (`replace_head` → `freeze_backbone` → `unfreeze_last` → `build_model` → `count/describe`).
3. Viết CLI `__main__.py`, chạy khám phá cả 3 model với weights ImageNet thật.
4. Viết test factory + pipeline, chạy `pytest -q`.
5. Thêm `vgg11` và `densenet121` vào registry, xử lý riêng head `Sequential` và `extra_trainable`.
6. Chốt config `model_config.yaml`, ghi model card và `CHANGELOG.md`.

## 10. Lệnh làm và kiểm tra

```powershell
# Bước làm 1 — khám phá từng backbone sau khi viết registry/factory
.\.venv\Scripts\python.exe -m src.models --model vgg11 --print-model
.\.venv\Scripts\python.exe -m src.models --config configs/model_config.yaml --model resnet18

# Bước làm 2 — train thử cấu hình chốt sau khi model xong
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --models vgg11 --lr 0.0001 --batch-size 32 --epochs 20

# Bước làm 3 — chạy test nghiệm thu
.\.venv\Scripts\python.exe -m pytest tests/unit/test_factory.py tests/integration/test_model_pipeline.py -q
```
