# AI Usage

> Hồ sơ trung thực về cách các công cụ AI được sử dụng trong project này.

## 1. Tools Used

| Vai trò | Tool / Model | Phiên bản | Khi nào |
|---------|--------------|-----------|---------|
| Code generation | Cursor + GPT-5-class agent | 2026-10 | Scaffold module, viết boilerplate |
| Documentation | Cursor + GPT-5-class agent | 2026-10 | Viết README, `AI_Usage.md`, `progress_status.md` |
| Debugging | Cursor + GPT-5-class agent | 2026-10 | Phân tích traceback, đề xuất root cause |
| Research | Cursor + GPT-5-class agent | 2026-10 | Đọc tài liệu Chương 3, chọn kiến trúc & hyperparameter |
| Review | Cursor + GPT-5-class agent | 2026-10 | Review code AI generate trước khi commit |

## 2. Plain Purposes

- **Code generation** — scaffold module từ thiết kế đã duyệt, viết hàm tiện ích.
- **Documentation** — nháp phần README, tóm tắt quyết định thiết kế.
- **Debugging** — đề xuất root cause, gợi ý log cần thêm.
- **Research** — tóm tắt nội dung Chương 3 (slide 1–25), chọn backbone & tài liệu tham khảo PyTorch.
- **Review** — review code AI đã viết, phát hiện edge case.

## 3. Student Contributions (con người làm gì)

- Định nghĩa bài toán và phạm vi (Practice 2, Chương 3).
- Quyết định dùng CIFAR-10 theo ví dụ trong slide 24.
- Quyết định stack CPU-only do máy không có GPU NVIDIA.
- Toàn bộ phương pháp đánh giá và chọn metric (accuracy, macro F1, confusion matrix).
- Toàn bộ review cuối cùng cho code AI generate.
- Toàn bộ commit (AI có thể nháp, người quyết định).
- Toàn bộ kết luận thí nghiệm.

## 4. Important AI-Generated Artifacts

| File | Nội dung | Prompt đã dùng (tóm tắt) |
|------|----------|--------------------------|
| `README.md` | Tài liệu project đầy đủ (10 mục chuẩn theo skill) | "Đọc kỹ đề Practice 2, Chapter 3 PDF, skill ai-fusion-workflow; viết README.md theo template readme.md" |
| `requirements.txt` | Pin dependency tương thích Python 3.13 + CPU | "Chọn version torch/torchvision/tensorboard phù hợp, kiểm tra bằng pip index" |
| `.gitignore` | Theo `templates/folder-structure.md` | "Sinh .gitignore theo convention của skill" |
| `progress_status.md` | Project Context, Module Log, Decisions Log | "Khởi tạo progress_status.md theo template progress-status.md" |
| `scripts/download_data.py` | Tải + giải nén CIFAR-10, có fallback link thủ công | "Lấy data về giúp tôi hoặc cho tôi link để tải" |
| `scripts/verify_data.py` | Kiểm tra MD5 archive + số mẫu, phân bố lớp, shape ảnh | "Verify dữ liệu sau khi tải, có assert để fail rõ ràng" → bổ sung kiểm tra MD5 |
| `data/README.md` | Hướng dẫn tải dataset + thông tin dataset | "Tài liệu data pipeline cho người clone repo" |
| `src/data/transforms.py` | Train/val transform (RandomCrop+Flip / Resize+Normalize ImageNet) | "Làm bước M1 data pipeline, nhớ xuất kết quả vào folder results và cập nhật docs" |
| `src/data/dataset.py` | `Cifar10Dataset` (in-memory) + `TransformSubset` | "Làm bước M1 data pipeline" |
| `src/data/split.py` | `stratified_indices` chia 45k/5k phân tầng, seed=42 | "Làm bước M1 data pipeline" |
| `src/data/dataloader.py` | `DataLoaderConfig` + `build_dataloaders` | "Làm bước M1 data pipeline" |
| `src/data/__main__.py` | CLI xuất artifact `config.yaml`/`metrics.json`/`logs.txt` | "Xuất kết quả vào folder results" |
| `src/utils/config.py` | Nạp YAML config, `resolve_path`, `deep_merge` | "Làm bước M1 data pipeline" |
| `src/utils/runs.py` | Tạo run dir, cấu hình logging, ghi JSON/YAML artifact | "Xuất kết quả vào folder results" |
| `configs/data_config.yaml` | Toàn bộ hyperparameter data pipeline | "Làm bước M1 data pipeline" |
| `tests/unit/test_transforms.py` | 9 test shape/determinism/normalization của transform | "Viết test cho M1" |
| `tests/unit/test_split.py` | 9 test về phân tầng, overlap, reproducibility, edge case | "Viết test cho M1" |
| `tests/unit/test_dataset.py` | 9 test về dataset, chỉ số âm, class balance, không mutate parent | "Viết test cho M1" |
| `tests/integration/test_dataloader_pipeline.py` | 8 test end-to-end DataLoader + config parse | "Viết test cho M1" |
| `scripts/plot_data_report.py` | Sinh biểu đồ phân bố lớp + lưới ảnh mẫu vào `results/runs/<ts>/plots/` | "Xuất kết quả vào folder results" |
| `src/data/prepare.py` | Xuất `data/processed/` (.npy + metadata) và `data/splits/` (.npz + .json), verify split bằng hash nội dung ảnh | "Sao folder processed với splits trống vậy bạn xem lại giúp tôi" |
| `tests/unit/test_prepare.py` | 11 test: file .npy khớp raw, split reproducible, split export khớp pipeline, phân tầng giữ nguyên | "Sao folder processed với splits trống vậy bạn xem lại giúp tôi" |
| `scripts/benchmark_loading.py` | Đo throughput data loading để quyết định có cache ảnh 224×224 | "Sao folder processed với splits trống vậy bạn xem lại giúp tôi" |
| `README.md` (v2) — mục "2b. Đối chiếu với đề bài" | Bảng ánh xạ 14 yêu cầu trong `đề.txt` (E1–E3, O1–O11) → file/module/trạng thái | "Đảm bảo các docs luôn bám sát đề" |
| `progress_status.md` (v2) | Thêm bảng đối chiếu đề bài, sửa shape `[B,3,32,32]` → `[B,3,224,224]`, cập nhật commit + số test | "Đảm bảo các docs luôn bám sát đề" |
| `docs/plan-m2-m4.md` | Plan chi tiết M2/M3/M4 — file cần tạo, chức năng từng hàm, test, tiêu chí, ước lượng thời gian train đã đo thật | "Cho tôi plan chia việc phần m2 m3 m4, làm file gì làm ở đâu file đó có chức năng gì" |
| `docs/PHASE2_PLAN.md` | Plan chi tiết M2/M3/M4 (theo format `PHASE2_PLAN.md`) — file nào, hàm nào, code mẫu, test, report template | "ghi phần m2 m3 m4 thôi" |
| `src/models/registry.py` | `BACKBONE_REGISTRY` (3 backbone + `block_units` introspect), `resolve_spec`, `get_module` | "làm và chạy phần m2 giúp tôi" (plan `docs/PHASE2_PLAN.md` §M2) |
| `src/models/factory.py` | `ModelConfig`, `build_model`, `replace_head` (Linear + Sequential), `freeze_backbone`, `unfreeze_last`, `count/describe_model` | "làm và chạy phần m2 giúp tôi" |
| `src/models/__main__.py` + `configs/model_config.yaml` | CLI khám phá (`--model/--strategy/--print-model`) + config mặc định resnet18 freeze | "làm và chạy phần m2 giúp tôi" |
| `tests/unit/test_factory.py` + `tests/integration/test_model_pipeline.py` | 17 unit + 6 integration test M2 | "làm và chạy phần m2 giúp tôi" |
| `docs/reports/m2_models_report.md` | Report M2: bảng params, architecture exploration, adaptation, model đã chọn | "làm và chạy phần m2 giúp tôi" |

## 5. Prompt Purpose (cho mỗi artifact)

- **`README.md`** — prompt yêu cầu AI đọc kỹ `đề.txt`, `Chapter 3 - TRAINING DEEP NEURAL NETWORKS (1).pdf` và skill `ai-fusion-workflow`, sau đó viết README 10 mục với Problem Statement cụ thể (CIFAR-10, 10 lớp, split 45k/5k/10k) và bảng tiêu chí đạt.
- **`requirements.txt`** — prompt yêu cầu AI kiểm tra version khả dụng trên PyPI/PyTorch index cho Python 3.13 (không có GPU) và pin lại.
- **`progress_status.md`** — prompt yêu cầu AI điền sẵn Phase 1 DEFINE với câu trả lời What/Why/Where-When-Which/How.

## 6. Student Modifications

- **`scripts/verify_data.py`** — AI viết bản đầu dùng `train_set.classes` (tên lớp dạng string) để so sánh với `Counter(targets)` (index dạng int) → `AssertionError` giả. Đã sửa thành `{idx: 5000 for idx in range(10)}`.
- **`src/data/dataloader.py` (bản đầu của AI)** — gán `val_subset.dataset.transform = eval_transform` để đổi transform cho tập validation. Đây là **bug**: `torch.utils.data.Subset` chỉ giữ tham chiếu tới dataset gốc, nên lệnh gán này làm thay đổi transform của *cả tập train* → tập train cũng mất augmentation. Đã sửa bằng cách tự viết `TransformSubset` trong `src/data/dataset.py`, giữ transform riêng cho từng split.
- **`src/utils/runs.py`** — bản đầu của AI dùng `logging.StreamHandler(sys.stdout)` mặc định, gây `UnicodeEncodeError: 'charmap' codec can't encode character '\u1edb'` trên console Windows (cp1252) — process chết giữa chừng. Đã sửa bằng cách `stream.reconfigure(encoding="utf-8", errors="replace")` cho stdout/stderr và `FileHandler(..., encoding="utf-8", errors="replace")`.
- **`src/data/__main__.py`** — bản đầu của AI viết log không dấu (`mau`, `that bai`) để né lỗi encoding trên. Sau khi sửa lỗi encoding ở trên, đã đổi lại toàn bộ log sang tiếng Việt có dấu.
- **`scripts/plot_data_report.py`** — bản đầu import `src.data...` không có `sys.path` bootstrap → `ModuleNotFoundError: No module named 'src'` khi chạy trực tiếp file. Đã thêm `sys.path.insert(0, parent.parent)`.
- **`requirements.txt`** — thêm `matplotlib==3.11.2` (chưa có sẵn) vì M1 cần vẽ biểu đồ kiểm chứng. Version được xác minh qua `pip index versions matplotlib`.
- **`src/data/prepare.py` — `verify_splits`** — bản đầu của AI so sánh overlap bằng cách lấy giao `set(train_idx) ∩ set(test_idx)`, cho ra `AssertionError: train/test overlap` **giả**. Nguyên nhân: `train_idx` trỏ vào tập train (0..49.999) còn `test_idx` trỏ vào tập test (0..9.999) — hai không gian chỉ số khác nhau, nên so sánh trực tiếp là vô nghĩa. Đã sửa bằng cách hash **nội dung ảnh** (`image.tobytes()`) của từng mẫu rồi so sánh tập hash giữa các split — vừa đúng ý nghĩa vừa phát hiện được cả rò rỉ ảnh lẫn chỉ số sai.
- **`src/data/prepare.py` — `export_processed`** — bản đầu của AI resize ảnh lên 224×224 trước khi lưu. Đã sửa thành giữ ảnh gốc 32×32, vì `RandomCrop(32, padding=4)` cần thao tác trên ảnh gốc và augmentation phải khác nhau mỗi epoch. Benchmark cho thấy workers=4 đạt 403 ảnh/s (~1.9 phút/epoch) nên cache 7 GB không đáng.
- **`README.md` / `data/README.md`** — bản đầu ghi `processed/` và `splits/` là "(giai đoạn sau)" tức thư mục rỗng. Đã sửa mô tả thành nội dung thật sau khi implement `src/data/prepare.py`.
- **`README.md` / `progress_status.md`** — bản đầu của AI ghi shape đầu vào là `[B, 3, 32, 32]`, trong khi transform thực tế resize lên **224×224** → shape thật là `[B, 3, 224, 224]`. Đã sửa cả hai file.
- **`README.md` — mục "Example Input → Output"** — bản đầu của AI đưa kết quả giả (`best_val_accuracy: 0.7412`, `test_accuracy: 0.7308`) trông như số đo thật, và dùng `next(iter(train_loader))["image"]` — API sai (`DataLoader` trả tuple, không dict). Đã sửa thành placeholder có nhãn cảnh báo rõ + `images, labels = next(iter(train_loader))` (đã kiểm chứng bằng code chạy thật).
- **`README.md` — mục 6.6 "Chạy training"** — bản đầu liệt kê lệnh `python -m src.training` và `configs/train_config.yaml` như đã có, trong khi các module đó mới chỉ là thư mục rỗng. Đã thêm ghi chú trạng thái "thiết kế dự kiến".
- **`README.md` — mục 4** — tham chiếu `docs/architecture.md` chưa tồn tại. Đã đổi sang `docs/data-flow.md` (đã có) và ghi rõ phần nào đã hiện thực.
- **`scripts/verify_data.py`** — bản đầu của AI chỉ kiểm tra số mẫu / phân bố lớp / shape, **không** kiểm tra tính toàn vẹn file tải về, dù docs và `download_data.py` đều nói "có kiểm tra MD5 tích hợp". Đã bổ sung `_verify_archive()` — hash MD5 theo chunk (không nạp hết 163 MB vào RAM), so với giá trị chuẩn `c58f30108f718f92721af3b95e74349a`, exit code 1 + hướng dẫn tải lại nếu sai. Đã chạy thật: PASS.
- **`docs/plan-m2-m4.md`** — bản đầu của AI ước lượng thời gian train bằng cảm tính ("10–20 phút/epoch"). Đã **đo thật** trên máy này (ResNet18, batch 32, 8 threads): freeze 224×224 = **36.6 min/epoch**, 64×64 = **4.8 min/epoch**; VGG11 224 = **144 min/epoch** (12 giờ cho 5 epoch → không khả thi). Nhờ đó mới phát hiện ra mâu thuẫn giữa README ("≤ 15 phút/epoch") và thực tế, dẫn tới quyết định 2 tầng 224/64.
- **`docs/plan-m2-m4.md` — cấu trúc head** — bản đầu của AI giả định mọi model có `model.fc`. Đã introspect thật: `resnet18.fc` = `Linear`, `densenet121.classifier` = `Linear`, nhưng **`vgg11.classifier` = `Sequential` 7 phần tử** (3 tầng Linear + ReLU + Dropout xen kẽ) → code thay head phải xử lý 2 dạng khác nhau, nếu không sẽ crash ngay khi chạy VGG11.
- **`docs/PHASE2_PLAN.md` — bản đầu của AI viết `freeze_backbone()` unfreeze toàn bộ `classifier`** mà không kiểm tra số params. Đã chạy code thật để đo: `classifier` của VGG11 chứa **103M params** → unfreeze cả sẽ khiến **92.8% model trainable**, mất hoàn toàn ý nghĩa của "freeze backbone". Đã sửa thành `freeze_head_full: false` (chỉ `Linear(4096,10)` = 40,970 params = 0.032%) và thêm test bắt lỗi này.
- **`docs/PHASE2_PLAN.md` — `unfreeze_last` dùng `feature_attr=""` cho resnet18** → sẽ lấy `children[-1:]` của model = `fc` chứ không phải `layer4`, tức fine-tune sẽ không mở block nào. Đã thay bằng `block_units` khai báo tập trung trong `registry.py` và **verify bằng chạy thật**: resnet18 `ft1` = 75.1% params trainable (đúng ~`layer4`), vgg11 `ft1` = 94.7%, densenet121 `ft1` = 31.1%.
- **`docs/PHASE2_PLAN.md` — `resolve_weights` dùng biến `self_name_normalized` chưa được định nghĩa** trong skeleton đầu tiên → sẽ `NameError` ngay khi gọi với `weights="IMAGENET1K_V1"`. Đã thay bằng mapping `_WEIGHT_ENUMS` ở cấp module, không cần hàm trợ.
- *(Sẽ bổ sung sau mỗi module: chỗ sửa logic, đổi optimizer, đổi hyperparameter vì kết quả thực tế khác kỳ vọng.)*
- **M2 — `src/models/__init__.py` (bản đầu của AI)** — import `resolve_weights` từ `registry` trong khi hàm này nằm ở `factory` → sẽ `ImportError` ngay khi `import src.models`. Đã sửa import cho đúng module.
- **M2 — test/CLI chạy trên Windows PowerShell** — lệnh mẫu trong plan dùng `tail -n 20` (không tồn tại trên PowerShell) → đổi thành chạy `pytest -q` trực tiếp. VGG11 tải weights ~500MB bị rớt mạng 1 lần ở 57.7% (`ConnectionAbortedError`) → chạy lại, torchvision tiếp tục từ cache và thành công.

## 7. Validation Method

- [x] Chạy code end-to-end trên môi trường sạch (venv mới tạo, dependency pin rõ ràng).
- [ ] So sánh output với baseline / đáp án đã biết.
- [x] Viết unit/integration test mà không có thay đổi thì sẽ fail (47 test cho M1; 23 test cho M2).
- [ ] Reproduce kết quả từ fresh clone + ghi lại command.
- [ ] Sanity-check metric theo Definition of Done.
- [x] **Đối chiếu docs với `đề.txt`**: mỗi yêu cầu trong đề (E1–E3, O1–O11) được ánh xạ tới file/module cụ thể trong bảng "Đối chiếu với đề bài"; mọi shape, số liệu và lệnh trong docs đã được đối chiếu lại với code thật.

### Kiểm chứng riêng cho M2 (model factory)

| Kiểm tra | Kết quả |
|----------|---------|
| `pytest tests/unit/test_factory.py` | **17 passed** in 31.8s (`weights=None`, không cần mạng) |
| `pytest tests/integration/test_model_pipeline.py` | **6 passed** in 42.6s |
| `python -m src.models --model resnet18` | Total 11,181,642 / trainable 5,130 (0.046%) / output `(1, 10)` — run `20261026-115633` |
| `python -m src.models --model vgg11` | Total 128,807,306 / trainable 40,970 (0.032%) / output `(1, 10)` — run `20261026-120218` |
| `python -m src.models --model densenet121` | Total 6,964,106 / trainable 10,250 (0.147%) / output `(1, 10)` — run `20261026-121817` |
| VGG11 `freeze_head_full=true` | 92.8% trainable → test bắt lỗi cấu hình này PASS |

### Kiểm chứng riêng cho M1 (data pipeline)

| Kiểm tra | Kết quả |
|----------|---------|
| `pytest -q` | **47 passed** in 97s (36 data pipeline + 11 prepare) |
| `python -m src.data` | train 45.000 / val 5.000 / test 10.000 — `validation.status = PASS` |
| Tổng train + val | 45.000 + 5.000 = 50.000 ✓ khớp tập train gốc |
| Phân bố lớp sau split | Mỗi lớp: 4.500 train / 500 val / 1.000 test — giữ đúng tỉ lệ |
| Overlap train/val | 0 mẫu trùng (kiểm tra bằng `set` intersection) |
| Reproducibility | Cùng `seed=42` → chỉ số split giống hệt |
| Shape batch | `[64, 3, 224, 224]` float32, label `[64]` int64 |
| Determinism val | Thứ tự batch val giống nhau qua 2 lần chạy |
| Shuffle train | Thứ tự batch train khác nhau qua 2 epoch |
| Biểu đồ | `plots/class_distribution.png`, `plots/sample_images.png` |
| `data/processed/` | train/test images + labels dạng `.npy`, metadata.json — shape `[50000,32,32,3]` uint8 |
| `data/splits/` | `cifar10_split_seed42.npz` + `.json` — 45.000 / 5.000 / 10.000 |
| Split export khớp pipeline | Test so `train_idx`/`val_idx` trong `.npz` với output của `stratified_indices` → khớp 100% |
| Split không trùng ảnh | Hash nội dung ảnh: overlap train/val = 0, train/test = 0, val/test = 0 |
| MD5 archive | `scripts/verify_data.py` → `c58f30108f718f92721af3b95e74349a` khớp chuẩn (162.6 MB) — PASS |
| Phân phối pixel | `mean 119.3`, `std 62.9`, 0.30% pixel = 0, 0.72% pixel = 255 → dữ liệu lành mạnh, không mờ/hỏng |
