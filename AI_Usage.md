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
| `scripts/verify_data.py` | Kiểm tra số mẫu, phân bố lớp, shape ảnh | "Verify dữ liệu sau khi tải, có assert để fail rõ ràng" |
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
- *(Sẽ bổ sung sau mỗi module: chỗ sửa logic, đổi optimizer, đổi hyperparameter vì kết quả thực tế khác kỳ vọng.)*

## 7. Validation Method

- [x] Chạy code end-to-end trên môi trường sạch (venv mới tạo, dependency pin rõ ràng).
- [ ] So sánh output với baseline / đáp án đã biết.
- [x] Viết unit/integration test mà không có thay đổi thì sẽ fail (36 test cho M1).
- [ ] Reproduce kết quả từ fresh clone + ghi lại command.
- [ ] Sanity-check metric theo Definition of Done.

### Kiểm chứng riêng cho M1 (data pipeline)

| Kiểm tra | Kết quả |
|----------|---------|
| `pytest -q` | 36 passed in 234s |
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
