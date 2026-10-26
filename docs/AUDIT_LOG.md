# Audit Log

> Nhật ký review code + docs theo yêu cầu skill `ai-fusion-workflow` (mục *Coursework & Audit Requirements*).
> Mỗi entry: ai audit, audit cái gì, phát hiện gì, đã xử lý ra sao. Chỉ ghi sự thật đã verify — không bịa số liệu.

## 2026-10-26 — Audit M2 (model factory) + docs hiện trạng

**Người audit:** AI agent (Cursor) theo yêu cầu user "coi cách docs có thiếu cái gì không".
**Phạm vi:** `src/models/*.py`, `configs/model_config.yaml`, `tests/unit/test_factory.py`, `tests/integration/test_model_pipeline.py`, toàn bộ docs (`README.md`, `docs/progress_status.md`, `docs/AI_Usage.md`, `docs/reports/m2_models_report.md`).

**Phương pháp:** đọc code + chạy test thật (`pytest` 23 passed / 43.8s) + đối chiếu docs với template skill và `đề.txt` + kiểm tra file tồn tại trên đĩa.

### Phát hiện (code)

| # | File | Vấn đề | Mức | Trạng thái |
|---|------|--------|-----|------------|
| A1 | `src/models/__init__.py` (bản đầu) | Import `resolve_weights` từ `registry` trong khi hàm nằm ở `factory` → `ImportError` ngay khi `import src.models` | Cao | ✅ Đã sửa trước commit `75455dd` |
| A2 | `docs/plan-m2-m4.md` skeleton `unfreeze_last` | `feature_attr=""` cho resnet18 → lấy `children[-1:]` = `fc`, không mở block nào khi fine-tune | Cao | ✅ Đã sửa: dùng `block_units` trong `registry.py`, verify chạy thật (resnet18 ft1 = 75.1% trainable ≈ `layer4`) |
| A3 | `docs/plan-m2-m4.md` skeleton `resolve_weights` | Dùng biến `self_name_normalized` chưa định nghĩa → `NameError` với `weights="IMAGENET1K_V1"` | Cao | ✅ Đã sửa: mapping `_WEIGHT_ENUMS` cấp module |
| A4 | `docs/plan-m2-m4.md` skeleton `freeze_backbone` | Unfreeze toàn bộ `classifier` mà không đo params | Cao | ✅ Đã sửa: đo thật → VGG11 `classifier` = 103M params (92.8% trainable) → mặc định `freeze_head_full: false` + test bắt lỗi |

### Phát hiện (docs — user xác nhận "thiếu nhiều")

| # | Vấn đề | Mức | Trạng thái |
|---|--------|-----|------------|
| D1 | `README.md` mục Example ghi `"params_total": 11689512` — số bịa, sai với ResNet18 thật (11,181,642) | Cao | ⏳ Sửa khi chốt báo cáo này (lấy từ `model_info.json`) |
| D2 | `README.md` ghi "chia 3 người" trong khi thực tế single-author, Module Log không phân công ai | Trung bình | ⏳ Sửa: ghi 1 người |
| D3 | `docs/progress_status.md` Module Log có **2 dòng M3 trùng nhau** (2026-10-26 và 2026-10-25) | Thấp | ⏳ Xoá dòng trùng |
| D4 | DoD checklist thiếu commit M2 `75455dd` | Thấp | ⏳ Bổ sung |
| D5 | `Architecture Snapshot` mô tả `src/training`, `src/evaluation` như đã có — thực tế còn rỗng | Trung bình | ⏳ Ghi rõ trạng thái M3/M4 |
| D6 | `progress_status.md` trích dẫn `checklists/dod.md` nhưng file **không tồn tại** (link chết) | Trung bình | ⏳ Tạo `checklists/dod.md` hoặc sửa trích dẫn |
| D7 | `progress_status.md` + `AI_Usage.md` nằm trong `docs/` thay vì root như template skill | Thấp | ⏳ Quyết định: giữ hay chuyển (đụng git history) |
| D8 | Thiếu `notebooks/` demo, `docs/AUDIT_LOG.md` (file này), báo cáo tổng hợp | Cao | ✅ Đang xử lý (commit này) |

### Kết luận audit

- Code M2: 4 lỗi logic đều đã bắt và sửa **trước khi commit** nhờ verify bằng chạy thật (không chỉ đọc). Test 23/23 pass.
- Docs: 8 vấn đề, trong đó D1 (số bịa) là nghiêm trọng nhất — xử lý trong commit này.
- Rủi ro còn lại: M3/M4 chưa build nên chưa thể audit training/eval; audit tiếp sau khi có `src/training/trainer.py`.

---

## 2026-10-25 — Audit M1 (data pipeline) + align docs với đề bài

**Người audit:** AI agent + user review.
**Phạm vi:** `src/data/*.py`, `scripts/verify_data.py`, `README.md`, `docs/progress_status.md`.
**Commit liên quan:** `a1a6b71` (docs align), `28aa1ff` (prepare step).

### Phát hiện đã xử lý

- `src/data/dataloader.py` (bản đầu): gán `val_subset.dataset.transform` làm hỏng transform của cả tập train (bug `Subset` giữ tham chiếu dataset gốc) → viết `TransformSubset` riêng. (Chi tiết: `docs/AI_Usage.md` §6)
- `src/utils/runs.py` (bản đầu): `UnicodeEncodeError` trên console Windows cp1252 → ép UTF-8 cho stdout/stderr + file handler.
- `scripts/verify_data.py` (bản đầu): claim "có kiểm tra MD5" nhưng không kiểm tra → bổ sung `_verify_archive` (chunk hash, so với `c58f30108f718f92721af3b95e74349a`, PASS thật).
- `src/data/prepare.py` (bản đầu): so overlap bằng giao chỉ số train/test (hai không gian khác nhau → báo động giả) → chuyển sang hash nội dung ảnh (overlap = 0 cả 3 cặp).
- `src/data/prepare.py` (bản đầu): resize 224 trước khi lưu (cache 7GB, phá `RandomCrop`) → giữ ảnh gốc 32×32.
- Docs: shape sai `[B,3,32,32]` → `[B,3,224,224]`; số liệu bịa trong Example → placeholder có nhãn; lệnh training chưa tồn tại → gắn nhãn "thiết kế dự kiến"; `docs/architecture.md` không tồn tại → đổi sang `data-flow.md`.

### Kết luận audit

- Test M1: 47/47 pass. Run `20261025-150847`: 45k/5k/10k, `validation.status = PASS`.
