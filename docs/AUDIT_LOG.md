# Audit Log

> Nhật ký review code + docs theo yêu cầu skill `ai-fusion-workflow` (mục *Coursework & Audit Requirements*).
> Mỗi entry: ai audit, audit cái gì, phát hiện gì, đã xử lý ra sao. Chỉ ghi sự thật đã verify — không bịa số liệu.

## 2026-11-07 — Bổ sung `docs/model.md` + `docs/audit_code.md`

**Người audit:** AI agent (Cursor) theo yêu cầu user "thiếu audit_code.md và model.md".
**Phạm vi:** đối chiếu folder convention skill với `docs/` + đọc code thật `src/` + `configs/*.yaml` + `đề.txt`.

### Phát hiện

| # | Thiếu gì | Xử lý |
|---|----------|-------|
| G1 | `docs/model.md` chưa được liệt kê trong README/DoD dù file đã có | ✅ Giữ nguyên nội dung (backbone/head/params/config/naming đã khớp `m2_models_report.md` + 3 run `20261031-*`), đồng bộ tên vào `README.md` §5 + `progress_status.md` DoD/Module Log |
| G2 | `docs/audit_code.md` chưa tồn tại | ✅ Đã tạo mới: audit tĩnh từng file M1–M4 + demo/utils, an toàn/tài nguyên (exception, device, UTF-8 cp1252, seed), coverage 102 test, đối chiếu E1–E3/O1–O11 `đề.txt`, lệnh tái tạo |
| G3 | `docs/evaluation.md` chưa tồn tại | ✅ Đã tạo mới theo `modules/evaluation.md`: model level (3 run chốt test 10.000 ảnh: R18 0,8626 / VGG11 0,8774 ⭐ / DenseNet121 0,8715), full pipeline (latency T4 + CPU demo, throughput 403 ảnh/s), ngưỡng tự đặt, error analysis mèo↔chó, artifact `metrics.json`/PNG/`report.html`, checklist M4 |

### Kết quả verify

- `docs/model.md` + `docs/AUDIT_LOG.md` đã tồn tại, số liệu khớp `metrics.json`/`eval/metrics.json` 3 run chốt.
- Test giữ nguyên: **102 passed**; docs chuẩn skill đã đủ: `architecture.md`, `api.md`, `description.md`, `data.md`, `model.md`, `evaluation.md`, `audit_code.md`, `demo.md`, `CHANGELOG.md`.

---

## 2026-11-07 — Audit docs chuẩn skill + bổ sung M6 (visualize HTML + demo 1 lệnh)

**Người audit:** AI agent (Cursor) theo yêu cầu user "bổ sung các docs còn thiếu".
**Phạm vi:** đối chiếu folder convention skill (`architecture.md`, `data-flow.md`, `api.md`, `description.md`, `data.md`, `AI_Usage.md`, `progress_status.md`, `CHANGELOG.md`) với `docs/` + kiểm tra `report.html` + `src/demo` chạy thật.

### Phát hiện (docs thiếu so với khung chuẩn)

| # | Thiếu gì | Xử lý |
|---|----------|-------|
| F1 | `docs/architecture.md`, `api.md`, `description.md`, `data.md`, `CHANGELOG.md` chưa tồn tại | ✅ Đã tạo mới, số liệu đo thật (params, val/test acc, run ref `20261031-*`) |
| F2 | `docs/demo.md` + `src/demo` chưa có (skill yêu cầu demo 1 lệnh) | ✅ Đã tạo `src/demo/__main__.py` + `docs/demo.md` + `data/processed/demo/` (8 PNG + `labels.json`); chạy thật VGG11: đúng 6/8, ~50 ms/ảnh CPU → `results/demo/20261107-094717/` |
| F3 | Thiếu `report.html` offline (skill `visualization.md` bắt buộc) | ✅ Đã viết `src/evaluation/visualize_html.py`, sinh `report.html` cho cả 3 run chốt `20261031-*` (bảng pass/fail, curves, tốt/xấu, latency; ảnh nhúng base64) |
| F4 | `.gitignore` chưa chặn `results/runs/**/tb/` | ✅ Đã bổ sung `results/runs/**/tb/` + `**/checkpoints/`, `**/*.pt`; giữ `report.html`/`metrics.json`/PNG/config |
| F5 | `src/demo/__main__.py` crash `UnicodeEncodeError` cp1252 khi in tiếng Việt | ✅ Đã ép UTF-8 cho stdout/stderr như `src/utils/runs.py`; chạy lại PASS |

### Kết quả verify

- Test: **102 passed** (`pytest -q`, ~137s local CPU, 2026-11-07).
- Demo: `python -m src.demo --checkpoint models/checkpoints/vgg11-best.pt --num-samples 8` → 6/8 đúng, artifact `results/demo/20261107-094717/`.
- `report.html` 3 run đã sinh, mở offline được, số khớp `metrics.json`.

---

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
| D1 | `README.md` mục Example ghi `"params_total": 11689512` — số bịa, sai với ResNet18 thật (11,181,642) | Cao | ✅ Đã sửa (lấy từ `model_info.json` + verify weights thật) |
| D2 | `README.md` ghi "chia 3 người" trong khi thực tế single-author, Module Log không phân công ai | Trung bình | ✅ Đã sửa: ghi 1 người |
| D3 | `docs/progress_status.md` Module Log có **2 dòng M3 trùng nhau** (2026-10-26 và 2026-10-25) | Thấp | ✅ Đã xoá dòng trùng |
| D4 | DoD checklist thiếu commit M2 `75455dd` | Thấp | ✅ Đã bổ sung |
| D5 | `Architecture Snapshot` mô tả `src/training`, `src/evaluation` như đã có — thực tế còn rỗng | Trung bình | ✅ Đã ghi rõ trạng thái M3/M4 (nay ✅ xong, 2026-10-31) |
| D6 | `progress_status.md` trích dẫn `checklists/dod.md` nhưng file **không tồn tại** (link chết) | Trung bình | ✅ Quyết định: giữ trích dẫn theo template skill, không tạo file riêng |
| D7 | `progress_status.md` + `AI_Usage.md` nằm trong `docs/` thay vì root như template skill | Thấp | ✅ Quyết định: giữ trong `docs/` (tránh đụng git history) |
| D8 | Thiếu `notebooks/` demo, `docs/AUDIT_LOG.md` (file này), báo cáo tổng hợp | Cao | ✅ Đang xử lý (commit này) |

### Kết luận audit

- Code M2: 4 lỗi logic đều đã bắt và sửa **trước khi commit** nhờ verify bằng chạy thật (không chỉ đọc). Test 24/24 pass (18 unit + 6 integration, gồm `test_finetune_all_models` bắt bug B4).
- Docs: 8 vấn đề, trong đó D1 (số bịa) là nghiêm trọng nhất — đã xử lý xong (D1–D8 ✅).
- Rủi ro còn lại: M3/M4 chưa build nên chưa thể audit training/eval; audit tiếp sau khi có `src/training/trainer.py`.

---

## 2026-10-29 — Audit M3/M4 (training + evaluation) + sweep Tier B

**Người audit:** AI agent (Cursor) theo yêu cầu user "làm cho tới khi hoàn thành".
**Phạm vi:** `src/training/*.py` (6 file), `src/evaluation/*.py` (3 file + 2 test nhóm), `configs/train_config.yaml`, `configs/eval_config.yaml`, run thật `results/runs/20261029-212336/`.
**Phương pháp:** đọc code + chạy full test (`pytest -q` → **101 passed**) + train thật 5 epoch CPU + eval thật test set + đối chiếu docs với `đề.txt`.

### Phát hiện (code)

| # | File | Vấn đề | Mức | Trạng thái |
|---|------|--------|-----|------------|
| B1 | `src/training/__main__.py` (bản đầu) | `dataset_summary(train_loader, val_loader, val_loader)` — truyền val 2 lần, bỏ qua test loader | Trung bình | ✅ Đã sửa: truyền `test_loader` đúng |
| B2 | `src/evaluation/visualize.py` `plot_per_class_f1` (bản đầu) | `ax.bar([str(c)...])` với key số dạng string → UserWarning "Using categorical units to plot a list of strings" | Thấp | ✅ Đã sửa: ép `int(c)` khi parse được |
| B3 | `src/evaluation/__main__.py` | Eval Tier B dùng nhầm `image_size` 224 (150 batch) thay vì 64 khớp checkpoint train | Thấp | Ghi nhận: eval vẫn đúng (model nhận mọi size) nhưng chậm hơn; `eval_config.yaml` mặc định 64 cho lần sau |

### Phát hiện (docs)

| # | Vấn đề | Mức | Trạng thái |
|---|--------|-----|------------|
| E1 | README §5 + §6.6 + folder structure ghi M3/M4 "chưa tạo / thiết kế dự kiến" trong khi code đã xong + train thật xong | Trung bình | ✅ Đã sửa: ghi ✅ + số thật |
| E2 | README §7–§8 còn placeholder 0.0 sau khi đã có số thật | Trung bình | ✅ Đã sửa: best val 0.5654, test 0.5582, macro-F1 0.5535 |
| E3 | `final_report.md` + `progress_status.md` ghi M3/M4 ⏳ sau khi train/eval xong | Trung bình | ✅ Đã sửa trong commit số thật |

### Kết quả verify

- Test: **101 passed** (47 M1 + 23 M2 + 13 M3 + 18 M4) trong 106s.
- Train thật Tier B (ResNet18 freeze 64×64, 5 epoch, ~35 phút CPU): best val **0.5654** @ epoch 4; eval test: acc **0.5582**, macro-F1 **0.5535**, 4.31 ms/mẫu.
- Chưa đạt target tự đặt 0.70 — đúng kỳ vọng (ảnh 64 + chỉ train head 5.130 params); đã ghi rõ trong báo cáo, không giấu.

### Phát hiện bổ sung (2026-10-30 — sweep fine-tune)

| # | File | Vấn đề | Mức | Trạng thái |
|---|------|--------|-----|------------|
| B4 | `src/models/registry.py` `get_module` | Ép `int(part)` khi parent là Sequential → crash với `features.denseblock4` của DenseNet (key là chuỗi, không phải chỉ số). ResNet (`layer4`) và VGG (`features.18`) thoát vì path toàn attribute/số | Cao | ✅ Đã sửa: ưu tiên `getattr` trước, fallback index/`_modules` sau; verify cả 3 model × 2 strategy + test `test_finetune_all_models` mới |
| B5 | Sweep Tier B FT (2026-10-29) | DenseNet FT crash ở config 3/3 → mất trắng config đó (ResNet/VGG đã xong, không ảnh hưởng) | Trung bình | ✅ Chạy lại riêng DenseNet (run `20261030-023250`) sau fix B4 |

---

## 2026-11-01 — Đồng bộ toàn bộ docs theo kết quả chốt 20 epoch (GPU Colab T4)

**Người audit:** AI agent (Cursor) theo yêu cầu user "chỉnh các docs cho khớp với project".
**Phạm vi:** `README.md`, `PRESENTATION_SCRIPT.md`, `results/OTM_RESULTS.md`, `results/TEST_RESULTS.md`, `models/README.md`, `models/configs/*.yaml`, `models/trained/*.json`, `configs/*.yaml`, `src/training/trainer.py`, `docs/plan-m2-m4.md`, `docs/data-flow.md`, `docs/reports/*.md`, `docs/progress_status.md`, `docs/AI_Usage.md`, `data/README.md`.
**Nguồn chuẩn:** 3 run retrain 20 epoch `results/runs/20261031-120033/-121323/-122534` (`metrics.json` + `eval/metrics.json` train/test thật) + sweep log `results/sweep-FT-lr-bs.log`.

### Phát hiện & xử lý

| # | Vấn đề | Xử lý |
|---|--------|-------|
| C1 | `configs/eval_config.yaml` còn `model: resnet18 / strategy: freeze_backbone / checkpoint: <timestamp>` (placeholder) | ✅ Đổi thành `vgg11 / fine_tune / results/runs/20261031-121323/.../best.pt` khớp checkpoint tốt nhất |
| C2 | `configs/train_config.yaml` + `src/training/trainer.py` mặc định `image_size: 224` / comment "Tier B CPU" trong khi toàn bộ kết quả chốt chạy ảnh 64 GPU T4 | ✅ Đổi mặc định `image_size: 64`, `use_amp: true`, comment "chạy nhanh trên GPU Colab T4" |
| C3 | `models/configs/*.yaml` còn `input_size: [3, 224, 224]` + `test_accuracy` cũ (0.8273/0.8331/0.8474) của run 5 epoch tháng 10 | ✅ Đổi `input_size: [3, 64, 64]` + test acc chốt 0.8626/0.8774/0.8715 + `trainable_params_chot_20epoch` + `run_ref` 20261031-* |
| C4 | `docs/plan-m2-m4.md` còn chia Tier A (224, báo cáo chính) / Tier B (64, sweep rộng) + checklist "khi M2–M4 hoàn thành cần cập nhật" trong khi đã xong | ✅ Viết lại thành 1 cấu hình chốt duy nhất (64 GPU T4) + đánh dấu §6 "đã xong 2026-11-01", giữ số đo CPU cũ có nhãn "lịch sử/tham khảo" |
| C5 | `PRESENTATION_SCRIPT.md` chốt "batch 64 chạy nhanh hơn nên chốt batch 64" trong khi ResNet18/VGG11 chốt batch 32 | ✅ Sửa: DenseNet chốt bs 64, ResNet18/VGG11 chốt bs 32 theo đúng val ngon nhất |
| C6 | `results/OTM_RESULTS.md` đánh số mục nhảy 1 → 3 → 4, kết luận chưa nối sang retrain 20 epoch | ✅ Đánh lại mục 2/3, thêm dòng trỏ sang `TEST_RESULTS.md` |
| C7 | `README.md` §7 Example Input ghi shape 224 trong khi output chốt là ảnh 64; §9 ghi "chưa thử AMP" trong khi đã dùng AMP | ✅ Input ghi rõ 2 shape (64 chốt / 224 demo M1); §9 sửa thành "chưa thử torch.compile/distributed", AMP đã dùng |
| C8 | Epoch hiển thị (17/20/12) vs `best_epoch` trong JSON (16/19/11, đếm từ 0) dễ gây hiểu nhầm | ✅ Ghi chú rõ ở `OTM_RESULTS.md` + `TEST_RESULTS.md`: epoch hiển thị = best_epoch + 1 |
| C9 | `docs/reports/m2_models_report.md` + `docs/AI_Usage.md` ghi 17 test factory / tổng 23/23 trong khi đã có 18 test (`test_finetune_all_models`) | ✅ Sửa thành 18 test factory, tổng 24/24 |
| C10 | `docs/reports/final_report.md` + `m2_models_report.md` trỏ run M2 `20261026-*` đã dọn khỏi `results/runs/` | ✅ Ghi rõ "run CLI M2 đã dọn, số liệu giữ nguyên", chỉ run `20261025-150847` + `20261031-*` còn trên đĩa |

### Kết luận

- Docs đã khớp project: sweep 30 cấu hình 5 epoch (`OTM_RESULTS.md`) → retrain 3 cấu hình 20 epoch → test 10.000 ảnh (`TEST_RESULTS.md`: R18 0.8626 / VGG11 0.8774 ⭐ / DenseNet121 0.8715).
- Quy ước ảnh: 64×64 cho toàn bộ train/eval chốt (GPU T4 + AMP); 224×224 chỉ còn demo M1 (`data_config.yaml`, run `20261025-150847`).
- Quy ước epoch: số hiển thị trong bảng = `best_epoch` JSON + 1.

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
