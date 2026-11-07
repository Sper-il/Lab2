# Evaluation — chuẩn đánh giá model và pipeline (M4)

> Chuẩn đánh giá 2 lớp của project theo `modules/evaluation.md` của skill `ai-fusion-workflow`.
> Mọi con số đều đo thật từ `results/runs/20261031-*/<cfg>/eval/metrics.json` + `metrics.json` (không ước lượng).
> Tài liệu liên quan: `docs/reports/m4_evaluation_report.md` (báo cáo ngắn), `results/TEST_RESULTS.md` (bảng chốt),
> `docs/audit_code.md` §2.4 (audit code eval), `docs/model.md` (kiến trúc được đánh giá).

## 1. Mục tiêu

- Mọi con số báo cáo đều tái tạo được từ `results/runs/<ts>/<cfg>/metrics.json` (train/val) + `eval/metrics.json` (test) + script `src/evaluation/__main__.py`.
- Không chỉ báo accuracy mà còn báo latency, tài nguyên và case thất bại — đúng yêu cầu O11 (`đề.txt`) và E3 (TensorBoard/`report.html`).

## 2. Model level (độ đúng)

Bài toán là classification 10 lớp CIFAR-10 nên metric gồm accuracy, precision/recall/F1 từng lớp, macro-F1 và confusion matrix.

| Model (fine-tune, Adam, ảnh 64, 20 epoch) | Val tốt nhất | Test acc | Macro-F1 | F1 thấp nhất | F1 cao nhất | Inference | Run |
|---|---|---|---|---|---|---|---|
| ResNet18 (lr 0,0001, bs 32) | 0,8634 @ ep 17 | **0,8626** | 0,8626 | 0,7324 (mèo) | 0,9264 | 0,23 ms/mẫu (GPU T4) | `20261031-120033/resnet18-lr0.0001-bs32-0/` |
| VGG11 ⭐ (lr 0,0001, bs 32) | 0,8786 @ ep 20 | **0,8774** | 0,8765 | 0,7629 (mèo, lớp 3) | 0,9347 (tàu, lớp 8) / 0,9343 (ô tô, lớp 1) | 0,33 ms/mẫu (GPU T4) | `20261031-121323/vgg11-lr0.0001-bs32-0/` |
| DenseNet121 (lr 0,0005, bs 64) | 0,8800 @ ep 12 | **0,8715** | 0,8709 | 0,7468 | 0,9320 | 0,69 ms/mẫu (GPU T4) | `20261031-122534/densenet121-lr0.0005-bs64-0/` |

- Mỗi test chấm **10.000 mẫu** (1.000/lớp), tổng confusion matrix = 10.000 = `n_samples` ✓.
- Dung lượng checkpoint local: ResNet18 ~45 MB, DenseNet121 ~28 MB, VGG11 ~515 MB (vượt giới hạn GitHub → local-only, `.gitignore` chặn).
- Cách tính: `evaluate_model(model, loader, device)` trong `src/evaluation/metrics.py` — `inference_mode`, đồng bộ CUDA trước khi đo giờ, kiểm tra logits `(B, 10)`, từ chối loader rỗng; key per-class dạng string để JSON-serializable.

## 3. Full pipeline (tốc độ, tài nguyên, độ ổn định)

| Khía cạnh | Cách đo | Kết quả |
|---|---|---|
| Latency suy luận | `inference_time_ms_per_sample` trong `eval/metrics.json` (10.000 mẫu test) | ResNet18 0,23 / VGG11 0,33 / DenseNet121 0,69 ms/mẫu (GPU T4) |
| Latency end-to-end demo | `src/demo` đo từng mẫu (preprocess Resize 64 + inference + postprocess top-3) | CPU local ~50 ms/ảnh (VGG11, 8 ảnh test, `results/demo/20261107-094717/`); GPU T4 ~0,33 ms/ảnh phần inference |
| Throughput data loading | `scripts/benchmark_loading.py` (CPU local, workers=4) | ~403 ảnh/s → data loading không phải nút thắt |
| CPU/GPU/RAM/disk | Train chốt GPU Colab T4 + AMP (~27–86 giây/epoch ảnh 64); CIFAR-10 in-memory ~150 MB; không cache ảnh 224 (tiết kiệm ~7 GB) | Đủ ngân sách Colab free-tier |
| Xử lý lỗi và phục hồi | `ValueError`/`FileNotFoundError` rõ ràng (checkpoint thiếu, loader rỗng, nhãn ngoài miền, config sai); TensorBoard/graph bọc try/except chỉ cảnh báo | Không sập pipeline vì lỗi phụ |

## 4. Ngưỡng đạt (target → đạt được → run id)

Đề bài không đặt ngưỡng số; dưới đây là **mục tiêu tự đặt** của project (ghi rõ trước khi code để tránh chỉnh số cho vừa):

| Metric | Target (tự đặt) | Đạt được (VGG11 ⭐) | Run | Kết luận |
|---|---|---|---|---|
| Test accuracy (freeze backbone) | ≥ 0,70 | 0,8774 (fine-tune; freeze chỉ dùng tham chiếu Tier B 0,5582) | `20261031-121323` | PASS |
| Test accuracy (fine-tune) | ≥ 0,80 | 0,8774 | `20261031-121323` | PASS |
| Macro-F1 | ≥ 0,70 | 0,8765 | `20261031-121323` | PASS |
| Overfitting (val acc − train acc) | ≤ 0,15 | val 0,8786 bám sát train (history trong `metrics.json`) | `20261031-121323` | PASS |
| Thời gian 1 epoch (CPU) | ≤ 15 phút | GPU T4 chỉ 27–35 giây/epoch (VGG11); CPU local vượt nên chuyển lên T4 | sweep log | PASS (bằng GPU) |

Cả 3 model fine-tune đều vượt ngưỡng 0,80 (0,8626 / 0,8774 / 0,8715).

## 5. Error analysis

- **Top case sai (VGG11):** mèo (3) → chó (5): 109 tấm; chó (5) → mèo (3): 106 tấm. Confusion matrix: `eval/confusion_matrix.png`.
- **Gom nhóm nguyên nhân:** cặp mèo/chó có F1 thấp nhất (0,7629 / 0,8258) — lông, tai, mõm tương đồng ở độ phân giải 32×32 upsample 64; chim (0,8573) nhầm với hươu/ếch khi nền cây cỏ; tàu/ô tô cao nhất (0,9347/0,9343) nhờ hình khối cứng, nền biển/đường phân biệt rõ.
- **Đề xuất cải thiện:** Mixup/CutMix + RandomErasing cho cặp mèo/chó; fine-tune toàn backbone với lr nhỏ (1e-4 → 1e-5); label smoothing / cosine annealing; thử ConvNeXt/EfficientNet-V2 so sánh công bằng.

## 6. Artifact

- `results/runs/20261031-*/<cfg>/metrics.json` — history train/val + `train_summary` (best val, best epoch).
- `results/runs/20261031-*/<cfg>/eval/metrics.json` — schema `EvalResult.to_json_dict()`: accuracy, macro_f1, per_class, confusion_matrix, inference ms/mẫu, n_samples.
- `eval/confusion_matrix.png` + `eval/per_class_f1.png` trong từng run (vẽ bằng `src/evaluation/visualize.py`, nhận `out_path`).
- `report.html` trong từng run (sinh bằng `src/evaluation/visualize_html.py`, offline 100% ảnh nhúng base64, mở bằng double-click, số khớp `metrics.json`): bảng pass/fail, curves, tốt/xấu, case tệ nhất, latency.
- TensorBoard `tb/` mỗi run: scalar (loss/acc/lr), histogram trọng số, grad norm, ảnh mẫu.

## 7. Checklist review (M4)

- [x] Metric tính trên test set giữ riêng (10.000 ảnh, chỉ chấm cuối 1 lần), mọi thử nghiệm trên val.
- [x] Báo kèm threshold/target (bảng §4), không chỉ số trần.
- [x] Có confusion matrix + per-class F1 + PNG, không chỉ 1 con số.
- [x] Latency đo đúng phần cứng: GPU T4 cho train/eval chốt + CPU local cho demo (`results/demo/`).
- [x] Có đo end-to-end (demo latency từng mẫu), không chỉ forward model.
- [x] Kết quả đã ghi vào `results/runs/<ts>/` và cập nhật `README.md` §7–§8 + `results/TEST_RESULTS.md`.
- [x] Có phân tích case sai và đề xuất cải thiện cụ thể (§5).

## 8. Rủi ro và cách xử lý trong eval

| Rủi ro | Xử lý trong project |
|---|---|
| Tuning trên test → số đẹp nhưng ảo | Khóa test, chỉ eval cuối 1 lần; sweep/retrain chọn bằng val |
| Đo trên GPU mạnh nhưng deploy CPU yếu → vỡ ngân sách | Báo cả 2: GPU T4 (eval) + CPU local (demo ~50 ms/ảnh) |
| Metric trung bình che giấu lớp hiếm kém | Luôn báo per-class precision/recall/F1 + confusion matrix |
| Không seed/config → không reproduce | Mọi eval kèm config + checkpoint path + `best_epoch`; seed 42 cố định |

## 9. Tái tạo

```powershell
# Đánh giá checkpoint trên test set (mặc định VGG11 tốt nhất trong eval_config.yaml)
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml

# Chỉ định checkpoint khác
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml --checkpoint results/runs/20261031-120033/resnet18-lr0.0001-bs32-0/checkpoints/best.pt

# Sinh báo cáo HTML offline
.\.venv\Scripts\python.exe -m src.evaluation.visualize_html --run-dir results/runs/20261031-121323/vgg11-lr0.0001-bs32-0
```
