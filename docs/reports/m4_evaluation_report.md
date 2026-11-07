# Báo cáo M4 — Evaluation & metrics

> Nguồn: `đề.txt` (O11 — Evaluate the Model on test dataset). Mọi con số có nguồn artifact đã push GitHub (trừ `*.pt` nặng — local only).

## 1. Yêu cầu đề bài được phủ

| # | Yêu cầu | Hiện thực |
|---|---------|-----------|
| O11 | Evaluate on test dataset | `evaluate_model(model, loader, device="cpu")` → `EvalResult` (accuracy, macro_f1, per-class, confusion_matrix, inference ms/mẫu, n_samples) + `to_json_dict()` đúng schema `metrics.json`; CLI `python -m src.evaluation --checkpoint <best.pt>` |

## 2. Files

| File | Chức năng |
|------|-----------|
| `src/evaluation/metrics.py` | `accuracy`, `per_class_metrics`, `macro_f1`, `confusion_matrix`, `classification_report` (có dòng `accuracy` + `macro avg`), `EvalResult`, `evaluate_model` |
| `src/evaluation/visualize.py` | `plot_confusion_matrix`, `plot_per_class_f1` (cả 2 nhận `out_path`), `plot_pr_curve`, `plot_confusion_examples`, `plot_train_curves` |
| `src/evaluation/__main__.py` | CLI eval: load checkpoint → test loader → `metrics.json` + PNG |
| `configs/eval_config.yaml` | checkpoint, model, batch, image_size, save_plots (mặc định ảnh 64 khớp checkpoint train chốt) |

## 3. Test

| Suite | Kết quả |
|-------|---------|
| `src/evaluation/test_metrics.py` (13 test) | acc/macro-F1/confusion/per-class/report — PASS |
| `src/evaluation/test_evaluation.py` (5 test) | end-to-end, schema JSON, plots, checkpoint reload, tổng CM = n_samples — PASS |

## 4. Kết quả eval chốt (3 cấu hình tốt nhất retrain 20 epoch, GPU Colab T4, ảnh 64, Adam, fine-tune → test 10.000 mẫu)

Chi tiết: `results/TEST_RESULTS.md`.

| Model | Cấu hình | Test acc | Macro-F1 | Best val | Nguồn |
|-------|----------|----------|----------|----------|-------|
| ResNet18 FT | Adam, lr 0.0001, bs 32, 20 epoch | **0.8626** | 0.8626 | 0.8634 @ ep 17 | `20261031-120033/resnet18-lr0.0001-bs32-0/eval/metrics.json` |
| VGG11 FT ⭐ | Adam, lr 0.0001, bs 32, 20 epoch | **0.8774** | 0.8765 | 0.8786 @ ep 20 | `20261031-121323/vgg11-lr0.0001-bs32-0/eval/metrics.json` |
| DenseNet121 FT | Adam, lr 0.0005, bs 64, 20 epoch | **0.8715** | 0.8709 | 0.8800 @ ep 12 | `20261031-122534/densenet121-lr0.0005-bs64-0/eval/metrics.json` |

Ngon nhất test là VGG11 với 87,74%.

Per-class F1 của VGG11: thấp nhất cat (lớp 3) 0.7629, cao nhất tàu (lớp 8) 0.9347 / ô tô (lớp 1) 0.9343. Confusion matrix tổng = 10.000 = số mẫu test ✓. Nhầm nhiều nhất: mèo → chó 109 tấm, chó → mèo 106 tấm.

Plots: `eval/confusion_matrix.png`, `eval/per_class_f1.png` trong từng run `20261031-*/<cfg>/eval/`. Báo cáo HTML offline: `report.html` trong từng run (sinh bằng `src/evaluation/visualize_html.py`, mở double-click, số khớp `metrics.json`).
