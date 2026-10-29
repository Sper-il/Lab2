# Báo cáo M4 — Evaluation & metrics

> Nguồn: `đề.txt` (O11 — Evaluate the Model on test dataset). Mọi con số có nguồn artifact; chỗ nào train chưa xong ghi rõ "chờ checkpoint".

## 1. Yêu cầu đề bài được phủ

| # | Yêu cầu | Hiện thực |
|---|---------|-----------|
| O11 | Evaluate on test dataset | `evaluate_model(model, loader, device="cpu")` → `EvalResult` (accuracy, macro_f1, per_class, confusion_matrix, inference ms/mẫu, n_samples) + `to_json_dict()` đúng schema `metrics.json`; CLI `python -m src.evaluation --checkpoint <best.pt>` |

## 2. Files

| File | Chức năng |
|------|-----------|
| `src/evaluation/metrics.py` | `accuracy`, `per_class_metrics`, `macro_f1`, `confusion_matrix`, `classification_report` (có dòng `accuracy` + `macro avg`), `EvalResult`, `evaluate_model` |
| `src/evaluation/visualize.py` | `plot_confusion_matrix`, `plot_per_class_f1` (cả 2 nhận `out_path`), `plot_pr_curve`, `plot_confusion_examples`, `plot_train_curves` |
| `src/evaluation/__main__.py` | CLI eval: load checkpoint → test loader → `metrics.json` + PNG |
| `configs/eval_config.yaml` | checkpoint, model, batch, image_size, save_plots |

## 3. Test

| Suite | Kết quả |
|-------|---------|
| `src/evaluation/test_metrics.py` (13 test) | acc/macro-F1/confusion/per-class/report — PASS |
| `src/evaluation/test_evaluation.py` (5 test) | end-to-end, schema JSON, plots, checkpoint reload, tổng CM = n_samples — PASS |

## 4. Kết quả eval thật

Chờ train Tier B xong (`results/runs/20261029-212336/.../checkpoints/best.pt`) rồi chạy:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml --checkpoint results/runs/20261029-212336/resnet18-lr0.001-bs32-0/checkpoints/best.pt
```

Kết quả sẽ điền vào bảng dưới + `final_report.md` + README §8 sau khi chạy xong.
