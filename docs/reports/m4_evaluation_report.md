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

## 4. Kết quả eval thật (Tier B, ResNet18 freeze 64×64 → test 10.000 mẫu)

`results/runs/20261029-212336/resnet18-lr0.001-bs32-0/eval/metrics.json`:

| Metric | Giá trị |
|--------|---------|
| Test accuracy | **0.5582** |
| Macro-F1 | **0.5535** |
| Inference | 4.31 ms/mẫu (CPU) |
| Best val (train) | 0.5654 @ epoch 4/5 |

Per-class F1 thấp nhất: cat (lớp 3) 0.442, deer (lớp 4) 0.424 — đúng cặp dễ nhầm cổ điển của CIFAR-10; cao nhất: automobile (lớp 1) 0.667, ship (lớp 8) 0.622. Confusion matrix tổng = 10.000 = số mẫu test ✓.

Plots: `eval/confusion_matrix.png`, `eval/per_class_f1.png`.

## 5. Kết quả eval FT (Tier B fine-tune — test 10.000 mẫu)

| Model | Test acc | Macro-F1 | Best val | Nguồn |
|-------|----------|----------|----------|-------|
| ResNet18 FT | **0.8273** ✅ | 0.8258 | 0.8296 @ ep5 | `20261029-220833/resnet18-*/eval/metrics.json` |
| VGG11 FT | **0.8331** ✅ (tốt nhất) | 0.8305 | 0.8316 @ ep5 | `20261029-220833/vgg11-*/eval/metrics.json` |
