# Báo cáo M3 — Training loop + TensorBoard

> Nguồn: `đề.txt` (E2, E3, O9, O10). Mọi con số có nguồn artifact; sweep 30 cấu hình 5 epoch chọn ra 3 cấu hình tốt nhất rồi retrain 20 epoch trên GPU Colab T4.

## 1. Yêu cầu đề bài được phủ

| # | Yêu cầu | Hiện thực |
|---|---------|-----------|
| E2 | Adjust hyperparameters | `configs/train_config.yaml` + CLI `--models/--lr/--batch-size/--epochs/--image-size/--strategy/--optimizer/--scheduler` (grid models×lr×bs trong `__main__.py`) + `notebooks/colab_sweep.ipynb` (sweep 30 cấu hình 5 epoch) + `notebooks/colab_best3_retrain.ipynb` (retrain 20 epoch) |
| E3 | TensorBoard | `SummaryWriter` tại `<run>/tb/`: scalar train/val loss+acc, lr, epoch time; grad norm; histogram trọng số (epoch đầu/cuối); ảnh mẫu denormalize; model graph (tắt mặc định) |
| O9 | Loss + optimizer | `build_criterion` (cross_entropy/label_smoothing/focal) + `build_optimizer` (adam/adamw/sgd, chỉ params `requires_grad=True`). Kết quả chốt chỉ xài **Adam** |
| O10 | Training loop | `Trainer.train_one_epoch`: `zero_grad → forward → loss → backward → step` (bám slide 8–9 Chương 3); `validate()` với `torch.no_grad()`; checkpoint `best.pt` theo val acc; early stopping theo patience |

## 2. Files

| File | Chức năng |
|------|-----------|
| `src/training/loss.py` | `build_criterion` + `FocalLoss` |
| `src/training/optimizer.py` | `build_optimizer`, `trainable_params` |
| `src/training/scheduler.py` | `build_scheduler` (none/cosine/step/plateau) |
| `src/training/tensorboard_utils.py` | `create_writer`, `log_scalars`, `log_weight_histograms`, `log_grad_norms`, `log_images`, `log_graph`, `tb_config` |
| `src/training/trainer.py` | `TrainConfig`, `Trainer` (train/val/checkpoint/history). Vòng train 6 bước và vòng validate đã note tiếng Việt trong code (khớp `PRESENTATION_SCRIPT.md` dòng 66–68) |
| `src/training/__main__.py` | CLI train + sweep, ghi `metrics.json` + `config.yaml` + `logs.txt` + `tb/` |
| `configs/train_config.yaml` | Mặc định ảnh 64×64, 5 epoch, fine-tune, Adam (`image_size: 64`, `use_amp: true` để chạy nhanh trên GPU) |
| `notebooks/colab_sweep.ipynb` | Sweep 30 cấu hình fine-tune Adam 5 epoch trên GPU Colab T4 (lr 0.0001/0.0005/0.001/0.003/0.005 × batch 32/64, workers = 2, AMP) |
| `notebooks/colab_best3_retrain.ipynb` | Retrain 3 cấu hình tốt nhất **20 epoch** trên GPU Colab T4 + chấm test (`confusion_matrix.png`, `per_class_f1.png`) |

## 3. Test

| Suite | Kết quả |
|-------|---------|
| `tests/unit/test_training_components.py` (9 test) | loss variants, optimizer chỉ trainable, scheduler decay, TB tags+ghi file — PASS |
| `tests/integration/test_training_loop.py` (4 test) | 1 epoch đổi weights, grad head, checkpoint reload, early stopping — PASS (FakeData, <10s) |

## 4. Sweep 30 cấu hình 5 epoch (GPU Colab T4, ảnh 64, Adam, fine-tune)

Chi tiết từng epoch: `results/sweep-FT-lr-bs.log`. Tổng hợp: `results/OTM_RESULTS.md`.

| Model | Cấu hình ngon nhất sweep | Val tốt nhất |
|-------|--------------------------|--------------|
| ResNet18 | lr 0.0001, batch 32 | 0.8456 @ ep 5 |
| VGG11 | lr 0.0001, batch 32 | 0.8538 @ ep 5 |
| DenseNet121 | lr 0.0005, batch 32/64 | 0.8618 @ ep 5 |

Kết luận sweep: lr nhỏ 0.0001–0.0005 hợp nhất; lr ≥ 0.003 tụt mạnh (rõ nhất VGG11: 0.8538 → 0.7708); batch 32 vs 64 chênh ~0.01, batch 64 chạy nhanh hơn.

## 5. Retrain 3 cấu hình tốt nhất 20 epoch (GPU Colab T4, AMP)

| Config | Kết quả 20 epoch | Ghi chú |
|--------|------------------|---------|
| ResNet18, Adam, lr 0.0001, bs 32 | **best val 0.8634 @ ep 17** (val_acc: 0.7742 → 0.8128 → 0.8330 → … → 0.8634 → 0.8604) | run `results/runs/20261031-120033/resnet18-lr0.0001-bs32-0/` |
| VGG11, Adam, lr 0.0001, bs 32 | **best val 0.8786 @ ep 20** (val_acc: 0.8044 → … → 0.8782 → 0.8786) | run `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/` |
| DenseNet121, Adam, lr 0.0005, bs 64 | **best val 0.8800 @ ep 12** (val_acc: 0.8218 → … → 0.8800 → … → 0.8696) | run `results/runs/20261031-122534/densenet121-lr0.0005-bs64-0/` |

Mỗi run ghi `metrics.json` + `config.yaml` + `logs.txt` + `tb/` + `checkpoints/best.pt` + `report.html` (sinh bằng `python -m src.evaluation.visualize_html --run-dir <run>` — 1 file offline duy nhất: bảng pass/fail, curves, mẫu tốt/xấu, latency). Đánh giá test xem báo cáo M4.

## 6. TensorBoard

Mở bằng `.\.venv\Scripts\tensorboard.exe --logdir results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/tb` → http://localhost:6006/ — các tag `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `lr`, `model/grad_norm`, `samples/train_images`.
