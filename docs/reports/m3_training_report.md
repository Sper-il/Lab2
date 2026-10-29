# Báo cáo M3 — Training loop + TensorBoard

> Nguồn: `đề.txt` (E2, E3, O9, O10). Mọi con số có nguồn artifact; chỗ nào train chưa xong ghi rõ "đang chạy".

## 1. Yêu cầu đề bài được phủ

| # | Yêu cầu | Hiện thực |
|---|---------|-----------|
| E2 | Adjust hyperparameters | `configs/train_config.yaml` + CLI `--models/--lr/--batch-size/--epochs/--image-size/--strategy/--optimizer/--scheduler` (grid models×lr×bs trong `__main__.py`) |
| E3 | TensorBoard | `SummaryWriter` tại `<run>/tb/`: scalar train/val loss+acc, lr, epoch time; grad norm; histogram trọng số (epoch đầu/cuối); ảnh mẫu denormalize; model graph (tắt mặc định) |
| O9 | Loss + optimizer | `build_criterion` (cross_entropy/label_smoothing/focal) + `build_optimizer` (adam/adamw/sgd, chỉ params `requires_grad=True`) |
| O10 | Training loop | `Trainer.train_one_epoch`: `zero_grad → forward → loss → backward → step` (bám slide 8–9 Chương 3); `validate()` với `torch.no_grad()`; checkpoint `best.pt` theo val acc; early stopping theo patience |

## 2. Files

| File | Chức năng |
|------|-----------|
| `src/training/loss.py` | `build_criterion` + `FocalLoss` |
| `src/training/optimizer.py` | `build_optimizer`, `trainable_params` |
| `src/training/scheduler.py` | `build_scheduler` (none/cosine/step/plateau) |
| `src/training/tensorboard_utils.py` | `create_writer`, `log_scalars`, `log_weight_histograms`, `log_grad_norms`, `log_images`, `log_graph`, `tb_config` |
| `src/training/trainer.py` | `TrainConfig`, `Trainer` (train/val/checkpoint/history) |
| `src/training/__main__.py` | CLI train + sweep, ghi `metrics.json` + `config.yaml` + `logs.txt` + `tb/` |
| `configs/train_config.yaml` | Mặc định Tier B 64×64, 5 epoch, freeze ResNet18 |

## 3. Test

| Suite | Kết quả |
|-------|---------|
| `tests/unit/test_training_components.py` (9 test) | loss variants, optimizer chỉ trainable, scheduler decay, TB tags+ghi file — PASS |
| `tests/integration/test_training_loop.py` (4 test) | 1 epoch đổi weights, grad head, checkpoint reload, early stopping — PASS (FakeData, <10s) |

## 4. Kết quả train thật (Tier B, ResNet18 freeze, 64×64)

| Epoch | Train loss/acc | Val loss/acc | Ghi chú |
|-------|----------------|--------------|---------|
| 1/5 | 1.4501 / 0.4993 | 1.3911 / 0.5288 | ~414s, *best* — run `results/runs/20261029-212336/resnet18-lr0.001-bs32-0/` |
| 2/5 | 1.3327 / 0.5416 | 1.3851 / 0.5336 | ~390s, *best* |
| 3/5 | 1.3248 / 0.5460 | 1.3150 / 0.5628 | ~364s, *best* |
| 4/5 | 1.3144 / 0.5486 | 1.2952 / 0.5654 | ~557s, *best* |
| 5/5 | 1.3099 / 0.5505 | 1.3531 / 0.5554 | ~356s (val tăng nhẹ — best vẫn epoch 4) |

**Best:** val acc **0.5654** ở epoch 4/5; `checkpoints/best.pt` (44.8 MB) + `last.pt`; `metrics.json` + `config.yaml` + `logs.txt` + `tb/` trong `results/runs/20261029-212336/resnet18-lr0.001-bs32-0/`.

## 5. TensorBoard

Mở bằng `.\.venv\Scripts\tensorboard.exe --logdir results/runs/20261029-212336/resnet18-lr0.001-bs32-0/tb` → http://localhost:6006/ — các tag `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `lr`, `model/grad_norm`, `samples/train_images`.
