# API / Interfaces — Practice 2

> Đầu vào, đầu ra và lệnh chạy của từng module. Chi tiết cài đặt xem code; số liệu xem `results/TEST_RESULTS.md`.
> Sơ đồ tổng thể: `docs/architecture.md`. Luồng dữ liệu M1: `docs/data-flow.md`.

## Quy ước chung

- Mọi config nằm trong YAML (`configs/*.yaml`), override được từ CLI (phục vụ E2 — chỉnh siêu tham số không cần sửa code).
- Mọi run ghi `metrics.json` + `config.yaml` + `logs.txt` vào `results/runs/<ts>/` (schema thống nhất để M5 tổng hợp).
- Seed cố định `42` cho split/shuffle; test set 10.000 ảnh chỉ dùng để đánh giá cuối.

## M1 — `src.data` (Data pipeline)

```powershell
.\.venv\Scripts\python.exe -m src.data --config configs/data_config.yaml
```

| Hàm | Input → Output |
|-----|---------------|
| `build_dataloaders(DataLoaderConfig) -> (train_loader, val_loader, test_loader)` | `data/raw/` → 3 loader: 704/79/157 batch (ảnh 224, bs 64, demo M1); ảnh 64: 1407/157/313 batch (bs 32) |
| `dataset_summary(train_loader, val_loader, test_loader) -> dict` | 3 loader → số batch/mẫu + shape/dtype tensor đầu tiên mỗi split |
| `stratified_indices(targets, val_size=5000, seed=42)` | nhãn train → `(train_idx 45k, val_idx 5k)`, mỗi lớp 4.500/500 |
| `build_transform("train"\|"val", image_size)` | tên split → transform (train có augment, val không) |

Artifact: `config.yaml`, `metrics.json` (`validation.status = PASS`), `logs.txt`, `plots/class_distribution.png`, `plots/sample_images.png`.

## M2 — `src.models` (Model factory)

```powershell
.\.venv\Scripts\python.exe -m src.models --model resnet18|vgg11|densenet121 [--print-model]
.\.venv\Scripts\python.exe -m src.models --config configs/model_config.yaml
```

| Hàm | Input → Output |
|-----|---------------|
| `build_model(ModelConfig) -> nn.Module` | `ModelConfig(name, num_classes=10, strategy, weights, freeze_until, freeze_head_full)` → model đã thay head + áp freeze/unfreeze, output `(B,10)` |
| `replace_head(model, name, num_classes)` | model gốc → model head 10 lớp (xử lý 2 dạng: `Linear` và `Sequential[7]` của VGG11) |
| `freeze_backbone(model, name, head_full=False)` | model → chỉ head trainable (VGG11 mặc định chỉ `Linear` cuối: 40.970 params) |
| `unfreeze_last(model, name, n_blocks=1, head_full=False)` | model → n block cuối + norm + head trainable |
| `count_parameters(model) -> dict` | model → `{params_total, params_trainable, params_trainable_pct}` |
| `describe_model(model, full=True) -> str` | model → chuỗi mô tả cấu trúc + bảng params (phục vụ O2) |
| `resolve_spec(name) / get_module(model, path)` | tên backbone → `BackboneSpec`; đường dẫn (`features.denseblock4`) → module con |

Số params đo thật (weights ImageNet): ResNet18 11.181.642 (freeze 5.130) / VGG11 128.807.306 (freeze 40.970) / DenseNet121 6.964.106 (freeze 10.250).

## M3 — `src.training` (Training loop + TensorBoard)

```powershell
# Train 1 model
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml
# So sánh nhiều model / sweep siêu tham số (E1 + E2)
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --models resnet18 vgg11 densenet121
.\.venv\Scripts\python.exe -m src.training --config configs/train_config.yaml --lr 1e-3 1e-4 --batch-size 32 64
# TensorBoard
.\.venv\Scripts\tensorboard.exe --logdir results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/tb
```

| Hàm / Class | Input → Output |
|-------------|---------------|
| `TrainConfig` (dataclass) | `model/epochs/batch_size/lr/optimizer/scheduler/loss/image_size/use_amp/...` — mọi hyperparameter chỉnh từ YAML/CLI |
| `Trainer(cfg, model, train_loader, val_loader, run_dir, writer)` | config + model + loader → `train()` → `{best_val_accuracy, best_epoch, history, stopped_early}` |
| `Trainer.train_one_epoch(epoch)` | 1 epoch: `zero_grad → forward → loss → backward → step` (đúng 6 bước slide 8–9 Ch.3), hỗ trợ AMP |
| `Trainer.validate()` | val loader → `{loss, acc}` (`torch.no_grad`), quyết định `best.pt` |
| `build_criterion(name)` | `cross_entropy` \| `label_smoothing` \| `focal` → loss fn |
| `build_optimizer(model, name, lr, ...)` | `adam` \| `adamw` \| `sgd` — chỉ params `requires_grad=True` |
| `build_scheduler(optimizer, name, epochs)` | `none` \| `cosine` \| `step` \| `plateau` |
| TensorBoard tags | `train/loss`, `train/accuracy`, `val/loss`, `val/accuracy`, `lr`, `epoch/time_seconds`, `model/grad_norm`, `model/param_histograms`, `samples/train_images` |

Cấu hình chốt: fine-tune + Adam, ảnh 64, AMP — ResNet18 (lr 0.0001, bs 32), VGG11 (lr 0.0001, bs 32), DenseNet121 (lr 0.0005, bs 64), 20 epoch.

## M4 — `src.evaluation` (Evaluation + visualize)

```powershell
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml --checkpoint <best.pt>
# Sinh báo cáo HTML offline cho 1 run đã có metrics
.\.venv\Scripts\python.exe -m src.evaluation.visualize_html --run-dir results/runs/20261031-121323/vgg11-lr0.0001-bs32-0
```

| Hàm | Input → Output |
|-----|---------------|
| `evaluate_model(model, loader, device="cpu", n_classes=10) -> EvalResult` | model + test loader → `EvalResult(accuracy, macro_f1, per_class, confusion_matrix, inference_time_ms_per_sample, n_samples)` |
| `accuracy / macro_f1 / per_class_metrics / confusion_matrix / classification_report` | `(pred, target)` → metric tương ứng (torch thuần, không cần scikit-learn) |
| `plot_confusion_matrix / plot_per_class_f1 / plot_pr_curve / plot_confusion_examples / plot_train_curves` | metric → PNG (`out_path` tùy chọn để test truyền vào) |
| `render_report(run_dir) -> report.html` (`visualize_html.py`) | `metrics.json` + `eval/metrics.json` + PNG → 1 file HTML offline duy nhất: bảng metric pass/fail, curves, mẫu tốt/xấu, case tệ nhất, latency |

Kết quả chốt (test 10.000): ResNet18 0.8626 / VGG11 **0.8774** ⭐ / DenseNet121 0.8715.

## Demo — `src.demo`

```powershell
.\.venv\Scripts\python.exe -m src.demo --checkpoint models/checkpoints/vgg11-best.pt --num-samples 8
```

Lấy N ảnh test thật → suy luận bằng checkpoint tốt nhất (VGG11) → in top-3 + latency/ảnh → lưu `results/demo/<ts>/` kèm `index.html` rút gọn. Chi tiết: `docs/demo.md`.

## Scripts tiện ích

| Script | Chức năng |
|--------|-----------|
| `scripts/download_data.py` | Tải + giải nén CIFAR-10 vào `data/raw/` |
| `scripts/verify_data.py` | Kiểm tra MD5 archive (`c58f...`), số mẫu, phân bố lớp, shape |
| `scripts/plot_data_report.py --run-dir <ts>` | Vẽ lại `class_distribution.png` + `sample_images.png` |
| `scripts/benchmark_loading.py` | Đo throughput DataLoader (quyết định không cache ảnh 224: 7 GB không đáng) |
