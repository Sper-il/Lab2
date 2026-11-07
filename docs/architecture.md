# Kiến trúc hệ thống — Practice 2 (CIFAR-10 + pre-trained torchvision)

> Snapshot kiến trúc đã hiện thực đầy đủ M1–M4. Mọi con số đều đo thật, nguồn artifact ghi kèm.
> Tài liệu liên quan: `docs/data-flow.md` (luồng M1 chi tiết), `docs/api.md` (interfaces từng module),
> `docs/plan-m2-m4.md` (kế hoạch M2–M4), `docs/eda.md` (phân tích dữ liệu).

## 1. Sơ đồ tổng thể

```text
[CIFAR-10 raw] ──► [Transforms] ──► [DataLoader] ──► [torchvision backbone] ──► [Head 10 lớp]
  data/raw/          resize 32→64      shuffle train     ResNet18 / VGG11 /        Linear → [B,10]
  50k train +        (train/eval chốt    704/79/157        DenseNet121
  10k test           GPU T4 + AMP)      batch             (weights="DEFAULT",
  MD5 PASS           resize 32→224      (bs 64, ảnh 64)    ImageNet)
  c58f3010...        (demo M1)                                 │
                                                        freeze_backbone ❄ / fine_tune 🔥
                                                        (chốt: fine_tune, Adam)
                                                               │
                                                               ▼
                                              [CrossEntropyLoss] + [Adam]
                                                               │
                                                               ▼
                                              [Trainer: train ⇄ val] ──► [best.pt theo val acc]
                                              train/ zero_grad→forward→loss→backward→step
                                              val/ torch.no_grad, đo loss/acc
                                                               │
                                                     ┌─────────┴─────────┐
                                                     ▼                   ▼
                                            [TensorBoard tb/]   [Test eval → metrics.json + PNG + report.html]
                                            scalar/hist/grad/    acc / macro-F1 / CM / per-class F1
                                            image/graph
```

## 2. Modules và trách nhiệm

| Module | Thư mục | Trách nhiệm | Trạng thái | Bằng chứng |
|--------|---------|-------------|-----------|------------|
| Data | `src/data/` | Tải/verify CIFAR-10, chia 45k/5k/10k (stratified, seed 42), transform, DataLoader | ✅ | run `20261025-150847`, 47 test pass |
| Models | `src/models/` | `registry.py` (tra cứu backbone) + `factory.py` (thay head, freeze/unfreeze) + CLI khám phá | ✅ | 24 test pass, verify weights thật cả 3 model |
| Training | `src/training/` | `Trainer` (train/val/checkpoint), loss/opt/scheduler factory, TensorBoard utils, CLI sweep | ✅ | 3 run 20 epoch GPU T4, 13 test pass |
| Evaluation | `src/evaluation/` | `evaluate_model` + plots + CLI eval + `visualize_html.py` (sinh `report.html`) | ✅ | 3 eval test 10k, 18 test pass |
| Demo | `src/demo/` | 1 lệnh chạy được trên checkpoint tốt nhất, đo latency live | ✅ | `docs/demo.md` |

## 3. Cấu trúc thư mục (thực tế)

```text
lab2/
├── data/{raw,processed,splits}   # raw: torchvision tải; processed: .npy 32×32; splits: .npz + .json (seed 42)
├── src/{data,models,training,evaluation,demo}  # + src/utils (config, runs)
├── models/{checkpoints,trained,configs}         # checkpoints *.pt LOCAL ONLY; *.json + *.yaml được push
├── configs/                      # data_config.yaml (ảnh 224, demo M1) | train_config.yaml (ảnh 64, chốt) | eval_config.yaml | model_config.yaml
├── notebooks/                    # eda + demo + colab_sweep + colab_best3_retrain
├── tests/                        # unit + integration; src/evaluation/test_*.py (nhóm M4)
├── docs/                         # architecture.md (file này) + data-flow.md + api.md + description.md + data.md + eda.md + plan-m2-m4.md + demo.md + CHANGELOG.md + AUDIT_LOG.md + AI_Usage.md + progress_status.md + reports/
├── results/runs/<ts>/            # config.yaml + metrics.json + logs.txt + tb/ + checkpoints/ + eval/ + report.html
├── results/OTM_RESULTS.md         # sweep 30 cấu hình 5 epoch
├── results/TEST_RESULTS.md        # test 10.000 ảnh 3 cấu hình 20 epoch
└── scripts/                      # download_data / verify_data / plot_data_report / benchmark_loading
```

## 4. Data flow (tóm tắt — chi tiết ở `docs/data-flow.md`)

1. `data/raw/cifar-10-batches-py/` → `Cifar10Dataset(in_memory=True)` (~147 MB RAM).
2. `stratified_indices(targets, 5000, seed=42)` → 45.000 train (4.500/lớp) / 5.000 val (500/lớp); test giữ nguyên 10.000.
3. `TransformSubset` giữ transform riêng từng split (không mutate dataset gốc).
4. Train: `RandomCrop(32,pad=4)` + `Flip` trên ảnh gốc 32 → `Resize` 64 (chốt) / 224 (demo M1) → `ToTensor` → `Normalize(ImageNet)`. Val/test: chỉ `Resize` + `Normalize`.
5. `DataLoader` → batch `[B,3,64,64]` (chốt) hoặc `[B,3,224,224]` (demo M1).
6. Backbone pre-trained (ImageNet) → head 10 lớp → logits `[B,10]` → `CrossEntropyLoss` + Adam → `Trainer` → `best.pt` → eval test → `metrics.json` + PNG + `report.html`.

## 5. Quyết định kiến trúc quan trọng (đã verify bằng chạy thật)

| Quyết định | Lý do | Nguồn |
|------------|-------|-------|
| Tách `registry.py` khỏi `factory.py` | Registry là dữ liệu (đổi khi thêm model), factory là logic (ổn định) | `docs/plan-m2-m4.md` §1 |
| `freeze_head_full: false` cho VGG11 | `classifier` VGG11 chứa 103M params — unfreeze cả thì 92.8% model trainable, mất ý nghĩa freeze | Đo thật, test bắt lỗi |
| `unfreeze_last` dùng **số block** (`freeze_until: 1`) | 3 model tổ chức block khác nhau (`layer4` vs `features[-3:]` vs `denseblock4`) — code chung, chỉ khác `block_units` | `registry.py` |
| Train chốt ảnh **64 + AMP trên GPU Colab T4** | Local CPU quá chậm (VGG11 224 ~144 phút/epoch); GPU T4 chỉ ~27–86 giây/epoch, vẫn đạt 86–88% test | `OTM_RESULTS.md`, `TEST_RESULTS.md` |
| Tự tính macro-F1 bằng torch thuần | Thêm scikit-learn chỉ để tính F1 là không đáng | `src/evaluation/metrics.py` |
| `*.pt` + `tb/` local-only, chỉ push `metrics.json`/PNG/config | VGG11 `best.pt` ~515 MB vượt giới hạn GitHub 100 MB; số liệu đã khóa trong JSON | `models/README.md` |

## 6. Milestones

| Module | Nội dung | Trạng thái | Commit |
|--------|----------|-----------|--------|
| M0 | Scaffold + dataset CIFAR-10 | ✅ 2026-10-25 | `f130862` |
| M1 | Data pipeline 45k/5k/10k | ✅ 2026-10-25 | `d0321eb` + `28aa1ff` |
| M1b | Docs bám sát `đề.txt` | ✅ 2026-10-25 | `a1a6b71` |
| M2 | Model factory 3 backbone | ✅ 2026-10-26 | `75455dd` (+`81fa425` fix B4) |
| M3 | Training loop + TensorBoard | ✅ 2026-10-31 | `2e98e31` |
| M4 | Evaluation + metrics | ✅ 2026-10-31 | `2e98e31` |
| M5 | Sweep 30 cấu hình + retrain 20 epoch + báo cáo | ✅ 2026-10-31 | `2e98e31` |

## 7. Rủi ro kỹ thuật (đã xảy ra → đã xử lý)

| Rủi ro | Tác động | Giảm thiểu |
|--------|----------|------------|
| Train CPU quá lâu (VGG11 224 ~12 giờ/5 epoch) | Trượt deadline | Chuyển toàn bộ lên GPU Colab T4 + AMP, ảnh 64 |
| VGG11 `classifier` 103M params phá vỡ ý nghĩa freeze | Transfer learning vô nghĩa | `freeze_head_full: false` + test bắt lỗi |
| `get_module` ép `int()` crash với DenseNet (`features.denseblock4` là key chuỗi) | Mất trắng config sweep | Ưu tiên `getattr` trước, fallback index sau (fix B4) |
| Overfit ảnh 32×32 upsample | Test thấp hơn val | Augmentation + weight decay + best checkpoint theo val |
