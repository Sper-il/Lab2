# models/ — weight & model card

> **Kết quả chốt 20 epoch (GPU Colab T4, ảnh 64, Adam, fine-tune):** cả 3 model đều vượt 86% test. Ngon nhất là **VGG11 test 0.8774**.

| Model | Cấu hình train | Best val | Test acc | Macro-F1 | Checkpoint gốc (local) |
|-------|----------------|----------|----------|----------|------------------------|
| ResNet18 | Adam, lr 0.0001, batch 32, 20 epoch | 0.8634 @ ep 17 | **0.8626** | 0.8626 | `results/runs/20261031-120033/resnet18-lr0.0001-bs32-0/checkpoints/best.pt` |
| VGG11 ⭐ | Adam, lr 0.0001, batch 32, 20 epoch | 0.8786 @ ep 20 | **0.8774** | 0.8765 | `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/checkpoints/best.pt` |
| DenseNet121 | Adam, lr 0.0005, batch 64, 20 epoch | 0.8800 @ ep 12 | **0.8715** | 0.8709 | `results/runs/20261031-122534/densenet121-lr0.0005-bs64-0/checkpoints/best.pt` |

Eval chi tiết: `results/runs/20261031-*/<cfg>/eval/metrics.json` + `confusion_matrix.png` + `per_class_f1.png` (đã push GitHub).
Tổng hợp: `results/TEST_RESULTS.md`. Sweep 30 cấu hình 5 epoch chọn ra 3 cấu hình trên: `results/OTM_RESULTS.md` + `results/sweep-FT-lr-bs.log`.

## Vì sao folder này "trống" trên GitHub?

`.gitignore` cố ý **không push file weight** (`*.pt`, `models/checkpoints/`, `models/trained/*.pt`,
`results/runs/*/checkpoints/`), vì:

1. VGG11 `best.pt` nặng **~491 MB** — vượt giới hạn file GitHub (100 MB), push là lỗi.
2. Weight tái tạo được: chạy lại `python -m src.training --config configs/train_config.yaml --models <name>`
   với cùng seed=42 là ra checkpoint tương đương; số liệu đã khóa trong `metrics.json` (đã push).
3. Local vẫn đầy đủ — checkpoint nằm ở `results/runs/<ts>/<cfg>/checkpoints/best.pt` (xem bảng trên).

## Bố cục sau khi lấp đầy (commit này)

```text
models/
├── README.md              # file này (được push)
├── checkpoints/           # bản copy *.pt để dùng nhanh (LOCAL ONLY, gitignore, không push)
│   ├── .gitkeep           # giữ folder hiện trên GitHub
│   ├── resnet18-best.pt   # copy từ run 20261031-120033 (local)
│   ├── vgg11-best.pt      # copy từ run 20261031-121323 (local)
│   └── densenet121-best.pt# copy từ run 20261031-122534 (local)
├── trained/               # model cuối đã eval + model card JSON (JSON được push, *.pt thì không)
│   ├── .gitkeep
│   ├── densenet121-final.json  # test 0.8715
│   ├── resnet18-final.json     # test 0.8626
│   └── vgg11-final.json        # card model tốt nhất (test 0.8774 ⭐)
└── configs/               # kiến trúc head + chiến lược freeze/unfreeze (được push)
    ├── resnet18-cifar10.yaml
    ├── vgg11-cifar10.yaml
    └── densenet121-cifar10.yaml
```

## Dùng model tốt nhất (VGG11, local)

```powershell
# Đánh giá lại checkpoint tốt nhất trên test set
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml `
  --checkpoint results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/checkpoints/best.pt
# hoặc bản copy tiện tay:
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml `
  --checkpoint models/checkpoints/vgg11-best.pt
```
