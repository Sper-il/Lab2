# models/ — weight & model card

> **Kết quả đã vượt 80%:** cả 3 model fine-tune Tier B (64×64, 5 epoch, CPU) đều đạt target tự đặt ≥ 0.80.

| Model | Strategy | Best val | Test acc | Macro-F1 | Checkpoint gốc (local) |
|-------|----------|----------|----------|----------|------------------------|
| ResNet18 | fine-tune block4 | 0.8296 @ ep5 | **0.8273** | 0.8258 | `results/runs/20261029-220833/resnet18-lr0.001-bs32-0/checkpoints/best.pt` (44.8 MB) |
| VGG11 | fine-tune | 0.8316 @ ep5 | **0.8331** | 0.8305 | `results/runs/20261029-220833/vgg11-lr0.001-bs32-1/checkpoints/best.pt` (491.4 MB) |
| DenseNet121 ⭐ | fine-tune | 0.8524 @ ep4 | **0.8474** | 0.8468 | `results/runs/20261030-023250/densenet121-lr0.001-bs32-0/checkpoints/best.pt` (27.1 MB) |

Eval chi tiết: `results/runs/<ts>/<cfg>/eval/metrics.json` + `confusion_matrix.png` + `per_class_f1.png` (đã push GitHub).

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
│   ├── resnet18-best.pt   # copy từ run 20261029-220833 (local)
│   ├── vgg11-best.pt      # copy từ run 20261029-220833 (local)
│   └── densenet121-best.pt# copy từ run 20261030-023250 (local)
├── trained/               # model cuối đã eval + model card JSON (JSON được push, *.pt thì không)
│   ├── .gitkeep
│   ├── densenet121-final.json  # card model tốt nhất (test 0.8474)
│   ├── resnet18-final.json
│   └── vgg11-final.json
└── configs/               # kiến trúc head + chiến lược freeze/unfreeze (được push)
    ├── resnet18-cifar10.yaml
    ├── vgg11-cifar10.yaml
    └── densenet121-cifar10.yaml
```

## Dùng model tốt nhất (DenseNet121, local)

```powershell
# Đánh giá lại checkpoint tốt nhất trên test set
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml `
  --checkpoint results/runs/20261030-023250/densenet121-lr0.001-bs32-0/checkpoints/best.pt
# hoặc bản copy tiện tay:
.\.venv\Scripts\python.exe -m src.evaluation --config configs/eval_config.yaml `
  --checkpoint models/checkpoints/densenet121-best.pt
```
