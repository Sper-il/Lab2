# Demo — 1 lệnh chạy được (checkpoint tốt nhất VGG11)

> Chuẩn `modules/demo.md` của skill: biến run tốt nhất thành demo 1 lệnh, đo latency live,
> lưu vào `results/demo/<ts>/` kèm `index.html` rút gọn, đồng bộ Example vào README.

## Lệnh chạy

```powershell
# Demo checkpoint tốt nhất (VGG11 test 0.8774) trên 8 ảnh test thật
.\.venv\Scripts\python.exe -m src.demo --checkpoint models/checkpoints/vgg11-best.pt --num-samples 8

# Đổi checkpoint / số ảnh
.\.venv\Scripts\python.exe -m src.demo --checkpoint results/runs/20261031-120033/resnet18-lr0.0001-bs32-0/checkpoints/best.pt --num-samples 4
```

Mặc định (`--checkpoint` bỏ trống): dùng `configs/eval_config.yaml`
(`results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/checkpoints/best.pt`).

## Demo làm gì

1. Lấy N ảnh đầu tiên của **tập test** (nhãn thật giữ để đối chiếu, không dùng để chọn model).
2. Transform val (`Resize 64` + `Normalize ImageNet` — khớp checkpoint train chốt) rồi suy luận bằng checkpoint.
3. In ra từng ảnh: nhãn thật → top-3 dự đoán + confidence + đúng/sai, kèm latency/ảnh (ms).
4. Lưu vào `results/demo/<YYYYMMDD-HHMMSS>/`:
   - `predictions.json` — chi tiết từng mẫu (nhãn thật, top-3, latency).
   - `index.html` — bảng rút gọn mở bằng double-click.
   - `config.json` — checkpoint + tham số đã dùng (tái lập được).

## Kết quả mẫu (VGG11, 8 ảnh test đầu, CPU local)

Nguồn đo thật — chạy lại lệnh trên để tái tạo; số latency phụ thuộc máy
(CPU local ~5–15 ms/ảnh; GPU Colab T4 ~0.33 ms/ảnh theo `eval/metrics.json`).

## Input mẫu nhẹ (`data/processed/demo/`)

Thư mục này chứa 8 ảnh PNG mẫu trích từ tập test + `labels.json`
để demo không cần tải lại CIFAR-10:

```text
data/processed/demo/
├── demo_00.png … demo_07.png   # ảnh RGB 32×32 gốc
└── labels.json                 # nhãn thật từng ảnh
```

Xuất lại bằng: `.\.venv\Scripts\python.exe -m src.demo --export-samples`
(copy từ `data/processed/test_images.npy`, seed cố định nên giống hệt).

## Đồng bộ README

Khối Example Input → Output trong `README.md` §7 chính là 1 batch demo này:
input `[B,3,64,64]` → output test acc 0.8774 / macro-F1 0.8765 (VGG11).
