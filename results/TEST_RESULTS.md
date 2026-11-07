# Kết quả chấm test 10.000 ảnh — 3 cấu hình tốt nhất 20 epoch

> Chấm 1 lượt duy nhất trên tập test giữ kín, mỗi lớp 1.000 ảnh. Train 20 epoch trên GPU Colab T4, ảnh 64, Adam, fine-tune, lấy checkpoint val tốt nhất rồi chấm test.

## 1. Bảng tổng hợp

| Model | Cấu hình | Val tốt nhất | Test accuracy | Macro F1 |
|-------|----------|--------------|---------------|----------|
| ResNet18 fine-tune, Adam | lr 0.0001, batch 32 | 0,8634 (epoch 17 = best_epoch 16) | 86,26% | 86,26% |
| **VGG11 fine-tune, Adam** | **lr 0.0001, batch 32** | **0,8786 (epoch 20 = best_epoch 19)** | **87,74%** | **87,65%** |
| DenseNet121 fine-tune, Adam | lr 0.0005, batch 64 | 0,8800 (epoch 12 = best_epoch 11) | 87,15% | 87,09% |

Ngon nhất test là VGG11 với 87,74%.

Nguồn:
- `results/runs/20261031-120033/resnet18-lr0.0001-bs32-0/eval/metrics.json`
- `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/eval/metrics.json`
- `results/runs/20261031-122534/densenet121-lr0.0005-bs64-0/eval/metrics.json`

## 2. F1 từng lớp của con ngon nhất (VGG11)

| Lớp | F1 |
|-----|-----|
| 0 máy bay | 0,90 |
| 1 ô tô | 0,93 |
| 2 chim | 0,86 |
| 3 mèo | 0,76 |
| 4 hươu | 0,84 |
| 5 chó | 0,83 |
| 6 ếch | 0,88 |
| 7 ngựa | 0,91 |
| 8 tàu | 0,93 |
| 9 xe tải | 0,92 |

Mèo thấp nhất 0,76. Tàu cao nhất 0,93.

## 3. Lỗi nhầm nhiều nhất (VGG11)

- Mèo đoán ra chó: 109 tấm.
- Chó đoán ra mèo: 106 tấm.
- Cộng hết ma trận nhầm đúng 10.000 tấm.

Hình xem ở `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/eval/confusion_matrix.png`, biểu đồ F1 ở `.../eval/per_class_f1.png`.
