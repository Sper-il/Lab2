# Kết quả huấn luyện Practice 2 — CIFAR-10 fine-tune bằng Adam

> File tổng hợp duy nhất. Chi tiết từng epoch của sweep 30 cấu hình xem ở `results/sweep-FT-lr-bs.log`.
> Cấu hình chung: ảnh 64, 5 epoch, optimizer Adam, fine-tune mở block cuối.

### 1. Sweep 30 cấu hình trên GPU Colab T4 (lr × batch-size)

Sweep 5 giá trị lr 0.0001, 0.0005, 0.001, 0.003, 0.005 nhân 2 batch-size 32/64, 5 epoch, workers = 2, có AMP.

### 1.1. ResNet18

| Cấu hình | lr | Batch | Val tốt nhất | Epoch đạt |
|----------|----|-------|--------------|-----------|
| resnet18-lr0.0001-bs32 | 0,0001 | 32 | 0,8456 | 5 |
| resnet18-lr0.0001-bs64 | 0,0001 | 64 | 0,8348 | 5 |
| resnet18-lr0.0005-bs32 | 0,0005 | 32 | 0,8384 | 5 |
| resnet18-lr0.0005-bs64 | 0,0005 | 64 | 0,8454 | 5 |
| resnet18-lr0.001-bs32 | 0,001 | 32 | 0,8314 | 5 |
| resnet18-lr0.001-bs64 | 0,001 | 64 | 0,8376 | 5 |
| resnet18-lr0.003-bs32 | 0,003 | 32 | 0,8168 | 5 |
| resnet18-lr0.003-bs64 | 0,003 | 64 | 0,8170 | 5 |
| resnet18-lr0.005-bs32 | 0,005 | 32 | 0,8016 | 5 |
| resnet18-lr0.005-bs64 | 0,005 | 64 | 0,8052 | 5 |

Ngon nhất ResNet18: lr 0,0001 batch 32 với 0,8456.

### 1.2. VGG11

| Cấu hình | lr | Batch | Val tốt nhất | Epoch đạt |
|----------|----|-------|--------------|-----------|
| vgg11-lr0.0001-bs32 | 0,0001 | 32 | 0,8538 | 5 |
| vgg11-lr0.0001-bs64 | 0,0001 | 64 | 0,8452 | 5 |
| vgg11-lr0.0005-bs32 | 0,0005 | 32 | 0,8502 | 5 |
| vgg11-lr0.0005-bs64 | 0,0005 | 64 | 0,8536 | 5 |
| vgg11-lr0.001-bs32 | 0,001 | 32 | 0,8218 | 5 |
| vgg11-lr0.001-bs64 | 0,001 | 64 | 0,8452 | 5 |
| vgg11-lr0.003-bs32 | 0,003 | 32 | 0,7920 | 3 |
| vgg11-lr0.003-bs64 | 0,003 | 64 | 0,8166 | 4 |
| vgg11-lr0.005-bs32 | 0,005 | 32 | 0,7708 | 5 |
| vgg11-lr0.005-bs64 | 0,005 | 64 | 0,7890 | 3 |

Ngon nhất VGG11: lr 0,0001 batch 32 với 0,8538.

### 1.3. DenseNet121

| Cấu hình | lr | Batch | Val tốt nhất | Epoch đạt |
|----------|----|-------|--------------|-----------|
| densenet121-lr0.0001-bs32 | 0,0001 | 32 | 0,8482 | 5 |
| densenet121-lr0.0001-bs64 | 0,0001 | 64 | 0,8496 | 5 |
| densenet121-lr0.0005-bs32 | 0,0005 | 32 | 0,8618 | 5 |
| densenet121-lr0.0005-bs64 | 0,0005 | 64 | 0,8618 | 5 |
| densenet121-lr0.001-bs32 | 0,001 | 32 | 0,8510 | 5 |
| densenet121-lr0.001-bs64 | 0,001 | 64 | 0,8566 | 4 |
| densenet121-lr0.003-bs32 | 0,003 | 32 | 0,8426 | 5 |
| densenet121-lr0.003-bs64 | 0,003 | 64 | 0,8478 | 4 |
| densenet121-lr0.005-bs32 | 0,005 | 32 | 0,8280 | 4 |
| densenet121-lr0.005-bs64 | 0,005 | 64 | 0,8332 | 5 |

Ngon nhất toàn sweep: DenseNet121 lr 0,0005 với 0,8618 ở cả batch 32 và 64.

## 3. Kết luận chỉnh 2 siêu tham số

1. **Learning rate quyết định rõ rệt.** Vùng lr nhỏ 0,0001 tới 0,001 cho điểm cao nhất ở cả 3 mô hình. Từ lr 0,003 trở lên điểm tụt mạnh, rõ nhất ở VGG11: lr 0,0001 được 0,8538 nhưng lr 0,005 chỉ còn 0,7708.
2. **Batch-size 32 và 64 chênh nhau rất ít.** ResNet18 hơn kém nhau khoảng 0,01, VGG11 và DenseNet cũng tương tự. Batch 64 chạy nhanh hơn hẳn nên là lựa chọn hợp lý trên GPU.
3. **Xếp hạng mô hình giữ nguyên.** DenseNet121 (0,8618) hơn VGG11 (0,8538) hơn ResNet18 (0,8456).
4. **Tốc độ trên GPU nhanh gấp nhiều lần CPU.** ResNet18 và VGG11 chỉ mất khoảng 27 tới 35 giây mỗi epoch, DenseNet121 mất khoảng 48 tới 86 giây mỗi epoch.

## 4. Cấu hình chốt

- DenseNet121, Adam, lr 0,0005, batch 64, val 0,8618 — ngon nhất mà chạy nhanh.
- Nếu máy yếu hơn: ResNet18, Adam, lr 0,0001, batch 32, val 0,8456.
