# Tổng hợp 3 lần train (Training runs)

Tất cả train thật trên CPU, 5 epoch, ảnh 64, chỉ dùng optimizer Adam.

| Lần chạy | Mô hình | Tốc độ học (lr) | Batch | Val tốt nhất |
|----------|---------|-----------------|-------|--------------|
| 1 | resnet18 fine-tune | 0,001 | 32 | 0,8296 |
| 2 | vgg11 fine-tune | 0,001 | 32 | 0,8316 |
| 3 | densenet121 fine-tune | 0,001 | 32 | 0,8524 |

Kết luận: fine-tune thắng, densenet 0,8524 ngon nhất.

## Chi tiết từng epoch (kết quả thật từ metrics.json)

### 1. ResNet18 fine-tune — Adam, lr 0,001, batch 32

| Epoch | Train loss | Train acc | Val loss | Val acc |
|-------|------------|-----------|----------|---------|
| 1 | 0,8830 | 0,6992 | 0,5776 | 0,7978 |
| 2 | 0,6959 | 0,7596 | 0,5993 | 0,7932 |
| 3 | 0,6409 | 0,7796 | 0,5183 | 0,8212 |
| 4 | 0,6076 | 0,7898 | 0,5074 | 0,8276 |
| 5 | 0,5813 | 0,7988 | 0,4865 | 0,8296 |

Tốt nhất ở epoch 5: val 0,8296. Nguồn: `results/runs/20261029-220833/resnet18-lr0.001-bs32-0/metrics.json`.

### 2. VGG11 fine-tune — Adam, lr 0,001, batch 32

| Epoch | Train loss | Train acc | Val loss | Val acc |
|-------|------------|-----------|----------|---------|
| 1 | 0,9435 | 0,6818 | 0,6107 | 0,8012 |
| 2 | 0,7671 | 0,7445 | 0,5637 | 0,8142 |
| 3 | 0,7103 | 0,7684 | 0,5501 | 0,8194 |
| 4 | 0,6749 | 0,7804 | 0,5287 | 0,8268 |
| 5 | 0,6353 | 0,7913 | 0,5189 | 0,8316 |

Tốt nhất ở epoch 5: val 0,8316. Nguồn: `results/runs/20261029-220833/vgg11-lr0.001-bs32-1/metrics.json`.

### 3. DenseNet121 fine-tune — Adam, lr 0,001, batch 32

| Epoch | Train loss | Train acc | Val loss | Val acc |
|-------|------------|-----------|----------|---------|
| 1 | 0,7555 | 0,7389 | 0,5221 | 0,8160 |
| 2 | 0,5682 | 0,8002 | 0,4994 | 0,8274 |
| 3 | 0,5102 | 0,8226 | 0,4639 | 0,8388 |
| 4 | 0,4716 | 0,8333 | 0,4221 | 0,8524 |
| 5 | 0,4508 | 0,8411 | 0,4350 | 0,8470 |

Tốt nhất ở epoch 4: val 0,8524. Nguồn: `results/runs/20261030-023250/densenet121-lr0.001-bs32-0/metrics.json`.

File gốc: `results/runs/<thư-mục-run>/metrics.json` + `tb/`.
