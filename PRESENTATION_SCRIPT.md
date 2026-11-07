# NÓI TRÌNH BÀY — Practice 2: Pre-trained model trên CIFAR-10 (10 phút)

> Nói thẳng, không vòng vo. Mọi con số đều chạy thật (train sweep + retrain trên GPU Colab T4, pipeline M1/demo/test trên local), test 10.000 mẫu.
> Đọc số kiểu: 0.8474 = "tám mươi tư phẩy bảy tư phần trăm".

---

## 1. Mở đầu · 40s

> "Em chào cô, em là Sper-il.
>
> Practice 2 - Hands-on practice with pre-trained neural network architectures
Exercises:
1. Thử nghiệm với các mô hình đã được huấn luyện trước khác nhau (VGG, DenseNet, v.v.).
2. Điều chỉnh các siêu tham số (Hyperparameters) như tốc độ học (Learning Rate), kích thước batch (Batch Size), v.v.
3.Sử dụng TensorBoard để theo dõi quá trình huấn luyện."

---

## 2. Bài toán · 50s

> "Bài toán đơn giản: đưa vào 1 ảnh màu 32×32, máy trả lời nó là con gì, xe gì — trong 10 loại: máy bay, ô tô, chim, mèo, hươu, chó, ếch, ngựa, tàu, xe tải.
>
> → mở `results/runs/20261025-150847/plots/class_distribution.png`
>
> Data có 60 nghìn ảnh: 50 nghìn train, 10 nghìn test, chia đều 10 lớp. Cột nào cũng cao bằng nhau — data sạch, không lệch lớp, MD5 check pass. Đây đúng là CIFAR-10 trong slide 24 của tài liệu.
>
> → mở `results/runs/20261025-150847/plots/sample_images.png`
>
> Ảnh hơi mờ — cái này bình thường, không phải data hỏng. Em đo rồi: pixel trung bình 119, độ lệch 63, gần như không có pixel chết. Nó mờ vì gốc chỉ có 32×32, phóng lên 64 (bản demo M1 phóng lên 224) thì nó nhòe ra thôi. Nhìn qua là thấy ngay mèo với chó, chim với hươu giống nhau — lát em quay lại chỗ này ở phần lỗi.
---

## 3. Chuẩn bị data · 1m30

> "Phần này em làm 4 việc.
>
> **Thứ nhất, tải data.** link tụi nhóm tụi em tải dữ liệu https://cave.cs.toronto.edu/kriz/cifar.html
>
> **Thứ hai, chia data.** Em chia 3 phần: 45 nghìn train, 5 nghìn validation, 10 nghìn test giữ kín tới cuối.
>
> → mở `data/splits/cifar10_split_seed42.json`
>
> Tại sao phải có validation riêng? Vì nếu vừa train vừa chấm trên test thì điểm nó ảo — model học thuộc test luôn. Chia xong em check: mỗi lớp đúng 4.500 / 500 / 1.000, trùng lặp giữa 3 tập bằng 0, seed 42 chạy lại ra y chang.
>
> **Thứ ba, transform.** Train thì RandomCrop + lật ngang trên ảnh gốc 32 rồi Resize lên 64×64 → chuyển sang Tensor → Normalize theo ImageNet. Val với test thì chỉ resize 64 + normalize, không augmentation. Ảnh 224×224 chỉ dùng cho pipeline demo M1, không dùng cho kết quả chốt.
>
> → mở `src/data/transforms.py`
>
> **Thứ tư, DataLoader.** Train thì xáo trộn mỗi epoch cho model khỏi học vẹt thứ tự, val với test thì không xáo. Batch 32 cho ResNet18/VGG11, batch 64 cho DenseNet121 (theo đúng 3 config chốt 20 epoch)."
> → mở `src/data/dataloader.py`
---

## 4. Model · 2m

> "Phần này có 3 bước: nạp, coi, sửa.
>
> **Nạp model.** Một dòng: `torchvision.models.resnet18(weights="DEFAULT")`. Nhóm em chọn 3 models: ResNet18, VGG11, DenseNet121.
> → mở `results/runs/20261025-150847/model_info.md`"

---

## 5. Train · 2m

> "Em viết 1 class `Trainer`, mỗi epoch chạy 2 vòng.
>
> **Vòng train** có 6 bước: đẩy data lên máy, xóa gradient cũ, chạy forward, tính loss, chạy backward, rồi cập nhật optimizer. Loss dùng CrossEntropy, optimizer chỉ xài Adam.
>
> **Vòng validate** dùng để đo rồi giữ lại bản tốt nhất theo validation.
>
> → mở `src/training/trainer.py`
>
> **Chỉnh siêu tham số.** Em fine-tune cả 3 model 5 epoch bằng Adam, rồi sweep learning rate nhân batch-size: 5 giá trị lr 0.0001, 0.0005, 0.001, 0.003, 0.005 nhân 2 batch-size 32/64, tổng 30 cấu hình chạy trên GPU Colab T4 có AMP.
>
> Kết quả validation tốt nhất (chi tiết ở `results/sweep-FT-lr-bs.log`):
>
> | Cấu hình | Val tốt nhất | Nói gì |
> |-----------|-------------|--------|
> | ResNet18 fine-tune, Adam, lr 0.0001, batch 32 | 0,8456 | lr nhỏ hợp ResNet nhất |
> | VGG11 fine-tune, Adam, lr 0.0001, batch 32 | 0,8538 | ngon nhất VGG |
> | DenseNet121 fine-tune, Adam, lr 0.0005, batch 64 | **0,8618** | ngon nhất sweep |
>
> Chốt: Adam + lr nhỏ từ 0.0001 tới 0.0005 là hợp nhất. Lr từ 0.003 trở lên tụt mạnh, rõ nhất ở VGG. Batch 32 với 64 chênh nhau có 0,01 — DenseNet chốt batch 64 cho nhanh, ResNet18 với VGG11 chốt batch 32 cho ngon nhất."

---

## 6. Chấm test · 1m30

> "Chấm trên 10.000 ảnh test. Chạy lại 3 cấu hình tốt nhất 20 epoch trên GPU Colab T4 rồi chấm test.
>
> → mở `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/eval/metrics.json`
>
> | Model | Test | F1 | Val tốt nhất |
> |-------|------|----|--------------|
> | ResNet18 fine-tune, Adam, lr 0.0001, batch 32 | 86,26% | 86,26% | 0,8634 |
> | **VGG11 fine-tune, Adam, lr 0.0001, batch 32** | **87,74%** | **87,65%** | 0,8786 |
> | DenseNet121 fine-tune, Adam, lr 0.0005, batch 64 | 87,15% | 87,09% | 0,8800 |
>
> Cả 3 con fine-tune bằng Adam đều qua mốc 86%: 86,26 — 87,74 — ngon nhất VGG11 87,74%.
>
> Soi lỗi con ngon nhất (VGG11): mèo điểm thấp nhất 0,76; tàu điểm cao nhất 0,93. Nhầm nhiều nhất là mèo đoán ra chó — 109 tấm — và chó đoán ra mèo — 106 tấm. Đúng như em đoán từ đầu: mấy con lông xù nhìn giống nhau là khó nhất.
>
> → mở `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/eval/confusion_matrix.png`
>
> Đường chéo là đoán đúng từng lớp. Hai ô mèo–chó sáng nhất ngoài đường chéo. Cộng hết ma trận đúng 10.000 tấm.
>
> → mở `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/eval/per_class_f1.png`"

---

## 7. Kết + demo · 1m

> "Tóm lại 3 ý.
>
> **Một**, cả 3 con fine-tune bằng Adam đều qua mốc 86%: ResNet 86,26 — VGG11 87,74 ngon nhất — DenseNet 87,15. Lr nhỏ 0.0001 tới 0.0005 là hợp nhất, lr từ 0.003 trở lên tụt mạnh.
>
> **Hai**, báo cáo HTML offline mở bằng double-click, không cần server:
> `results/runs/20261031-121323/vgg11-lr0.0001-bs32-0/report.html` — có bảng pass/fail, curves, mẫu tốt/xấu, case tệ nhất, latency. Sinh bằng `python -m src.evaluation.visualize_html --run-dir ...`.
>
> **Ba**, demo 1 lệnh chạy được trên checkpoint tốt nhất:
> `python -m src.demo --checkpoint models/checkpoints/vgg11-best.pt --num-samples 8` — chạy trên 8 ảnh test thật, in top-3 + latency từng ảnh, lưu `results/demo/<ts>/predictions.json + index.html`. Chi tiết ở `docs/demo.md`.
>
> Tài liệu đầy đủ theo khung chuẩn: `docs/architecture.md` (kiến trúc), `docs/api.md` (interfaces), `docs/description.md` + `docs/data.md` (define/data), `docs/CHANGELOG.md` (version model), `docs/demo.md` (demo). Em cảm ơn cô."
