# NÓI TRÌNH BÀY — Practice 2: Pre-trained model trên CIFAR-10 (10 phút)

> Nói thẳng, không vòng vo. Mọi con số đều chạy thật trên máy này, test 10.000 mẫu.
> Đọc số kiểu: 0.8474 = "tám mươi tư phẩy bảy tư phần trăm".

---

## 1. Mở đầu · 40s

> "Em chào cô, em là Sper-il.
>
> Hôm nay em trình bày bài Practice 2 — xài model pre-trained có sẵn để phân loại ảnh CIFAR-10.
>
> Nói thẳng vấn đề: mình chỉ có 50 nghìn ảnh nhỏ 32×32. Train từ đầu thì overfit chắc. Nên em mượn kiến thức của model đã học trên 1 triệu ảnh ImageNet, rồi chỉnh lại cho bài của mình. Tất cả em chạy code thật trên CPU, không có gì làm tay.
>
> Code em để hết trên GitHub, cô muốn kiểm tra gì cũng được."

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
> Ảnh hơi mờ — cái này bình thường, không phải data hỏng. Em đo rồi: pixel trung bình 119, độ lệch 63, gần như không có pixel chết. Nó mờ vì gốc chỉ có 32×32, phóng lên 224 thì nó nhòe ra thôi. Nhìn qua là thấy ngay mèo với chó, chim với hươu giống nhau — lát em quay lại chỗ này ở phần lỗi.
>
> Em tự đặt mốc: freeze phải được 70%, fine-tune phải được 80%. Nói trước là đề không cho mốc nào hết, mốc này em tự đặt để biết mình đang ở đâu."

---

## 3. Chuẩn bị data · 1m30

> "Phần này em làm 4 việc.
>
> **Thứ nhất, tải data.** Một dòng `torchvision.datasets` là xong, verify MD5 khớp. 163 MB, em quăng hết vào RAM cho chạy nhanh.
>
> **Thứ hai, chia data.** Em chia 3 phần: 45 nghìn train, 5 nghìn validation, 10 nghìn test giữ kín tới cuối.
>
> → mở `data/splits/cifar10_split_seed42.json`
>
> Tại sao phải có validation riêng? Vì nếu vừa train vừa chấm trên test thì điểm nó ảo — model học thuộc test luôn. Chia xong em check: mỗi lớp đúng 4.500 / 500 / 1.000, trùng lặp giữa 3 tập bằng 0, seed 42 chạy lại ra y chang.
>
> **Thứ ba, transform.** Train thì crop + lật ngang cho nó đa dạng, rồi resize lên 224, normalize theo chuẩn ImageNet. Val với test thì chỉ resize + normalize, không augmentation.
>
> → mở `src/data/transforms.py`
>
> **Thứ tư, DataLoader.** Train thì xáo trộn, val/test thì không. 4 worker.
>
> Xong phần data: 47 test pass, run ghi PASS."

---

## 4. Model · 2m

> "Phần này có 3 bước: nạp, coi, sửa.
>
> **Nạp model.** Một dòng: `torchvision.models.resnet18(weights="DEFAULT")`. Em làm với 3 con: ResNet18, VGG11, DenseNet121 — đúng yêu cầu thử nhiều model.
>
> **Coi model.** Em in cấu trúc ra đếm tham số:
>
> → mở `results/runs/20261026-115633/model_info.json`
>
> | Model | Tổng | Train khi freeze | Còn lại |
> |-------|------|-----------------|---------|
> | ResNet18 | 11,2 triệu | 5.130 | 0,046% |
> | VGG11 | 128,8 triệu | 40.970 | 0,032% |
> | DenseNet121 | 7,0 triệu | 10.250 | 0,147% |
>
> Cả 3 đều ra đúng 10 lớp. Có 1 cú lừa ở đây: cái đuôi của VGG11 không phải 1 lớp mà là 1 cụm 7 lớp, chứa tới 103 triệu tham số. Mở hết cụm đó ra thì còn gọi gì là freeze nữa — nên em chỉ mở đúng lớp cuối cùng, còn 41 nghìn tham số.
>
> **Sửa model.** Ba thao tác: thay đuôi thành Linear ra 10 lớp; khóa backbone lại chỉ train đuôi — gọi là transfer learning; rồi mở block cuối ra train tiếp — gọi là fine-tuning.
>
> Máy em không có GPU nên em chia 2 kiểu: ảnh 224 đúng chuẩn để báo cáo, ảnh 64×64 để chạy sweep cho nhanh. Đo thật rồi: ảnh 224 thì 1 epoch 37 phút, ảnh 64 thì 5 phút.
>
> Xong phần model: 24 test pass. Có 1 bug với DenseNet — code lấy module con bị crash vì key của nó là chữ chứ không phải số. Em sửa rồi, có test giữ lại."

---

## 5. Train · 2m

> "Em viết 1 class `Trainer`, mỗi epoch chạy 2 vòng.
>
> **Vòng train** có 6 bước đúng slide: đẩy data lên máy, xóa gradient cũ — PyTorch nó cộng dồn nên phải xóa — chạy forward, tính loss, chạy backward, rồi optimizer cập nhật. Loss dùng CrossEntropy, optimizer dùng Adam.
>
> **Vòng validate** khác 3 chỗ: gọi `model.eval()` cho nó tắt dropout, bọc `no_grad()` cho nhẹ, và không học gì hết — chỉ đo rồi giữ lại bản tốt nhất theo validation.
>
> **TensorBoard.** Mỗi lần chạy ghi riêng 1 folder: loss, accuracy, learning rate, gradient, histogram cân nặng, ảnh mẫu.
>
> → mở TensorBoard `results/runs/20261030-023250/densenet121-lr0.001-bs32-0/tb`
>
> **Chỉnh siêu tham số.** Em chạy thật 2 sweep: 3 model fine-tune 5 epoch, với sweep learning rate nhân batch-size.
>
> Kết quả validation tốt nhất:
>
> | Cấu hình | Val tốt nhất | Nói gì |
> |-----------|-------------|--------|
> | ResNet18 freeze, lr 1e-3 | 0,5654 | mốc để so |
> | ResNet18 fine-tune | 0,8296 | mở block cuối lên hẳn 0,83 |
> | VGG11 fine-tune | 0,8316 | ngang ResNet |
> | DenseNet121 fine-tune | **0,8524** | ngon nhất |
> | lr 1e-2 quá lớn | 0,4674 | toang — chứng tỏ phải chỉnh lr |
> | lr 1e-4 quá nhỏ | 0,5558 | học chậm mà chắc |
>
> Chốt: lr 1e-3 là hợp nhất cho freeze. Batch 32 với 64 chênh nhau có 0,01 — không đáng kể."

---

## 6. Chấm test · 1m30

> "Chấm trên 10.000 ảnh test, model chưa thấy bao giờ.
>
> → mở `results/runs/20261030-023250/densenet121-lr0.001-bs32-0/eval/metrics.json`
>
> | Model | Test | F1 | Tốc độ |
> |-------|------|----|--------|
> | ResNet18 freeze | 55,82% | 55,35% | 4,3 ms/ảnh |
> | ResNet18 fine-tune | 82,73% | 82,58% | 13,9 ms/ảnh |
> | VGG11 fine-tune | 83,31% | 83,05% | 20,6 ms/ảnh |
> | **DenseNet121 fine-tune** | **84,74%** | **84,68%** | 8,6 ms/ảnh |
>
> Nói thẳng: freeze chỉ được 55,8%, rớt mốc 70%. Nhưng cái này em lường trước — ảnh test có 64×64 mà chỉ train mỗi cái đuôi 5 nghìn tham số thì đòi hỏi gì. Quan trọng là cả 3 con fine-tune đều qua mốc 80%: 82,7 — 83,3 — ngon nhất DenseNet 84,74%.
>
> Soi lỗi con ngon nhất: mèo điểm thấp nhất 0,72; ô tô cao nhất 0,92. Nhầm nhiều nhất là mèo đoán ra chó — 147 tấm — và chó đoán ra mèo — 99 tấm. Đúng như em đoán từ đầu: mấy con lông xù nhìn giống nhau là khó nhất.
>
> → mở `.../eval/confusion_matrix.png`
>
> Đường chéo là đoán đúng từng lớp. Hai ô mèo–chó sáng nhất ngoài đường chéo. Cộng hết ma trận đúng 10.000 tấm.
>
> → mở `.../eval/per_class_f1.png`"

---

## 7. Chốt · 30s

> "Em chốt 3 câu.
>
> Một, mượn kiến thức ImageNet là đúng: chỉ train đuôi thì rẻ mà yếu — 55,8%. Mở thêm block cuối thì lên 82 tới 85%. Ngon nhất là DenseNet121: 84,74%.
>
> Hai, learning rate quyết định: 1e-2 thì toang xuống 0,47; 1e-4 thì chậm; 1e-3 là vừa.
>
> Ba, em nói luôn điểm yếu: ảnh sweep chỉ 64 nên điểm thấp hơn ảnh 224 chuẩn; train CPU nên VGG 1 epoch 23 phút; mới 5 epoch nên chưa hết cỡ. Muốn ngon hơn thì ảnh 224, nhiều epoch hơn, thêm scheduler.
>
> Code, 102 test pass, số liệu, TensorBoard — em để hết trên GitHub. Em xong, cô hỏi gì em trả lời."

---

## TRA CỨU (đừng đọc) — số nào nằm ở đâu

| Nói câu gì | Mở file nào |
|-----------|-------------|
| 50k train / 10k test | `data/README.md`, run `20261025-150847/metrics.json` |
| MD5 pass | `scripts/verify_data.py` — mã `c58f...49a` |
| pixel 119/63 | `data/README.md` |
| chia 45k/5k/10k, trùng 0, seed 42 | `data/splits/cifar10_split_seed42.json` |
| transform | `src/data/transforms.py` |
| tham số 3 model, ra 10 lớp | `results/runs/20261026-*/model_info.json` |
| VGG đuôi 103M | `docs/reports/m2_models_report.md` |
| Trainer 6 bước | `src/training/trainer.py`, hàm `train_one_epoch` |
| eval + no_grad | `src/training/trainer.py`, hàm `validate` |
| lưu best/last | `src/training/trainer.py`, hàm `save_checkpoint` |
| tag TensorBoard | `src/training/tensorboard_utils.py` |
| val tốt nhất 4 cấu hình | `results/runs/*/metrics.json` — `train_summary.best_val_accuracy` |
| sweep lr×bs | `results/runs/20261030-043458/sweep_summary.json` |
| test 4 model | `results/runs/*/eval/metrics.json` — `test.accuracy`, `test.macro_f1` |
| mèo F1 0,72 thấp nhất | DenseNet `eval/metrics.json` — `per_class["3"].f1` |
| mèo↔chó 147/99 | DenseNet `eval/metrics.json` — `confusion_matrix[3][5]`, `[5][3]` |
| 102 test pass | gõ `pytest -q` — 47 + 24 + 13 + 18 |
| bug DenseNet | `src/models/registry.py`, test `test_finetune_all_models` |
