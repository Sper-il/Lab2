# LỜI THOẠI TRÌNH BÀY — Practice 2: Pre-trained Neural Network Architectures trên CIFAR-10 (10 phút)

> Nguồn số liệu: `results/runs/` (4 eval thật, test 10.000 mẫu). Quy ước đọc số: đọc 0.8474 thành "tám mươi tư phẩy bảy mươi tư phần trăm".
> Ánh xạ đề: E1–E3 = 3 bài tập trong `đề.txt`; O1–O11 = 7 bước (Step 1–7) trong đề, chi tiết ở slide 23–24 tài liệu Chương 3.

---

## PHẦN 1 — Mở đầu · 40s · `0:00`

> "Em chào cô. Em tên là Sper-il.
>
> Hôm nay em trình bày Practice 2 — thực hành với các kiến trúc mạng nơ-ron pre-trained, trên tập dữ liệu CIFAR-10.
>
> Em chọn bài này vì nó trả lời một câu hỏi rất thực tế: khi mình chỉ có 50 nghìn ảnh nhỏ 32×32, làm sao tận dụng được kiến thức của model đã học trên hơn 1 triệu ảnh ImageNet, thay vì train từ đầu và overfit ngay. Toàn bộ quá trình — nạp model, thay head, freeze, fine-tune, tinh chỉnh siêu tham số, theo dõi TensorBoard — em chạy bằng code thật trên CPU, không làm thủ công.
>
> Source em đẩy lên GitHub công khai, cô kiểm tra bất cứ lúc nào ạ."

---

## PHẦN 2 — Bài toán & mục tiêu · 50s · `0:40`

> "Bài toán của em: một ảnh màu 32×32, máy phải trả lời là vật gì trong 10 loại — máy bay, ô tô, chim, mèo, hươu, chó, ếch, ngựa, tàu, xe tải.
>
> → mở `data/README.md` (phân bố lớp)
>
> Tập dữ liệu có 60 nghìn ảnh: 50 nghìn train, 10 nghìn test, chia đều 10 lớp — mỗi lớp 5 nghìn train, 1 nghìn test.
>
> → mở `results/runs/20261025-150847/plots/class_distribution.png`
>
> Mười cột cao đều nhau — dataset cân bằng hoàn hảo, đã verify MD5 archive khớp chuẩn. Đây là CIFAR-10, đúng dataset ví dụ ở slide 24 tài liệu Chương 3.
>
> → mở `results/runs/20261025-150847/plots/sample_images.png`
>
> Cô thấy ảnh hơi mờ — đó không phải dữ liệu hỏng. Em đo trên 256 ảnh đầu: mean 119, std 63, chỉ 0,3% pixel đen tuyệt đối. Ảnh mờ vì CIFAR-10 gốc chỉ 32×32, resize lên 224 chỉ nội suy chứ không tạo thêm chi tiết. Và em dự đoán trước: cặp mèo–hươu–chim nhìn giống nhau thế này sẽ là chỗ khó nhất — em sẽ quay lại ở phần 6.
>
> Trước khi code, em tự đặt mục tiêu: freeze backbone đạt tối thiểu 70% test accuracy, fine-tune đạt tối thiểu 80%. Em nói rõ luôn: đề bài không cho ngưỡng nào, đây là mốc tự đặt để biết khi nào dừng. Có mốc thì em biết kết quả của mình đang ở đâu."

---

## PHẦN 3 — Chuẩn bị dữ liệu · 1m30 · `1:30`

> "Giai đoạn một em làm bốn việc, đúng Step 5 của đề và slide 4–5 Chương 3.
>
> **Một, tải và verify.** Em tải qua `torchvision.datasets` với `download=True`, verify MD5 archive `c58f...49a` khớp — PASS. File gốc nặng 163 MB, nạp hết vào RAM vì máy này I/O là nút thắt.
>
> **Hai, chia ba tập — không phải hai.** Train 45 nghìn, validation 5 nghìn lấy từ tập train, test 10 nghìn giữ kín.
>
> → mở `data/splits/cifar10_split_seed42.json`
>
> Nếu chỉ chia hai tập thì có một cái bẫy: model học thuộc luôn phần test, điểm đẹp lên nhưng đẹp sai nghĩa. Em chia phân tầng — mỗi lớp đúng 4.500 train, 500 val, 1.000 test — và seed cố định 42 để chạy lại ra đúng kết quả cũ. Overlap train/val, train/test, val/test đều bằng 0, kiểm bằng hash nội dung ảnh.
>
> **Ba, transform.** Train: RandomCrop 32 padding 4 + lật ngang, rồi resize lên 224, chuyển tensor, normalize theo mean/std ImageNet. Val và test: chỉ resize + normalize, không augmentation để kết quả xác định.
>
> → mở `src/data/transforms.py` (hàm `build_train_transform`, `build_val_transform`)
>
> **Bốn, DataLoader.** Shuffle chỉ ở train, 4 worker, batch 64 cho Tier A.
>
> Kết quả M1: 47 test pass, run `20261025-150847` ghi `validation.status = PASS`."

---

## PHẦN 4 — Mô hình: nạp, khám phá, adapt · 2m · `3:00`

> "Phần này trả lời Step 2, 3, 4 của đề và slide 6, 20, 21 Chương 3.
>
> **Nạp model — Step 2.** Một dòng duy nhất: `torchvision.models.resnet18(weights="DEFAULT")`. Em làm với cả ba backbone: ResNet18, VGG11, DenseNet121 — đúng yêu cầu E1 thử nhiều pre-trained model.
>
> **Khám phá — Step 3.** Em in cấu trúc và đếm tham số:
>
> → mở `results/runs/20261026-115633/model_info.json` (và 2 file tương tự)
>
> | Model | Tổng params | Trainable khi freeze | Tỉ lệ |
> |-------|-------------|---------------------|-------|
> | ResNet18 | 11.181.642 | 5.130 | 0,046% |
> | VGG11 | 128.807.306 | 40.970 | 0,032% |
> | DenseNet121 | 6.964.106 | 10.250 | 0,147% |
>
> Output cả ba đều ra đúng `(1, 10)`. Chỗ này có một cái bẫy em bắt được nhờ introspect thật: `classifier` của VGG11 không phải 1 lớp Linear mà là Sequential 7 phần tử chứa tới 103 triệu params. Unfreeze cả là mất ý nghĩa freeze — nên em chỉ mở lớp Linear cuối, còn 40.970 params.
>
> **Adapt — Step 4.** Ba thao tác đúng slide 20–21: một, thay head cuối thành Linear ra 10 lớp; hai, freeze backbone chỉ train head — transfer learning; ba, unfreeze block cuối cho fine-tuning.
>
> Vì máy không có GPU nên em chia hai tầng: Tier A ảnh 224 đúng chuẩn ImageNet để báo cáo, Tier B ảnh 64×64 để sweep nhanh — đã đo thật, freeze ResNet18 Tier A mất 37 phút mỗi epoch, Tier B chỉ 5 phút.
>
> Kết quả M2: 24 test pass, trong đó có test bắt bug DenseNet — hàm lấy module con crash vì key của DenseNet là chuỗi `denseblock4` chứ không phải chỉ số. Em sửa bằng cách ưu tiên attribute trước."

---

## PHẦN 5 — Huấn luyện: loop, TensorBoard, tinh chỉnh · 2m · `5:00`

> "Phần này trả lời Step 6, E2 và E3 của đề, slide 7–15 Chương 3.
>
> Về code, em có lớp `Trainer` trong `src/training/trainer.py`. Mỗi epoch gồm hai vòng.
>
> **Vòng train** đúng 6 bước slide 8–9: chuyển data lên device, zero gradient — vì PyTorch cộng dồn gradient nên phải xóa trước mỗi batch — forward, tính loss CrossEntropyLoss, backward lan truyền ngược, rồi optimizer Adam bước cập nhật trọng số.
>
> **Vòng validate** khác ở ba điểm. Một: gọi `model.eval()` để tắt dropout và dùng running stats của BatchNorm. Hai: bọc trong `torch.no_grad()` để không tính gradient, nhanh và nhẹ bộ nhớ. Ba: model không học gì trên tập validation — chỉ đo, rồi chọn best checkpoint theo val accuracy.
>
> **TensorBoard — E3.** Mỗi run ghi vào thư mục `tb/` riêng: loss và accuracy train/val, learning rate, gradient norm, histogram trọng số, ảnh mẫu đã denormalize.
>
> → mở TensorBoard `results/runs/20261030-023250/densenet121-lr0.001-bs32-0/tb`
>
> **Tinh chỉnh — E2.** Em chạy sweep thật, không smoke test: 3 model fine-tune Tier B, mỗi model 5 epoch, cộng sweep lr nhân batch-size 2×2 trên ResNet18 freeze.
>
> Kết quả train — best val accuracy:
>
> | Config | Best val | Nhận xét |
> |--------|----------|----------|
> | ResNet18 freeze, lr 1e-3 | 0,5654 @ epoch 4 | baseline |
> | ResNet18 fine-tune | 0,8296 @ epoch 5 | vượt xa freeze |
> | VGG11 fine-tune | 0,8316 @ epoch 5 | ngang ResNet |
> | DenseNet121 fine-tune | **0,8524 @ epoch 4** | tốt nhất |
> | lr 1e-2 (quá lớn) | 0,4674 | mất ổn định — bằng chứng phải tune lr |
> | lr 1e-4 (quá nhỏ) | 0,5558 | hội tụ chậm nhưng ổn định |
>
> Kết luận E2 của em: lr 1e-3 tối ưu cho freeze; batch 32 và 64 chênh nhau chỉ khoảng 0,01."

---

## PHẦN 6 — Đánh giá & phân tích lỗi · 1m30 · `7:00`

> "Đây là Step 7 của đề — đánh giá trên test set 10.000 mẫu, model chưa từng thấy.
>
> → mở `results/runs/20261030-023250/densenet121-lr0.001-bs32-0/eval/metrics.json`
>
> | Model | Test accuracy | Macro-F1 | Suy luận |
> |-------|--------------|----------|----------|
> | ResNet18 freeze | 55,82% | 55,35% | 4,3 ms/mẫu |
> | ResNet18 fine-tune | 82,73% | 82,58% | 13,9 ms/mẫu |
> | VGG11 fine-tune | 83,31% | 83,05% | 20,6 ms/mẫu |
> | **DenseNet121 fine-tune** | **84,74%** | **84,68%** | 8,6 ms/mẫu |
>
> Freeze chỉ đạt 55,8% — dưới mốc 70% em tự đặt, nhưng đúng kỳ vọng vì ảnh Tier B chỉ 64×64 mà chỉ train có 5.130 params. Cả ba model fine-tune đều vượt mốc 80%: 82,7 — 83,3 — và tốt nhất là DenseNet121 với 84,74%.
>
> Phân tích lỗi trên model tốt nhất: F1 thấp nhất là mèo 0,72, chó–mèo nhầm nhau; cao nhất là ô tô 0,92 và tàu 0,90. Cặp nhầm nhiều nhất là mèo thật bị đoán thành chó — 147 mẫu — và ngược lại 99 mẫu. Đúng như em dự đoán ở phần 2: nhóm vật nuôi lông xù giống nhau là chỗ khó nhất của CIFAR-10.
>
> → mở `results/runs/20261030-023250/densenet121-lr0.001-bs32-0/eval/confusion_matrix.png`
>
> Đường chéo chính là tỉ lệ đúng từng lớp; hai ô mèo–chó sáng nhất ngoài đường chéo. Tổng ma trận bằng đúng 10.000 mẫu test.
>
> → mở `results/runs/20261030-023250/densenet121-lr0.001-bs32-0/eval/per_class_f1.png`"

---

## PHẦN 7 — Kết luận · 30s · `8:30`

> "Em tóm tắt ba điều.
>
> Một, transfer learning hoạt động đúng lý thuyết slide 20: chỉ train head thì rẻ — 5 nghìn params — nhưng chỉ đạt 55,8%. Mở thêm block cuối để fine-tune thì nhảy lên 82–85%. DenseNet121 tốt nhất: 84,74% test accuracy, macro-F1 84,68%.
>
> Hai, siêu tham số quyết định: lr 1e-2 quá lớn làm val rớt xuống 0,47; lr 1e-4 hội tụ chậm; 1e-3 là điểm ngọt cho freeze.
>
> Ba, hạn chế em nói thẳng: Tier B ảnh 64 nên điểm thấp hơn Tier A 224 chuẩn ImageNet; train CPU nên VGG mất 23 phút mỗi epoch; mới 5 epoch nên chưa hội tụ hẳn. Hướng tiếp theo: ảnh 224, nhiều epoch hơn, scheduler cosine, và thử EfficientNet.
>
> Toàn bộ code, 102 test pass, số liệu và TensorBoard log em đẩy lên GitHub. Em cảm ơn cô và sẵn sàng trả lời câu hỏi ạ."

---

## BẢNG TRA CỨU (KHÔNG đọc ra) — vị trí file cho từng con số

| Nói gì trong lời thoại | File | Dòng / khóa |
|----------------------|------|-------------|
| 50k train / 10k test, 10 lớp | `data/README.md`, run `20261025-150847` | `metrics.json` |
| MD5 archive PASS | `scripts/verify_data.py` | `c58f30108f718f92721af3b95e74349a` |
| mean 119, std 63, 0,3% pixel đen | `data/README.md` | đo 256 ảnh đầu |
| 45k/5k/10k, overlap 0, seed 42 | `data/splits/cifar10_split_seed42.json` | test `test_prepare.py` |
| transform train/val | `src/data/transforms.py` | `build_train_transform` / `build_val_transform` |
| params 3 model + output (1,10) | `results/runs/20261026-*/model_info.json` | `params_total`, `params_trainable`, `output_shape` |
| VGG classifier 103M params | `docs/reports/m2_models_report.md` | § freeze_head_full |
| Trainer 6 bước | `src/training/trainer.py` | `train_one_epoch` 112–140 |
| `model.eval()` + `no_grad()` ở validate | `src/training/trainer.py` | 144–156 |
| save `best.pt`/`last.pt` | `src/training/trainer.py` | `save_checkpoint` 159–169 |
| TB tags | `src/training/tensorboard_utils.py` | `tb_config` |
| best val 4 config | `results/runs/*/metrics.json` | `train_summary.best_val_accuracy` |
| sweep E2 4 config | `results/runs/20261030-043458/sweep_summary.json` | 4 runs |
| test acc/F1/inference 4 model | `results/runs/*/eval/metrics.json` | `test.accuracy`, `test.macro_f1`, `test.inference_time_ms_per_sample` |
| per-class F1 mèo 0,72 thấp nhất | DenseNet `eval/metrics.json` | `test.per_class["3"].f1` |
| nhầm mèo↔chó 147/99 | DenseNet `eval/metrics.json` | `test.confusion_matrix[3][5]`, `[5][3]` |
| 102 test pass | `pytest -q` | 47 M1 + 24 M2 + 13 M3 + 18 M4 |
| bug DenseNet `get_module` | `src/models/registry.py`, `tests/unit/test_factory.py` | `test_finetune_all_models` |
