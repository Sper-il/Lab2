# Mô tả project / module (DEFINE)

> Bản tóm tắt định nghĩa bài toán theo chuẩn `modules/description.md` của skill `ai-fusion-workflow`.
> **Nguồn chuẩn đầy đủ:** bảng "Project Context" trong `docs/progress_status.md` (luôn cập nhật song song).
> Mọi con số dưới đây đều đã đo thật — nguồn artifact ghi kèm từng mục.

## 1. Project

| Mục | Nội dung |
|-----|----------|
| Tên đầy đủ | Practice 2 — Hands-on with Pre-trained Neural Network Architectures |
| Slug | `lab2` |
| Tagline | Fine-tune backbone ImageNet trên CIFAR-10: so 3 kiến trúc, sweep siêu tham số, giám sát bằng TensorBoard |
| Bài toán | Phân loại ảnh CIFAR-10 (10 lớp: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) bằng model pre-trained ImageNet |
| Vì sao quan trọng | CIFAR-10 chỉ 50.000 ảnh 32×32 — train từ đầu dễ overfit; transfer learning (giữ feature extractor, thay head) là cách chuẩn, đúng nội dung Chương 3 (slide 20–21) |

## 2. Người dùng mục tiêu

Sinh viên môn AI/ML — Viện CNTT&ĐT, TSU TP.HCM. Kỳ vọng ở output: chạy được end-to-end, số liệu trung thực có nguồn, giải thích được từng bước từ data → model → train → eval.

## 3. Input / Output

- **Input:** ảnh RGB 32×32 × 3 → resize 64×64 (train/eval chốt, GPU T4 + AMP) hoặc 224×224 (demo M1 chuẩn ImageNet); batch `[B,3,64,64]` / `[B,3,224,224]`, `float32`, normalize theo ImageNet mean/std. Nguồn: `torchvision.datasets.CIFAR-10` (`data/raw/`), MD5 `c58f30108f718f92721af3b95e74349a` — PASS.
- **Output:** logits `[B,10]` → `argmax` ra nhãn + softmax probability.
- **Chia dữ liệu:** 45.000 train (4.500/lớp) / 5.000 val (500/lớp, stratified, seed 42) / 10.000 test giữ kín tới cuối. Overlap 3 cặp = 0 (hash nội dung ảnh).

## 4. Tính năng cốt lõi (Must / Should / Could)

- **Must:** nạp pre-trained ResNet18/VGG11/DenseNet121 → thay head 10 lớp → freeze/fine-tune → training loop 6 bước → validation + `best.pt` → TensorBoard → eval test + confusion matrix + `metrics.json`. (Tất cả ✅)
- **Should:** config YAML mọi experiment, bảng so sánh sweep, augmentation, `metrics.json` + `config.yaml` mỗi run. (Tất cả ✅)
- **Could:** scheduler, histogram trọng số, gradient clipping, ONNX. (Scheduler ✅, ONNX chưa làm — xem `README.md` §10.)

## 5. Tiêu chí công nghệ (Where / When / Which)

- **Where:** train nặng trên **GPU Colab T4 + AMP** (sweep 30 cấu hình 5 epoch + retrain 20 epoch, ảnh 64, ~27–86 giây/epoch); local CPU-only cho pipeline M1/demo/test.
- **When:** chạy offline theo batch experiment, mỗi run sinh `results/runs/<timestamp>/`.
- **Which:** Python 3.13, PyTorch 2.14.1 + torchvision 0.29.1 + TensorBoard 2.21.0 (pin trong `requirements.txt`); metric: val accuracy (chọn best), test accuracy + macro-F1 + confusion matrix (báo cáo cuối).
- **Target tự đặt** (đề bài không cho ngưỡng — đã ghi rõ nguồn gốc): freeze ≥ 0.70, fine-tune ≥ 0.80, val−train gap ≤ 0.15. Kết quả: cả 3 model fine-tune đều vượt 0.86 ✅.

## 6. Mô tả từng module (Describe)

| Module | Làm gì | Input → Output | Tiêu chí thành công |
|--------|--------|----------------|---------------------|
| M1 Data | Tải/verify/chia/transform/dataloader CIFAR-10 | `data/raw/*.tar.gz` → 3 loader + `metrics.json` (`validation.status = PASS`) | 47 test pass; train+val = 50.000; overlap = 0 |
| M2 Model | Factory 3 backbone + thay head + freeze/unfreeze + CLI khám phá | `ModelConfig` → model output `(B,10)` + `model_info.json` | 24 test pass; verify weights thật cả 3 model |
| M3 Training | `Trainer` + loss/opt/scheduler + TensorBoard + CLI sweep | config YAML → `best.pt` + `metrics.json` + `tb/` | loss giảm; val acc tăng; 13 test pass |
| M4 Evaluation | `evaluate_model` + plots + CLI eval + `report.html` | `best.pt` → test acc/macro-F1/CM + PNG + HTML | tổng CM = 10.000; 18 test pass |
| M5 Experiment | Sweep 30 cấu hình → retrain 3 tốt nhất 20 epoch → `OTM_RESULTS.md` + `TEST_RESULTS.md` | log sweep → bảng so sánh + kết luận lr/bs | VGG11 test **0.8774** tốt nhất |
| Demo | 1 lệnh chạy checkpoint tốt nhất trên ảnh thật, đo latency | `--checkpoint + --num-samples` → dự đoán + `results/demo/<ts>/index.html` | chạy được, latency đo thật |
