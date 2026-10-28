
import torch
import numpy as np


def _prepare_tensors(pred, target):
    """Chuyển pred và target về tensor 1 chiều kiểu long."""
    pred = torch.as_tensor(pred).detach().cpu().long().reshape(-1)
    target = torch.as_tensor(target).detach().cpu().long().reshape(-1)

    if pred.numel() != target.numel():
        raise ValueError("pred và target phải có cùng số phần tử.")

    return pred, target


def accuracy(pred, target):
    """Tính tỷ lệ dự đoán đúng."""
    pred, target = _prepare_tensors(pred, target)

    if target.numel() == 0:
        raise ValueError("Không thể tính accuracy trên dữ liệu rỗng.")

    return (pred == target).float().mean().item()


def confusion_matrix(pred, target, n_classes=10):
    """
    Hàng là lớp thật, cột là lớp dự đoán.
    Trả về numpy array có kích thước (n_classes, n_classes).
    """
    pred, target = _prepare_tensors(pred, target)

    if n_classes <= 0:
        raise ValueError("n_classes phải lớn hơn 0.")

    if pred.numel() > 0:
        if (
            pred.min() < 0 or target.min() < 0
            or pred.max() >= n_classes
            or target.max() >= n_classes
        ):
            raise ValueError("Nhãn nằm ngoài phạm vi lớp hợp lệ.")

    cm = torch.zeros(
        (n_classes, n_classes), dtype=torch.int64
    )

    if pred.numel() > 0:
        indices = target * n_classes + pred
        cm = torch.bincount(
            indices, minlength=n_classes * n_classes
        ).reshape(n_classes, n_classes)

    return cm.numpy()


def per_class_metrics(pred, target, n_classes=10):
    """Tính precision, recall, F1 và support cho từng lớp."""
    cm = torch.as_tensor(
        confusion_matrix(pred, target, n_classes)
    ).float()

    tp = cm.diag()
    predicted_count = cm.sum(dim=0)
    actual_count = cm.sum(dim=1)

    precision = torch.where(
        predicted_count > 0,
        tp / predicted_count.clamp(min=1),
        torch.zeros_like(tp)
    )

    recall = torch.where(
        actual_count > 0,
        tp / actual_count.clamp(min=1),
        torch.zeros_like(tp)
    )

    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall
        / (precision + recall).clamp(min=1e-12),
        torch.zeros_like(tp)
    )

    result = {}

    for i in range(n_classes):
        result[i] = {
            "p": precision[i].item(),
            "r": recall[i].item(),
            "f1": f1[i].item(),
            "support": int(actual_count[i].item())
        }

    return result


def macro_f1(pred, target, n_classes=10):
    """F1 macro: trung bình F1 của tất cả các lớp."""
    metrics = per_class_metrics(pred, target, n_classes)

    if not metrics:
        return 0.0

    return sum(
        item["f1"] for item in metrics.values()
    ) / len(metrics)


def classification_report(pred, target, n_classes=10,
                          class_names=None):
    """Tạo báo cáo phân loại, không cần scikit-learn."""
    metrics = per_class_metrics(pred, target, n_classes)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    if len(class_names) != n_classes:
        raise ValueError("class_names phải có n_classes phần tử.")

    lines = [
        f"{'Class':<16}{'Precision':>12}{'Recall':>12}"
        f"{'F1-score':>12}{'Support':>10}"
    ]

    total_support = sum(
        item["support"] for item in metrics.values()
    )

    weighted_p = weighted_r = weighted_f1 = 0.0

    for i, item in metrics.items():
        name = str(class_names[i])
        p, r = item["p"], item["r"]
        f1, support = item["f1"], item["support"]

        lines.append(
            f"{name:<16}{p:>12.4f}{r:>12.4f}"
            f"{f1:>12.4f}{support:>10}"
        )

        weighted_p += p * support
        weighted_r += r * support
        weighted_f1 += f1 * support

    acc = accuracy(pred, target) if total_support else 0.0
    macro = macro_f1(pred, target, n_classes)

    if total_support:
        weighted_p /= total_support
        weighted_r /= total_support
        weighted_f1 /= total_support

    lines.extend([
        "-" * 62,
        f"Accuracy: {acc:.4f}",
        f"Macro F1: {macro:.4f}",
        f"Weighted Precision: {weighted_p:.4f}",
        f"Weighted Recall: {weighted_r:.4f}",
        f"Weighted F1: {weighted_f1:.4f}"
    ])

    return "\n".join(lines)


def evaluate_model(model, loader, device, n_classes=10):
    """
    Đánh giá model trên loader.
    Giả định mỗi batch là (inputs, targets), model trả về logits.
    """
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).long().reshape(-1)

            logits = model(inputs)

            if logits.ndim != 2 or logits.shape[1] != n_classes:
                raise ValueError(
                    "Model phải trả logits dạng (batch_size, n_classes)."
                )

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

    if not all_targets:
        raise ValueError("DataLoader không có dữ liệu.")

    pred = torch.cat(all_preds)
    target = torch.cat(all_targets)
    probs = torch.cat(all_probs)

    per_class = per_class_metrics(pred, target, n_classes)
    cm = confusion_matrix(pred, target, n_classes)

    return {
        "accuracy": accuracy(pred, target),
        "macro_f1": macro_f1(pred, target, n_classes),
        "per_class": per_class,
        "confusion_matrix": cm,
        "classification_report": classification_report(
            pred, target, n_classes
        ),
        "predictions": pred.numpy(),
        "targets": target.numpy(),
        "probabilities": probs.numpy()
    }

