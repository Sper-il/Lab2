
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PLOT_DIR = Path("plots")


def _save_plot(filename):
    """Lưu biểu đồ vào thư mục plots."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_confusion_matrix(cm, class_names=None):
    """Vẽ heatmap ma trận nhầm lẫn."""
    cm = np.asarray(cm)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm phải là ma trận vuông.")

    n_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    if len(class_names) != n_classes:
        raise ValueError("Số tên lớp không khớp ma trận.")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of samples")

    ax.set(
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix"
    )
    ax.set_xticks(range(n_classes), labels=class_names,
                  rotation=45, ha="right")
    ax.set_yticks(range(n_classes), labels=class_names)

    threshold = cm.max() / 2 if cm.size else 0

    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    return _save_plot("confusion_matrix.png")


def plot_per_class_f1(metrics):
    """Vẽ biểu đồ F1-score theo lớp."""
    class_ids = list(metrics.keys())
    scores = [metrics[c]["f1"] for c in class_ids]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([str(c) for c in class_ids], scores)

    ax.set(
        xlabel="Class",
        ylabel="F1-score",
        title="F1-score per class",
        ylim=(0, 1.05)
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    return _save_plot("per_class_f1.png")


def plot_pr_curve(probabilities, targets, class_names=None):
    """
    Vẽ Precision-Recall curve one-vs-rest.
    probabilities: (N, C), xác suất của từng lớp.
    targets: (N,), nhãn thật.
    """
    probs = np.asarray(probabilities)
    targets = np.asarray(targets).reshape(-1)

    if probs.ndim != 2 or len(targets) != len(probs):
        raise ValueError("Kích thước probabilities hoặc targets không hợp lệ.")

    n_classes = probs.shape[1]

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]

    if len(class_names) != n_classes:
        raise ValueError("Số tên lớp không khớp.")

    if len(targets) == 0:
        raise ValueError("Không thể vẽ PR curve với dữ liệu rỗng.")

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = 0

    for class_id in range(n_classes):
        binary_target = (targets == class_id).astype(int)

        # PR curve không có ý nghĩa nếu lớp này không có
        # mẫu dương hoặc không có mẫu âm.
        if binary_target.sum() == 0 or binary_target.sum() == len(targets):
            continue

        order = np.argsort(-probs[:, class_id])
        sorted_target = binary_target[order]

        tp = np.cumsum(sorted_target)
        fp = np.cumsum(1 - sorted_target)

        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / binary_target.sum()

        # Thêm điểm xuất phát để đường cong đầy đủ hơn.
        precision = np.concatenate(([1.0], precision))
        recall = np.concatenate(([0.0], recall))

        ax.plot(
            recall, precision,
            label=str(class_names[class_id])
        )
        plotted += 1

    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curve",
        xlim=(0, 1.01),
        ylim=(0, 1.05)
    )
    ax.grid(True, linestyle="--", alpha=0.4)

    if plotted:
        ax.legend(title="Class")

    return _save_plot("pr_curve.png")


def plot_confusion_examples(
    images, predictions, targets, class_names=None, max_images=16
):
    """Hiển thị những ảnh model dự đoán sai."""
    images = np.asarray(images)
    predictions = np.asarray(predictions).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    if len(images) != len(predictions) or len(images) != len(targets):
        raise ValueError("images, predictions và targets phải cùng số mẫu.")

    if max_images <= 0:
        raise ValueError("max_images phải lớn hơn 0.")

    wrong = np.flatnonzero(predictions != targets)[:max_images]

    if len(wrong) == 0:
        raise ValueError("Không có ảnh dự đoán sai để hiển thị.")

    n_cols = min(4, len(wrong))
    n_rows = int(np.ceil(len(wrong) / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3 * n_cols, 3 * n_rows),
        squeeze=False
    )

    def label_name(label):
        if class_names is not None:
            return str(class_names[int(label)])
        return str(int(label))

    for ax in axes.flat:
        ax.axis("off")

    for ax, idx in zip(axes.flat, wrong):
        img = images[idx]

        # Hỗ trợ ảnh grayscale (H, W) hoặc (H, W, 1),
        # ảnh RGB (H, W, 3), và tensor dạng (C, H, W).
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            img = np.transpose(img, (1, 2, 0))

        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)

        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(
            f"True: {label_name(targets[idx])}\n"
            f"Pred: {label_name(predictions[idx])}"
        )

    return _save_plot("misclassified.png")


def plot_train_curves(history):
    """
    Vẽ loss/accuracy từ dictionary history.
    Hỗ trợ các khóa train_loss, val_loss,
    train_acc, val_acc.
    """
    if not history:
        raise ValueError("history không được rỗng.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Biểu đồ loss
    loss_found = False

    for key in ("train_loss", "val_loss"):
        if key in history:
            axes[0].plot(history[key], label=key)
            loss_found = True

    axes[0].set(
        xlabel="Epoch",
        ylabel="Loss",
        title="Training and Validation Loss"
    )
    axes[0].grid(True, linestyle="--", alpha=0.4)

    if loss_found:
        axes[0].legend()

    # Biểu đồ accuracy
    acc_found = False

    for key in ("train_acc", "val_acc"):
        if key in history:
            axes[1].plot(history[key], label=key)
            acc_found = True

    axes[1].set(
        xlabel="Epoch",
        ylabel="Accuracy",
        title="Training and Validation Accuracy"
    )
    axes[1].grid(True, linestyle="--", alpha=0.4)

    if acc_found:
        axes[1].legend()

    if not loss_found and not acc_found:
        plt.close(fig)
        raise ValueError("history không có khóa loss hoặc accuracy hợp lệ.")

    return _save_plot("train_curves.png")
