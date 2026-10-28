import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    macro_f1,
    per_class_metrics,
)


def test_accuracy_perfect():
    p = torch.tensor([0, 1, 2, 3])
    t = torch.tensor([0, 1, 2, 3])
    assert accuracy(p, t) == 1.0


def test_accuracy_wrong():
    p = torch.tensor([0, 1, 2, 3])
    t = torch.tensor([3, 2, 1, 0])
    assert accuracy(p, t) == 0.0


def test_accuracy_half():
    p = torch.tensor([0, 1, 2, 3])
    t = torch.tensor([0, 1, 3, 2])
    assert accuracy(p, t) == 0.5


def test_macro_f1_perfect():
    p = torch.tensor([0, 0, 1, 1, 2, 2])
    t = torch.tensor([0, 0, 1, 1, 2, 2])
    assert macro_f1(p, t, n_classes=3) == pytest.approx(1.0)


def test_macro_f1_hand_computed():
    # Lớp 0: 1 tp, 0 fp, 1 fn -> p=1, r=0.5, f1=2/3
    # Lớp 1: 1 tp, 1 fp, 0 fn -> p=0.5, r=1, f1=2/3
    p = torch.tensor([0, 1, 1])
    t = torch.tensor([0, 0, 1])
    assert macro_f1(p, t, n_classes=2) == pytest.approx(2 / 3, rel=1e-4)


def test_confusion_matrix_diagonal():
    p = torch.tensor([0, 1, 2, 0, 1, 2])
    t = torch.tensor([0, 1, 2, 0, 1, 2])
    cm = confusion_matrix(p, t, n_classes=3)
    assert np.array_equal(np.diag(cm), np.array([2, 2, 2]))


def test_confusion_matrix_off_diagonal():
    p = torch.tensor([0, 1])
    t = torch.tensor([1, 0])
    cm = confusion_matrix(p, t, n_classes=2)
    assert cm[1, 0] == 1
    assert cm[0, 1] == 1
    assert cm[0, 0] == 0


def test_confusion_matrix_shape():
    p = torch.tensor([0, 1, 2])
    t = torch.tensor([0, 1, 2])
    cm = confusion_matrix(p, t, n_classes=10)
    assert cm.shape == (10, 10)


def test_confusion_matrix_rows_sum():
    p = torch.tensor([0, 1, 0, 2, 2, 2])
    t = torch.tensor([0, 0, 1, 1, 2, 2])
    cm = confusion_matrix(p, t, n_classes=3)
    assert cm.sum(axis=1).tolist() == [2, 2, 2]


def test_per_class_support_sums():
    p = torch.tensor([0, 1, 2, 0, 1, 2])
    t = torch.tensor([0, 1, 2, 0, 1, 2])
    pcm = per_class_metrics(p, t, n_classes=3)
    total_support = sum(v["support"] for v in pcm.values())
    assert total_support == 6


def test_metrics_empty_input_raises():
    with pytest.raises(ValueError):
        accuracy(torch.tensor([]), torch.tensor([]))


def test_label_out_of_range_raises():
    p = torch.tensor([0, 1, 5])
    t = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError):
        confusion_matrix(p, t, n_classes=3)


def test_classification_report_contains_rows():
    p = torch.tensor([0, 1, 2, 0, 1, 2])
    t = torch.tensor([0, 1, 2, 0, 1, 2])
    report = classification_report(p, t, n_classes=3)
    assert "accuracy" in report
    assert "macro avg" in report