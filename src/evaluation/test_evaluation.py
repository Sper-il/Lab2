import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import evaluate_model
from src.evaluation.visualize import plot_confusion_matrix, plot_per_class_f1


class DummyModel(nn.Module):
    """Model nhỏ: flatten -> linear, output 3 lớp."""

    def __init__(self, n_classes=3):
        super().__init__()
        self.fc = nn.Linear(3 * 4 * 4, n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


def _make_loader(n=20, n_classes=3, batch_size=8):
    torch.manual_seed(0)
    x = torch.randn(n, 3, 4, 4)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_evaluate_model_end_to_end():
    model = DummyModel(n_classes=3)
    loader = _make_loader()
    result = evaluate_model(model, loader, device="cpu", n_classes=3)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert result.n_samples == 20
    assert set(result.per_class.keys()) == {"0", "1", "2"}


def test_metrics_json_schema(tmp_path):
    model = DummyModel(n_classes=3)
    loader = _make_loader()
    result = evaluate_model(model, loader, n_classes=3)
    schema = result.to_json_dict()
    assert "accuracy" in schema
    assert "macro_f1" in schema
    assert "per_class" in schema
    assert "confusion_matrix" in schema
    assert "inference_time_ms_per_sample" in schema
    # ghi ra file được
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(schema))
    assert json.loads(p.read_text())["accuracy"] == schema["accuracy"]


def test_plots_generated(tmp_path):
    cm = [[5, 1], [2, 4]]
    p1 = plot_confusion_matrix(cm, out_path=tmp_path / "cm.png")
    assert Path(p1).exists() and Path(p1).stat().st_size > 0

    per_class = {
        "0": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 6},
        "1": {"precision": 0.6, "recall": 0.7, "f1": 0.65, "support": 6},
    }
    p2 = plot_per_class_f1(per_class, out_path=tmp_path / "f1.png")
    assert Path(p2).exists() and Path(p2).stat().st_size > 0


def test_checkpoint_reload_consistency(tmp_path):
    model = DummyModel(n_classes=3)
    loader = _make_loader()
    r1 = evaluate_model(model, loader, n_classes=3)

    ckpt = tmp_path / "best.pt"
    torch.save(model.state_dict(), ckpt)

    model2 = DummyModel(n_classes=3)
    model2.load_state_dict(torch.load(ckpt))
    r2 = evaluate_model(model2, loader, n_classes=3)

    assert r1.accuracy == pytest.approx(r2.accuracy)
    assert r1.macro_f1 == pytest.approx(r2.macro_f1)


def test_confusion_matrix_sum_equals_samples():
    model = DummyModel(n_classes=3)
    loader = _make_loader(n=20)
    result = evaluate_model(model, loader, n_classes=3)
    total = sum(sum(row) for row in result.confusion_matrix)
    assert total == 20