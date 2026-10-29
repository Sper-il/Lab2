"""Test integration M3: 1 epoch that, checkpoint, early stopping (FakeData, <10s)."""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision import transforms

from src.training.trainer import TrainConfig, Trainer


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.fc = nn.Linear(4 * 4 * 4, 10)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


def _loaders(n_train=32, n_val=16):
    tf = transforms.Compose([transforms.ToTensor()])
    train = DataLoader(FakeData(n_train, (3, 16, 16), 10, tf), batch_size=8)
    val = DataLoader(FakeData(n_val, (3, 16, 16), 10, tf), batch_size=8)
    return train, val


def _trainer(tmp_path, **kw):
    cfg = TrainConfig(epochs=1, batch_size=8, lr=1e-3, device="cpu", **kw)
    train_loader, val_loader = _loaders()
    run_dir = Path(tmp_path) / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return Trainer(cfg, TinyCNN(), train_loader, val_loader, run_dir=run_dir)


def test_one_epoch_runs_and_updates_weights(tmp_path):
    t = _trainer(tmp_path)
    before = {k: v.clone() for k, v in t.model.state_dict().items()}
    out = t.train()
    assert out["epochs_run"] == 1
    changed = any(not torch.equal(before[k], v) for k, v in t.model.state_dict().items())
    assert changed


def test_gradients_flow_to_head(tmp_path):
    t = _trainer(tmp_path)
    t.train_one_epoch(0)
    assert t.model.fc.weight.grad is not None


def test_checkpoint_saved_and_reload(tmp_path):
    t = _trainer(tmp_path)
    t.train()
    best = t.run_dir / "checkpoints" / "best.pt"
    assert best.exists()
    m2 = TinyCNN()
    ckpt = torch.load(best, map_location="cpu", weights_only=False)
    m2.load_state_dict(ckpt["state_dict"])
    assert "best_val_acc" in ckpt


def test_early_stopping_triggers(tmp_path):
    cfg = TrainConfig(epochs=10, batch_size=8, lr=0.0, device="cpu",
                      early_stopping_patience=1)
    train_loader, val_loader = _loaders()
    run_dir = Path(tmp_path) / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    t = Trainer(cfg, TinyCNN(), train_loader, val_loader, run_dir=run_dir)
    out = t.train()
    assert out["stopped_early"] is True
    assert out["epochs_run"] < 10
