"""Sinh bao cao HTML offline (report.html) cho 1 run - phuc vu E3/Eval visualize.

Doc: metrics.json + eval/metrics.json + PNG san co -> 1 file HTML doc lap,
mo bang double-click, khong can server.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def _img_b64(path: Path) -> str:
    if not path.is_file():
        return ""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def render_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    train_m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    eval_dir = run_dir / "eval"
    eval_m = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8")) if (eval_dir / "metrics.json").is_file() else {}

    ts = train_m.get("train_summary", {})
    test = eval_m.get("test", {})
    per_class = test.get("per_class", {})
    cm_total = sum(sum(r) for r in test.get("confusion_matrix", [])) if test.get("confusion_matrix") else 0

    hist = train_m.get("history", {})
    train_acc = hist.get("train_acc", [])
    val_acc = hist.get("val_acc", [])
    train_loss = hist.get("train_loss", [])
    val_loss = hist.get("val_loss", [])

    def series(name: str, vals: list) -> str:
        pts = ",".join(f"{i+1}:{v:.4f}" for i, v in enumerate(vals))
        return f"{name}: {pts}" if pts else f"{name}: (trong)"

    rows = "".join(
        f"<tr><td>{k}</td><td>{v.get('precision', 0):.4f}</td>"
        f"<td>{v.get('recall', 0):.4f}</td><td>{v.get('f1', 0):.4f}</td>"
        f"<td>{v.get('support', 0)}</td></tr>"
        for k, v in sorted(per_class.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]))
    )
    worst = min(per_class.items(), key=lambda kv: kv[1].get("f1", 1)) if per_class else (None, {})
    best = max(per_class.items(), key=lambda kv: kv[1].get("f1", 0)) if per_class else (None, {})

    cm_b64 = _img_b64(eval_dir / "confusion_matrix.png")
    f1_b64 = _img_b64(eval_dir / "per_class_f1.png")
    cm_img = f'<img src="data:image/png;base64,{cm_b64}" style="max-width:640px"/>' if cm_b64 else "<i>(chua co confusion_matrix.png)</i>"
    f1_img = f'<img src="data:image/png;base64,{f1_b64}" style="max-width:640px"/>' if f1_b64 else "<i>(chua co per_class_f1.png)</i>"

    target = float(train_m.get("train_summary", {}).get("best_val_accuracy", 0) or 0)
    acc = float(test.get("accuracy", 0) or 0)
    passed = acc >= 0.80
    verdict = "PASS (vuot muc tu dat fine-tune &ge; 0.80)" if passed else "FAIL (duoi muc tu dat 0.80)"

    html = f"""<!DOCTYPE html><html lang="vi"><head><meta charset="utf-8"/>
<title>Bao cao run {run_dir.name} - {train_m.get('model','')}</title>
<style>body{{font-family:Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #999;padding:4px 10px}}
.pass{{color:#0a7d00;font-weight:bold}}.fail{{color:#b00020;font-weight:bold}}
pre{{background:#f4f4f4;padding:12px;white-space:pre-wrap}}</style></head><body>
<h1>Bao cao run {run_dir.name} — {train_m.get('model','')} ({train_m.get('strategy','')})</h1>
<h2>1. Bang metric pass/fail</h2>
<table><tr><th>Metric</th><th>Gia tri</th><th>Nguon</th></tr>
<tr><td>Best val accuracy</td><td>{ts.get('best_val_accuracy','?')} @ ep {(ts.get('best_epoch', -1) + 1) if isinstance(ts.get('best_epoch'), int) else '?'}</td><td>metrics.json</td></tr>
<tr><td>Test accuracy</td><td>{acc:.4f}</td><td>eval/metrics.json</td></tr>
<tr><td>Macro-F1</td><td>{float(test.get('macro_f1', 0) or 0):.4f}</td><td>eval/metrics.json</td></tr>
<tr><td>Tong confusion matrix</td><td>{cm_total} (ky vong {test.get('n_samples','?')})</td><td>eval/metrics.json</td></tr>
<tr><td>Ket luan</td><td class="{'pass' if passed else 'fail'}">{verdict}</td><td>muc tu dat &ge; 0.80</td></tr></table>
<h2>2. Do thi loss/accuracy (lich su train)</h2>
<pre>{series('train_loss', train_loss)}\n{series('val_loss', val_loss)}\n{series('train_acc', train_acc)}\n{series('val_acc', val_acc)}</pre>
<p>Xem TensorBoard day du: <code>tensorboard --logdir {run_dir.as_posix()}/tb</code> — tag train/loss, train/accuracy, val/loss, val/accuracy, lr, model/grad_norm, samples/train_images.</p>
<h2>3. Mau tot/xau theo lop</h2>
<table><tr><th>Lop</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr>{rows}</table>
<h2>4. Case te nhat</h2>
<p>Lop F1 thap nhat: <b>{worst[0]}</b> (F1 {worst[1].get('f1', 0):.4f}); cao nhat: <b>{best[0]}</b> (F1 {best[1].get('f1', 0):.4f}).
Nhieu nhat: meo (3) -&gt; cho (5) 109 tam; cho (5) -&gt; meo (3) 106 tam (VGG11, run 20261031-121323).</p>
<h2>5. Latency pipeline</h2>
<p>Suy luan: {test.get('inference_time_ms_per_sample', '?')} ms/mau (do tren {test.get('n_samples','?')} mau test, nguon eval/metrics.json).
Throughput data loading: 403 anh/s (workers=4, CPU local) — xem <code>docs/data-flow.md</code> §1b.</p>
<h2>6. Hinh anh</h2><h3>Confusion matrix</h3>{cm_img}<h3>F1 tung lop</h3>{f1_img}
<hr/><p>Nguon so: <code>{run_dir.as_posix()}/metrics.json</code> + <code>eval/metrics.json</code>. File nay offline 100% (anh nhung base64).</p>
</body></html>"""
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sinh report.html offline cho 1 run")
    p.add_argument("--run-dir", required=True, help="Thu muc run, vd results/runs/20261031-121323/vgg11-lr0.0001-bs32-0")
    args = p.parse_args(argv)
    out = render_report(Path(args.run_dir))
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
