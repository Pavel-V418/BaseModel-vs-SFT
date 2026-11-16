"""Графики обучения: loss, token accuracy, learning rate, grad norm.

Запуск:
    python -m src.plots results/<run>_live.json                 # -> results/<run>_curves.png
    python -m src.plots results/<run>_log_history.json --out x.png

Понимает оба формата:
    *_live.json         — пишет LiveLogCallback во время обучения (есть точка base «до»);
    *_log_history.json  — trainer.state.log_history после обучения (без точки «до»).
В ноутбуке: from src.plots import plot_history; plot_history(path) — вернёт Figure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TRAIN_COLOR = "#2a78d6"   # train — синий
EVAL_COLOR = "#eb6834"    # validation — оранжевый
BASE_COLOR = "#6b6a63"    # base-модель (до обучения) — серый пунктир
MUTED = "#8a897f"


def load_history(src) -> dict:
    """Путь к *_live.json / *_log_history.json или уже загруженный объект -> dict."""
    if isinstance(src, (str, Path)):
        with open(src, encoding="utf-8") as f:
            src = json.load(f)
    if isinstance(src, list):  # trainer.state.log_history
        src = {"history": [{**r, "phase": "train"} for r in src]}
    return src


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _series(rows, key):
    pts = [(r["step"], r[key]) for r in rows if key in r and _finite(r[key])]
    return [p[0] for p in pts], [p[1] for p in pts]


def _ema(ys, alpha=0.2):
    out, m = [], None
    for y in ys:
        m = y if m is None else alpha * y + (1 - alpha) * m
        out.append(m)
    return out


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color="#1a1a19")
    ax.grid(True, color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel("шаг", fontsize=9, color=MUTED)


def plot_history(src, out: str | Path | None = None, title: str | None = None):
    """Рисует 2x2: loss, token accuracy, learning rate, grad norm. Возвращает Figure."""
    import matplotlib.pyplot as plt

    h = load_history(src)
    rows = h.get("history", [])
    before = [r for r in rows if r.get("phase") == "before"]
    after = [r for r in rows if r.get("phase") == "after"]
    train = [r for r in rows if r.get("phase") == "train"]
    train_logs = [r for r in train if "loss" in r]
    eval_logs = [r for r in train if "eval_loss" in r]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    (ax_loss, ax_acc), (ax_lr, ax_gn) = axes

    # --- 1. loss
    xs, ys = _series(train_logs, "loss")
    if xs:
        ax_loss.plot(xs, ys, color=TRAIN_COLOR, alpha=0.25, linewidth=1)
        ax_loss.plot(xs, _ema(ys), color=TRAIN_COLOR, linewidth=2, label="train (сглаж.)")
    xs, ys = _series(eval_logs, "eval_loss")
    if xs:
        ax_loss.plot(xs, ys, color=EVAL_COLOR, linewidth=2, marker="o", markersize=5,
                     label="validation")
        i = min(range(len(ys)), key=ys.__getitem__)
        ax_loss.annotate(f"лучший {ys[i]:.3f}", (xs[i], ys[i]), textcoords="offset points",
                         xytext=(0, -16), ha="center", fontsize=9, color="#1a1a19")
    if before and _finite(before[0].get("eval_loss")):
        b = before[0]["eval_loss"]
        ax_loss.axhline(b, color=BASE_COLOR, linestyle="--", linewidth=1.2,
                        label=f"base (до SFT) {b:.3f}")
    _style(ax_loss, "Loss (только по SQL)")
    ax_loss.legend(frameon=False, fontsize=9)

    # --- 2. token accuracy
    xs, ys = _series(train_logs, "mean_token_accuracy")
    if xs:
        ax_acc.plot(xs, ys, color=TRAIN_COLOR, alpha=0.25, linewidth=1)
        ax_acc.plot(xs, _ema(ys), color=TRAIN_COLOR, linewidth=2, label="train (сглаж.)")
    xs, ys = _series(eval_logs, "eval_mean_token_accuracy")
    if xs:
        ax_acc.plot(xs, ys, color=EVAL_COLOR, linewidth=2, marker="o", markersize=5,
                    label="validation")
    if before and _finite(before[0].get("eval_mean_token_accuracy")):
        b = before[0]["eval_mean_token_accuracy"]
        ax_acc.axhline(b, color=BASE_COLOR, linestyle="--", linewidth=1.2,
                       label=f"base (до SFT) {b:.3f}")
    _style(ax_acc, "Token accuracy (доля верно предсказанных токенов SQL)")
    if ax_acc.lines:
        ax_acc.legend(frameon=False, fontsize=9)

    # --- 3. learning rate
    xs, ys = _series(train_logs, "learning_rate")
    if xs:
        ax_lr.plot(xs, ys, color=TRAIN_COLOR, linewidth=2)
    _style(ax_lr, "Learning rate")
    ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    # --- 4. grad norm (nan/inf в fp16 бывают — считаем и выкидываем)
    xs, ys = _series(train_logs, "grad_norm")
    n_bad = sum(1 for r in train_logs if "grad_norm" in r and not _finite(r["grad_norm"]))
    if xs:
        ax_gn.plot(xs, ys, color=TRAIN_COLOR, linewidth=1.5)
    _style(ax_gn, "Grad norm" + (f"  (nan/inf: {n_bad})" if n_bad else ""))

    # --- заголовок: прогресс
    step = max((r["step"] for r in rows), default=0)
    max_steps = h.get("max_steps") or 0
    epoch = max((r.get("epoch", 0) or 0 for r in rows), default=0)
    phase = {"before": "оценка base", "train": "обучение", "after": "итоговая оценка",
             "done": "готово"}.get(h.get("phase"), h.get("phase") or "")
    parts = [title or h.get("run_name") or "run"]
    if max_steps:
        parts.append(f"шаг {step}/{max_steps} ({100 * step / max_steps:.0f}%)")
    if epoch:
        parts.append(f"эпоха {epoch:.2f}")
    if phase:
        parts.append(phase)
    if after and _finite(after[0].get("eval_loss")) and before:
        parts.append(f"eval_loss {before[0]['eval_loss']:.3f} → {after[0]['eval_loss']:.3f}")
    fig.suptitle("  ·  ".join(parts), x=0.01, ha="left", fontsize=12)
    fig.tight_layout()

    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120, bbox_inches="tight")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="results/<run>_live.json или *_log_history.json")
    ap.add_argument("--out", default=None, help="куда сохранить PNG")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    p = Path(args.path)
    stem = p.name.replace("_live.json", "").replace("_log_history.json", "")
    out = args.out or p.with_name(f"{stem}_curves.png")
    plot_history(p, out=out)
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
