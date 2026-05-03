"""
Compare all trained models and generate a summary report.
Usage: python scripts/compare.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"

# ── Load all metrics ──────────────────────────────────────────────────────────
models_data = {}
for f in sorted(RESULTS_DIR.glob("metrics_*.json")):
    with open(f) as fp:
        d = json.load(fp)
    models_data[d["model"]] = d

if not models_data:
    print("❌  No metrics found. Run train.py for each model first.")
    sys.exit(1)

names   = list(models_data.keys())
metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
labels  = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]

# ── Print table ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<20} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
print("="*65)
for name, d in models_data.items():
    print(f"{name:<20} {d['accuracy']:>9.4f} {d['precision']:>10.4f} "
          f"{d['recall']:>8.4f} {d['f1_score']:>8.4f} {d['roc_auc']:>8.4f}")
print("="*65)

# Find best model
best = max(models_data.items(), key=lambda x: x[1]["roc_auc"])
print(f"\n🏆  Best model : {best[0].upper()}  (AUC = {best[1]['roc_auc']:.4f})\n")

# ── Bar chart comparison ──────────────────────────────────────────────────────
x      = np.arange(len(metrics))
width  = 0.8 / len(names)
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, d) in enumerate(models_data.items()):
    vals   = [d[m] for m in metrics]
    offset = (i - len(names) / 2 + 0.5) * width
    bars   = ax.bar(x + offset, vals, width, label=name.upper(),
                    color=colors[i % len(colors)], alpha=0.87)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.95, 1.002)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — Bot Detection on Cresci-17", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))

plt.tight_layout()
out = RESULTS_DIR / "model_comparison.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"📊  Comparison chart saved → {out}")

# ── Radar chart ───────────────────────────────────────────────────────────────
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i, (name, d) in enumerate(models_data.items()):
    vals = [d[m] for m in metrics]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=2, label=name.upper(), color=colors[i % len(colors)])
    ax.fill(angles, vals, alpha=0.07, color=colors[i % len(colors)])

ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
ax.set_ylim(0.95, 1.0)
ax.set_title("Radar — Model Comparison", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
ax.grid(alpha=0.4)

out_radar = RESULTS_DIR / "model_radar.png"
fig.savefig(out_radar, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"📡  Radar chart saved    → {out_radar}")