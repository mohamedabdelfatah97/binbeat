"""
benchmark.py
Generates all comparison plots for the README and portfolio:
  1. Accuracy + F1 comparison bar chart
  2. Model size vs accuracy scatter (efficiency plot)
  3. Per-class F1 heatmap
  4. Confusion matrices (3 subplots)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
SUBTEXT   = "#8b949e"

MODEL_COLORS = {
    "mlp": "#58a6ff",   # blue
    "cnn": "#3fb950",   # green
    "bnn": "#f78166",   # coral/red
}

def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK_BG,
        "axes.facecolor":    PANEL_BG,
        "axes.edgecolor":    BORDER,
        "axes.labelcolor":   TEXT,
        "axes.titlecolor":   TEXT,
        "xtick.color":       SUBTEXT,
        "ytick.color":       SUBTEXT,
        "text.color":        TEXT,
        "grid.color":        BORDER,
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "legend.facecolor":  PANEL_BG,
        "legend.edgecolor":  BORDER,
        "font.family":       "monospace",
        "font.size":         11,
    })

# ── Plot 1: Accuracy + F1 bar chart ───────────────────────────────────────────
def plot_accuracy_f1(metrics):
    apply_dark_style()
    models  = list(metrics.keys())
    colors  = [MODEL_COLORS[m] for m in models]
    labels  = [m.upper() for m in models]

    acc      = [metrics[m]["accuracy"]    for m in models]
    f1_macro = [metrics[m]["f1_macro"]    for m in models]
    f1_w     = [metrics[m]["f1_weighted"] for m in models]

    x     = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG)

    b1 = ax.bar(x - width,     acc,      width, label="Accuracy",    color=colors, alpha=0.9)
    b2 = ax.bar(x,             f1_macro, width, label="F1 Macro",    color=colors, alpha=0.6)
    b3 = ax.bar(x + width,     f1_w,     width, label="F1 Weighted", color=colors, alpha=0.35)

    # value labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=8.5, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance: Accuracy & F1", fontsize=14, pad=14)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    # custom legend for metric shading
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#aaaaaa", alpha=0.9,  label="Accuracy"),
        Patch(facecolor="#aaaaaa", alpha=0.6,  label="F1 Macro"),
        Patch(facecolor="#aaaaaa", alpha=0.35, label="F1 Weighted"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "01_accuracy_f1.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: Size vs Accuracy (efficiency) ────────────────────────────────────
def plot_efficiency(metrics):
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(DARK_BG)

    for name, m in metrics.items():
        x   = m["model_size_kb"]
        y   = m["accuracy"]
        col = MODEL_COLORS[name]
        ax.scatter(x, y, s=220, color=col, zorder=5, edgecolors=TEXT, linewidths=0.8)
        ax.annotate(
            name.upper(),
            (x, y),
            textcoords="offset points",
            xytext=(10, 6),
            fontsize=12,
            color=col,
            fontweight="bold",
        )

    ax.set_xlabel("Model Size (KB)  ←  smaller is better", fontsize=11)
    ax.set_ylabel("Test Accuracy  →  higher is better",   fontsize=11)
    ax.set_title("Efficiency: Size vs Accuracy", fontsize=14, pad=14)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

    # annotation: top-right is ideal
    ax.text(0.97, 0.06, "↖ ideal region: small & accurate",
            transform=ax.transAxes, ha="right", fontsize=9, color=SUBTEXT, style="italic")

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "02_efficiency.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: Per-class F1 heatmap ──────────────────────────────────────────────
def plot_f1_heatmap(metrics):
    apply_dark_style()

    models  = list(metrics.keys())
    # collect all class labels in consistent order
    classes = sorted(next(iter(metrics.values()))["f1_per_class"].keys())

    data = np.array([
        [metrics[m]["f1_per_class"].get(c, 0.0) for c in classes]
        for m in models
    ])

    cmap = LinearSegmentedColormap.from_list(
        "bw_blue", ["#0d1117", "#1f4e8c", "#58a6ff", "#cae8ff"]
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.upper() for m in models], fontsize=12)
    ax.set_title("Per-class F1 Score by Model", fontsize=14, pad=14)

    for i in range(len(models)):
        for j in range(len(classes)):
            val = data[i, j]
            color = TEXT if val < 0.5 else DARK_BG
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=SUBTEXT)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "03_f1_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: Confusion matrices (3 side-by-side) ───────────────────────────────
def plot_confusion_matrices(metrics):
    apply_dark_style()

    models  = list(metrics.keys())
    classes = sorted(next(iter(metrics.values()))["f1_per_class"].keys())

    cmap = LinearSegmentedColormap.from_list(
        "cm_blue", ["#0d1117", "#1f4e8c", "#58a6ff"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Confusion Matrices", fontsize=15, color=TEXT, y=1.02)

    for ax, name in zip(axes, models):
        cm  = np.array(metrics[name]["confusion_matrix"])
        # normalize row-wise so color = recall per class
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm  = cm / row_sums

        im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, fontsize=9)
        ax.set_yticklabels(classes, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True",      fontsize=10)
        ax.set_title(name.upper(), fontsize=13, color=MODEL_COLORS[name], pad=8)

        for i in range(len(classes)):
            for j in range(len(classes)):
                raw = cm[i, j]
                val = cm_norm[i, j]
                color = TEXT if val < 0.55 else DARK_BG
                ax.text(j, i, f"{raw}", ha="center", va="center",
                        fontsize=8, color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "04_confusion_matrices.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    if not os.path.exists(metrics_path):
        print("ERROR: results/metrics.json not found. Run evaluate.py first.")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = json.load(f)

    print("Generating benchmark plots...")
    plot_accuracy_f1(metrics)
    plot_efficiency(metrics)
    plot_f1_heatmap(metrics)
    plot_confusion_matrices(metrics)
    print(f"\nAll plots saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()