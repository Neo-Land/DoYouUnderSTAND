"""Draw the README result charts from the committed report files.

Usage:
    python scripts/make_figures.py

Reads reports/signs/{logreg,gru,live}.json and reports/poses/metrics.json (numbers only) and writes
docs/figures/asl_results.png and docs/figures/jojo_results.png. Re-run after retraining.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dyu.config import ROOT

SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1 and 2 (validated pair)
OUT = ROOT / "docs" / "figures"


def load(path):
    return json.loads((ROOT / path).read_text())


def style(ax, title, subtitle):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=10)
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=14, fontweight="bold", color=TEXT)
    ax.text(0, 1.05, subtitle, transform=ax.transAxes, fontsize=10, color=MUTED)


def label_bars(ax, bars, fmt="{:.1f}%"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=9.5, color=TEXT)


def asl_chart():
    logreg, gru, live = load("reports/signs/logreg.json"), load("reports/signs/gru.json"), load("reports/signs/live.json")
    groups = ["Logistic regression\n(baseline)", "GRU sequence model\n(mean of 3 seeds)", "GRU, live pipeline\n(app cuts out the sign)"]
    top1 = [100 * logreg["test"]["top1"], 100 * gru["test"]["top1"], 100 * live["top1"]]
    top5 = [100 * logreg["test"]["top5"], 100 * gru["test"]["top5"], 100 * live["top5"]]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    x, w = range(len(groups)), 0.28
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], top1, w, color=BLUE, label="Top-1")
    b5 = ax.bar([i + w / 2 + 0.01 for i in x], top5, w, color=ORANGE, label="Top-5")
    label_bars(ax, b1)
    label_bars(ax, b5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups, color=TEXT, fontsize=10)
    style(ax, "ASL word signs: accuracy on 11 signers the model never saw",
          "100-sign subset of ASL Citizen, 1,261 test videos. Chance: 1% top-1.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=2, fontsize=10, labelcolor=TEXT)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    fig.tight_layout()
    fig.savefig(OUT / "asl_results.png", facecolor=SURFACE)


def jojo_chart():
    m = load("reports/poses/metrics.json")
    best = m[m["best"]]
    steps = ["Baseline\n3 phone sessions", "+ wrist-crop hands\n3 phone sessions",
             "+ webcam, dismiss pose\n5 sessions, 8 classes"]
    acc = [81.1, 91.2, 100 * best["accuracy"]]   # earlier rows are from docs/DEVLOG.md
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    bars = ax.bar(range(3), acc, 0.32, color=BLUE)
    label_bars(ax, bars)
    ax.set_xticks(range(3))
    ax.set_xticklabels(steps, color=TEXT, fontsize=10)
    ax.set_xlim(-0.6, 2.6)
    style(ax, "JoJo poses: accuracy on held-out recording sessions",
          "Leave-one-session-out. The last bar is a harder test than the second.")
    fig.tight_layout()
    fig.savefig(OUT / "jojo_results.png", facecolor=SURFACE)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    asl_chart()
    jojo_chart()
    print(f"Saved {OUT / 'asl_results.png'} and {OUT / 'jojo_results.png'}")


if __name__ == "__main__":
    main()
