"""Thinking tokens and accuracy drop by position in the composition (appendix figure).

For every model, two box rows over the k prompt positions of MATH-500 at k=20: the
section tokens each position received and its accuracy drop against isolation. A box
spans the seeds (quartiles and whiskers over the 10 batching seeds); the last position is
the dummy, drawn grey, because its section holds the answer recap rather than a task.
Reads `position_effect_seeds.csv` (position_effect_csv.py).

    python experiments/IV-position-effect/position_effect.py
    python experiments/IV-position-effect/position_effect.py --k 10
    python experiments/IV-position-effect/position_effect.py --datasets mmlu
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cbook import boxplot_stats
from matplotlib.ticker import FuncFormatter

from llm import paths
from llm.plotting import pt_to_in, use_paper_style
from llm.specs import MODEL_NAMES

ROWS = [["qwen3-4b-thinking", "qwen3-8b", "qwen3-14b", "qwen3-8-27b"],
        ["r1-qwen-distill-7b", "r1-llama-distill-8b", "r1-qwen3-8b"]]
DUMMY_COLOR = "#d0d4da"
STATS = ("whislo", "q1", "med", "q3", "whishi")


def style(ax):
    ax.set_facecolor("#f8f9fa")
    ax.set_axisbelow(True)
    ax.grid(True, color="#e3e5e8", linewidth=0.8)
    for side in ("top", "left"):
        ax.spines[side].set_color("#2b3a55")
    for side in ("right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", left=True, direction="out", length=3, labelsize=8,
                   labelcolor="#7a8699", color="#2b3a55", pad=1.5)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)


def box_stats(sub, col, scale, p):
    """Quartiles and whiskers over the seeds within each dataset, averaged over datasets."""
    per_dataset = [boxplot_stats(g[col].dropna().to_numpy() * scale)[0]
                   for _, g in sub[sub.position == p].groupby("dataset") if g[col].notna().any()]
    return {s: np.mean([d[s] for d in per_dataset]) for s in STATS} | {"fliers": []}


def boxes(ax, sub, col, scale, colors, k):
    positions = np.arange(1, k + 1)
    stats = [box_stats(sub, col, scale, p) for p in positions]
    b = ax.bxp(stats, positions=positions, widths=0.6, patch_artist=True, showfliers=False,
               medianprops=dict(color="black", linewidth=0.8),
               whiskerprops=dict(color="#2b3a55", linewidth=0.8),
               capprops=dict(color="#2b3a55", linewidth=0.8),
               boxprops=dict(edgecolor="#2b3a55", linewidth=0.6), zorder=3)
    for p, box in zip(positions, b["boxes"]):
        box.set_facecolor(DUMMY_COLOR if p == k else colors[p - 1])
        if p == k:   # the dummy: its section holds the answer recap, not a task
            box.set_alpha(0.5)
            for line in b["whiskers"][2 * (p - 1):2 * p] + b["caps"][2 * (p - 1):2 * p]:
                line.set_alpha(0.5)
    ax.set_xlim(0.4, k + 0.6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--datasets", nargs="+", default=["math500"])
    args = parser.parse_args()
    k = args.k
    use_paper_style()

    out_dir = paths.results("IV-position-effect")
    df = pd.read_csv(out_dir / "position_effect_seeds.csv")
    df = df[(df.k == k) & df.dataset.isin(args.datasets)]

    positions = np.arange(1, k + 1)
    # yellow (first) -> blue / purple (last task); tokens in plasma, accuracy drop in viridis
    tok_colors = plt.cm.plasma(np.linspace(0.95, 0, k - 1))
    drop_colors = plt.cm.viridis(np.linspace(1, 0, k - 1))
    # every position gets a tick; at large k only every 5th and the dummy get a label
    labelled = set(positions) if k <= 8 else {1, *range(5, k, 5), k}
    tick_labels = [("d." if p == k else str(p)) if p in labelled else "" for p in positions]
    tok_fmt = FuncFormatter(lambda y, _: f"{y / 1000:g}k" if y >= 1000 else f"{y:g}")
    drop_fmt = FuncFormatter(lambda y, _: f"{y:g}%")

    fig = plt.figure(figsize=(2 * pt_to_in(397.48), 2 * 4.6))
    gs = fig.add_gridspec(5, 4, height_ratios=[1, 1, 0.22, 1, 1], hspace=0.12, wspace=0.32,
                          top=0.94)
    for r, models in enumerate(ROWS):
        for c, model in enumerate(models):
            sub = df[df.model == model]
            ax_tok = fig.add_subplot(gs[3 * r, c])
            ax_drop = fig.add_subplot(gs[3 * r + 1, c], sharex=ax_tok)
            style(ax_tok)
            style(ax_drop)
            boxes(ax_tok, sub, "mean_thinking_tokens", 1.0, tok_colors, k)
            boxes(ax_drop, sub, "accuracy_drop", 100.0, drop_colors, k)
            ax_tok.yaxis.set_major_formatter(tok_fmt)
            ax_drop.yaxis.set_major_formatter(drop_fmt)
            ax_tok.set_ylim(bottom=0)

            # the shared position axis, once, on top of the pair
            ax_tok.set_xticks(positions)
            top = ax_tok.secondary_xaxis("top")
            top.set_xticks(positions)
            top.set_xticklabels(tick_labels, fontsize=8, color="#7a8699")
            top.tick_params(axis="x", direction="out", length=2, color="#2b3a55", pad=1)
            top.spines["top"].set_color("#2b3a55")
            ax_drop.set_xlabel(MODEL_NAMES.get(model, model), fontsize=10, labelpad=3,
                               fontweight="bold")
            if c == 0:
                ax_tok.set_ylabel("Thinking\ntokens", fontsize=10, labelpad=3)
                ax_drop.set_ylabel("Accuracy\ndrop", fontsize=10, labelpad=3)

    fig.align_ylabels()
    fig.suptitle("Position in composition", fontsize=11, y=0.99)
    out = out_dir / f"position-effect-k{k}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
