#!/usr/bin/env python3
"""The appendix heatmaps: disattenuated correlation of the accuracy drop across k and
across models.

    python experiments/V-fragility-correlation/plot_fragility_correlation.py

Reads fragility_correlation_{crossk,crossmodel}.csv (written by fragility_correlation.py)
and averages ``r_disatt`` arithmetically per k pair (over datasets and models) and per
model pair (over datasets and k). Writes fragility-correlation-cross-k-and-model.pdf to
$RUN_DIR/results/V-fragility-correlation/.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator

from llm import paths
from llm.plotting import pt_to_in, use_paper_style

AXIS_COLOR = "#42505c"
# Model names as printed in this figure of the paper.
LABELS = {
    "qwen3-4b-thinking": "Qwen3 4B Thinking", "qwen3-8b": "Qwen3 8B",
    "qwen3-14b": "Qwen3 14B", "qwen3-8-27b": "Qwen3.8 27B",
    "r1-qwen-distill-7b": "R1 Qwen 7B", "r1-llama-distill-8b": "R1 Llama 8B",
    "r1-qwen3-8b": "R1 Qwen3 8B",
}


def symmetric(s: pd.Series) -> pd.DataFrame:
    """A pair-indexed series as a symmetric matrix with an empty diagonal."""
    a, b = s.index.names
    labels = list(dict.fromkeys(list(s.index.get_level_values(a)) +
                                list(s.index.get_level_values(b))))
    M = pd.DataFrame(np.nan, index=labels, columns=labels)
    for (x, y), r in s.items():
        M.loc[x, y] = M.loc[y, x] = r
    return M


def draw(ax, M: pd.DataFrame, norm, cmap, fontsize=7):
    n = len(M)
    im = ax.imshow(np.ma.masked_invalid(M.values), cmap=cmap, norm=norm,
                   aspect="equal", interpolation="nearest")
    ax.grid(False)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(n), [str(v) for v in M.columns])
    ax.set_yticks(range(n), [str(v) for v in M.index])
    for i in range(n):
        for j in range(n):
            v = M.values[i, j]
            if np.isnan(v):
                continue
            r, g, b, _ = cmap(norm(v))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fontsize,
                    color="white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.5 else AXIS_COLOR)
    ax.tick_params(axis="both", which="major", direction="out", length=5, labelsize=10)
    ax.set_anchor("N")
    return im


def main() -> None:
    out = paths.results("V-fragility-correlation")
    crossk = pd.read_csv(out / "fragility_correlation_crossk.csv")
    crossmodel = pd.read_csv(out / "fragility_correlation_crossmodel.csv")
    by_k = crossk.groupby(["k1", "k2"])["r_disatt"].mean()
    by_model = crossmodel.groupby(["model1", "model2"])["r_disatt"].mean()

    use_paper_style()
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    norm = Normalize(vmin=0.0, vmax=0.9)
    Mk = symmetric(by_k)
    Mm = symmetric(by_model).rename(index=LABELS, columns=LABELS)

    width = pt_to_in(600)
    fig, (ax_k, ax_m) = plt.subplots(
        1, 2, figsize=(width, width * 0.6), layout="constrained",
        gridspec_kw={"width_ratios": [len(Mk), len(Mm)]})
    draw(ax_k, Mk, norm, cmap)
    im = draw(ax_m, Mm, norm, cmap)
    ax_k.set_title("Across $k$", fontsize=12)
    ax_m.set_title("Across models", fontsize=12)
    ax_m.tick_params(axis="x", labelrotation=45)
    for label in ax_m.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")

    cbar = fig.colorbar(im, ax=[ax_k, ax_m], shrink=0.8, pad=0.02)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(direction="out", length=5, labelsize=10)
    cbar.ax.yaxis.set_major_locator(FixedLocator([0.0, 0.3, 0.6, 0.9]))
    cbar.set_label("Disattenuated Pearson correlation", fontsize=14, labelpad=6)

    path = out / "fragility-correlation-cross-k-and-model.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
