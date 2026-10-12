"""The main-text token-ratio figure: MMLU-Redux large, four more datasets small.

Token ratio (composed section tokens over isolated output tokens) against the number of
tasks per composition, on log-log axes, one colour per isolated-trace-length bin, each
with its power law through (1, 1). The large panel prints each line's law, ∝ k^m.
Qwen3 4B Thinking.

    python experiments/II-token-ratio/token_ratio_main.py
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from llm import paths
from llm.plotting import pt_to_in, use_paper_style

from ratio_plot import TITLES, bin_legend, by_bin, plot_bins, slope_labels, sparse_ticks
from trace_data import figure_points

MAIN = "mmlu"
SMALL = ["gpqa-diamond", "integrals", "math500", "gsm8k"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="qwen3-4b-thinking")
    args = parser.parse_args()
    use_paper_style()

    width_in = pt_to_in(700)
    fig = plt.figure(figsize=(width_in, width_in * 0.36), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1])
    fig.get_layout_engine().set(wspace=0.06, hspace=0.12)
    ax_main = fig.add_subplot(gs[:, 0])
    small_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in (1, 2)]

    d = by_bin(figure_points(MAIN, args.model))
    main_fits = plot_bins(ax_main, d, markersize=8)
    ax_main.yaxis.set_major_locator(FixedLocator(np.array([0.05, 0.10, 0.15, 0.25, 0.40, 0.70])))
    ax_main.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax_main.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ax_main.set_xticks([2, 3, 4, 6, 8, 10, 12, 15, 20])
    legend = bin_legend(ax_main, fontsize=12)
    ax_main.set_xlabel(r"Log-scale tasks per composition on $\mathbf{MMLU\text{-}Redux}$",
                       fontsize=14, labelpad=6)
    ax_main.set_ylabel("Log-scale Token Ratio", fontsize=14, labelpad=6)
    ax_main.tick_params(axis="both", which="major", direction="out", length=5, labelsize=12)

    for ax, ds in zip(small_axes, SMALL):
        d = by_bin(figure_points(ds, args.model))
        plot_bins(ax, d, markersize=5.5, linewidth=1.0)
        sparse_ticks(ax, d, n=5, round_y=True)
        ax.set_ylabel(TITLES[ds], fontsize=12, labelpad=2)
        ax.tick_params(axis="both", which="major", direction="out", length=3, labelsize=8, pad=2)

    # the slope labels go on last, into the settled layout, which then stays fixed
    fig.canvas.draw()
    fig.set_layout_engine("none")
    slope_labels(ax_main, main_fits, fontsize=14, markersize=8,
                 avoid=[legend.get_window_extent()])

    out = paths.results("II-token-ratio") / "token-ratio-integrals-plus-benchmarks.pdf"
    fig.savefig(out)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
