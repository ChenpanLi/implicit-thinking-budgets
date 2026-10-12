"""The appendix token-ratio grids: one 2x3 figure per model, the six per-term-trace datasets.

    python experiments/III-budget-forcing/plot_grids.py                     # every model
    python experiments/III-budget-forcing/plot_grids.py --model qwen3-14b

Reads composed-bins.csv and forced-bins.csv (build_points.py) and writes
<model>-composed-vs-forced-token-ratio-vs-accuracy-drop-grid.pdf. Points with no token
reduction (or a negative one) are left out. A panel's y axis turns symmetric-log when the
forced points reach far past the composed ones (draw.symlog_y).
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator, NullLocator

from draw import (LEGEND_STYLE, OUT, circle_boxes, draw_panel, legend_entries, load_points,
                  place_k_labels, signed_formatter, symlog_y, thin_xticklabels)
from llm.plotting import pt_to_in, use_paper_style
from llm.specs import DATASETS, MODELS

GRID_DATASETS = ["integrals", "mmlu", "onewordqa", "gpqa-diamond", "gsm8k", "math500"]
CIRCLE = 14           # marker area, points^2
# Height over width. With the x label and the legend below the panels, this gives the
# 606 x 379 pt PDF the appendix pages are laid out for.
ASPECT = 0.623


def grid(model, points, forced):
    width_in = pt_to_in(600)
    fig, axes = plt.subplots(2, 3, figsize=(width_in, width_in * ASPECT), layout="constrained")

    panels = []
    for ax, name in zip(axes.flat, GRID_DATASETS):
        p, f = points[name], forced[name]
        p = p.loc[(p.saved_tokens < 10.0) & (p.saved_tokens > 0)]
        f = f.loc[(f.saved_tokens < 10.0) & (f.saved_tokens > 0)]
        k_labels, circles, comp_ys, ys = draw_panel(ax, p, f, circle_size=CIRCLE,
                                                    diamond_size=12, line_width=0.9)
        panels.append((ax, k_labels, circles))
        ax.set_title(DATASETS[name].title, fontsize=10, fontweight="bold", pad=4)

        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        ax.xaxis.set_minor_locator(NullLocator())

        # y axis: symmetric-log when the forced points dwarf the composed ones; otherwise
        # integer ticks if the visible range is > 3, else 0.25 steps
        y_lo, y_hi = ax.get_ylim()
        if symlog_y(ax, comp_ys, ys):
            pass
        elif y_hi - y_lo > 3:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax.yaxis.set_major_formatter(signed_formatter(0))
        else:
            ax.yaxis.set_major_locator(MultipleLocator(0.25))
            ax.yaxis.set_major_formatter(signed_formatter(2))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="both", which="major", direction="out", length=4, labelsize=9)

    # the x label belongs to the bottom middle panel, so the layout keeps room for it
    # above the legend (a figure-level supxlabel would share the legend's slot)
    axes[1, 1].set_xlabel(r"Token reduction ($1 - r_i(k)$)", fontsize=11, labelpad=4)
    fig.supylabel("Accuracy drop (pp)", fontsize=11)
    handles, labels = legend_entries(size=7)
    fig.legend(handles, labels, title="Isolated trace len.", loc="outside lower center", ncol=5,
               fontsize=10, title_fontsize=10, columnspacing=1.4, **LEGEND_STYLE)

    # k labels last: the constrained layout has to settle first
    fig.canvas.draw()
    for ax, k_labels, circles in panels:
        thin_xticklabels(ax)
        place_k_labels(ax, k_labels, circle_boxes(ax, circles, CIRCLE), fontsize=6)

    out = OUT / f"{model}-composed-vs-forced-token-ratio-vs-accuracy-drop-grid.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[grid] -> {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", nargs="+", default=MODELS, choices=MODELS)
    args = parser.parse_args()
    use_paper_style()
    points, forced = load_points()
    for model in args.model:
        grid(model, points[model], forced[model])


if __name__ == "__main__":
    main()
