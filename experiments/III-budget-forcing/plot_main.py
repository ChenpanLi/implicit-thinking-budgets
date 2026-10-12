"""The main-text token-ratio figure: MMLU-Redux and MATH-500 on Qwen3 4B Thinking.

    python experiments/III-budget-forcing/plot_main.py

Reads composed-bins.csv and forced-bins.csv (build_points.py) and writes
mmlu-token-ratio-vs-accuracy-drop.pdf and math500-token-ratio-vs-accuracy-drop.pdf, set side
by side in the paper. Only the MMLU-Redux panel carries the legend; the MATH-500 panel's k
labels keep clear of the spot where that legend sits, so the two panels read as one figure.
A panel's y axis turns symmetric-log by the same rule as in the appendix grids
(draw.symlog_y): MATH-500 does, MMLU-Redux stays linear.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from draw import (OUT, circle_boxes, draw_panel, legend_entries, load_points, place_k_labels,
                  symlog_y)
from llm.plotting import pt_to_in, use_paper_style
from llm.specs import DATASETS

MODEL = "qwen3-4b-thinking"
PANELS = ["mmlu", "math500"]           # the first one carries the legend
LABEL_MARGIN = 7 ** 2                  # the k labels avoid a circle slightly larger than s=36


def panel(dataset, points, forced, legend_box=None):
    """One panel. Without `legend_box` it draws the legend and returns its box; with one,
    it draws none and keeps its k labels clear of that box."""
    width_in = pt_to_in(400)
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.62))
    k_labels, circles, comp_ys, ys = draw_panel(ax, points, forced)

    ax.xaxis.set_major_locator(FixedLocator([0.0, 0.20, 0.40, 0.60, 0.70, 0.80, 0.90]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.xaxis.set_minor_locator(NullLocator())

    # y axis: symmetric-log, as in the appendix grids, when the forced points dwarf the
    # composed ones (draw.symlog_y); otherwise even steps covering both conditions
    if not symlog_y(ax, comp_ys, ys):
        span = max(ys.max(), 0) - min(ys.min(), 0)
        step = 2 if span <= 14 else 5 if span <= 40 else 10
        lo, hi = np.floor(min(ys.min(), 0) / step) * step, np.ceil(max(ys.max(), 0) / step) * step
        ax.yaxis.set_major_locator(FixedLocator(np.arange(lo, hi + step, step)))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "0" if v == 0 else f"{int(v):+d}".replace("-", "−")))
        ax.set_ylim(lo - step / 4, max(ys.max(), 0) + step / 4)

    leg = None
    if legend_box is None:
        handles, labels = legend_entries(high_first=True)
        leg = ax.legend(handles, labels, title="Isolated \ntrace len.", fontsize=12, title_fontsize=12,
                        loc="best", frameon=True, facecolor="#ffffff", edgecolor="#ced4da",
                        framealpha=1.0, borderpad=0.6, labelspacing=0.4, handlelength=1.0,
                        handletextpad=0.5, borderaxespad=0.5, shadow={"color": "#303559", "alpha": 0.7})
        leg.set_zorder(5)
    ax.set_xlabel("Token reduction (1 - $r_i(k)$)", fontsize=14, labelpad=6)
    ax.set_ylabel(r"Accuracy drop (pp) on $\mathbf{" + DATASETS[dataset].title.replace("-", r"\text{-}") + "}$", fontsize=12, labelpad=6)
    ax.tick_params(axis="y", which="major", left=True, direction="out", length=5, labelsize=12)
    ax.tick_params(axis="x", which="major", bottom=True, direction="out", length=5, labelsize=12)

    # k labels last: they need the final axes, ticks and legend position to avoid them
    fig.canvas.draw()
    boxes = circle_boxes(ax, circles, LABEL_MARGIN)
    if leg is not None:
        if any(leg.get_window_extent().overlaps(b) for b in boxes):
            # no free corner inside: hang the legend outside, right of the axes
            leg.set_loc("upper left")
            leg.set_bbox_to_anchor((1.02, 1.0), transform=ax.transAxes)
            fig.canvas.draw()
        legend_box = leg.get_window_extent()
    place_k_labels(ax, k_labels, [legend_box] + boxes, same_spot_px=14, keep_inside=False)

    out = OUT / f"{dataset}-token-ratio-vs-accuracy-drop.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[main] -> {out}")
    return legend_box


def main():
    use_paper_style()
    points, forced = load_points()
    legend_box = None
    for dataset in PANELS:
        legend_box = panel(dataset, points[MODEL][dataset], forced[MODEL][dataset], legend_box)


if __name__ == "__main__":
    main()
