"""All eight datasets: accuracy drop against composition size k, one panel per dataset.

Writes all-datasets-composed-sets-drop.pdf (appendix, "Compositional degradation is
observable over a wide range of datasets and models") and
all-datasets-composed-sets-drop.csv with every plotted point.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from drop import LEGEND, SEED, TITLES, drop_points, load_drop, points_frame
from llm import paths
from llm.plotting import MODEL_COLORS, pt_to_in, use_paper_style

NAME = "all-datasets-composed-sets-drop"
NCOLS = NROWS = 3


def main() -> None:
    use_paper_style()
    out = paths.results("I-success-conditioned-drop")
    rng = np.random.default_rng(SEED)
    points = []
    for dataset in TITLES:
        points += drop_points(dataset, load_drop(dataset), rng)
    points = points_frame(points)
    points.to_csv(out / f"{NAME}.csv", index=False)

    width_in = pt_to_in(600)
    panel_h = 0.75 * width_in / NCOLS
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(width_in, panel_h * NROWS + 0.3), squeeze=False)
    for ax, dataset in zip(axes.flat, TITLES):
        ax.set_facecolor("#f8f9fa")
        ax.spines["bottom"].set_visible(True)
        ks = {1}
        for model, color in MODEL_COLORS.items():
            stats = points[(points["dataset"] == dataset) & (points["model"] == model)]
            stats = stats.set_index("k").sort_index()
            if stats.empty:
                continue
            # The drop is drawn downwards, so the interval's upper bound becomes the lower bar.
            yerr = np.vstack([stats["hi_pp"] - stats["drop_pp"], stats["drop_pp"] - stats["lo_pp"]])
            ks.update(stats.index)
            ax.errorbar(stats.index, -stats["drop_pp"], yerr=yerr, color=color, linewidth=1.2,
                        marker="o", markersize=3, capsize=0, capthick=0.8, elinewidth=0.8)
        ax.set_title(TITLES[dataset], fontsize=9, fontweight="bold", pad=4)
        ax.set_xticks(sorted(ks))
        lo, hi = ax.get_ylim()
        # A minimum span, so flat datasets do not magnify noise; never clip a curve.
        ax.set_ylim(min(lo, -10), max(hi, 2))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))
        ax.tick_params(axis="y", which="major", left=True, direction="out", length=3, labelsize=7)
        ax.tick_params(axis="x", which="major", bottom=True, direction="out", length=3, labelsize=7)
    for ax in axes.flat[len(TITLES):]:
        ax.set_visible(False)

    fig.supxlabel("tasks per composition", fontsize=10, y=0.02)
    fig.supylabel("Accuracy drop", fontsize=10, x=0.01)
    fig.tight_layout(h_pad=1.0, w_pad=0.6)
    handles = [Line2D([0], [0], color=color, linewidth=1.2, marker="o", markersize=3,
                      label=LEGEND[model]) for model, color in MODEL_COLORS.items()]
    legend = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=5,
                        fontsize=7, frameon=True, facecolor="#ffffff", edgecolor="#ced4da",
                        framealpha=1.0, borderpad=0.6, labelspacing=0.4, handlelength=1.6,
                        columnspacing=1.2, borderaxespad=0.5,
                        shadow={"color": "#303559", "alpha": 0.7})
    legend.set_zorder(5)
    fig.savefig(out / f"{NAME}.pdf", bbox_inches="tight")

    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(points.round(2).to_string(index=False))
    print(f"wrote {out / NAME}.pdf and .csv")


if __name__ == "__main__":
    main()
