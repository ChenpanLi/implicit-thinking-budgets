"""GPQA-Diamond: accuracy drop against composition size k, one line per model.

Writes gpqa-composed-sets-drop.pdf (main paper, right panel of the first results figure)
and gpqa-composed-sets-drop.csv with every plotted point.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from drop import SEED, drop_points, load_drop, points_frame
from llm import paths
from llm.plotting import MODEL_COLORS, pt_to_in, use_paper_style

DATASET = "gpqa-diamond"
NAME = "gpqa-composed-sets-drop"


def main() -> None:
    use_paper_style()
    out = paths.results("I-success-conditioned-drop")
    points = points_frame(drop_points(DATASET, load_drop(DATASET), np.random.default_rng(SEED)))
    points.to_csv(out / f"{NAME}.csv", index=False)

    fig, ax = plt.subplots(figsize=(pt_to_in(400), 4.582814445828144))
    ax.set_facecolor("#f8f9fa")
    ax.spines["bottom"].set_visible(True)
    for model, color in MODEL_COLORS.items():
        stats = points[points["model"] == model].set_index("k").sort_index()
        if stats.empty:
            continue
        # The drop is drawn downwards, so the interval's upper bound becomes the lower bar.
        yerr = np.vstack([stats["hi_pp"] - stats["drop_pp"], stats["drop_pp"] - stats["lo_pp"]])
        ax.errorbar(stats.index, -stats["drop_pp"], yerr=yerr, label=model, color=color,
                    linewidth=1.5, marker="o", capsize=0, capthick=1, elinewidth=1)

    ax.set_xticks([1, 3, 6, 10])
    ax.set_xlabel("tasks per composition", fontsize=14, labelpad=6)
    ax.set_yticks(np.arange(0, -70, -10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y}%"))
    ax.set_ylabel(r"Accuracy drop on $\mathbf{GPQA\text{-}Diamond}$", fontsize=14, labelpad=6)
    ax.tick_params(axis="y", which="major", left=True, direction="out", length=5, labelsize=12)
    ax.tick_params(axis="x", which="major", bottom=True, direction="out", length=5, labelsize=9)
    fig.savefig(out / f"{NAME}.pdf", bbox_inches="tight")

    with pd.option_context("display.width", 200):
        print(points.round(2).to_string(index=False))
    print(f"wrote {out / NAME}.pdf and .csv")


if __name__ == "__main__":
    main()
