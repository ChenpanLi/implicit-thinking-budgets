"""GPQA-Diamond: accuracy drop per model at k=10, one bar per model.

Writes gpqa-composed-sets-drop-k10.pdf (main paper, left panel of the first results
figure) and gpqa-composed-sets-drop-k10.csv, and prints the Claude Opus 5 drop the
results section quotes ("loses about 7 pp at k=10").
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from drop import LEGEND, SEED, TITLES, load_drop, point, points_frame
from llm import paths
from llm.plotting import MODEL_COLORS, pt_to_in, use_paper_style

DATASET = "gpqa-diamond"
K = 10
NAME = "gpqa-composed-sets-drop-k10"
SPINE = "#2b3a55"


def main() -> None:
    use_paper_style()
    out = paths.results("I-success-conditioned-drop")
    df = load_drop(DATASET)
    rng = np.random.default_rng(SEED)
    points = points_frame([point(DATASET, model, K, df[(df["model"] == model) & (df["k"] == K)]["d"], rng)
                           for model in MODEL_COLORS])
    points.to_csv(out / f"{NAME}.csv", index=False)

    fig, ax = plt.subplots(figsize=(pt_to_in(400), 4.582814445828144))
    ax.set_facecolor("#f8f9fa")
    ax.set_axisbelow(True)
    ax.grid(True, color="#e3e5e8", linewidth=1.0)
    for side in ("top", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE)
    for side in ("right", "bottom"):
        ax.spines[side].set_visible(False)

    # Smallest drop on the right; the drop is drawn downwards, so the interval flips.
    bars = points.sort_values("drop_pp", ascending=False, kind="stable").reset_index(drop=True)
    for i, row in bars.iterrows():
        ax.bar(i, -row["drop_pp"], width=0.7, color=MODEL_COLORS[row["model"]],
               label=LEGEND[row["model"]], zorder=3)
        ax.errorbar(i, -row["drop_pp"],
                    yerr=[[row["hi_pp"] - row["drop_pp"]], [row["drop_pp"] - row["lo_pp"]]],
                    fmt="none", ecolor="black", elinewidth=1.0, capsize=0, capthick=1.0, zorder=4)

    x = np.arange(len(bars))
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_xlabel(f"Random compositions of {K} tasks", fontsize=14, labelpad=6)
    ax.set_xlim(-0.6, len(bars) - 0.4)
    top = ax.secondary_xaxis("top")
    top.set_xticks(x)
    top.set_xticklabels([])
    top.set_xlabel("Model", fontsize=14, labelpad=6)
    top.tick_params(axis="x", direction="out", length=5, color=SPINE)
    top.spines["top"].set_color(SPINE)

    ax.set_ylim(-70, 0)
    ax.set_yticks(np.arange(0, -80, -10))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}%"))
    ax.set_ylabel(r"Accuracy drop on $\mathbf{" + TITLES[DATASET].replace("-", r"\text{-}") + "}$",
                  fontsize=14, labelpad=6)
    ax.tick_params(axis="y", which="major", left=True, direction="out", length=5,
                   labelsize=12, labelcolor="#7a8699", color=SPINE)
    ax.tick_params(axis="x", which="major", bottom=False, direction="out", length=5,
                   labelsize=9, color=SPINE)
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True, fontsize=11,
              borderaxespad=0.8, framealpha=1.0, edgecolor="#d0d4da",
              handlelength=1.0, handleheight=1.0)
    fig.savefig(out / f"{NAME}.pdf", bbox_inches="tight")

    print(bars.round(2).to_string(index=False))
    opus = points.set_index("model").loc["claude-opus-5"]
    print(f"Claude Opus 5 on GPQA-Diamond at k={K}: drop {opus['drop_pp']:.1f} pp "
          f"(95% CI {opus['lo_pp']:.1f} to {opus['hi_pp']:.1f}), n={opus['n']} composed slots")
    print(f"wrote {out / NAME}.pdf and .csv")


if __name__ == "__main__":
    main()
