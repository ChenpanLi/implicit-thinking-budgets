"""OLMo 3 7B Think after SFT, DPO and RLVR: the two per-term-trace figures of one dataset.

A. Token ratio against k: one scatter per stage with its log-log fit through (1, 1) and
   its law, written horizontally beside the line as in the token-ratio figures of
   II-token-ratio. DPO's line is dashed, so it stays visible where it meets RLVR's.
B. Accuracy change against token reduction (1 - r(k)): one line per stage, a few k
   labelled on each.

Per task and k (prompt size, the dummy included):

    r_i(k)  = mean section tokens over the task's occurrences / mean isolated thinking tokens
    change  = 100 * (composed accuracy - isolated accuracy), in percentage points

A point is the mean over tasks. Echo reads and unreadable answer lists are left out of
accuracy; tasks with no isolated thinking are dropped. Also writes the points to
`olmo3-<dataset>-token-ratio-points.csv`.

    python experiments/VII-olmo-post-training/token_ratio_figures.py                  # GPQA-Diamond
    python experiments/VII-olmo-post-training/token_ratio_figures.py --dataset math500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from llm import paths
from llm.plotting import pt_to_in, use_paper_style
from llm.specs import OLMO, TRACE_KS

from olmo_data import frames

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "II-token-ratio"))
from ratio_plot import slope_labels  # noqa: E402

COLORS = dict(zip(OLMO, ["#1B2F5E", "#6FB3D9", "#C8413F"]))
NAMES = dict(zip(OLMO, ["SFT", "DPO", "RLVR"]))
LEGEND_TITLE = "OLMo 3 7B\nThink"
K_LABELS = [2, 4, 10, 20]       # figure B: the k written next to each line's points
K_LABEL_OFFSET = (3, 4)         # figure B: label offset from its point, in points
MIN_SEEDS = 10                  # a k enters only once every batching seed has finished
TITLES = {
    "gpqa-diamond": r"$\mathbf{GPQA\text{-}Diamond}$",
    "math500": r"$\mathbf{MATH\text{-}500}$",
    "mmlu": r"$\mathbf{MMLU\text{-}Redux}$",
}


def points(dataset: str, model: str) -> pd.DataFrame:
    """One row per k: mean token ratio, mean accuracy change (pp), tasks, seeds."""
    f = frames(dataset, model)
    pos = f["positions"].copy()
    pos.loc[pos.echo_read, "correct"] = pd.NA            # echo reads are not graded
    per = (pos.groupby(["k", "item_id"])
           .agg(composed_tokens=("tokens", "mean"), composed_acc=("correct", "mean"))
           .reset_index().merge(f["solo"], on="item_id"))
    per = per[per.solo_thinking_tokens > 0]
    per["ratio"] = per.composed_tokens / per.solo_thinking_tokens
    per["acc_change_pp"] = 100 * (per.composed_acc.astype(float) - per.solo_accuracy)
    out = per.groupby("k").agg(ratio=("ratio", "mean"), acc_change_pp=("acc_change_pp", "mean"),
                               n_items=("item_id", "nunique"))
    out["seeds"] = pos.groupby("k").seed.nunique()
    return out


def origin_slope(p) -> float:
    """Log-log least squares with no intercept: the line passes through (1, 1)."""
    lx, ly = np.log2(p.index.values.astype(float)), np.log2(p.ratio.values)
    return float(np.dot(lx, ly) / np.dot(lx, lx))


def figure_ratio(pts, slopes, dataset):
    width_in = pt_to_in(400)
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.62))
    ax.set_facecolor("#f8f9fa")
    k_max = max(p.index.max() for p in pts.values())
    fits = []
    for model, color in COLORS.items():
        p = pts[model]
        ax.loglog(p.index, p.ratio, "o", color=color, markersize=8,
                  markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        x_fit = np.logspace(np.log10(2.0), np.log10(p.index.max()), 200)
        ax.loglog(x_fit, x_fit ** slopes[model], "--" if NAMES[model] == "DPO" else "-",
                  color=color, linewidth=1.5, label=NAMES[model],
                  zorder=2.5 if NAMES[model] == "DPO" else 2)
        fits.append((color, slopes[model], x_fit, x_fit ** slopes[model],
                     p.index.values.astype(float), p.ratio.values))

    ax.set_xlim(1.5, k_max + 1.5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_locator(FixedLocator([k for k in TRACE_KS if k <= k_max and k != 18]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v))}"))
    ratios = np.concatenate([p.ratio.values for p in pts.values()])
    cands = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
    ax.yaxis.set_major_locator(FixedLocator(
        cands[(cands >= ratios.min() * 0.9) & (cands <= ratios.max() * 1.1)]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

    handles = [Line2D([], [], linestyle="none", marker="o", markersize=9, markerfacecolor=c,
                      markeredgecolor="white", markeredgewidth=1.0) for c in COLORS.values()]
    leg = ax.legend(handles, list(NAMES.values()), title=LEGEND_TITLE, fontsize=12,
                    title_fontsize=12, loc="lower left", frameon=True, facecolor="#ffffff",
                    edgecolor="#ced4da", framealpha=1.0, borderpad=0.6, labelspacing=0.4,
                    handlelength=1.0, handletextpad=0.5, borderaxespad=0.5,
                    shadow={"color": "#303559", "alpha": 0.7})
    leg.set_zorder(5)
    ax.set_xlabel(f"Log-scale tasks per composition on {TITLES.get(dataset, dataset)}",
                  fontsize=14, labelpad=6)
    ax.set_ylabel("Log-scale Token Ratio", fontsize=14, labelpad=6)
    ax.tick_params(axis="both", which="major", direction="out", length=5, labelsize=12)
    fig.canvas.draw()
    slope_labels(ax, fits, fontsize=12, markersize=8, avoid=[leg.get_window_extent()])
    return fig


def figure_accuracy(pts, dataset):
    width_in = pt_to_in(400)
    fig, ax = plt.subplots(figsize=(width_in, width_in * 0.62))
    ax.set_facecolor("#f8f9fa")
    ax.axhline(0, color="#42505c", linewidth=0.8, zorder=1)
    for model, color in COLORS.items():
        p = pts[model]
        ax.plot(1 - p.ratio.values, p.acc_change_pp.values, "-o", color=color, linewidth=1.5,
                markersize=6, markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        # the stages reach a given k at different reductions: label each line's own points
        for k in K_LABELS:
            if k in p.index:
                ax.annotate(f"k={k}", (1 - p.ratio[k], p.acc_change_pp[k]),
                            textcoords="offset points", xytext=K_LABEL_OFFSET, ha="left",
                            va="bottom", fontsize=12, color=color, zorder=4)

    red = np.concatenate([1 - p.ratio.values for p in pts.values()])
    chg = np.concatenate([p.acc_change_pp.values for p in pts.values()])
    ax.set_xlim(red.min() - 0.04, red.max() + 0.04)
    ax.xaxis.set_major_locator(FixedLocator(
        np.round(np.arange(np.ceil(red.min() * 10) / 10, red.max() + 0.001, 0.1), 2)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    y_lo, y_hi = min(chg.min(), 0) - 1.5, max(chg.max(), 0) + 2.5
    step = 2 if y_hi - y_lo <= 16 else 5
    ax.yaxis.set_major_locator(FixedLocator(range(int(np.floor(y_lo / step) * step), int(y_hi) + 1, step)))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{int(v):+d}".replace("-", "−")))
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("Token reduction (1 - $r_i(k)$)", fontsize=14, labelpad=6)
    ax.set_ylabel(f"Accuracy drop (pp) on {TITLES.get(dataset, dataset)}", fontsize=12, labelpad=6)
    ax.tick_params(axis="y", which="major", left=True, direction="out", length=5, labelsize=12)
    ax.tick_params(axis="x", which="major", bottom=True, direction="out", length=5, labelsize=12)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="gpqa-diamond", choices=list(TITLES))
    args = parser.parse_args()
    use_paper_style()

    pts = {}
    for model in OLMO:
        p = points(args.dataset, model)
        short = p.loc[p.seeds < MIN_SEEDS]
        if len(short):
            print(f"[{NAMES[model]}] left out, fewer than {MIN_SEEDS} seeds: "
                  + ", ".join(f"k={k} ({s})" for k, s in short.seeds.items()))
        pts[model] = p.loc[(p.seeds >= MIN_SEEDS) & p.index.isin(TRACE_KS)]
    slopes = {m: origin_slope(p) for m, p in pts.items()}
    print(pd.DataFrame({NAMES[m]: p.ratio for m, p in pts.items()}).round(3).to_string())
    print(pd.DataFrame({NAMES[m]: p.acc_change_pp for m, p in pts.items()}).round(2).to_string())
    print("exponents (fit through (1, 1)):", {NAMES[m]: round(s, 3) for m, s in slopes.items()})

    out = paths.results("VII-olmo-post-training")
    stem = f"olmo3-{args.dataset}"
    pd.concat([p.assign(model=m) for m, p in pts.items()]).reset_index().to_csv(
        out / f"{stem}-token-ratio-points.csv", index=False)
    figure_ratio(pts, slopes, args.dataset).savefig(out / f"{stem}-token-ratio-by-k.pdf",
                                                    bbox_inches="tight")
    figure_accuracy(pts, args.dataset).savefig(
        out / f"{stem}-token-ratio-vs-accuracy-drop.pdf", bbox_inches="tight")
    print(f"-> {out / f'{stem}-token-ratio-by-k.pdf'}\n-> {out / f'{stem}-token-ratio-vs-accuracy-drop.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
