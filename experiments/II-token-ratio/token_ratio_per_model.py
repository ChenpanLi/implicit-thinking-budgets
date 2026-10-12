"""Token ratio against k on all six per-term-trace datasets, one figure per model.

Each panel is one dataset; each colour one isolated-trace-length bin; each line a power
law through (1, 1) fitted to that bin's points, labelled with its law, ∝ k^m. The appendix shows one such figure per
model. Also writes every point drawn (and the pooled `all` row) to
`token-ratio-points.csv`.

    python experiments/II-token-ratio/token_ratio_per_model.py
    python experiments/II-token-ratio/token_ratio_per_model.py --models qwen3-8b
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from llm import paths
from llm.plotting import pt_to_in, use_paper_style
from llm.specs import MODELS

from ratio_plot import TITLES, bin_legend, by_bin, plot_bins, slope_labels, sparse_ticks
from trace_data import figure_points

DATASETS = ["integrals", "mmlu", "onewordqa", "gpqa-diamond", "gsm8k", "math500"]


def figure(model: str, points: dict[str, pd.DataFrame]):
    width_in = pt_to_in(600)
    fig, axes = plt.subplots(2, 3, figsize=(width_in, width_in * 0.62), layout="constrained")
    fig.get_layout_engine().set(wspace=0.04, hspace=0.04)
    fits = []
    for ax, ds in zip(axes.ravel(), DATASETS):
        d = by_bin(points[ds])
        fits.append(plot_bins(ax, d))
        sparse_ticks(ax, d, n=4)
        ax.set_title(TITLES[ds], fontsize=11, pad=4)
        ax.tick_params(axis="both", which="major", direction="out", length=3, labelsize=9, pad=2)
    legend = bin_legend(axes.ravel()[3], fontsize=10, borderaxespad=0.4)
    fig.supxlabel("Tasks per composition (log scale)", fontsize=12)
    fig.supylabel("Token ratio (log scale)", fontsize=12)
    # the slope labels go on last, into the settled layout, which then stays fixed
    fig.canvas.draw()
    fig.set_layout_engine("none")
    for ax, panel_fits in zip(axes.ravel(), fits):
        slope_labels(ax, panel_fits, fontsize=9.5, markersize=6,
                     avoid=[legend.get_window_extent()] if ax is axes.ravel()[3] else ())
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models", nargs="+", default=MODELS)
    args = parser.parse_args()
    use_paper_style()
    out = paths.results("II-token-ratio")
    rows = []
    for model in args.models:
        points = {ds: figure_points(ds, model) for ds in DATASETS}
        fig = figure(model, points)
        fig.savefig(out / f"{model}-token-ratio-all-datasets.pdf")
        plt.close(fig)
        print(f"-> {out / f'{model}-token-ratio-all-datasets.pdf'}")
        rows += [p.assign(dataset=ds, model=model) for ds, p in points.items()]
    csv = out / "token-ratio-points.csv"
    table = pd.concat(rows)[["dataset", "model", "bin", "k", "ratio", "n_items"]]
    if csv.exists() and set(args.models) != set(MODELS):   # keep the other models' rows
        old = pd.read_csv(csv)
        table = pd.concat([old[~old.model.isin(args.models)], table])
    table.sort_values(["model", "dataset", "bin", "k"]).to_csv(csv, index=False)
    print(f"-> {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
