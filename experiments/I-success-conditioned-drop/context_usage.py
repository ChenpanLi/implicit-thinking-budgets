"""Share of composed responses by how much of their output budget they used, per dataset.

Reads context-records.csv (from context_records.py) and writes context_budget_usage.csv and
context-budget-usage.pdf, the context-usage figure of the main results section. A
truncated response is its own bin; every other response falls into a bin by its share of
the budget.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from llm import paths, specs
from llm.plotting import pt_to_in, use_paper_style

EDGES = [0, 0.10, 0.25, 0.50, 0.75, 1.0]
COLS = ["pct_lt10", "pct_10_25", "pct_25_50", "pct_50_75", "pct_75_100"]
# Figure: datasets top to bottom, merged bins -> (legend label, colour).
ORDER = ["gsm8k", "math500", "aime", "integrals", "mmlu", "gpqa-diamond", "onewordqa", "strategyqa"]
SHORT = {"gsm8k": "GSM8K", "math500": "MATH-500", "aime": "AIME", "integrals": "Integrals",
         "mmlu": "MMLU-R.", "gpqa-diamond": "GPQA-D.", "onewordqa": "OneWordQA",
         "strategyqa": "StrategyQA"}
BINS = [
    (["pct_lt10", "pct_10_25"], "<25%", "#1B2F5E"),
    (["pct_25_50", "pct_50_75"], "25–75%", "#2F6FB0"),
    (["pct_75_100"], "75–99%", "#6FB3D9"),
    (["pct_truncated"], "truncated (100%)", "#C8413F"),
]
WIDTH_IN = pt_to_in(397) / 2.75   # half the text width, labels and legend included
HEIGHT_IN = 1.9
FS = 7.5


def usage_table(rec: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset plus an "all" row: the percentage of responses in each bin."""
    frac = rec["budget_share"].clip(upper=0.9999)
    rec = rec.assign(bin=pd.cut(frac, EDGES, labels=COLS, right=False))
    rows = []
    for key, g in list(rec.groupby("dataset")) + [("all", rec)]:
        ok = g[~g["truncated"]]
        row = {"dataset": key,
               "dataset_name": specs.DATASETS[key].title if key in specs.DATASETS else "All datasets",
               "n_models": g["model"].nunique(), "n_attempts": len(g),
               "n_truncated": int(g["truncated"].sum()), "pct_truncated": 100 * g["truncated"].mean()}
        for col in COLS:
            row[col] = 100 * (ok["bin"] == col).sum() / len(g)
        row["median_output_tokens"] = g["out_tokens"].median()
        row["median_budget_share"] = g["budget_share"].median()
        row["tokens_exact"] = bool(g["exact_tokens"].all())
        rows.append(row)
    table = pd.DataFrame(rows)
    assert np.allclose(table[COLS + ["pct_truncated"]].sum(axis=1), 100)
    return table


def plot(table: pd.DataFrame, path) -> None:
    """One stacked horizontal bar per dataset, the exact truncated share beside it."""
    use_paper_style()
    plt.rcParams.update({"grid.linewidth": 0.6})
    df = table.set_index("dataset").loc[ORDER]
    shares = pd.DataFrame({label: df[cols].sum(axis=1) for cols, label, _ in BINS})
    colors = {label: color for _, label, color in BINS}

    fig, ax = plt.subplots(figsize=(WIDTH_IN, HEIGHT_IN))
    y = np.arange(len(df))[::-1]
    left = np.zeros(len(df))
    for _, label, _ in BINS:
        width = shares[label].to_numpy()
        ax.barh(y, width, left=left, height=0.62, color=colors[label],
                edgecolor="white", linewidth=0.4, zorder=2)
        left = left + width
    for yi, (_, row) in zip(y, df.iterrows()):
        pct = row["pct_truncated"]
        text = "0%" if row["n_truncated"] == 0 else (f"{pct:.2f}%" if pct < 1 else f"{pct:.1f}%")
        ax.text(1.02, yi, text, transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=FS, color=colors["truncated (100%)"])

    ax.set_yticks(y, df.index.map(SHORT), fontsize=FS)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
    ax.set_xlabel("share of attempts", fontsize=FS, labelpad=2)
    ax.tick_params(axis="both", labelsize=FS, length=2.5, width=0.6, pad=2)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.minorticks_off()
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)

    handles = [Patch(facecolor=c, edgecolor="none", label=l) for _, l, c in BINS]
    handles = [handles[i] for i in (0, 2, 1, 3)]   # row-wise order for two columns
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False,
              fontsize=FS, handlelength=0.9, handleheight=0.9, handletextpad=0.3,
              columnspacing=0.6, borderaxespad=0, title_fontsize=FS, alignment="left")

    # Shrink the figure by the labels' overhang, so the saved PDF is WIDTH_IN x HEIGHT_IN.
    for _ in range(2):
        fig.canvas.draw()
        bb = fig.get_tightbbox()
        w, h = fig.get_size_inches()
        fig.set_size_inches(w - (bb.width - WIDTH_IN), h - (bb.height - HEIGHT_IN))
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)


def main() -> None:
    out = paths.results("I-success-conditioned-drop")
    rec = pd.read_csv(out / "context-records.csv")
    table = usage_table(rec)
    table.round(4).to_csv(out / "context_budget_usage.csv", index=False)
    plot(table, out / "context-budget-usage.pdf")
    with pd.option_context("display.width", 250):
        print(table.round(2).to_string(index=False))
    print(f"wrote {out}/context_budget_usage.csv and context-budget-usage.pdf")


if __name__ == "__main__":
    main()
