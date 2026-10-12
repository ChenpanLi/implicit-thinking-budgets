"""Power-law fits of the token ratio through (1, 1): every model on every dataset.

The paper summarises how fast the token ratio falls with k by one fit per (model,
dataset) over all of its tasks:

    log r(k) = m log k        (no intercept: a composition of one task has r = 1)

The ratio of a task at k is its mean section tokens over its mean isolated thinking
tokens (`isolated-a`: output tokens minus the re-tokenized answer minus one; the first
pass `isolated-a-v1` for Qwen3.8 27B on GPQA-Diamond, as in the paper), so tasks
with no isolated thinking are left out; a point is the mean ratio over the tasks at
that k, every k the per-term trace ran. R^2 is the centred one (it goes negative when
the line through (1, 1) does worse than a flat one).

Writes `power-law-fits.csv` (one row per fit) and prints the summaries the paper reports:
per model (median and range of m, mean R^2) and over all fits (median and range of m,
median R^2, range of the per-model mean R^2).

    python experiments/II-token-ratio/power_law_fits.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llm import paths
from llm.specs import MODEL_NAMES, MODELS, TRACE_DATASETS

from trace_data import fit_solo, frames


def ratio_by_k(dataset: str, model: str) -> pd.Series:
    """Mean token ratio over tasks, per k (composed section / isolated thinking tokens)."""
    m = frames(dataset, model)["items_k"].merge(fit_solo(dataset, model), on="item_id")
    m = m[(m.solo_thinking_tokens > 0) & m.split1_output_tokens.notna()]
    return (m.tokens / m.solo_thinking_tokens).groupby(m.k).mean()


def fit(ratio: pd.Series) -> dict:
    x, y = np.log(ratio.index.values.astype(float)), np.log(ratio.values)
    m = float((x * y).sum() / (x * x).sum())
    r2 = 1 - ((y - m * x) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return {"n_k": len(x), "k_min": int(ratio.index.min()), "k_max": int(ratio.index.max()),
            "slope": m, "r2": float(r2)}


def main() -> int:
    fits = pd.DataFrame([{"dataset": ds, "model": model, **fit(ratio_by_k(ds, model))}
                         for model in MODELS for ds in TRACE_DATASETS])
    out = paths.results("II-token-ratio") / "power-law-fits.csv"
    fits.to_csv(out, index=False, float_format="%.6g")
    print(fits.pivot(index="dataset", columns="model", values="slope").round(3).to_string())
    print(fits.pivot(index="dataset", columns="model", values="r2").round(3).to_string())
    print(f"-> {out}\n")

    per_model = fits.groupby("model").agg(median_slope=("slope", "median"),
                                          min_slope=("slope", "min"),
                                          max_slope=("slope", "max"), mean_r2=("r2", "mean"))
    print(per_model.rename(index=MODEL_NAMES).round(2).to_string())
    print(f"\nall fits: median slope {fits.slope.median():.2f}, "
          f"range [{fits.slope.min():.2f}, {fits.slope.max():.2f}], "
          f"median R^2 {fits.r2.median():.2f}, per-model mean R^2 "
          f"[{per_model.mean_r2.min():.2f}, {per_model.mean_r2.max():.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
