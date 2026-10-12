"""The accuracy drop of confidently solved items under composition, shared by the figures.

One row per composed slot (an item at one position of one composition, on one trial):

    d = isolated-b accuracy of the item  -  1/0 outcome of the item in the composition

An unparseable composed answer counts as wrong, because the isolated accuracies divide by
every trial, parsed or not. Items were selected on isolated-a (3 of 3 trials correct), so
the baseline is isolated-b, the independent re-run. The k=1 point is isolated-a minus
isolated-b over the same items, which shows how much of the drop is selection noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from llm import paths, specs
from llm.core.analysis.item_accuracy import (compare_atomic_to_composition,
                                             load_composition_items, load_sanity_items)
from llm.plotting import MODEL_COLORS

KS = [3, 6, 10, 20]
N_BOOT = 2000
SEED = 0

# Legend names and panel titles as printed in the paper's figures.
LEGEND = {
    "r1-qwen-distill-7b": "R1 Qwen 7B",
    "r1-llama-distill-8b": "R1 Llama 8B",
    "r1-qwen3-8b": "R1 Qwen3 8B",
    "qwen3-4b-thinking": "Qwen3 4B Thinking",
    "qwen3-8b": "Qwen3 8B",
    "qwen3-14b": "Qwen3 14B",
    "qwen3-8-27b": "Qwen3.8 27B",
    "claude-opus-5": "Claude Opus 5",
}
TITLES = {
    "gpqa-diamond": "GPQA-Diamond",
    "integrals": "Integrals",
    "mmlu": "MMLU-Redux",
    "onewordqa": "OneWordQA",
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "strategyqa": "StrategyQA",
    "aime": "AIME",
}

POINT_COLUMNS = ["dataset", "model", "k", "n", "drop_pp", "lo_pp", "hi_pp"]


def load_drop(dataset: str) -> pd.DataFrame:
    """Every composed slot of ``dataset`` with its isolated accuracies and ``d``."""
    filter_name = specs.dataset(dataset).filter
    items = load_composition_items(paths.runs_glob("composed", dataset), filter_name=filter_name)
    items = items[items["model"].isin(MODEL_COLORS) & items["k"].isin(KS)]
    iso_a = load_sanity_items(dataset, filter_name, paths.runs_glob("isolated-a", dataset))
    iso_b = load_sanity_items(dataset, filter_name, paths.runs_glob("isolated-b", dataset))

    df = pd.concat([compare_atomic_to_composition(g, iso_a, model=m)
                    for m, g in items.groupby("model", sort=False)])
    key = ["model", "question_id"]
    df["isolated_a"] = df["atomic_accuracy"]
    df["isolated_b"] = (iso_b.set_index(key)["accuracy"]
                        .reindex(pd.MultiIndex.from_frame(df[key])).to_numpy())
    df["composed_correct"] = df["correct"].fillna(False).astype(float)
    df["d"] = df["isolated_b"] - df["composed_correct"]
    missing = df["d"].isna().sum()
    if missing:
        raise SystemExit(f"{dataset}: {missing} composed slots have no isolated-b accuracy")
    return df[["model", "question_id", "k", "isolated_a", "isolated_b",
               "composed_correct", "d"]].reset_index(drop=True)


def mean_ci(d, rng, n_boot: int = N_BOOT, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of ``d`` and its 95% percentile bootstrap interval.

    The bootstrap resamples single values of ``d`` (slots), as the figures in the paper
    did; it does not cluster the slots of one item.
    """
    d = np.asarray(d, dtype=float)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].sum(axis=1) / len(d)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return d.mean(), lo, hi


def k1_values(model_df: pd.DataFrame) -> np.ndarray:
    """Isolated-a minus isolated-b, once per item that appears in the model's compositions."""
    items = model_df.drop_duplicates("question_id").dropna(subset=["isolated_b"])
    return (items["isolated_a"] - items["isolated_b"]).to_numpy()


def drop_points(dataset: str, df: pd.DataFrame, rng) -> list[dict]:
    """One point per (model, k), k=1 included, drawing the bootstrap from ``rng``.

    The order of the draws (models in legend order; per model the composed k ascending,
    then k=1) is the one the paper's figures used, so the intervals come out the same.
    """
    points = []
    for model in MODEL_COLORS:
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue
        groups = [(k, g["d"].to_numpy()) for k, g in model_df.groupby("k")]
        groups.append((1, k1_values(model_df)))
        for k, d in groups:
            points.append(point(dataset, model, k, d, rng))
    return points


def point(dataset: str, model: str, k: int, d, rng) -> dict:
    """A plotted point as a CSV row: the drop and its interval in percentage points."""
    mean, lo, hi = mean_ci(d, rng)
    return {"dataset": dataset, "model": model, "k": int(k), "n": len(d),
            "drop_pp": 100 * mean, "lo_pp": 100 * lo, "hi_pp": 100 * hi}


def points_frame(points: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(points, columns=POINT_COLUMNS).sort_values(
        ["dataset", "model", "k"], kind="stable").reset_index(drop=True)
