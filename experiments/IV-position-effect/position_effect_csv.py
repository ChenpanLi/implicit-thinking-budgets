"""Thinking tokens and accuracy drop per prompt position and seed -> one CSV.

Reads every per-term-trace run of each (dataset, model). One row per (k, position,
seed): the mean section tokens of that position and its mean accuracy drop (composed
correctness minus the item's isolated accuracy from `isolated-a`). The last position is
the dummy, kept with role=dummy; the fillers that pad one batch per seed are dropped.
Echo reads and unreadable answer lists are left out of accuracy. A (dataset, model)
without an isolated run gets NaN drop.

    python experiments/IV-position-effect/position_effect_csv.py        # MMLU-Redux and MATH-500
    python experiments/IV-position-effect/position_effect_csv.py --datasets mmlu gsm8k
"""
from __future__ import annotations

import argparse
import gc
import sys

import pandas as pd

from llm import paths
from llm.core.analysis.per_term_trace import load_experiment, load_solo


def summarise(dataset: str, model: str) -> list[dict]:
    try:
        _, pos = load_experiment(dataset, model, include_non_actual=True)
    except FileNotFoundError as err:
        print(f"[skip] {dataset} {model}: {err}", file=sys.stderr)
        return []
    pos = pos[pos.role != "filler"].copy()
    pos.loc[pos.echo_read, "correct"] = pd.NA
    try:
        solo = load_solo(dataset, model)[["item_id", "solo_accuracy"]]
    except FileNotFoundError as err:
        print(f"[no solo] {dataset} {model}: {err}", file=sys.stderr)
        solo = pd.DataFrame({"item_id": pos.item_id.unique(), "solo_accuracy": float("nan")})
    pos = pos.merge(solo, on="item_id", how="left")
    pos["correct"] = pos["correct"].astype("Float64")
    pos["drop"] = pos["correct"] - pos["solo_accuracy"]
    rows = []
    keys = ["k", "position", "role", "seed"]
    for key, g in pos.groupby(keys):
        rows.append({"dataset": dataset, "model": model, **dict(zip(keys, key)),
                     "n": len(g), "n_seeds": g.seed.nunique(),
                     "mean_thinking_tokens": g["tokens"].mean(),
                     "composed_accuracy": g["correct"].mean(),
                     "solo_accuracy": g.loc[g["drop"].notna(), "solo_accuracy"].mean(),
                     "accuracy_drop": g["drop"].mean()})
    print(f"[ok] {dataset} {model}", file=sys.stderr, flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=["mmlu", "math500"])
    parser.add_argument("--models", nargs="+", default=None,
                        help="default: every model with per-term-trace runs on the dataset")
    args = parser.parse_args()
    rows = []
    for ds in args.datasets:
        for model in args.models or paths.models_with_runs("per-term-trace", ds):
            rows += summarise(ds, model)
            gc.collect()   # one tokenizer at a time
    df = pd.DataFrame(rows).sort_values(["dataset", "model", "k", "position", "seed"])
    out = paths.results("IV-position-effect") / "position_effect_seeds.csv"
    df.to_csv(out, index=False)
    print(f"-> {out}: {len(df)} rows, {df.groupby(['dataset', 'model']).ngroups} (dataset, model)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
