"""The per-term-trace numbers of one (dataset, model), cached as CSVs.

Loading a model's per-term-trace runs and its isolated baseline re-tokenizes every
isolated answer, which takes minutes per (dataset, model). `frames()` does it once and
keeps three small tables under `$RUN_DIR/results/II-token-ratio/cache/<dataset>/<model>/`:

* `items_k.csv` -- one row per (k, item) over the actual positions: mean section
  tokens over the item's occurrences, mean correctness (unreadable answer lists left
  out), number of occurrences.
* `attempts.csv` -- one row per response: k, seed, think_closed, n_sections, and how
  many of its positions (actual ones, and all of them including the dummy) the trace
  named at least once.
* `solo.csv` -- `load_solo`: per item, isolated thinking tokens and accuracy
  (`isolated-a`) and mean isolated output tokens (`isolated-b`).

For Qwen3.8 27B on GPQA-Diamond the paper's figures use the rerun and its power-law fits
the first pass. `fit_solo()` gives the
first-pass baseline for that pair (`isolated-a-v1`, `isolated-b-v1`, see
`llm.specs.baseline_stages`), cached as `solo-first-pass.csv`, and `solo.csv` for every
other pair.

`figure_points()` turns them into the points the token-ratio figures draw.

    python experiments/II-token-ratio/trace_data.py            # warm every cache
"""
from __future__ import annotations

import argparse
import gc
import sys

import pandas as pd

from llm import paths
from llm.core.analysis.per_term_trace import load_experiment, load_solo
from llm.specs import MODELS, TRACE_DATASETS, baseline_stages

NAMES = ("items_k", "attempts", "solo")
BINS = ["Low", "Med.", "High"]


def cache_dir(dataset: str, model: str):
    out = paths.results("II-token-ratio") / "cache" / dataset / model
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build(dataset: str, model: str) -> dict[str, pd.DataFrame]:
    attempts, positions = load_experiment(dataset, model, include_non_actual=True)
    actual = positions[positions.role == "actual"]
    items_k = (actual.groupby(["k", "item_id"])
               .agg(tokens=("tokens", "mean"), acc=("correct", "mean"), n=("tokens", "size"))
               .reset_index())
    keys = ["k", "seed", "doc_id"]
    is_actual = positions.role == "actual"
    named = positions.assign(named=positions.visits > 0, is_actual=is_actual,
                             actual_named=(positions.visits > 0) & is_actual)
    per = named.groupby(keys).agg(n_positions=("position", "size"), n_named=("named", "sum"),
                                  n_actual=("is_actual", "sum"),
                                  n_actual_named=("actual_named", "sum"))
    attempts = attempts[keys + ["n_sections", "think_closed", "readable"]].merge(
        per.reset_index(), on=keys, how="left")
    return {"items_k": items_k, "attempts": attempts, "solo": load_solo(dataset, model)}


def frames(dataset: str, model: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """{"items_k", "attempts", "solo"} for one (dataset, model), built on first use."""
    out = cache_dir(dataset, model)
    if refresh or not all((out / f"{n}.csv").exists() for n in NAMES):
        for name, frame in _build(dataset, model).items():
            frame.to_csv(out / f"{name}.csv", index=False)
    return {n: pd.read_csv(out / f"{n}.csv") for n in NAMES}


def fit_solo(dataset: str, model: str) -> pd.DataFrame:
    """`load_solo` on the isolated stages the power-law fits were computed from."""
    stages = baseline_stages(dataset, model)
    if stages["isolated-a"] == "isolated-a":
        return frames(dataset, model)["solo"]
    path = cache_dir(dataset, model) / "solo-first-pass.csv"
    if not path.exists():
        load_solo(dataset, model, stages=(stages["isolated-a"], stages["isolated-b"])).to_csv(
            path, index=False)
    return pd.read_csv(path)


def figure_points(dataset: str, model: str) -> pd.DataFrame:
    """Per (bin, k): the mean token ratio the token-ratio figures draw.

    The ratio of an item at k is its mean section tokens over its mean isolated output
    tokens in the independent `isolated-b` pass. The bins are quintiles of those output
    tokens, taken over the (k, item) rows: Low the first, Med. the middle three, High
    the last. `all` pools every item.
    """
    f = frames(dataset, model)
    m = f["items_k"].merge(f["solo"], on="item_id")
    m["ratio"] = m.tokens / m.split1_output_tokens
    q = pd.qcut(m.split1_output_tokens, q=5, labels=False)
    groups = {"Low": q == 0, "Med.": q.between(1, 3), "High": q == 4, "all": q.notna() | q.isna()}
    out = [m[sel].groupby("k").agg(ratio=("ratio", "mean"), n_items=("item_id", "nunique"))
           .reset_index().assign(bin=name) for name, sel in groups.items()]
    return pd.concat(out, ignore_index=True)[["bin", "k", "ratio", "n_items"]]




def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=TRACE_DATASETS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--refresh", action="store_true", help="rebuild existing caches")
    args = parser.parse_args()
    for dataset in args.datasets:
        for model in args.models:
            try:
                frames(dataset, model, args.refresh)
                print(f"[ok] {dataset} {model}", file=sys.stderr, flush=True)
            except FileNotFoundError as err:
                print(f"[skip] {dataset} {model}: {err}", file=sys.stderr, flush=True)
            gc.collect()   # one tokenizer at a time
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
