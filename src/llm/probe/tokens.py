"""Output tokens per item at each prompt size k, read off the isolated and random runs.

    python -m llm.probe.tokens                                    # every pair
    python -m llm.probe.tokens --dataset gsm8k --model qwen3-8b   # updates those rows

For one composition the cost per item is its generated tokens divided by the number
of items in it; the table holds the mean of that over every occurrence at a k (so a
short tail batch counts by how many items it really held). ``cost_ratio`` divides by
the k=1 value: it is the ``cost_k`` the fragility-aware assignment prices each k at.
Only output tokens are counted. Writes ``$RUN_DIR/probe/token-cost.csv`` (or
``--out``), one row per (dataset, model, k).

The stored table the paper's assignments read was assembled pair by pair from the
trials that existed at the time (for some pairs 3 or 5 seeds per side). This module
pools both passes of each side, so a rebuild moves the cost ratios by up to ~0.01.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from llm.probe import common


def _per_item_cost(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["output_tokens"] = pd.to_numeric(frame["output_tokens"], errors="coerce")
    frame["per_item"] = frame["output_tokens"] / frame["k"].clip(lower=1)
    grouped = frame.groupby("nominal_k")
    return pd.DataFrame({
        "tokens_per_item": grouped["per_item"].mean(),
        "tokens_per_prompt": grouped["output_tokens"].mean(),
        "mean_realised_k": grouped["k"].mean(),
        "n_occurrences": grouped.size(),
        "n_prompts": grouped["composition_id"].nunique(),
    })


def build_pair(dataset: str, model: str, *, isolated=None, random=None) -> pd.DataFrame:
    """One pair's rows, from the stages given (default: both passes of each side)."""
    isolated, random = common.stages(dataset, model, isolated, random)
    control = common.occurrences(isolated, dataset, model)
    composed = common.occurrences(random, dataset, model)
    parts = pd.concat([_per_item_cost(control), _per_item_cost(composed)])
    sizes = common.sizes(dataset)
    missing = [k for k in sizes if k not in parts.index]
    if missing:
        raise SystemExit(f"{dataset}/{model}: no token measurement at k={missing}")
    parts = parts.loc[list(sizes)].reset_index(names="k")
    parts.insert(0, "model", model)
    parts.insert(0, "domain", dataset)
    solo = float(parts.loc[parts["k"] == 1, "tokens_per_item"].iloc[0])
    parts["cost_ratio"] = parts["tokens_per_item"] / solo
    return parts


def cost_vector(table: pd.DataFrame, dataset: str, model: str) -> list[float]:
    """Tokens per item at each of the dataset's sizes (k=1 first), from the table."""
    part = table[(table["domain"] == dataset) & (table["model"] == model)]
    if part.empty:
        raise SystemExit(f"the token cost table has no row for {dataset}/{model}; "
                         "run python -m llm.probe.tokens")
    lookup = {int(k): float(v) for k, v in zip(part["k"], part["tokens_per_item"])}
    return [lookup[k] for k in common.sizes(dataset)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--model", nargs="*", default=None)
    parser.add_argument("--out", default=None,
                        help="the table to write or update (default: $RUN_DIR/probe/token-cost.csv)")
    common.stage_args(parser)
    args = parser.parse_args(argv)

    frames = []
    for dataset, model in common.pairs(args.dataset, args.model):
        frames.append(build_pair(dataset, model, isolated=args.isolated, random=args.random))
        rows = frames[-1]
        print(f"  {dataset}/{model}: " + " ".join(
            f"k{int(k)}={r:.3f}" for k, r in zip(rows["k"], rows["cost_ratio"])), flush=True)
    table = pd.concat(frames, ignore_index=True)

    path = Path(args.out) if args.out else common.token_cost_file()
    if path.is_file():
        # Rebuilt rows replace their own and leave every other pair alone.
        old = pd.read_csv(path)
        keys = set(zip(table["domain"], table["model"]))
        old = old[[key not in keys for key in zip(old["domain"], old["model"])]]
        table = pd.concat([old, table], ignore_index=True)
    table = table.sort_values(["domain", "model", "k"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"wrote {path} ({len(table)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
