"""Per-item accuracy drop and per-k survival label for one (dataset, model).

    python -m llm.probe.build_labels                                   # every pair
    python -m llm.probe.build_labels --dataset gsm8k --model qwen3-8b
    python -m llm.probe.build_labels --dataset gpqa-diamond --model claude-opus-5 --random random-a

Per item and per k of the dataset's ladder, from the pooled isolated and random runs:

    accuracy_k = trials correct at k / trials at k        (an unreadable answer is wrong)
    drop_k     = solo_accuracy - accuracy_k
    label_k    = 1 if drop_k <= tau else 0                ("no worse than alone")

With tau = 0 the label is exactly "lost no accuracy". ``largest_surviving_k`` is the
largest k whose label is 1, or 0. Trials are bucketed by the batching pass, so the
short tail batch of a pass counts at that pass's k. Writes
``$RUN_DIR/probe/labels/<ds>/<model>.csv`` and a ``.json`` manifest recording the
runs read and the trial counts per item.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from llm import specs
from llm.probe import common

DEFAULT_TAU = 0.0


def _histogram(counts: pd.Series) -> dict[str, int]:
    values = pd.to_numeric(counts, errors="coerce").dropna().astype(int)
    return {str(int(k)): int(v) for k, v in sorted(values.value_counts().items())}


def build(dataset: str, model: str, *, isolated=None, random=None, tau: float = DEFAULT_TAU
          ) -> tuple[pd.DataFrame, dict]:
    isolated, random = common.stages(dataset, model, isolated, random)
    control = common.occurrences(isolated, dataset, model)
    composed = common.occurrences(random, dataset, model)
    rungs = common.ladder(dataset)
    found = [k for k in rungs if k in set(composed["nominal_k"])]
    if found != list(rungs):
        raise SystemExit(f"{dataset}/{model}: the random runs cover k={found}, not {list(rungs)}")

    solo = common.accuracy(control, ["question_id"]).rename(columns={
        "n_trials": "n_solo_trials", "n_correct": "n_solo_correct",
        "accuracy": "solo_accuracy"})
    per_k = common.accuracy(composed, ["question_id", "nominal_k"])

    pool = common.records(dataset)
    table = pd.DataFrame({
        "domain": dataset,
        "model": model,
        "question_id": [str(record["id"]) for record in pool],
        "category": [record.get("category") for record in pool],
    }).set_index("question_id")
    table = table.join(solo, how="left")
    for k in rungs:
        at_k = per_k.xs(k, level="nominal_k")
        table[f"n_trials_k{k}"] = at_k["n_trials"]
        table[f"accuracy_k{k}"] = at_k["accuracy"]
        table[f"drop_k{k}"] = table["solo_accuracy"] - table[f"accuracy_k{k}"]
        table[f"label_k{k}"] = (table[f"drop_k{k}"] <= tau).astype("float")
        table.loc[table[f"accuracy_k{k}"].isna(), f"label_k{k}"] = np.nan

    labels = table[[f"label_k{k}" for k in rungs]]
    measured = labels.notna().all(axis=1)
    largest = np.zeros(len(table))
    for k in rungs:
        largest = np.where(table[f"label_k{k}"].to_numpy() == 1, k, largest)
    table["largest_surviving_k"] = np.where(measured, largest, np.nan)
    table["n_labels"] = labels.notna().sum(axis=1)

    tail = composed[composed["k"] != composed["nominal_k"]]
    manifest = {
        "dataset": dataset,
        "model": model,
        "tau": tau,
        "k_ladder": list(rungs),
        "label_rule": f"label_k = 1 iff solo_accuracy - accuracy_k <= {tau}",
        "unparsed_counts_as": "wrong",
        "bucketing": "batching pass (nominal_k)",
        "n_items": int(len(table)),
        "n_items_fully_measured": int(measured.sum()),
        "tail_sizes": {str(key): int(value) for key, value
                       in tail["k"].value_counts().sort_index().items()},
        "isolated_stages": list(isolated),
        "random_stages": list(random),
        "isolated_runs": sorted(control["run_id"].unique().tolist()),
        "random_runs": sorted(composed["run_id"].unique().tolist()),
        "solo_trials_per_item": _histogram(table["n_solo_trials"]),
        "trials_per_item_by_k": {str(k): _histogram(table[f"n_trials_k{k}"]) for k in rungs},
        "filter": specs.dataset(dataset).filter,
        "solo_accuracy_mean": float(table["solo_accuracy"].mean()),
        "accuracy_by_k": {str(k): float(table[f"accuracy_k{k}"].mean()) for k in rungs},
        "positive_rate_by_k": {str(k): float(table[f"label_k{k}"].mean()) for k in rungs},
    }
    return table.reset_index(), manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="*", default=None)
    parser.add_argument("--model", nargs="*", default=None)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    common.stage_args(parser)
    args = parser.parse_args(argv)

    failed = []
    for dataset, model in common.pairs(args.dataset, args.model):
        try:
            table, manifest = build(dataset, model, isolated=args.isolated,
                                    random=args.random, tau=args.tau)
        except SystemExit as error:
            print(f"[{dataset}/{model}] skipped: {error}")
            failed.append(f"{dataset}/{model}")
            continue
        out = common.labels_file(dataset, model)
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        hists = {"solo": manifest["solo_trials_per_item"], **manifest["trials_per_item_by_k"]}
        uneven = {side: h for side, h in hists.items() if len(h) > 1}
        print(f"[{dataset}/{model}] {manifest['n_items_fully_measured']}/{manifest['n_items']}"
              f" items measured at every k; trials per item "
              + ", ".join(f"{side} {next(iter(h))}" for side, h in hists.items())
              + f" -> {out}")
        if uneven:
            print(f"    warning: uneven trial counts {uneven}")
    if failed:
        print(f"not built: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
