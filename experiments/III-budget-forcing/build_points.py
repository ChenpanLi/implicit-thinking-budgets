"""The points of the token-ratio figures: composed and budget-forced, per (dataset, model, k, bin).

    python experiments/III-budget-forcing/build_points.py                       # every cell
    python experiments/III-budget-forcing/build_points.py --model qwen3-8b      # one model's cells
    python experiments/III-budget-forcing/build_points.py --refresh             # recompute cached cells

Every (dataset, model) cell is written to `parts/` as soon as it is done and reused on the
next call (`--refresh` recomputes it), so the slow export can be spread over several
processes with `--model` / `--dataset`. Once every cell is there, the two tables are joined:

* `composed-bins.csv` -- the per-term-trace runs. Per item and k, the token reduction
  `1 - T_i(k) / T_i(1)` (its section's thinking tokens against its isolated thinking
  tokens) and the accuracy change against isolated-a; averaged per k over the items of each
  length group. Groups are quintiles of the isolated-b output length: the lowest is `low`,
  the middle three `medium`, the top one `high`.
* `forced-bins.csv` -- the budget-forcing runs (each item alone, thinking cut at the budget
  it had at k). Per (k, bin): the mean token ratio (forced thinking / isolated thinking),
  the forced and the isolated accuracy, and the drop. Bins are cut at the 20th and 80th
  percentiles of the isolated-b output length. An unreadable answer counts as incorrect.

k counts the trailing dummy item of the per-term trace: k=2 is one real item plus the dummy.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from llm import paths
from llm.core.analysis.per_term_trace import (MCQ_DATASETS, _echo_reads, _field,
                                              load_experiment, load_solo, reference_run)
from llm.specs import MODELS, TRACE_DATASETS
from llm.specs import dataset as dataset_spec

OUT = paths.results("III-budget-forcing")
PARTS = OUT / "parts"
COMPOSED_CSV = OUT / "composed-bins.csv"
FORCED_CSV = OUT / "forced-bins.csv"
_SEED_RE = re.compile(r"-s(\d+)-k1-")


def _records(trace, filter_name):
    """The trace records of one extraction filter."""
    with open(trace, encoding="utf-8") as handle:
        for line in handle:
            if filter_name in line:
                r = json.loads(line)
                if r.get("filter") == filter_name:
                    yield r


def _correct(record):
    """The record's one item: True / False, or None when its answer was unreadable."""
    terms = _field(record.get("term_correct"))
    if terms:
        return bool(terms[0])
    verdict = record.get("verdict")
    if verdict in ("CORRECT", "WRONG", "INCORRECT"):
        return verdict == "CORRECT"
    return None


# --------------------------------------------------------------------------
# composed: the per-term-trace runs
# --------------------------------------------------------------------------
def composed_points(dataset, model, solo):
    _, positions = load_experiment(dataset, model)                 # actual positions only
    positions.loc[positions.echo_read, "correct"] = pd.NA          # echo reads are not graded

    per = (positions.groupby(["k", "item_id"])
           .agg(composed_tokens=("tokens", "mean"), composed_acc=("correct", "mean"))
           .reset_index().merge(solo, on="item_id"))
    per = per[per.solo_thinking_tokens > 0]
    per["saved_tokens"] = 1 - per.composed_tokens / per.solo_thinking_tokens
    per["acc_change_pp"] = 100 * (per.composed_acc - per.solo_accuracy)

    # quintiles of the independent isolated-b output length
    length = solo.set_index("item_id").split1_output_tokens.reindex(per.item_id.unique()).dropna()
    edges = np.quantile(length.values, np.linspace(0, 1, 6))
    quintile = pd.Series(np.minimum(np.searchsorted(edges, length.values, side="right") - 1, 4),
                         index=length.index)
    per["group"] = per.item_id.map(quintile).map(
        {0: "low", 1: "medium", 2: "medium", 3: "medium", 4: "high"})

    out = (per.groupby(["group", "k"])
           .agg(saved_tokens=("saved_tokens", "mean"), acc_change_pp=("acc_change_pp", "mean"),
                n_items=("item_id", "nunique"))
           .reset_index().assign(dataset=dataset, model=model))
    return out[["dataset", "model", "group", "k", "saved_tokens", "acc_change_pp", "n_items"]]


# --------------------------------------------------------------------------
# forced: the budget-forcing runs
# --------------------------------------------------------------------------
def budgets_path(dataset, model):
    """The drawn budget of every (k, item), stored beside the batch it was built into."""
    return paths.batch("budget-forcing", dataset, model).with_name(f"{model}.budgets.jsonl")


def forcing_run(dataset, model):
    """The one budget-forcing run of a (dataset, model), checked against its own run.json."""
    runs = paths.runs("budget-forcing", dataset, model)
    if len(runs) != 1:
        raise SystemExit(f"budget-forcing/{dataset}/{model} holds {len(runs)} runs; expected one")
    meta = json.loads((runs[0] / "run.json").read_text())["per_term_budget"]
    if meta["model"] != model or meta["variant"] != "section-outside":
        raise SystemExit(f"{runs[0]} is a {meta['variant']} run of {meta['model']}, "
                         f"not a section-outside run of {model}")
    return runs[0]


def solo_table(dataset, model):
    """Per item: isolated accuracy over the isolated-a trials (unreadable counts as wrong)."""
    run = reference_run(dataset, "isolated-a", model)
    by_item = defaultdict(dict)
    for r in _records(run / "trace.jsonl", dataset_spec(dataset).filter):
        match = _SEED_RE.search(str(r.get("doc_id")))
        seed = int(match.group(1)) if match else int(r.get("trial") or 0)
        by_item[_field(r["component_ids"])[0]][seed] = bool(_correct(r))
    return pd.DataFrame([{"item_id": item, "solo_acc": float(np.mean(list(seeds.values())))}
                         for item, seeds in by_item.items()])


def forced_rows(dataset, model):
    """One row per (k, item): the forced run's thinking tokens and outcome, and its budget."""
    run = forcing_run(dataset, model)
    outcome = {r["doc_id"]: bool(_correct(r))
               for r in _records(run / "trace.jsonl", dataset_spec(dataset).filter)}
    if dataset in MCQ_DATASETS:
        for doc_id in _echo_reads(run, 1):
            outcome[doc_id] = False
    with open(run / "generations.jsonl", encoding="utf-8") as handle:
        generations = [json.loads(line) for line in handle if line.strip()]
    capped = pd.DataFrame({
        "doc_id": [g["doc_id"] for g in generations],
        "capped_correct": [outcome.get(g["doc_id"], False) for g in generations],
        "thinking_tokens": [g["thinking_tokens"] for g in generations],
        "cap": [g["thinking_budget"] for g in generations],
        "free_tokens": [g.get("free_tokens") or 0 for g in generations],
    })
    with open(budgets_path(dataset, model), encoding="utf-8") as handle:
        budgets = pd.DataFrame([json.loads(line) for line in handle if line.strip()])
    budgets = budgets[["k", "item_id", "budget_tokens"]].copy()
    budgets["doc_id"] = [f"ptb-k{k:02d}-{item}" for k, item in zip(budgets.k, budgets.item_id)]
    frame = budgets.merge(capped, on="doc_id", how="inner", validate="one_to_one")
    if (frame.cap != frame.budget_tokens).any():
        raise SystemExit(f"{dataset}/{model}: a run's cap differs from its drawn budget")
    return frame.merge(solo_table(dataset, model), on="item_id", how="left")


def forced_points(dataset, model, solo):
    frame = forced_rows(dataset, model).merge(
        solo[["item_id", "solo_thinking_tokens", "split1_output_tokens"]], on="item_id", how="left")
    frame = frame[(frame.solo_thinking_tokens > 0) & frame.split1_output_tokens.notna()]
    lo, hi = np.quantile(frame.drop_duplicates("item_id").split1_output_tokens, [0.2, 0.8])
    capped = frame.capped_correct.astype(float)
    frame = frame.assign(
        length_bin=np.where(frame.split1_output_tokens <= lo, "low",
                            np.where(frame.split1_output_tokens > hi, "high", "medium")),
        token_ratio=(frame.thinking_tokens + frame.free_tokens) / frame.solo_thinking_tokens,
        accuracy_pct=100 * capped,
        solo_accuracy_pct=100 * frame.solo_acc,
        accuracy_drop_pp=100 * (frame.solo_acc - capped))
    out = (frame.groupby(["k", "length_bin"])
           .agg(n_items=("item_id", "nunique"), token_ratio=("token_ratio", "mean"),
                accuracy_pct=("accuracy_pct", "mean"), solo_accuracy_pct=("solo_accuracy_pct", "mean"),
                accuracy_drop_pp=("accuracy_drop_pp", "mean"))
           .reset_index())
    out.insert(0, "model", model)
    out.insert(0, "dataset", dataset)
    out["bin_edges_tokens"] = f"{lo:.0f}/{hi:.0f}"
    return out


# --------------------------------------------------------------------------
def part(kind, model, dataset):
    return PARTS / f"{kind}__{model}__{dataset}.csv"


def export_cell(model, dataset):
    print(f"[points] {model} / {dataset}", flush=True)
    solo = load_solo(dataset, model)            # slow: re-tokenizes every isolated answer
    PARTS.mkdir(parents=True, exist_ok=True)
    forced_points(dataset, model, solo).to_csv(part("forced", model, dataset), index=False)
    composed_points(dataset, model, solo).to_csv(part("composed", model, dataset), index=False)


def join_parts():
    """Write both tables once every cell is there; otherwise say which are missing."""
    cells = [(m, d) for m in MODELS for d in TRACE_DATASETS]
    missing = [f"{m}/{d}" for m, d in cells
               if not (part("forced", m, d).exists() and part("composed", m, d).exists())]
    if missing:
        print(f"[points] {len(missing)} cells still missing, tables not written: {', '.join(missing)}")
        return
    forced = pd.concat([pd.read_csv(part("forced", m, d)) for m, d in cells])
    composed = pd.concat([pd.read_csv(part("composed", m, d)) for m, d in cells])
    forced.sort_values(["dataset", "model", "k", "length_bin"]).to_csv(FORCED_CSV, index=False)
    composed.sort_values(["dataset", "model", "k", "group"]).to_csv(COMPOSED_CSV, index=False)
    print(f"[points] -> {COMPOSED_CSV}\n[points] -> {FORCED_CSV}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--dataset", nargs="+", default=TRACE_DATASETS, choices=TRACE_DATASETS)
    parser.add_argument("--refresh", action="store_true", help="recompute cells already in parts/")
    args = parser.parse_args()
    for model in args.model:
        for dataset in args.dataset:
            done = part("forced", model, dataset).exists() and part("composed", model, dataset).exists()
            if args.refresh or not done:
                export_cell(model, dataset)
    join_parts()


if __name__ == "__main__":
    main()
