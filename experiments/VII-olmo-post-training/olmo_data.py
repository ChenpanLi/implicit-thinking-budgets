"""The OLMo 3 per-term-trace and isolated numbers, cached as CSVs.

Per (dataset, model), under `$RUN_DIR/results/VII-olmo-post-training/cache/<dataset>/<model>/`:

* `positions.csv.gz` -- one row per (response, actual position) of the per-term-trace
  runs: k, seed, item, section tokens, `correct` (empty when the answer list could not
  be read) and `echo_read` (the graded letters are the prompt's format example).
* `solo.csv` -- `load_solo`: per item, isolated thinking tokens and accuracy (`isolated-a`).
* `solo_trials.csv` -- one row per isolated trial: item, correct (empty when unparsed).

Building a cache re-tokenizes every isolated answer (minutes per dataset and model).

    python experiments/VII-olmo-post-training/olmo_data.py       # warm every cache
"""
from __future__ import annotations

import argparse
import gc
import sys

import pandas as pd

from llm import paths
from llm.core.analysis.per_term_trace import _field, _lenient, load_experiment, load_solo, reference_run
from llm.specs import OLMO, OLMO_DATASETS, dataset as dataset_spec

FILES = {"positions": "positions.csv.gz", "solo": "solo.csv", "solo_trials": "solo_trials.csv"}


def solo_trials(dataset: str, model: str) -> pd.DataFrame:
    """Every isolated-a trial of every item, graded as the analyses read it."""
    control = reference_run(dataset, "isolated-a", model)
    if control is None:
        raise FileNotFoundError(f"no isolated-a run of {model} on {dataset}")
    rows = []
    for r in _lenient(control / "trace.jsonl", dataset_spec(dataset).filter):
        c = _field(r.get("term_correct")) or []
        rows.append({"item_id": _field(r["component_ids"])[0], "doc_id": r["doc_id"],
                     "correct": bool(c[0]) if c else None})
    return pd.DataFrame(rows)


def _build(dataset: str, model: str) -> dict[str, pd.DataFrame]:
    _, positions = load_experiment(dataset, model)
    positions = positions[["k", "seed", "doc_id", "position", "item_id", "tokens", "correct",
                           "echo_read"]]
    return {"positions": positions, "solo": load_solo(dataset, model),
            "solo_trials": solo_trials(dataset, model)}


def frames(dataset: str, model: str, refresh: bool = False) -> dict[str, pd.DataFrame]:
    """{"positions", "solo", "solo_trials"} for one (dataset, model), built on first use."""
    out = paths.results("VII-olmo-post-training") / "cache" / dataset / model
    out.mkdir(parents=True, exist_ok=True)
    if refresh or not all((out / f).exists() for f in FILES.values()):
        for name, frame in _build(dataset, model).items():
            frame.to_csv(out / FILES[name], index=False)
    # `correct` is empty where the answer could not be read: keep that as NA, not False.
    return {name: pd.read_csv(out / f, dtype={"correct": "boolean"} if name != "solo" else None)
            for name, f in FILES.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", default=OLMO_DATASETS)
    parser.add_argument("--refresh", action="store_true", help="rebuild existing caches")
    args = parser.parse_args()
    for dataset in args.datasets:
        for model in OLMO:
            frames(dataset, model, args.refresh)
            print(f"[ok] {dataset} {model}", file=sys.stderr, flush=True)
            gc.collect()   # one tokenizer at a time
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
