"""The five outer folds of each dataset, stratified by category.

    python -m llm.probe.make_splits                     # every batching dataset
    python -m llm.probe.make_splits --dataset gsm8k

Fold f is the held-out set and the other four together are the training set, so the
five held-out sets tile the pool. A split belongs to the dataset, not the model: every
model is cross-fitted on the same folds. Within each category (in sorted order) the
items are shuffled by a seeded RNG and dealt round-robin, with an offset carried
across categories so small ones do not always start at fold 0. Writes
``$RUN_DIR/probe/splits/<ds>/kfold5-seed0-by-category.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from llm import paths, specs
from llm.probe import common

SEED = 0


def deal(item_ids: list[str], categories: list, *, folds: int, seed: int) -> dict[str, int]:
    """``{item id: fold}``, dealt round-robin inside each category."""
    rng = np.random.default_rng(seed)
    strata: dict[object, list[str]] = defaultdict(list)
    for item_id, category in zip(item_ids, categories):
        strata[category].append(item_id)
    assignment: dict[str, int] = {}
    offset = 0
    for key in sorted(strata, key=lambda value: "" if value is None else str(value)):
        ids = strata[key]
        for position, index in enumerate(rng.permutation(len(ids))):
            assignment[ids[index]] = (position + offset) % folds
        offset = (offset + len(ids)) % folds
    return assignment


def build(dataset: str) -> dict:
    pool = common.records(dataset)
    ids = [str(record["id"]) for record in pool]
    fold_of = deal(ids, [record.get("category") for record in pool],
                   folds=common.N_FOLDS, seed=SEED)
    splits = []
    for fold in range(common.N_FOLDS):
        train = [i for i in ids if fold_of[i] != fold]
        test = [i for i in ids if fold_of[i] == fold]
        splits.append({"fold": fold, "n_train": len(train), "n_test": len(test),
                       "train": train, "test": test})
    return {
        "kind": "splits",
        "domain": dataset,
        "scheme": common.SCHEME,
        "n_folds": common.N_FOLDS,
        "seed": SEED,
        "stratify": "category",
        "train_share": "majority",
        "convention": "fold f is the held-out TEST set; the other folds together "
                      "are the TRAIN set",
        "n_items": len(ids),
        "item_ids": ids,
        "fold_of_item": [fold_of[i] for i in ids],
        "splits": splits,
        "source_pool": str(paths.pool(dataset)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", nargs="*", default=specs.BATCHING_DATASETS)
    args = parser.parse_args(argv)
    for dataset in args.dataset:
        record = build(dataset)
        out = common.splits_file(dataset)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        sizes = ", ".join(str(s["n_test"]) for s in record["splits"])
        print(f"[{dataset}] {record['n_items']} items, held-out folds of {sizes} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
