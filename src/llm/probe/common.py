"""What every probe stage shares: the (dataset, model) pairs, the files, the run reader.

Layout under ``$RUN_DIR/probe/``:

    embeddings/<ds>/qwen--qwen3-embedding-8b.safetensors (+ .json)   embed_items
    labels/<ds>/<model>.csv (+ .json)                                build_labels
    splits/<ds>/kfold5-seed0-by-category.json                        make_splits
    heads/<prefix>-<ds>-<model>-fold<F>/                             train
    token-cost.csv                                                   tokens
    assignments/{measured,probe}/<ds>/<model>.csv                    assign_k

A head is named ``<prefix>-<ds>-<model>-fold<F>``: outer fold ``F`` is held out, and
the prefix names the inner fold the head held out of the remaining four fifths
(``kb-emb`` is inner fold 0, ``kb-embv1`` .. ``kb-embv3`` the others).

Two kinds of run are read, each pooled over two independent passes: the isolated
stages (every item alone, k=1) and the random stages (the pool partitioned into
random batches at every k of the dataset's ladder).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm import paths, specs
from llm.core.analysis.item_accuracy import (
    _graded_ids,
    _records,
    _solo_parsed_correct,
)

EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
SCHEME = "kfold5-seed0-by-category"
N_FOLDS = 5
# Inner fold j of the four the training slice is cut into is held out by this head.
PREFIXES = ("kb-emb", "kb-embv1", "kb-embv2", "kb-embv3")

ISOLATED = ("isolated-a", "isolated-b")
RANDOM = ("random-a", "random-b")

# Every pair the batching tables report. Opus (GPQA-Diamond only) has one random
# pass rather than two, so it is never part of "every pair": name it with --model.
PAIRS = [(ds, m) for ds in specs.BATCHING_DATASETS for m in specs.MODELS]
OPUS_PAIR = ("gpqa-diamond", specs.OPUS)


def ladder(dataset: str) -> tuple[int, ...]:
    """The k values of the dataset's random stages, without k=1."""
    return tuple(specs.dataset(dataset).random_ks)


def sizes(dataset: str) -> tuple[int, ...]:
    """The prompt sizes an item can be assigned, k=1 first."""
    return (1,) + ladder(dataset)


def pairs(datasets=None, models=None) -> list[tuple[str, str]]:
    """The pairs matching the filters; Opus only when it is named."""
    candidates = PAIRS + ([OPUS_PAIR] if models and specs.OPUS in models else [])
    return [(d, m) for d, m in candidates
            if (not datasets or d in datasets) and (not models or m in models)]


# ------------------------------------------------------------------ files

def embeddings_file(dataset: str) -> Path:
    slug = EMBED_MODEL.replace("/", "--").lower()
    return paths.probe("embeddings", dataset, f"{slug}.safetensors")


def labels_file(dataset: str, model: str) -> Path:
    return paths.probe("labels", dataset, f"{model}.csv")


def splits_file(dataset: str) -> Path:
    return paths.probe("splits", dataset, f"{SCHEME}.json")


def head_name(prefix: str, dataset: str, model: str, fold: int) -> str:
    return f"{prefix}-{dataset}-{model}-fold{fold}"


def head_dir(prefix: str, dataset: str, model: str, fold: int) -> Path:
    return paths.probe("heads", head_name(prefix, dataset, model, fold))


def token_cost_file() -> Path:
    return paths.probe("token-cost.csv")


def assignment_file(arm: str, dataset: str, model: str) -> Path:
    return paths.probe("assignments", arm, dataset, f"{model}.csv")


# ------------------------------------------------------------------ the pool

def records(dataset: str) -> list[dict]:
    """The item pool, in file order."""
    with paths.pool(dataset).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def embed_text(dataset: str, record: dict) -> str:
    """The item alone, as the embedding model reads it: no numbering, no instructions."""
    family = specs.dataset(dataset).family
    if family == "math":
        return str(record["integrand_latex_integral"])
    if family == "mcq":
        from llm.core.mcq.answers import LETTERS

        lines = [str(record["question"]).strip()]
        lines += [f"{LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(record["choices"])]
        return "\n".join(lines)
    if family == "oneword":
        return " ".join(str(record["question"]).split())
    return str(record["question"]).strip()


def load_embeddings(dataset: str) -> tuple[np.ndarray, dict[str, int]]:
    """The embedding matrix and ``{item id: row}``."""
    from safetensors.numpy import load_file

    path = embeddings_file(dataset)
    if not path.is_file():
        raise SystemExit(f"no embeddings at {path}; run python -m llm.probe.embed_items")
    meta = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    rows = {str(item): row for row, item in enumerate(meta["item_ids"])}
    return load_file(str(path))["embeddings"].astype(np.float32), rows


def load_splits(dataset: str, fold: int) -> tuple[list[str], list[str]]:
    """``(train ids, held-out ids)`` of one outer fold."""
    path = splits_file(dataset)
    if not path.is_file():
        raise SystemExit(f"no split file at {path}; run python -m llm.probe.make_splits")
    split = json.loads(path.read_text())["splits"][fold]
    return split["train"], split["test"]


# ------------------------------------------------------------------ the runs

def _output_tokens(record: dict) -> int | None:
    usage = record.get("token_usage") or {}
    for key in ("vllm_output_tokens", "output_tokens", "completion_tokens"):
        if usage.get(key) is not None:
            return int(usage[key])
    return None


def stage_runs(stage: str, dataset: str, model: str) -> list[Path]:
    """The finished runs of one stage; a named stage with none is an error."""
    runs = [d for d in paths.runs(stage, dataset, model) if (d / "trace.jsonl").is_file()]
    if not runs:
        raise SystemExit(f"{dataset}/{model}: no finished {stage} run under "
                         f"{paths.model_runs_dir(stage, dataset, model)}")
    return runs


def occurrences(stages, dataset: str, model: str) -> pd.DataFrame:
    """One row per (attempt, position) over every run of the given stages.

    ``correct`` is None where the answer list could not be read; callers count that
    as wrong. ``nominal_k`` is the k of the batching pass, so the one short tail batch
    of a pass still counts at its pass's k. A one-item record graded as a whole
    answer (the integrals isolated runs) is read through its verdict.
    """
    filter_name = specs.dataset(dataset).filter
    rows = []
    for stage in stages:
        for run_dir in stage_runs(stage, dataset, model):
            meta = json.loads((run_dir / "run.json").read_text())
            run_id = meta.get("run_id", run_dir.name)
            for record in _records(run_dir, filter_name):
                atoms, _ = _graded_ids(record)
                if not atoms:
                    continue
                term_correct = list(record.get("term_correct") or ())
                if term_correct and len(term_correct) != len(atoms):
                    term_correct = []
                if not term_correct and len(atoms) == 1:
                    parsed, correct = _solo_parsed_correct(record)
                    term_correct = [correct] if parsed else []
                k = int(record.get("k") or len(atoms))
                nominal = record.get("nominal_k")
                tokens = _output_tokens(record)
                for position, atom in enumerate(atoms):
                    rows.append({
                        "run_id": run_id,
                        "composition_id": record.get("problem_id") or record.get("doc_id"),
                        "k": k,
                        "nominal_k": int(nominal) if nominal is not None else k,
                        "batch_seed": record.get("batch_seed"),
                        "question_id": str(atom),
                        "correct": bool(term_correct[position]) if term_correct else None,
                        "output_tokens": tokens,
                    })
    frame = pd.DataFrame(rows)
    return frame.sort_values(["question_id", "nominal_k", "batch_seed", "composition_id"],
                             kind="stable").reset_index(drop=True)


def accuracy(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Trials, correct answers and accuracy per key; an unreadable answer is wrong."""
    correct = frame["correct"].fillna(False).astype(bool)
    grouped = frame.assign(_correct=correct).groupby(keys, dropna=False)
    out = grouped.agg(n_trials=("_correct", "size"), n_correct=("_correct", "sum"))
    out["accuracy"] = out["n_correct"] / out["n_trials"]
    return out


def stage_args(parser) -> None:
    """``--isolated`` / ``--random``: which stages are pooled into one measurement."""
    parser.add_argument("--isolated", nargs="+", default=None, metavar="STAGE",
                        help="isolated stages pooled for the k=1 side "
                             "(default: both passes, see baseline_stages)")
    parser.add_argument("--random", nargs="+", default=None, metavar="STAGE",
                        help="random stages pooled for the k>1 side "
                             "(default: both passes, see baseline_stages)")


def stages(dataset: str, model: str, isolated=None, random=None) -> tuple[list, list]:
    """The isolated and random stages of a pair: as given, else both passes.

    The defaults go through ``specs.baseline_stages``, which names the first-pass
    stages (``*-v1``) for the one pair whose regular stages hold a later rerun.
    """
    names = specs.baseline_stages(dataset, model)
    return ([names.get(s, s) for s in (isolated or ISOLATED)],
            [names.get(s, s) for s in (random or RANDOM)])
