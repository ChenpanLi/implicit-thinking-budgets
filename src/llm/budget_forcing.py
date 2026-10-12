"""Budget forcing: every item asked alone, its thinking capped at its own per-term budget.

    python -m llm.budget_forcing build --dataset gsm8k --model qwen3-8b
    python -m llm.budget_forcing run --dataset gsm8k --model qwen3-8b [--dry-run]

The question: if the accuracy drop under composition comes from the thinking each item
gets squeezed into, then asking an item alone but capping its thinking at what it got
inside a k-item prompt should reproduce the drop as k grows.

**build** reads the model's per-term-trace runs (``runs/per-term-trace/<ds>/<model>/``,
cut by ``python -m llm.sections``). For every prompt size k, each pool item occurs once
per batching seed (10 seeds). One occurrence per (k, item) is drawn uniformly at random,
seeded by (dataset, model, k, item) alone, and its budget is the item's labelled section
plus all of that response's unlabelled thinking (the read-through before the first label,
and the token or so before ``<think>``). Only actual positions count -- the dummy and the
fillers are not measurements -- and a k is used only when complete (every item seen once
per seed over all 10 seeds). A section of 0 tokens stays 0 and is flagged
``zero_section``. The k=1 documents are the seed-0 rows of the model's isolated batch,
given a new id (``ptb-k<K>-<item>``) and the budget fields. Writes, beside each other:

    batches/budget-forcing/<ds>/<model>.jsonl            the capped k=1 documents
    batches/budget-forcing/<ds>/<model>.budgets.jsonl    one drawn budget per (k, item)
    batches/budget-forcing/<ds>/<model>.manifest.json    written last; marks a finished build

**run** sends those documents through the budget-forcing driver (``llm.core.thinking``)
in one engine: an upper cap only (a model that stops early is taken at its word, one
still thinking at its cap is cut off, the block is closed and an ``Answer:`` line is
cued), 512 answer tokens after the forced close, the section-marker prompt at k=1 (the
wording the budgets were measured under), and grading through the dataset's own task
module. The run lands in ``runs/budget-forcing/<ds>/<model>/<run-id>/``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from llm import paths
from llm.specs import TRACE_SEEDS, Dataset, dataset as dataset_spec

STAGE = "budget-forcing"
# The only budget the paper uses: the item's section plus the unlabelled thinking.
VARIANT = "section-outside"
BUDGET_KEY = "budget_tokens"
# Room for the answer after the forced close. Short on purpose: the `Answer:` cue
# otherwise lets a capped model keep reasoning in the open for ~2k tokens.
ANSWER_TOKENS = 512
DRAW_SEED = 0


def documents_path(dataset: str, model: str) -> Path:
    return paths.batch(STAGE, dataset, model)


def budgets_path(dataset: str, model: str) -> Path:
    return documents_path(dataset, model).with_suffix(".budgets.jsonl")


def manifest_path(dataset: str, model: str) -> Path:
    return documents_path(dataset, model).with_suffix(".manifest.json")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------
def draw_rng(dataset: str, model: str, k: int, item: str) -> random.Random:
    """A generator for one (k, item) cell, independent of every other cell."""
    return random.Random(f"{dataset}|{model}|k{k}|{item}|draw{DRAW_SEED}")


def load_occurrences(dataset: str, model: str):
    """Actual positions of every cut per-term-trace run, with the response's unlabelled
    thinking (`outside_tokens`) and this position's share of it; echo reads graded NA."""
    import pandas as pd

    from llm.core.analysis.per_term_trace import load_experiment

    attempts, positions = load_experiment(dataset, model, include_non_actual=True)
    key = ["k", "doc_id", "trial"]
    labelled = positions.groupby(key).agg(section_sum=("tokens", "sum"),
                                          n_positions=("tokens", "size")).reset_index()
    response = attempts[key + ["thinking_tokens"]].merge(labelled, on=key, validate="one_to_one")
    response["unlabelled"] = response.thinking_tokens - response.section_sum
    if (response.unlabelled < 0).any():
        raise SystemExit(f"{dataset}/{model}: sections exceed the thinking in "
                         f"{(response.unlabelled < 0).sum()} responses")
    positions = positions.merge(response[key + ["section_sum", "n_positions", "unlabelled"]],
                                on=key, validate="many_to_one")
    positions["outside_share"] = [
        u * t / total if total > 0 else u / n
        for u, t, total, n in zip(positions.unlabelled, positions.tokens,
                                  positions.section_sum, positions.n_positions)]
    positions["outside_tokens"] = positions.unlabelled
    positions = positions[positions.role == "actual"].copy()
    positions.loc[positions.echo_read, "correct"] = pd.NA
    return positions, sorted(set(attempts.run))


def budget_of(occurrence) -> int:
    return int(occurrence.tokens + occurrence.outside_tokens)


def complete_ks(positions) -> tuple[list[int], dict[int, str]]:
    """The k values with every item seen once per seed over exactly TRACE_SEEDS seeds."""
    used, skipped = [], {}
    for k, frame in positions.groupby("k"):
        seeds = sorted(frame.seed.unique())
        if len(seeds) != TRACE_SEEDS:
            skipped[int(k)] = f"{len(seeds)} of {TRACE_SEEDS} seeds ({seeds})"
            continue
        per_item = frame.groupby("item_id").seed.agg(["count", "nunique"])
        bad = per_item[(per_item["count"] != TRACE_SEEDS) | (per_item["nunique"] != TRACE_SEEDS)]
        if len(bad):
            skipped[int(k)] = f"{len(bad)} items not seen once per seed (first: {bad.index[0]})"
            continue
        used.append(int(k))
    return used, skipped


def control_set(dataset: str, model: str) -> Path:
    """The model's isolated batch, whose seed-0 rows are the solo documents.

    The control batching does not depend on the model, so isolated-b's seed-0 rows
    are the same documents where an isolated-a file is missing.
    """
    for stage in ("isolated-a", "isolated-b"):
        path = paths.batch(stage, dataset, model)
        if path.exists():
            return path
    raise SystemExit(f"no isolated batch for {model} on {dataset}; build it with "
                     f"python -m llm.batches --stage isolated-a ... first")


def control_rows(dataset: str, model: str) -> dict[str, dict]:
    """Seed-0 rows of the model's k=1 isolated batch, by item id."""
    out = {}
    for row in read_jsonl(control_set(dataset, model)):
        if int(row.get("k", 1)) == 1 and int(row["batch_seed"]) == 0:
            out[row["component_ids"][0]] = row
    if not out:
        raise SystemExit(f"{control_set(dataset, model)} holds no seed-0 k=1 rows")
    return out


def _bool(value):
    """A nullable boolean as JSON: True, False or None for NA."""
    import pandas as pd

    return None if pd.isna(value) else bool(value)


def build(dataset: str, model: str) -> dict:
    positions, runs = load_occurrences(dataset, model)
    ks, skipped = complete_ks(positions)
    if not ks:
        raise SystemExit(f"{dataset}/{model}: no complete k in the per-term-trace runs: {skipped}")
    controls = control_rows(dataset, model)

    budgets, documents = [], []
    missing_control: set[str] = set()
    zero_by_k: Counter = Counter()
    items_by_k: Counter = Counter()
    for k in ks:
        frame = positions[positions.k == k].sort_values("seed")
        for item, occ in frame.groupby("item_id", sort=True):
            occ = occ.reset_index(drop=True)
            pick = occ.iloc[draw_rng(dataset, model, k, item).randrange(len(occ))]
            graded = occ.correct.dropna()
            entry = {
                "dataset": dataset, "model": model, "k": k, "item_id": item,
                BUDGET_KEY: budget_of(pick),
                "budget_variant": VARIANT,
                "section_tokens": int(pick.tokens),
                "outside_share": float(pick.outside_share),
                "outside_tokens": int(pick.outside_tokens),
                "budget_seed": int(pick.seed),
                "budget_visits": int(pick.visits),
                "zero_section": int(pick.tokens) == 0,
                "composed_correct": _bool(pick.correct),
                "composed_acc_10": float(graded.mean()) if len(graded) else None,
                "composed_n_graded": int(len(graded)),
                "section_tokens_by_seed": [int(t) for t in occ.tokens],
                "budget_tokens_by_seed": [budget_of(o) for o in occ.itertuples()],
                # None where the answer list was unreadable (or an echo read).
                "composed_correct_by_seed": [_bool(c) for c in occ.correct],
                "section_tokens_mean": float(occ.tokens.mean()),
                "section_tokens_min": int(occ.tokens.min()),
                "section_tokens_max": int(occ.tokens.max()),
                "source_doc_id": pick.doc_id,
                "draw_seed": DRAW_SEED,
            }
            budgets.append(entry)
            items_by_k[k] += 1
            zero_by_k[k] += entry["zero_section"]
            row = controls.get(item)
            if row is None:
                missing_control.add(item)
                continue
            doc = dict(row)
            doc["control_id"] = row["id"]
            doc["id"] = f"ptb-k{k:02d}-{item}"
            doc.update({key: entry[key] for key in (
                BUDGET_KEY, "budget_variant", "section_tokens", "outside_share", "outside_tokens",
                "budget_seed", "budget_visits", "zero_section",
                "composed_correct", "composed_acc_10", "section_tokens_mean",
                "source_doc_id", "draw_seed")})
            doc["budget_k"] = k
            documents.append(doc)

    if missing_control:
        print(f"[warn] {len(missing_control)} items have budgets but no seed-0 control row "
              f"and are left out (first: {sorted(missing_control)[:3]})", file=sys.stderr)
    manifest_path(dataset, model).unlink(missing_ok=True)  # unfinished until rewritten
    _write_jsonl(budgets_path(dataset, model), budgets)
    _write_jsonl(documents_path(dataset, model), documents)
    manifest = {
        "dataset": dataset, "tree": dataset_spec(dataset).tree, "model": model,
        "draw_seed": DRAW_SEED, "budget_key": BUDGET_KEY, "variant": VARIANT,
        "budget_unit": "section tokens plus all of the response's unlabelled thinking "
                       "(preamble + outside)",
        "per_term_runs": runs,
        "ks": ks, "ks_skipped": {str(k): why for k, why in sorted(skipped.items())},
        "k_counts_dummy": True,
        "n_items_by_k": {str(k): items_by_k[k] for k in ks},
        "zero_section_share_by_k": {str(k): zero_by_k[k] / items_by_k[k] for k in ks},
        "control_set": str(control_set(dataset, model)),
        "n_items_without_control": len(missing_control),
        "n_documents": len(documents),
        "budgets": str(budgets_path(dataset, model)),
        "documents": str(documents_path(dataset, model)),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path(dataset, model).write_text(json.dumps(manifest, indent=2) + "\n",
                                              encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def spec_for(ds: Dataset):
    """The forcing driver's view of a dataset: its task module, filters in YAML order."""
    from llm.core.thinking.runner import ExperimentSpec

    module = ds.task_utils()
    if ds.per_term:
        ensembles = (("boxed_last", "extract_boxed_list_last"),
                     ("final_answer", "extract_boxed_list_final_answer"))
        env_run_id = "LMEVAL_INTEGRALS_RUN_ID"
    else:
        ensembles = (("lenient", "extract_lenient"), ("explicit", "extract_explicit"))
        env_run_id = module.ENV_RUN_ID
    return ExperimentSpec(name=ds.tree, module_path=f"lm_tasks.{ds.tree}.utils",
                          ensembles=ensembles, env_run_id=env_run_id, stage="evaluation")


def run(dataset: str, model: str, passthrough: list[str]) -> int:
    from llm.core.thinking.runner import build_parser, run_per_document
    from llm.run import task_env

    ds = dataset_spec(dataset)
    manifest_file = manifest_path(dataset, model)
    if not manifest_file.exists():
        raise SystemExit(f"no finished budget build ({manifest_file} is missing); run "
                         f"python -m llm.budget_forcing build --dataset {dataset} "
                         f"--model {model} first")
    manifest = json.loads(manifest_file.read_text())
    documents = documents_path(dataset, model)
    docs = read_jsonl(documents)
    if len(docs) != manifest["n_documents"]:
        raise SystemExit(f"{documents} holds {len(docs)} documents, its manifest says "
                         f"{manifest['n_documents']}; rebuild it")

    # The task module reads these when it renders and grades.
    env = task_env(ds, section_markers=True)
    os.environ.update(env)
    spec = spec_for(ds)
    output_root = paths.model_runs_dir(STAGE, dataset, model)
    runner_args = build_parser(spec, __doc__).parse_args([
        "--model", model, "--dataset", str(documents),
        "--budget-mode", "upper", "--answer-tokens", str(ANSWER_TOKENS),
        "--output-root", str(output_root),
        *passthrough,
    ])
    if runner_args.limit is not None:
        docs = docs[:runner_args.limit]
    module = spec.load_module()
    return run_per_document(
        spec, runner_args, documents=docs, budget_key=BUDGET_KEY,
        output_root=output_root, dataset_name=dataset, run_id_extra=(VARIANT,),
        run_config_extra={
            "per_term_budget": manifest,
            "section_markers": True,
            "section_markers_prompt": module.MARKER_PROMPT_VERSION,
            "task_env": env,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["build", "run"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--force", action="store_true",
                        help="build: rebuild even if the manifest exists (the draws do not change)")
    args, passthrough = parser.parse_known_args(argv)
    dataset_spec(args.dataset)

    if args.command == "run":
        # Any further flag goes to the forcing driver: --dry-run, --progress,
        # --model-variant, --limit, --keep-thinking.
        return run(args.dataset, args.model, passthrough)

    if passthrough:
        parser.error(f"build takes no {' '.join(passthrough)}")
    if manifest_path(args.dataset, args.model).exists() and not args.force:
        print(f"[build] {manifest_path(args.dataset, args.model)} exists; --force to rebuild")
        return 0
    info = build(args.dataset, args.model)
    print(f"[build] {args.dataset}/{args.model}: {info['n_documents']} documents over "
          f"k {info['ks']}")
    for k, why in info["ks_skipped"].items():
        print(f"    k={k:>2}: skipped, {why}")
    print(f"[build] -> {info['documents']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
