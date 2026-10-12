"""Per-item accuracy drops, and the per-policy metrics of the batching tables.

Every item is measured in two independent passes, each with its own generation seed
and its own batching seeds:

    pass A = isolated-a (k=1) + random-a (random batches of each k)
    pass B = isolated-b (k=1) + random-b

An item's drop at k is its isolated accuracy minus its accuracy in random batches of
nominal size k, both read off the trace's per-item grades. A trial whose answer list
could not be parsed counts as wrong for each of its items, on both sides.

    from llm.core.analysis import fragility
    a = fragility.load_outcomes("isolated-a", "mmlu", "qwen3-8b")
    r = fragility.load_outcomes("random-a", "mmlu", "qwen3-8b")
    table = fragility.drop_table(a, r, ks=(3, 6, 10, 20))

`policy_metrics` reads one batching policy's accuracy and output tokens per item the
way the batching tables report them: one value per batching seed, averaged over every
seed of every run of the listed stages.
"""
from __future__ import annotations

from statistics import mean

import pandas as pd

from llm import paths
from llm.core.analysis import eval_metrics
from llm.core.analysis.item_accuracy import _graded_ids, _records, _solo_parsed_correct
from llm.specs import dataset as dataset_spec


def models_with(dataset: str, stages) -> list[str]:
    """The models that have runs in every one of `stages` on `dataset`."""
    present = [set(paths.models_with_runs(stage, dataset)) for stage in stages]
    return sorted(set.intersection(*present)) if present else []


def load_outcomes(stage: str, dataset: str, model: str) -> pd.DataFrame:
    """One row per (record, position): ``question_id``, ``nominal_k``, ``correct``.

    ``correct`` is ``None`` where the record's answer list could not be attributed to
    its items. ``nominal_k`` is the size of the pass a batch belongs to, so the short
    tail batch of a pass still counts at that pass's k.
    """
    filter_name = dataset_spec(dataset).filter
    rows = []
    for run_dir in paths.runs(stage, dataset, model):
        for record in _records(run_dir, filter_name):
            atoms, _ = _graded_ids(record)
            if not atoms:
                continue
            term_correct = list(record.get("term_correct") or ())
            if term_correct and len(term_correct) != len(atoms):
                term_correct = []
            if not term_correct and len(atoms) == 1:
                # A k=1 record graded as a whole answer (the integrals tree).
                parsed, correct = _solo_parsed_correct(record)
                term_correct = [correct] if parsed else []
            realised_k = int(record.get("k") or len(atoms))
            nominal = record.get("nominal_k")
            nominal_k = int(nominal) if nominal is not None else realised_k
            for position, atom in enumerate(atoms):
                rows.append((str(atom), nominal_k,
                             bool(term_correct[position]) if term_correct else None))
    if not rows:
        raise SystemExit(f"no graded items in runs/{stage}/{dataset}/{model}")
    return pd.DataFrame(rows, columns=["question_id", "nominal_k", "correct"])


def item_accuracy(frame: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Accuracy per key, an unparsed attempt counting as wrong."""
    correct = frame["correct"].fillna(False).astype(bool)
    return frame.assign(_c=correct).groupby(keys)["_c"].mean()


def drop_table(solo: pd.DataFrame, batched: pd.DataFrame, ks) -> pd.DataFrame:
    """One row per item: isolated accuracy ``solo`` and ``drop_k<k>`` for every k."""
    table = pd.DataFrame(item_accuracy(solo, ["question_id"]).rename("solo"))
    per_k = item_accuracy(batched, ["question_id", "nominal_k"]).unstack("nominal_k")
    for k in ks:
        table[f"drop_k{k}"] = table["solo"] - per_k.get(k)
    return table


def trial_counts(frame: pd.DataFrame, keys: list[str]) -> str:
    """How many trials the keys have, e.g. ``10x196`` (196 items with 10 trials)."""
    counts = frame.groupby(keys).size().value_counts().sort_index()
    return ",".join(f"{int(n)}x{int(c)}" for n, c in counts.items())


# --------------------------------------------------------------------------
# batching policies
# --------------------------------------------------------------------------

def _seed_values(counters: eval_metrics.Counters) -> tuple[float, float, int] | None:
    """(item accuracy, output tokens per item, items) of one batching seed."""
    if not counters.items:
        return None
    if counters.terms:
        accuracy = counters.terms_correct / counters.terms
    else:
        accuracy = counters.correct / counters.graded
    return accuracy, counters.completion_tokens / counters.items, counters.items


def policy_metrics(stages, dataset: str, model: str, k: int | None = None) -> dict | None:
    """Accuracy and output tokens per item of one policy, averaged over its seeds.

    Every batching seed of every run under ``runs/<stage>/<dataset>/<model>/`` for each
    of `stages` contributes one value, and the values are averaged with equal weight.
    With `k`, only the prompts that held exactly k items count, whichever pass they
    came from (in a random pass the short tail batch of a larger k can hold k items).

    Returns ``acc`` and ``tok`` (fractions and tokens), ``n_seeds``, ``items`` (items
    answered per seed), ``mean_k`` and ``frac_alone`` (share of items asked alone),
    or ``None`` when no run exists.
    """
    filter_name = dataset_spec(dataset).filter
    seeds, sizes = [], {}
    for stage in stages:
        for run_dir in paths.runs(stage, dataset, model):
            summary = eval_metrics.analyse_run(run_dir, filter_name)
            if summary.overall is None:
                raise SystemExit(f"{run_dir}: {summary.note}")
            if k is None:
                block = summary.overall
            else:
                block = next((b for b in summary.per_k if b.k == k), None)
                if block is None:
                    continue
            seeds += [v for v in map(_seed_values, block.per_seed.values()) if v]
            for one in summary.per_k:
                sizes[one.k] = sizes.get(one.k, 0) + one.totals.items
    if not seeds:
        return None
    total = sum(sizes.values())
    return {
        "acc": mean(a for a, _, _ in seeds),
        "tok": mean(t for _, t, _ in seeds),
        "n_seeds": len(seeds),
        "items": seeds[0][2],
        "mean_k": sum(size * n for size, n in sizes.items()) / total,
        "frac_alone": sizes.get(1, 0) / total,
    }
