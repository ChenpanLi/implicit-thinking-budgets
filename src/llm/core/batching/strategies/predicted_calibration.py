"""Batch every item at the largest k a per-item selection says it tolerates.

The selection is a CSV (one row per item and fold, with the chosen k), written by
``python -m llm.probe.assign_k`` and passed as ``--stats predicted-largest-k=<csv>``
with ``--param column=pred_largest_k_lagrangian``. It comes either from the measured
fragility of each item (``fragility-measured``) or from the probe's prediction for
an item it never saw in training (``fragility-probe``).

The pool is tiered by k and cut tier by tier (``_calibration_common.assign``),
**smallest k first**, a tier's remainder topped up from the **larger** tiers only
-- an item that tolerates k=10 is safe in a batch of 3 and not the reverse -- and
the last tier's remainder emitted short the way ``random`` keeps its tail.

One pass per fold, and why they must not be mixed
-------------------------------------------------
The selection is cross-validated: per fold *f*, it speaks only for the items fold
*f* held out from the fitting. That makes each fold's held-out set a **separate
dataset**, and this strategy yields one pass per fold rather than one pass over
the pool:

* a pass covers exactly the items of its own fold's held-out set, declared with
  ``Pass.covers`` so the engine checks it rather than trusting it;
* **no batch ever holds items from two different folds' held-out sets.** If it
  did, an item would share a prompt whose size was chosen by a predictor trained
  on that very item, and the number at the end would not be a prediction result;
* every row carries ``fold``, stamped by the pass, so a run's own trace says
  which fold each answer belongs to.

Under the five-fold split the five held-out fifths tile the pool exactly once per
seed; the manifest says so per pass (``n_items_covered``, ``covers_whole_pool``).

The items the selection cannot speak for
----------------------------------------
Two kinds, counted separately and both batched at the **fallback k**:

* items whose largest k is 0 -- the selection expects any composition to cost
  accuracy;
* items the selection carries with ``predicted`` false -- they are in the fold's
  held-out set but no k could be chosen for them.

Neither is evidence that any k is safe, so both go to ``--fallback-k``: 1 by
default, which asks them on their own and composes nothing.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from ..registry import BatchingContext, Pass, strategy
from ._calibration_common import UNBATCHED_K, assign, check_fallback, one_param

DEFAULT_KS = (3, 6, 10, 20)


def k_values(params: Mapping[str, Any]) -> list[int]:
    """``--k`` as a sorted list of ints, however it was written.

    ``--k 3 20`` arrives as a list and ``--param k=3,20`` as a string; both mean
    the same thing. The result is **sorted and required to be distinct**, and
    both properties are load-bearing rather than tidiness: the same list names
    the build (:func:`build_name`), so ``--k 20 3`` and ``--k 3 20`` have to be
    one name rather than two holding the same partitions, and a repeated k
    would be two identical passes carrying one label -- the same reason
    ``llm.batches.resolve_seeds`` refuses a repeated seed.

    Read here rather than through ``BatchingContext.int_list_param`` because the
    build's *name* is computed from the raw parameters, before any context
    exists, and the name has to list exactly the k values the passes were built
    at. One parser serves both.
    """
    value = params.get("k")
    if value is None:
        ks = list(DEFAULT_KS)
    elif isinstance(value, (list, tuple)):
        ks = [int(v) for v in value]
    else:
        ks = [int(part) for part in str(value).replace(",", " ").split()]
    if not ks:
        raise SystemExit("--k needs at least one value")
    if len(set(ks)) != len(ks):
        raise SystemExit(f"repeated k in {ks}; each k is one full pass over the pool")
    for k in ks:
        if k < 1:
            raise SystemExit(f"--k must be at least 1, got {k}")
    return sorted(ks)


# Written into each pass's manifest entry (``budget``, ``sorted_by``) and nothing
# else; the selection's own accuracy budget is set in ``llm.probe.assign_k``.
DEFAULT_BUDGET = 0.02

# Where an item the selection cannot speak for goes: asked on its own, composing
# nothing.
DEFAULT_FALLBACK_K = 1

# The name the selection table is addressed by (``--stats predicted-largest-k=...``).
TABLE = "predicted-largest-k"

# The columns of that table this strategy reads; the k column is ``--param column=``.
ID_COLUMN = "question_id"
FOLD_COLUMN = "fold"
MODEL_COLUMN = "model"
PREDICTED_COLUMN = "predicted"

# Announced once per (model, settings) rather than once per seed: the tiering is
# the same at every seed, which differ only in how the tiers are shuffled.
_announced: set = set()


def settings(params: Mapping[str, Any]) -> tuple[float, list[int], int, str]:
    """``(budget, ks, fallback_k, k_column)`` -- what this strategy *is*.

    Read from the parameters rather than from the context, because the same
    values decide the name the build is filed under and :func:`build_name` is
    called before any file is opened; a name that disagreed with the partition
    would be worse than no name at all.
    """
    budget = one_param(params, "budget", DEFAULT_BUDGET, float, "--param budget=")
    fallback_k = one_param(params, "fallback_k", DEFAULT_FALLBACK_K, int, "--fallback-k")
    k_column = one_param(params, "column", None, str, "--param column=")
    if not k_column:
        raise SystemExit("predicted-calibration needs --param column=<k column of the "
                         "selection>, e.g. pred_largest_k_lagrangian")
    ks = k_values(params)
    check_fallback(fallback_k, ks, what="the selection cannot speak for")
    return budget, ks, fallback_k, k_column


def build_name(params: Mapping[str, Any]) -> str:
    """``predicted-calibration-lagrangian-fb1-k3-6-10-20``.

    Every setting that changes the partition is in it and none is left implicit:
    the selection column, the fallback (it moves every item the selection cannot
    speak for) and the ladder.
    """
    _budget, ks, fallback_k, k_column = settings(params)
    tag = k_column.replace("pred_largest_k", "").strip("_")
    return (f"predicted-calibration-{tag}-fb{fallback_k}-k"
            + "-".join(str(k) for k in ks))


def _as_int(value: Any) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _is_false(value: Any) -> bool:
    return str(value).strip().lower() in {"false", "0", "0.0", "no"}


def read_selection(rows: Sequence[Mapping[str, str]], model: str, ks: Sequence[int],
                   k_column: str) -> tuple[dict[int, dict[str, int]], dict[int, set[str]],
                                           dict[int, int]]:
    """``({fold: {item id: k}}, {fold: unpredicted ids}, {fold: n rows})`` for one model.

    Only the rows of ``model`` are read: a selection is a property of the
    (domain, model) pair, and another model's predicted k is not a prediction
    about this one. A file with no row for it stops the build rather than
    batching by somebody else's numbers.

    A k outside the ladder is refused rather than rounded down. The selection's
    own ladder and the pass's have to be the same four rungs, because an item
    assigned k=20 by a predictor and then batched at 10 because the ladder stops
    there is neither the prediction nor a measurement of it.
    """
    if not rows:
        raise SystemExit(f"the {TABLE!r} selection table is empty")
    header = set(rows[0])
    for column in (ID_COLUMN, FOLD_COLUMN, k_column):
        if column not in header:
            raise SystemExit(
                f"the {TABLE!r} selection table has no {column!r} column; found "
                f"{', '.join(sorted(header))}. It should be a file written by "
                "`python -m llm.probe.assign_k`.")

    ladder = {int(k) for k in ks}
    by_fold: dict[int, dict[str, int]] = {}
    unpredicted: dict[int, set[str]] = {}
    counts: dict[int, int] = {}
    models_seen: set[str] = set()
    for row in rows:
        if MODEL_COLUMN in header:
            models_seen.add(str(row.get(MODEL_COLUMN, "")))
            if str(row.get(MODEL_COLUMN, "")) != model:
                continue
        fold = _as_int(row.get(FOLD_COLUMN))
        item = str(row.get(ID_COLUMN, "")).strip()
        if fold is None or not item:
            continue
        counts[fold] = counts.get(fold, 0) + 1
        selected = by_fold.setdefault(fold, {})
        if item in selected or item in unpredicted.setdefault(fold, set()):
            raise SystemExit(
                f"the {TABLE!r} selection table names item {item!r} twice in fold "
                f"{fold} for model {model!r}. One fold's selection is one k per "
                "item; two rows make the batch size arbitrary.")
        if PREDICTED_COLUMN in header and _is_false(row.get(PREDICTED_COLUMN)):
            unpredicted[fold].add(item)
            continue
        k = _as_int(row.get(k_column))
        if k is None or k == 0:
            # The predictor put this item below the cut at every rung: it expects
            # any composition to cost accuracy. A real outcome, and the fallback
            # tier is where it goes -- told apart from "could not be predicted"
            # in the manifest because they are different facts.
            unpredicted.setdefault(fold, set())
            selected[item] = 0
            continue
        if k not in ladder:
            raise SystemExit(
                f"the {TABLE!r} selection assigns item {item!r} (fold {fold}) to "
                f"k={k}, which is not one of the batch sizes {sorted(ladder)}. The "
                "selection's ladder and the pass's must be the same rungs.")
        selected[item] = k

    if not counts:
        known = ", ".join(sorted(m for m in models_seen if m)) or "(none)"
        raise SystemExit(
            f"the {TABLE!r} selection table has no rows for model {model!r}, so "
            "there is nothing to batch it by -- and another model's predictions "
            "are not predictions about this one. Models in the table: " + known)
    for fold in by_fold:
        unpredicted.setdefault(fold, set())
    return by_fold, unpredicted, counts


@strategy(
    "predicted-calibration",
    description="Batch every item at the largest k a cross-validated predictor "
                "says it tolerates within a degradation budget, one pass per "
                "fold over that fold's held-out items and never across folds.",
    uses_model=True,
    uses_stats=True,
    name_from_params=build_name,
)
def predicted_calibration(ctx: BatchingContext) -> Iterator[Pass]:
    """One pass per cross-validation fold, over that fold's held-out items."""
    budget, ks, fallback_k, k_column = settings(ctx.params)
    rows = ctx.stats.table(TABLE)
    by_fold, unpredicted, counts = read_selection(rows, ctx.model, ks, k_column)

    index_of = {str(item["id"]): index for index, item in enumerate(ctx.items)}
    partition_ks = sorted({*(int(k) for k in ks), fallback_k})
    announcement = (ctx.model, budget, tuple(ks), fallback_k, k_column)
    say = announcement not in _announced
    if say:
        _announced.add(announcement)
        print(f"[batch] selection: {k_column} at budget {budget:g}, "
              f"{len(by_fold)} fold(s), {sum(counts.values())} row(s) for "
              f"{ctx.model}")

    for fold in sorted(by_fold):
        selected = by_fold[fold]
        unknown = sorted((set(selected) | unpredicted[fold]) - set(index_of))
        if unknown:
            raise SystemExit(
                f"the {TABLE!r} selection names {len(unknown)} item(s) that are "
                f"not in the pool (first: {unknown[0]!r}). The selection "
                "and the pool have drifted apart; rebuild one of them rather "
                "than batching a pool this selection was not made for.")

        # The universe of this pass: exactly the fold's held-out items, and
        # nothing else. `covers` hands it to the engine, which then holds the
        # pass to partitioning it -- every one of these items once, and no item
        # from another fold's set.
        universe = sorted({index_of[item] for item in selected}
                          | {index_of[item] for item in unpredicted[fold]})

        tiers: dict[int, list[int]] = {k: [] for k in partition_ks}
        n_no_surviving_k = 0
        for item, k in selected.items():
            if k == 0:
                n_no_surviving_k += 1
                tiers[fallback_k].append(index_of[item])
            else:
                tiers[k].append(index_of[item])
        for item in unpredicted[fold]:
            tiers[fallback_k].append(index_of[item])

        # A fresh stream per fold rather than consuming ctx.rng in sequence, so
        # that adding a fourth fold later cannot re-draw the first three -- the
        # same rule `random` follows for its per-k passes.
        rng = ctx.derive("fold", fold)
        groups, borrowed, short = assign(tiers, partition_ks, rng)

        if say:
            tiering = "  ".join(f"k={k}x{len(tiers[k])}" for k in partition_ks)
            print(f"[batch] fold {fold}  : {len(universe)} held-out item(s)  "
                  f"[{tiering}]  -> {len(groups)} batches")
            if n_no_surviving_k or unpredicted[fold]:
                print(f"[batch]   fallback: {n_no_surviving_k + len(unpredicted[fold])} "
                      f"item(s) at k={fallback_k}"
                      + (", asked on their own and composed with nothing"
                         if fallback_k == UNBATCHED_K else "")
                      + f" -- {n_no_surviving_k} predicted to tolerate no k, "
                      f"{len(unpredicted[fold])} the predictor could not be run on")
            filled = {k: count for k, count in borrowed.items() if count}
            if filled:
                print("[batch]   filled  : " + "  ".join(
                    f"k={k} borrowed {count} item(s) from larger tiers"
                    for k, count in filled.items()))
            if short:
                print(f"[batch]   tail    : {len(short)} short batch(es) of {short}")

        yield Pass(
            groups=groups,
            # The fold is the pass, so it is the label -- which becomes part of
            # every row id and lands on every trace record as `batch_pass`.
            label=f"fold{fold}",
            # No single target size: the sizes are per item by construction. The
            # real k is on every row, as always.
            nominal_k=None,
            covers=universe,
            # Carried onto every row and from there onto every trace record, so
            # a run says which fold each answer belongs to without anything
            # parsing it back out of the label.
            stamp={"fold": int(fold)},
            info={
                "fold": int(fold),
                "sorted_by": f"{k_column} at a degradation budget of {budget:g}",
                "selection_table": TABLE,
                "budget": budget,
                "k_column": k_column,
                "k_ladder": list(ks),
                "fallback_k": fallback_k,
                "n_heldout_items": len(universe),
                "n_items_by_batch_size": {str(k): len(tiers[k]) for k in partition_ks},
                "n_borrowed_into_k": {str(k): count for k, count in borrowed.items()},
                "n_predicted": len(selected) - n_no_surviving_k,
                "n_no_surviving_k": n_no_surviving_k,
                "n_unpredicted": len(unpredicted[fold]),
                "unmeasured_placement": (
                    "asked on their own (fallback k=1), composed with nothing"
                    if fallback_k == UNBATCHED_K
                    else f"batched at the fallback k={fallback_k}"),
                "short_batches": short,
            },
        )
