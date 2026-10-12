"""Turning a batching strategy into a dataset file, and checking it as it goes.

The engine owns everything a strategy must not have to think about: seeding,
the loop over seeds, proving that a pass really is a partition, stamping the
identity fields, and building the manifest. What is left for a strategy is the
one decision that distinguishes it from every other -- which items go together.

The invariant this module exists to enforce is the one the whole comparison
rests on: **within one pass, every pool item appears exactly once**. Two
strategies can then be compared on their scores directly, because both were
graded on the same corpus; a strategy that dropped the hard items, or showed
the easy ones twice, would otherwise beat an honest one for a reason that has
nothing to do with batching. :func:`validate_pass` is therefore fatal, never a
warning.

The other thing worth stating: **row ids must be unique across the whole file**.
A file holds every seed and every pass, so the same item is in it several times
over -- and each task layer's ``_STASH`` in ``lm_tasks/<tree>/utils.py`` is keyed
on the document id, with a ``consumed`` counter picking the filter ensemble. A
repeated id there silently crosses two documents' extractions. :func:`build`
asserts uniqueness rather than trusting the id format.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import BatchingContext, Pass, Strategy

# Fields the engine owns on every row. A record builder that returned one of
# these would be quietly overwritten, so `build` refuses instead.
IDENTITY_FIELDS = (
    "id", "stage", "k", "strategy", "batch_seed", "batch_pass", "batch_index",
    "nominal_k", "batching_model", "rng_seed",
    # Written only for a pass with an `actual` mask (see `Pass.actual`).
    "component_actual", "n_actual",
)

# The stage every batching row carries. It is what selects the variable-k
# lm-eval task (`<prefix>_evaluation`), which -- unlike the per-k compositional
# tasks -- declares no `k` in its metadata and therefore does not narrow the
# file to one k.
STAGE = "evaluation"

# How a record builder is called: given the items of one group, in prompt
# order, return the domain fields of one composition (the question block, the
# component lists, the answer key). Identity is added by the engine.
RecordBuilder = Callable[[Sequence[Mapping[str, Any]]], dict]


@dataclass
class BuildResult:
    rows: list[dict]
    manifest: dict


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL pool or dataset, skipping `//` comment lines."""
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def manifest_beside(dataset: Path) -> dict | None:
    """The batching manifest written beside a dataset file, if there is one.

    The runner copies it verbatim into ``run.json``, so a finished
    run says which strategy, which seeds and which pool it measured without
    anyone having to find the build command. ``None`` for every other kind of
    dataset, which is also what tells a batching run from a compositional one
    after the fact.
    """
    manifest = Path(dataset).parent / f"{Path(dataset).stem}.manifest.json"
    if not manifest.exists():
        return None
    with manifest.open(encoding="utf-8") as handle:
        return json.load(handle)


def covered_indices(pass_: Pass, n_items: int, *, where: str) -> list[int]:
    """The indices this pass must partition: its own ``covers``, else the pool.

    A declared subset is checked here rather than trusted: a repeated or
    out-of-range index in it would make the coverage test below vacuous in
    exactly the direction that hides a bug.
    """
    if pass_.covers is None:
        return list(range(n_items))
    declared = [int(index) for index in pass_.covers]
    bad = [index for index in declared if index < 0 or index >= n_items]
    if bad:
        raise SystemExit(
            f"{where}: the pass declares it covers index {bad[0]}, which is not "
            f"an index into a pool of {n_items} items")
    if len(set(declared)) != len(declared):
        raise SystemExit(
            f"{where}: the pass declares the same index twice in `covers`; the "
            "set of items a pass partitions cannot hold a duplicate")
    if not declared:
        raise SystemExit(f"{where}: the pass declares it covers no items")
    return declared


def validate_pass(pass_: Pass, n_items: int, *, where: str) -> None:
    """Fail unless the pass is a partition of the items it claims to cover.

    Three separate failures, each named on its own, because they have three
    different causes: an item used twice (the strategy drew instead of
    partitioning), an item never used (an off-by-one at the tail, or a filter
    the strategy applied and forgot to undo), and an empty or out-of-range
    group (an index bug). All three produce a file that runs and grades and
    means nothing, which is why none of them is a warning.

    "The items it claims to cover" is the whole pool unless the pass set
    ``covers`` (see :class:`~llm.core.batching.registry.Pass`); a pass that
    declared a subset is held to that subset just as strictly, and an item from
    outside it is a failure of its own -- that is the check that keeps two
    cross-validation folds' held-out sets from being batched into each other.
    """
    universe = covered_indices(pass_, n_items, where=where)
    allowed = set(universe)
    masks = _masks(pass_, where=where)
    seen = Counter()
    for position, group in enumerate(pass_.groups):
        if len(group) == 0:
            raise SystemExit(f"{where}: group {position} is empty")
        if len(set(group)) != len(group):
            # Under a mask a non-actual item may repeat one the partition holds
            # elsewhere, but never one in its own prompt: the same question
            # twice in one prompt is two answers to one question.
            raise SystemExit(
                f"{where}: group {position} holds the same pool index twice; a "
                "prompt never shows one question in two positions"
            )
        for index, is_actual in zip(group, masks[position]):
            if not isinstance(index, int) or index < 0 or index >= n_items:
                raise SystemExit(
                    f"{where}: group {position} holds {index!r}, which is not an "
                    f"index into a pool of {n_items} items"
                )
            if not is_actual:
                # A dummy or a filler: in range and unique within the prompt,
                # checked above, and outside the partition by construction.
                continue
            if index not in allowed:
                raise SystemExit(
                    f"{where}: group {position} holds pool index {index}, which is "
                    f"not among the {len(allowed)} items this pass declared it "
                    "covers. A pass that narrows its universe must stay inside "
                    "it -- batching an item from another pass's set together "
                    "with this one is exactly what `covers` exists to refuse."
                )
            seen[index] += 1

    repeated = sorted(i for i, count in seen.items() if count > 1)
    if repeated:
        raise SystemExit(
            f"{where}: {len(repeated)} pool items appear more than once in the "
            f"same pass (first: index {repeated[0]}, {seen[repeated[0]]} times). "
            "A pass must be a partition -- every item exactly once -- or the "
            "strategies being compared are not graded on the same corpus."
        )
    missing = [i for i in universe if i not in seen]
    if missing:
        scope = ("the whole pool" if pass_.covers is None
                 else f"the {len(universe)} items it declared it covers")
        raise SystemExit(
            f"{where}: {len(missing)} of {len(universe)} items are in no group "
            f"(first: index {missing[0]}). A pass must cover {scope}."
        )


def _masks(pass_: Pass, *, where: str) -> list[list[bool]]:
    """Each group's actual-position mask: the pass's own, or all True.

    A mask that does not match its group's length, or a group with no actual
    position at all, is refused -- the first would misattribute every answer
    after the mismatch, the second is a prompt that measures nothing.
    """
    if pass_.actual is None:
        return [[True] * len(group) for group in pass_.groups]
    if len(pass_.actual) != len(pass_.groups):
        raise SystemExit(
            f"{where}: the pass has {len(pass_.groups)} groups but "
            f"{len(pass_.actual)} actual masks; the two must be parallel")
    masks = []
    for position, (group, mask) in enumerate(zip(pass_.groups, pass_.actual)):
        mask = [bool(flag) for flag in mask]
        if len(mask) != len(group):
            raise SystemExit(
                f"{where}: group {position} has {len(group)} items but a mask of "
                f"{len(mask)}; every position needs exactly one flag")
        if not any(mask):
            raise SystemExit(
                f"{where}: group {position} has no actual position; a prompt "
                "of fillers and dummies alone measures nothing")
        masks.append(mask)
    return masks


def _pass_label(pass_: Pass, index: int) -> str:
    label = str(pass_.label or f"p{index}").strip()
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in label)
    if not safe:
        raise SystemExit(f"pass {index} has an unusable label {pass_.label!r}")
    return safe


def build(
    *,
    strategy: Strategy,
    pool: Sequence[Mapping[str, Any]],
    pool_path: Path,
    model: str,
    seeds: Sequence[int],
    params: Mapping[str, Any],
    stats: Any,
    record_builder: RecordBuilder,
    experiment: str,
    command: str | None = None,
) -> BuildResult:
    """Run a strategy over every seed and return the rows plus the manifest.

    One call produces one model's whole file: every seed, every pass the
    strategy yields for that seed, in that order. The model is handed to the
    strategy (through the context) rather than used to slice the output, so a
    strategy that reads this model's statistics and one that ignores the model
    entirely both go through this same path.
    """
    if not pool:
        raise SystemExit(f"the item pool at {pool_path} is empty")
    if not seeds:
        raise SystemExit("no seeds: --seeds N or --seed-list is required")

    # What this build is filed under: the registered name, plus the settings of
    # a strategy whose parameters are the condition (`predicted-calibration-...`).
    # The *registered* name still keys the RNG below, deliberately: two settings
    # of one strategy then draw the same stream, so their partitions differ only
    # by the setting and not also by a re-shuffle.
    name = strategy.instance_name(params)

    rows: list[dict] = []
    pass_manifest: list[dict] = []

    for seed in seeds:
        # Keyed on the strategy and the seed, *not* on the model: the batching is
        # the same for every model at a given seed, so a difference between two
        # models' scores is a difference between the models, not between draws.
        key = f"{strategy.name}:{seed}"
        rng = random.Random(key)
        ctx = BatchingContext(items=list(pool), model=model, seed=seed, rng=rng,
                              params=dict(params), stats=stats, key=key)
        passes = strategy(ctx)
        if not passes:
            raise SystemExit(
                f"strategy {strategy.name!r} yielded no passes at seed {seed}; "
                "a strategy must yield at least one partition of the pool"
            )

        labels_seen: set[str] = set()
        for pass_index, pass_ in enumerate(passes):
            where = f"strategy {strategy.name!r}, seed {seed}, pass {pass_index}"
            validate_pass(pass_, len(pool), where=where)
            label = _pass_label(pass_, pass_index)
            if label in labels_seen:
                raise SystemExit(
                    f"{where}: two passes of one seed share the label {label!r}; "
                    "labels distinguish rows within a seed and must be unique"
                )
            labels_seen.add(label)

            sizes = Counter(len(group) for group in pass_.groups)
            stamp = dict(pass_.stamp or {})
            stamp_clash = sorted(set(stamp) & set(IDENTITY_FIELDS))
            if stamp_clash:
                raise SystemExit(
                    f"{where}: the pass stamps engine-owned field(s) "
                    f"{stamp_clash}; identity belongs to the engine so that a "
                    "row's id, stage and k cannot disagree with its batching"
                )
            masks = _masks(pass_, where=where) if pass_.actual is not None else None
            for batch_index, group in enumerate(pass_.groups):
                items = [pool[i] for i in group]
                record = record_builder(items)
                clash = sorted(set(record) & set(IDENTITY_FIELDS))
                if clash:
                    raise SystemExit(
                        f"the record builder returned engine-owned field(s) "
                        f"{clash}; identity is stamped by the engine so a row's "
                        "id, stage and k cannot disagree with its batching"
                    )
                over_record = sorted(set(record) & set(stamp))
                if over_record:
                    raise SystemExit(
                        f"{where}: the pass stamps field(s) {over_record} that the "
                        "record builder already wrote. A stamp says what the pass "
                        "is; it must not overwrite what the composition is."
                    )
                rows.append({
                    "id": f"eval-{name}-s{seed}-{label}-{batch_index:05d}",
                    "stage": STAGE,
                    # The actual size of this composition. Never the nominal
                    # one: under a dynamic strategy they differ on most rows,
                    # and every downstream figure keys on the real k.
                    "k": len(items),
                    "strategy": name,
                    **record,
                    # Which pass this row belongs to. The model answers every
                    # seed in the file, so these are what a run's logs are
                    # grouped by afterwards.
                    "batch_seed": seed,
                    "batch_pass": label,
                    "batch_index": batch_index,
                    "nominal_k": pass_.nominal_k,
                    "batching_model": model,
                    "rng_seed": key,
                    # Which positions count, for a masked pass only -- so an
                    # unmasked file carries no mask fields at all.
                    **({"component_actual": masks[batch_index],
                        "n_actual": sum(masks[batch_index])}
                       if masks is not None else {}),
                    # What this pass is, beyond its label: the fold a
                    # fold-restricted pass belongs to, and anything else a
                    # strategy wants its own rows to carry. Last, so a typo in a
                    # stamp cannot shadow identity -- the two clash checks above
                    # have already refused that case outright.
                    **stamp,
                })

            pass_manifest.append({
                "seed": seed,
                "label": label,
                "nominal_k": pass_.nominal_k,
                # How many items this pass partitions, and whether that was the
                # whole pool. A fold-restricted pass covers only its fold's
                # items, and nothing downstream should have to infer that from
                # the row count.
                "n_items_covered": (len(pool) if pass_.covers is None
                                    else len(set(int(i) for i in pass_.covers))),
                "covers_whole_pool": pass_.covers is None,
                "stamp": dict(pass_.stamp or {}),
                "n_batches": len(pass_.groups),
                "k_min": min(sizes),
                "k_max": max(sizes),
                "k_counts": {str(k): sizes[k] for k in sorted(sizes)},
                "rng_key": key,
                "info": dict(pass_.info),
                **({"n_actual_slots": sum(sum(m) for m in masks),
                    "n_nonactual_slots": sum(len(m) - sum(m) for m in masks)}
                   if masks is not None else {}),
            })

    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        duplicate = next(i for i, c in Counter(ids).items() if c > 1)
        raise SystemExit(
            f"duplicate row id {duplicate!r}. Ids must be unique across the whole "
            "file: the task layer stashes a document's raw response under its id, "
            "so a repeat crosses two documents' extractions."
        )

    manifest = {
        "experiment": experiment,
        "stage": STAGE,
        # The name this build is filed under, settings and all -- what appears
        # on every row -- and beside it the registered name `--strategy` took, so
        # a reader can group every setting of one strategy without parsing the
        # first.
        "strategy": name,
        "strategy_base": strategy.name,
        "strategy_description": strategy.description,
        "uses_model": strategy.uses_model,
        "uses_stats": strategy.uses_stats,
        "model": model,
        "per_model_rng": False,
        "seeds": list(seeds),
        "params": {k: v for k, v in params.items()},
        "pool": str(pool_path),
        "pool_sha256": sha256_of(pool_path),
        "n_pool_items": len(pool),
        "n_rows": len(rows),
        "n_passes": len(pass_manifest),
        "passes": pass_manifest,
        "statistics_read": dict(getattr(stats, "read_tables", {})),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": command,
    }
    return BuildResult(rows=rows, manifest=manifest)

