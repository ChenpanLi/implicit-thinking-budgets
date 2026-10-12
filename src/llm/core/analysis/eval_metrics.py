"""Per-seed accuracy and output tokens of one run, read off its ``trace.jsonl``.

A run of a batching stage holds every batching seed of the batch file, i.e. several
complete passes over the same pool, so every count here is kept per (seed, k) and
the caller averages over seeds (:func:`llm.core.analysis.fragility.policy_metrics`).

Two conventions decide what is counted:

* **A generation is counted once for cost.** lm-eval calls ``process_results``
  once per filter ensemble on the *same* response, so every generation is in
  ``trace.jsonl`` twice (``boxed_last``/``final_answer`` for integrals,
  ``lenient``/``explicit`` elsewhere). Token counting keys on
  ``(task, doc_id, trial)`` and ignores the repeat; accuracy is read from one
  chosen ensemble only, for the same reason.
* **Only the output half of a request is counted.** Prompt tokens fall with k
  (one instruction is shared over k items) while output tokens rise with it, so a
  combined total would mix the two; the token cost of a batching is what the
  model generated.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from llm.core.serving.token_stash import normalize_usage


@dataclass
class Counters:
    """The raw counts of one (seed, k) cell. Every field is a count, never a rate."""

    items: int = 0               # summed batch sizes of the generations
    completion_tokens: int = 0
    graded: int = 0              # graded prompts (a batch is one)
    correct: int = 0             # prompts whose every term was right
    terms: int = 0               # items graded individually, where the tree does that
    terms_correct: int = 0

    def add(self, other: "Counters") -> None:
        for name, value in vars(other).items():
            setattr(self, name, getattr(self, name) + value)


@dataclass(frozen=True)
class MetricBlock:
    """A whole run (``k`` None) or one k of it: counts per seed and summed over seeds."""

    k: int | None
    per_seed: dict[Any, Counters]
    totals: Counters


@dataclass(frozen=True)
class RunSummary:
    overall: MetricBlock | None
    per_k: tuple[MetricBlock, ...]
    note: str | None = None


def iter_trace(path: Path) -> Iterator[dict[str, Any]]:
    """Stream ``trace.jsonl``, skipping a half-written last line."""
    with Path(path).open("rb") as handle:
        for raw in handle:
            if not raw.endswith(b"\n"):
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def aggregate_trace(path: Path, filter_name: str | None = None) -> dict[tuple, Counters]:
    """Fold a trace into ``{(task, batch_seed, k): Counters}``.

    ``filter_name`` chooses the extraction ensemble the accuracy is read from (default:
    the first one the file mentions). Token counts do not depend on it: a generation
    costs what it cost however its answer was later extracted.
    """
    cells: dict[tuple[str, Any, int], Counters] = defaultdict(Counters)
    seen: set[tuple[str, str, Any]] = set()
    filters: list[str] = []

    for record in iter_trace(path):
        task = record.get("task", "unknown")
        ensemble = record.get("filter") or "unknown"
        if ensemble not in filters:
            filters.append(ensemble)
        chosen = filter_name or filters[0]

        k = record.get("k")
        k = int(k) if isinstance(k, (int, float)) else 1
        cell = cells[(task, record.get("batch_seed"), k)]

        key = (task, str(record.get("doc_id")), record.get("trial"))
        if key not in seen:
            seen.add(key)
            cell.items += k
            cell.completion_tokens += normalize_usage(record.get("token_usage"))["completion"] or 0

        if ensemble != chosen:
            continue
        cell.graded += 1
        cell.correct += record.get("verdict") == "CORRECT"
        n_terms = record.get("n_terms")
        if isinstance(n_terms, (int, float)) and n_terms:
            cell.terms += int(n_terms)
            cell.terms_correct += int(record.get("n_correct") or 0)
    return dict(cells)


def _block(k: int | None, per_seed: dict[Any, Counters]) -> MetricBlock:
    seeds = sorted(per_seed, key=lambda s: (s is None, s))
    totals = Counters()
    for seed in seeds:
        totals.add(per_seed[seed])
    return MetricBlock(k=k, per_seed={seed: per_seed[seed] for seed in seeds}, totals=totals)


def analyse_run(run_dir: Path, filter_name: str | None = None) -> RunSummary:
    """The run's counts per seed, overall (pooled over k) and per k."""
    trace = Path(run_dir) / "trace.jsonl"
    if not trace.is_file():
        return RunSummary(overall=None, per_k=(), note="no trace.jsonl yet -- the run is "
                          "still starting, or it died before grading")
    by_seed: dict[Any, Counters] = defaultdict(Counters)
    by_k_seed: dict[int, dict[Any, Counters]] = defaultdict(lambda: defaultdict(Counters))
    for (_task, seed, k), counters in aggregate_trace(trace, filter_name).items():
        by_seed[seed].add(counters)
        by_k_seed[k][seed].add(counters)
    if not by_seed:
        return RunSummary(overall=None, per_k=(), note="trace.jsonl is empty")
    return RunSummary(overall=_block(None, dict(by_seed)),
                      per_k=tuple(_block(k, dict(by_k_seed[k])) for k in sorted(by_k_seed)))
