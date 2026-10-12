"""What the Hub pool builders share: loading a split and counting rejects."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def load_rows(dataset: str, config: str | None, split: str,
              revision: str) -> list[dict[str, Any]]:
    """Every raw row of one split at a pinned revision."""
    import datasets as hf_datasets

    rows = hf_datasets.load_dataset(dataset, config, split=split, revision=revision)
    return [dict(row) for row in rows]


def clean_rows(rows: list[dict[str, Any]], is_clean: Callable[[dict], bool],
               reject_reason: Callable[[dict], str],
               ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The rows that make well-posed items, plus a count of why the others do not."""
    kept: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for row in rows:
        if is_clean(row):
            kept.append(row)
        else:
            reason = reject_reason(row)
            rejected[reason] = rejected.get(reason, 0) + 1
    return kept, rejected
