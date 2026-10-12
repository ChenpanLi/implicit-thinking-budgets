"""MATH-500: all 500 problems, grouped by subject in a fixed order.

Problems carrying ``[asy]`` figure source are kept and marked with ``has_figure``.
"""
from __future__ import annotations

from llm.core.shortmath.math500 import (
    ANSWER_FIELD, BROAD_CATEGORY, CATEGORY_ORDER, CONFIG, DATASET, LEVELS,
    QUESTION_FIELD, RECORD_ID_FIELD, SPLIT, has_figure, is_clean, level_of,
    reject_reason, subject_of,
)
from llm.core.shortmath.rendering import clean_problem, render_block

from ._hub import clean_rows, load_rows

REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"


def build_record(index: int, row: dict) -> dict:
    question = clean_problem(row[QUESTION_FIELD])
    answer = str(row[ANSWER_FIELD]).strip()
    block, _ = render_block([question])
    return {
        "id": f"q-{index:04d}",
        "stage": "atomic",
        "k": 1,
        "question": question,
        "correct_answers": [answer],
        "accepted_answers": [[answer]],
        "reference_answers": [answer],
        "reference_answer": answer,
        "n_accepted": 1,
        "questions_block": block,
        "category": subject_of(row),
        "broad_category": BROAD_CATEGORY,
        "level": level_of(row),
        "has_figure": has_figure(row),
        "source_dataset": DATASET,
        "source_split": SPLIT,
        "source_record_id": row.get(RECORD_ID_FIELD),
    }


def group_by_subject(rows: list[dict]) -> dict[str, list[dict]]:
    """The rows by subject, the subjects of ``CATEGORY_ORDER`` first."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(subject_of(row), []).append(row)
    ordered = {name: grouped.pop(name) for name in CATEGORY_ORDER if name in grouped}
    ordered.update({name: grouped[name] for name in sorted(grouped)})
    return ordered


def build() -> tuple[list[dict], dict]:
    rows = load_rows(DATASET, CONFIG, SPLIT, REVISION)
    clean, rejected = clean_rows(rows, is_clean, reject_reason)
    by_subject = group_by_subject(clean)
    records: list[dict] = []
    for subject_rows in by_subject.values():
        for row in subject_rows:
            records.append(build_record(len(records), row))
    available = {name: len(subject_rows) for name, subject_rows in by_subject.items()}
    manifest = {
        "pool_size": len(records),
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "source_revision": REVISION,
        "seed": 42,
        "per_subject": None,
        "levels_kept": list(LEVELS),
        "n_dropped_by_level": 0,
        "no_figures": False,
        "n_dropped_figures": 0,
        "categories": available,
        "available_per_subject": available,
        "category_order": list(by_subject),
        "levels": {str(level): sum(r["level"] == level for r in records) for level in LEVELS},
        "n_with_figure": sum(1 for r in records if r["has_figure"]),
        "rejected": rejected,
    }
    return records, manifest
