"""GSM8K: the whole test split, 1319 problems.

The answer is read from the ``####`` marker, thousands separators removed, and the
problem text is stored exactly as the prompt renders it.
"""
from __future__ import annotations

from llm.core.shortmath.gsm8k import (
    BROAD_CATEGORY, CATEGORY, CONFIG, DATASET, QUESTION_FIELD, SPLIT,
    final_answer, is_clean, reject_reason,
)
from llm.core.shortmath.rendering import clean_problem, render_block

from ._hub import clean_rows, load_rows

REVISION = "740312add88f781978c0658806c59bc2815b9866"


def build_record(index: int, row: dict, source_index: int) -> dict:
    question = clean_problem(row[QUESTION_FIELD])
    answer = final_answer(row)
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
        "category": CATEGORY,
        "broad_category": BROAD_CATEGORY,
        "level": None,
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "source_index": source_index,
    }


def build() -> tuple[list[dict], dict]:
    rows = load_rows(DATASET, CONFIG, SPLIT, REVISION)
    clean, rejected = clean_rows(rows, is_clean, reject_reason)
    chosen = [(i, row) for i, row in enumerate(rows) if is_clean(row)]
    records = [build_record(position, row, source_index)
               for position, (source_index, row) in enumerate(chosen)]
    manifest = {
        "pool_size": len(records),
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "source_revision": REVISION,
        "seed": 42,
        "n_requested": None,
        "n_clean": len(clean),
        "categories": {CATEGORY: len(records)},
        "rejected": rejected,
    }
    return records, manifest
