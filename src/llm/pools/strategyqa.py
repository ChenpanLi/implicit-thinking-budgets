"""StrategyQA: the 687-question ``test`` split of ChilleD/StrategyQA (yes/no answers).

StrategyQA's own test set ships without answers; this upload's ``test`` split is a
held-out part of the annotated questions.
"""
from __future__ import annotations

from llm.core.oneword.rendering import render_block
from llm.core.oneword.strategyqa import (
    BROAD_CATEGORY, CATEGORY, CONFIG, DATASET, QUESTION_FIELD, RECORD_ID_FIELD, SPLIT,
    accepted_of, is_clean, reject_reason,
)

from ._hub import clean_rows, load_rows

REVISION = "705562638fe1d8ca6bb98c66fc8f94d45fda8c83"


def build_record(index: int, row: dict, source_index: int) -> dict:
    question = " ".join(str(row[QUESTION_FIELD]).split())
    answers = accepted_of(row) or []
    block, _ = render_block([question])
    return {
        "id": f"q-{index:04d}",
        "stage": "atomic",
        "k": 1,
        "question": question,
        "correct_answers": answers,
        "accepted_answers": [answers],
        "reference_answers": [answers[0]],
        "reference_answer": answers[0],
        "n_accepted": len(answers),
        "questions_block": block,
        "category": CATEGORY,
        "broad_category": BROAD_CATEGORY,
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "source_split": SPLIT,
        "source_index": source_index,
        "source_id": str(row.get(RECORD_ID_FIELD) or ""),
    }


def build() -> tuple[list[dict], dict]:
    rows = load_rows(DATASET, CONFIG, SPLIT, REVISION)
    clean, rejected = clean_rows(rows, is_clean, reject_reason)
    chosen = [(i, row) for i, row in enumerate(rows) if is_clean(row)]
    records = [build_record(position, row, source_index)
               for position, (source_index, row) in enumerate(chosen)]
    balance = {"yes": 0, "no": 0}
    for record in records:
        balance[record["reference_answer"]] += 1
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
        "answer_balance": balance,
        "rejected": rejected,
    }
    return records, manifest
