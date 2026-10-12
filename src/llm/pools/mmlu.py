"""MMLU-Redux 2.0: 19 subjects, 74 questions each (the smallest subject's clean count).

Only rows the Redux annotators marked ``ok`` are kept, and each subject is drawn
down to the same size with its own seeded RNG. Options keep their original order.
"""
from __future__ import annotations

import random

from llm.core.mcq.answers import LETTERS
from llm.core.mcq.mmlu import (
    CLEAN_ERROR_TYPE, DATASET, broad_category_of, is_clean, subcategory_of,
)
from llm.core.mcq.rendering import render_block

from ._hub import load_rows

REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"
SEED = 42

# The subjects of the pool, in the order they are written out.
SUBJECTS = [
    "logical_fallacies", "college_chemistry", "human_sexuality", "professional_law",
    "business_ethics", "human_aging", "formal_logic", "philosophy", "machine_learning",
    "miscellaneous", "high_school_european_history", "public_relations",
    "conceptual_physics", "professional_accounting", "astronomy", "security_studies",
    "high_school_biology", "international_law", "marketing",
]


def clean_subject(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """The rows annotated ``ok`` and well formed, plus why the others were dropped."""
    keep = (CLEAN_ERROR_TYPE,)
    kept, rejected = [], {}
    for row in rows:
        if is_clean(row, keep):
            kept.append(row)
            continue
        reason = row.get("error_type")
        reason = "malformed" if reason in keep else str(reason)
        rejected[reason] = rejected.get(reason, 0) + 1
    return kept, rejected


def build_record(index: int, row: dict, subject: str) -> dict:
    choices = list(row["choices"])
    answer = int(row["answer"])
    letter = LETTERS[answer]
    block, _ = render_block([row["question"]], [choices])
    return {
        "id": f"q-{index:04d}",
        "stage": "atomic",
        "k": 1,
        "question": row["question"],
        "choices": choices,
        "n_choices": len(choices),
        "answer_index": answer,
        "answer_letter": letter,
        "reference_answers": [letter],
        "reference_answer": letter,
        "questions_block": block,
        "category": subject,
        "subcategory": subcategory_of(subject),
        "broad_category": broad_category_of(subject),
        "source_dataset": DATASET,
        "source_config": subject,
        "source_url": row.get("source"),
    }


def build() -> tuple[list[dict], dict]:
    clean: dict[str, list[dict]] = {}
    rejected: dict[str, int] = {}
    for subject in SUBJECTS:
        kept, reasons = clean_subject(load_rows(DATASET, subject, "test", REVISION))
        clean[subject] = kept
        for reason, count in reasons.items():
            rejected[reason] = rejected.get(reason, 0) + count
    available = {subject: len(rows) for subject, rows in clean.items()}
    per_category = min(available.values())

    records: list[dict] = []
    for subject in SUBJECTS:
        rng = random.Random(f"{SEED}:{subject}")
        indices = sorted(rng.sample(range(len(clean[subject])), per_category))
        for i in indices:
            records.append(build_record(len(records), clean[subject][i], subject))

    manifest = {
        "pool_size": len(records),
        "source_dataset": DATASET,
        "source_revision": REVISION,
        "seed": SEED,
        "keep_error_types": [CLEAN_ERROR_TYPE],
        "shuffle_choices": False,
        "per_category": per_category,
        "categories": {subject: per_category for subject in SUBJECTS},
        "available_per_category": available,
        "category_order": list(SUBJECTS),
        "broad_categories": {subject: broad_category_of(subject) for subject in SUBJECTS},
        "rejected": rejected,
    }
    return records, manifest
