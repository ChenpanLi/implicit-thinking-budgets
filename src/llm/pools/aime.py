"""AIME 2024 + 2025: 60 problems merged from two Hub uploads.

The two years come from different datasets, each pinned to its own revision. After
the merge the problems are put in contest order (year, paper, problem number), and
that order assigns the ``q-NNNN`` ids. The contest (e.g. ``2024 AIME I``) is the
category and the problem number the level.
"""
from __future__ import annotations

from llm.core.shortmath.aime import (
    BROAD_CATEGORY, QUESTION_FIELD, SOURCES, contest_of, final_answer, is_clean,
    problem_number_of, reject_reason, sort_key, year_of,
)
from llm.core.shortmath.rendering import clean_problem, render_block

from ._hub import load_rows

REVISIONS = {
    "HuggingFaceH4/aime_2024": "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
    "yentinglin/aime_2025": "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
}


def build_record(index: int, row: dict) -> dict:
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
        "category": contest_of(row),
        "broad_category": BROAD_CATEGORY,
        "level": problem_number_of(row),
        "year": year_of(row),
        "source_dataset": row["_source_dataset"],
        "source_config": row["_source_config"],
        "source_split": row["_source_split"],
        "source_index": row["_source_index"],
        "source_url": str(row.get("url") or ""),
    }


def build() -> tuple[list[dict], dict]:
    clean: list[dict] = []
    rejected: dict[str, int] = {}
    sources: dict[str, dict] = {}
    for dataset, config, split in SOURCES:
        raw = load_rows(dataset, config, split, REVISIONS[dataset])
        kept = 0
        for index, row in enumerate(raw):
            if is_clean(row):
                clean.append({**row, "_source_dataset": dataset, "_source_config": config,
                              "_source_split": split, "_source_index": index})
                kept += 1
            else:
                reason = reject_reason(row)
                rejected[reason] = rejected.get(reason, 0) + 1
        sources[dataset] = {"config": config, "split": split,
                            "revision": REVISIONS[dataset],
                            "n_rows": len(raw), "n_clean": kept}
    clean.sort(key=sort_key)
    records = [build_record(position, row) for position, row in enumerate(clean)]

    categories: dict[str, int] = {}
    levels: dict[str, int] = {}
    for record in records:
        categories[record["category"]] = categories.get(record["category"], 0) + 1
        levels[str(record["level"])] = levels.get(str(record["level"]), 0) + 1
    manifest = {
        "pool_size": len(records),
        "year": None,
        "seed": 42,
        "n_requested": None,
        "n_clean": len(clean),
        "sources": sources,
        "categories": categories,
        "levels": levels,
        "rejected": rejected,
    }
    return records, manifest
