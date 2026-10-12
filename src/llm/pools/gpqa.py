"""GPQA-Diamond: the GPQA-main pool restricted to the 198 diamond questions.

The pool is built from ``gpqa_main`` and then filtered to the Record IDs of
``gpqa_diamond``, so every question keeps the ``q-NNNN`` id and the option order it
has in the full main pool. Options are permuted per question with an RNG keyed on
its Record ID (GPQA stores the correct answer first). Three main rows whose four
answers are not distinct are dropped; two of them are diamond questions, which
leaves 196.

``Idavidrein/gpqa`` is gated: the Hub account in use must have accepted its terms
and ``HF_TOKEN`` must be set.
"""
from __future__ import annotations

import random

from llm.core.mcq.answers import LETTERS
from llm.core.mcq.gpqa import (
    BROAD_CATEGORY, CONFIG, DATASET, DOMAINS, QUESTION_FIELD, RECORD_ID_FIELD, SPLIT,
    domain_of, is_clean, options_of, reject_reason, subdomain_of,
)
from llm.core.mcq.rendering import render_block

from ._hub import clean_rows, load_rows

REVISION = "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
RESTRICT_TO = "gpqa_diamond"
SEED = 42


def build_record(index: int, row: dict) -> dict:
    options = options_of(row)
    order = list(range(len(options)))
    random.Random(f"{SEED}:{row.get(RECORD_ID_FIELD)}:choices").shuffle(order)
    options = [options[i] for i in order]
    answer = order.index(0)  # the raw file puts the correct answer first
    letter = LETTERS[answer]
    question = str(row[QUESTION_FIELD]).strip()
    block, _ = render_block([question], [options])
    return {
        "id": f"q-{index:04d}",
        "stage": "atomic",
        "k": 1,
        "question": question,
        "choices": options,
        "n_choices": len(options),
        "answer_index": answer,
        "answer_letter": letter,
        "reference_answers": [letter],
        "reference_answer": letter,
        "questions_block": block,
        "category": domain_of(row),
        "subcategory": subdomain_of(row),
        "broad_category": BROAD_CATEGORY,
        "source_dataset": DATASET,
        "source_record_id": row.get(RECORD_ID_FIELD),
    }


def group_by_domain(rows: list[dict]) -> dict[str, list[dict]]:
    """The rows by high-level domain, the domains of ``DOMAINS`` first."""
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(domain_of(row), []).append(row)
    ordered = {name: grouped.pop(name) for name in DOMAINS if name in grouped}
    ordered.update({name: grouped[name] for name in sorted(grouped)})
    return ordered


def build() -> tuple[list[dict], dict]:
    clean, rejected = clean_rows(load_rows(DATASET, CONFIG, SPLIT, REVISION),
                                 is_clean, reject_reason)
    by_domain = group_by_domain(clean)
    records: list[dict] = []
    for domain_rows in by_domain.values():
        for row in domain_rows:
            records.append(build_record(len(records), row))

    keep = {str(row.get(RECORD_ID_FIELD))
            for row in load_rows(DATASET, RESTRICT_TO, SPLIT, REVISION)}
    records = [r for r in records if r["source_record_id"] in keep]

    subdomains: dict[str, int] = {}
    for record in records:
        subdomains[record["subcategory"]] = subdomains.get(record["subcategory"], 0) + 1
    manifest = {
        "pool_size": len(records),
        "source_dataset": DATASET,
        "source_config": CONFIG,
        "restricted_to_config": RESTRICT_TO,
        "source_split": SPLIT,
        "source_revision": REVISION,
        "seed": SEED,
        "shuffle_choices": True,
        "per_domain": None,
        "restricted_to": RESTRICT_TO,
        "restricted_to_size": len(keep),
        "domains": {name: sum(r["category"] == name for r in records) for name in by_domain},
        "available_per_domain": {name: len(rows) for name, rows in by_domain.items()},
        "domain_order": list(by_domain),
        "subdomains": subdomains,
        "rejected": rejected,
    }
    return records, manifest
