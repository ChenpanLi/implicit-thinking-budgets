"""The MMLU subject taxonomy, and which MMLU-Redux rows are usable.

Two things the pool builder (:mod:`llm.pools.mmlu`) needs and that nothing else
should define differently: which broad group a subject belongs to, and whether a
raw Hub row passed review.

The subject -> subcategory -> category mapping is Hendrycks et al.'s own
(`categories.py` in the MMLU release), reproduced verbatim. It is recorded on
every problem as `broad_category` so a figure can group 57 subjects into
something a legend can hold, without the *dataset* having to choose a
granularity up front.
"""
from __future__ import annotations

from typing import Any

DATASET = "edinburgh-dawg/mmlu-redux-2.0"

# The annotation MMLU-Redux gives a question whose original MMLU labelling
# survived manual review. Everything else is a question with a known defect --
# ambiguous wording, no correct option, several correct options, or a wrong
# ground truth -- and including any of it would mean "the model got it wrong"
# could mean the benchmark instead of the model.
CLEAN_ERROR_TYPE = "ok"

SUBCATEGORIES: dict[str, str] = {
    "abstract_algebra": "math",
    "anatomy": "health",
    "astronomy": "physics",
    "business_ethics": "business",
    "clinical_knowledge": "health",
    "college_biology": "biology",
    "college_chemistry": "chemistry",
    "college_computer_science": "computer science",
    "college_mathematics": "math",
    "college_medicine": "health",
    "college_physics": "physics",
    "computer_security": "computer science",
    "conceptual_physics": "physics",
    "econometrics": "economics",
    "electrical_engineering": "engineering",
    "elementary_mathematics": "math",
    "formal_logic": "philosophy",
    "global_facts": "other",
    "high_school_biology": "biology",
    "high_school_chemistry": "chemistry",
    "high_school_computer_science": "computer science",
    "high_school_european_history": "history",
    "high_school_geography": "geography",
    "high_school_government_and_politics": "politics",
    "high_school_macroeconomics": "economics",
    "high_school_mathematics": "math",
    "high_school_microeconomics": "economics",
    "high_school_physics": "physics",
    "high_school_psychology": "psychology",
    "high_school_statistics": "math",
    "high_school_us_history": "history",
    "high_school_world_history": "history",
    "human_aging": "health",
    "human_sexuality": "culture",
    "international_law": "law",
    "jurisprudence": "law",
    "logical_fallacies": "philosophy",
    "machine_learning": "computer science",
    "management": "business",
    "marketing": "business",
    "medical_genetics": "health",
    "miscellaneous": "other",
    "moral_disputes": "philosophy",
    "moral_scenarios": "philosophy",
    "nutrition": "health",
    "philosophy": "philosophy",
    "prehistory": "history",
    "professional_accounting": "business",
    "professional_law": "law",
    "professional_medicine": "health",
    "professional_psychology": "psychology",
    "public_relations": "politics",
    "security_studies": "politics",
    "sociology": "culture",
    "us_foreign_policy": "politics",
    "virology": "health",
    "world_religions": "philosophy",
}

CATEGORIES: dict[str, tuple[str, ...]] = {
    "STEM": ("physics", "chemistry", "biology", "computer science", "math", "engineering"),
    "humanities": ("history", "philosophy", "law"),
    "social sciences": ("politics", "culture", "economics", "geography", "psychology"),
    "other": ("other", "business", "health"),
}

_BROAD_OF_SUBCATEGORY = {
    sub: broad for broad, subs in CATEGORIES.items() for sub in subs
}

SUBJECTS: tuple[str, ...] = tuple(sorted(SUBCATEGORIES))


def subcategory_of(subject: str) -> str:
    """The fine group (`math`, `health`, ...) a subject belongs to."""
    try:
        return SUBCATEGORIES[subject]
    except KeyError:
        raise KeyError(
            f"unknown MMLU subject {subject!r}; known subjects are "
            f"{', '.join(SUBJECTS[:5])} ... ({len(SUBJECTS)} in all)"
        ) from None


def broad_category_of(subject: str) -> str:
    """The broad group (STEM / humanities / social sciences / other)."""
    return _BROAD_OF_SUBCATEGORY[subcategory_of(subject)]


def is_clean(row: dict[str, Any], keep: tuple[str, ...] = (CLEAN_ERROR_TYPE,)) -> bool:
    """Whether a raw MMLU-Redux row passed manual review.

    Structural sanity is checked here too rather than downstream: a row whose
    answer index does not address one of its own options is unusable whatever
    its annotation says. Every row of MMLU-Redux 2.0 currently passes that
    check, which is exactly why a silent regression would be invisible.
    """
    if row.get("error_type") not in keep:
        return False
    choices = row.get("choices") or []
    answer = row.get("answer")
    return isinstance(answer, int) and 0 <= answer < len(choices) and len(choices) >= 2
