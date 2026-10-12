"""GPQA-main's own taxonomy and row check -- the counterpart of :mod:`llm.core.mcq.mmlu`.

Small, because GPQA is: one file, 448 graduate-level multiple-choice questions,
three high-level domains and sixteen subdomains, and no per-subject config to
enumerate. What it does share with the MMLU module is the *shape*: a
``category`` / ``subcategory`` / ``broad_category`` triple every downstream
figure and draw already keys on, and an ``is_clean`` predicate that decides
which raw rows may enter the pool at all.

The one structural difference from MMLU-Redux, and the reason ``is_clean`` here
looks at the options rather than at an annotation: GPQA ships no per-row quality
label. What it ships instead is four free-text answers -- one correct, three
distractors -- which are rendered into options by this repo rather than by the
dataset. So the defect worth rejecting is a *structural* one: a row whose four
answers are not four distinct non-empty strings has no single correct letter
once rendered, and a model that picks the duplicate of the correct answer would
be scored wrong for giving a right answer. Three rows of ``gpqa_main`` are like
that today.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# The Hub dataset and the config within it. `gpqa_main` is the 448-question
# main set; the pool builder (:mod:`llm.pools.gpqa`) restricts it to the
# Record IDs of `gpqa_diamond` (198).
DATASET = "Idavidrein/gpqa"
CONFIG = "gpqa_main"
# GPQA ships one split and calls it `train`, although it is an evaluation set.
SPLIT = "train"

# The raw columns this repo reads. Everything else in the 78-column file is
# provenance about the human validation, which the pool does not carry.
QUESTION_FIELD = "Question"
CORRECT_FIELD = "Correct Answer"
INCORRECT_FIELDS = ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
DOMAIN_FIELD = "High-level domain"
SUBDOMAIN_FIELD = "Subdomain"
RECORD_ID_FIELD = "Record ID"

# The three high-level domains, in the order a figure should show them. Fixed
# here rather than derived from whatever the file happens to hold: a per-build
# order would make two builds' category axes incomparable.
DOMAINS = ("Physics", "Chemistry", "Biology")

# Every question in GPQA main is a natural-science question, so the coarsest
# level of the MMLU taxonomy collapses to one value here. It is still written
# onto every record: `component_broad_categories` is a field the shared
# composition builder fills, and a tree that left it out would need a special
# case in every loader that reads it.
BROAD_CATEGORY = "STEM"


def options_of(row: Mapping[str, Any]) -> list[str]:
    """One row's four answers, correct first, exactly as the file spells them.

    Correct *first* is the raw file's own order and is never the order the
    prompt shows: the pool builder permutes them per row with a seeded RNG
    before rendering. Anything reading this function has to permute too, which
    is why the correct answer's position is stated here rather than left to be
    discovered.
    """
    return [str(row.get(CORRECT_FIELD) or "").strip(),
            *(str(row.get(name) or "").strip() for name in INCORRECT_FIELDS)]


def is_clean(row: Mapping[str, Any]) -> bool:
    """Whether a raw row can become a well-posed four-option question.

    Three things are required, and each failure produces a question that runs,
    scores and plots normally while measuring nothing:

    * a non-empty question,
    * four non-empty answers,
    * four *distinct* answers -- otherwise two letters are correct and the
      grader, which compares one letter, marks one of them wrong.
    """
    if not str(row.get(QUESTION_FIELD) or "").strip():
        return False
    options = options_of(row)
    if len(options) != 4 or any(not option for option in options):
        return False
    return len({option for option in options}) == 4


def reject_reason(row: Mapping[str, Any]) -> str:
    """Why :func:`is_clean` refused a row, for the manifest's rejection counts."""
    if not str(row.get(QUESTION_FIELD) or "").strip():
        return "empty_question"
    options = options_of(row)
    if any(not option for option in options):
        return "empty_option"
    if len(set(options)) != 4:
        return "duplicate_options"
    return "ok"


def domain_of(row: Mapping[str, Any]) -> str:
    """The row's high-level domain, which is this tree's ``category``."""
    return str(row.get(DOMAIN_FIELD) or "uncategorized").strip() or "uncategorized"


def subdomain_of(row: Mapping[str, Any]) -> str:
    """The row's subdomain, which is this tree's ``subcategory``."""
    return str(row.get(SUBDOMAIN_FIELD) or "unknown").strip() or "unknown"
