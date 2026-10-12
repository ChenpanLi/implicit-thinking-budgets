r"""The MATH-500 pool: its subjects, its levels, and what counts as usable.

The counterpart of :mod:`llm.core.mcq.gpqa` and :mod:`llm.core.oneword.strategyqa`.
``HuggingFaceH4/MATH-500`` is the 500-problem evaluation subset of Hendrycks
MATH used by the PRM800K line of work: each row carries the problem, a worked
solution, the final ``answer`` already extracted as LaTeX, a ``subject``, a
``level`` from 1 to 5 and a ``unique_id``.

Three properties of the pool shape everything downstream:

* **The answer is LaTeX, not a number.** 311 of the 500 are bare integers; the
  rest are fractions, surds, tuples, intervals, unions, column vectors and ten
  ``\text{...}`` words. That is what :mod:`llm.core.shortmath.latex` exists for.
* **The subject is the category, the level is kept beside it.** ``category`` is
  the subject (Algebra, Precalculus, ...), which is what a category-balanced
  draw balances on and what every per-category figure groups by;
  ``broad_category`` is the constant ``math``, because MATH-500 is entirely
  mathematics and the field is one the shared composition builder fills anyway.
  ``level`` is carried through as its own field and lands on every composition
  as ``component_levels``, so accuracy can be split by difficulty without
  joining back to the pool.
* **42 problems carry an ``[asy]`` figure**, whose Asymptote source is part of
  the problem text. They are kept: a text model is expected to read the source,
  every published MATH-500 number includes them, and dropping them would make
  this pool incomparable with the literature. ``has_figure`` is recorded on the
  record so a later analysis can split on it, and the renderer collapses the
  blank lines inside the figure so that a blank line still means "next problem"
  (see :mod:`llm.core.shortmath.rendering`).
"""
from __future__ import annotations

from typing import Any

DATASET = "HuggingFaceH4/MATH-500"
CONFIG = None
SPLIT = "test"

QUESTION_FIELD = "problem"
ANSWER_FIELD = "answer"
SUBJECT_FIELD = "subject"
LEVEL_FIELD = "level"
RECORD_ID_FIELD = "unique_id"

BROAD_CATEGORY = "math"

#: The seven subjects, in the order figures lay them out: the three that make up
#: two thirds of the pool first, then the rest alphabetically. Written down so
#: that a figure with seven bars puts them in a stable order rather than in
#: whatever order a dict happened to iterate.
CATEGORY_ORDER: tuple[str, ...] = (
    "Algebra",
    "Intermediate Algebra",
    "Prealgebra",
    "Counting & Probability",
    "Geometry",
    "Number Theory",
    "Precalculus",
)

#: The difficulty bands the dataset ships, 1 (easiest) to 5.
LEVELS: tuple[int, ...] = (1, 2, 3, 4, 5)

FIGURE_MARKER = "[asy]"


def subject_of(row: dict[str, Any]) -> str:
    return str(row.get(SUBJECT_FIELD) or "Uncategorized").strip()


def level_of(row: dict[str, Any]) -> int | None:
    """The problem's difficulty band, or ``None`` if the row does not say.

    ``None`` rather than a default: a level nobody recorded is not level 1, and
    a per-level split that silently filled one in would report a band that does
    not exist.
    """
    try:
        level = int(row.get(LEVEL_FIELD))
    except (TypeError, ValueError):
        return None
    return level if level in LEVELS else None


def has_figure(row: dict[str, Any]) -> bool:
    return FIGURE_MARKER in str(row.get(QUESTION_FIELD, ""))


def is_clean(row: dict[str, Any]) -> bool:
    """Is this row usable as a problem?

    Shallow on purpose -- MATH-500 is a curated subset and ships no per-row
    quality annotation, so the only things checkable here are that a problem was
    asked and that a non-empty answer was written down for it. A row failing
    this is dropped and *counted* by the generator rather than silently skipped.
    """
    if not str(row.get(QUESTION_FIELD, "")).strip():
        return False
    return bool(str(row.get(ANSWER_FIELD, "")).strip())


def reject_reason(row: dict[str, Any]) -> str:
    """Why a row is not usable, for the manifest's rejection counts."""
    if not str(row.get(QUESTION_FIELD, "")).strip():
        return "empty_problem"
    return "empty_answer"
