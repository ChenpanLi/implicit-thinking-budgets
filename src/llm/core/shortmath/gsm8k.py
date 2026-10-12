"""The GSM8K pool: where it comes from, and what counts as usable.

The counterpart of :mod:`llm.core.mcq.gpqa` and :mod:`llm.core.oneword.strategyqa`,
and smaller than either, because GSM8K needs almost no editorial judgement: its
test split is 1319 grade-school word problems, each with a worked solution whose
last line is ``#### <integer>``. There are no options to render, no annotation
column to filter on and no per-row quality label.

What *is* here is the part every downstream script has to agree on: how to read
the answer out of the solution field, the one predicate that says whether a row
is usable, and the category story.

**There is one category, and that is a fact about GSM8K rather than a gap.**
The dataset ships no subject, topic or difficulty label -- every problem is
arithmetic word-problem reasoning -- so ``category`` and ``broad_category`` are
the constant ``arithmetic`` on every record. The constant is still written out,
because ``component_categories`` is a field the shared composition builder fills
and the batching engine's category-balanced draws read; leaving it out would
need a special case in every reader. What follows for analysis is that a GSM8K
composition has nothing to balance *across*, so the ``random`` draw here is
pool-uniform rather than category-first, unlike GPQA's.
"""
from __future__ import annotations

import re
from typing import Any

DATASET = "openai/gsm8k"
CONFIG = "main"
SPLIT = "test"

QUESTION_FIELD = "question"
SOLUTION_FIELD = "answer"

#: The only category GSM8K has, written onto every record. See the module docstring.
CATEGORY = "arithmetic"
BROAD_CATEGORY = "math"

# The final-answer marker GSM8K's solutions end with. The integer after it is
# the answer; the `<<12*2=24>>` calculator annotations earlier in the solution
# are not, which is why this is anchored to the marker rather than looking for
# the last number in the text.
_MARKER = re.compile(r"####\s*(.+?)\s*$", re.S)
# The answer as the file writes it: an integer, occasionally with a thousands
# comma (14 of the 1319). No test answer carries a decimal point, and one that
# did would still be read correctly here.
_NUMBER = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?")


def final_answer(row: dict[str, Any]) -> str | None:
    """The canonical answer of one raw row, or ``None`` if it has none.

    The thousands commas are removed, because the answer is stored as a number
    and every comparison downstream reads it as one; leaving them in would make
    the *reference* the thing that needs lenient parsing.
    """
    solution = str(row.get(SOLUTION_FIELD, ""))
    marker = _MARKER.search(solution)
    if marker is None:
        return None
    match = _NUMBER.fullmatch(marker.group(1).strip())
    if match is None:
        return None
    return match.group(0).replace(",", "")


def is_clean(row: dict[str, Any]) -> bool:
    """Is this row usable as a problem?

    Deliberately shallow -- GSM8K ships no quality annotation to defer to, so
    the only things checkable here are that a question was asked and that a
    single numeric final answer was written down for it. A row failing this is
    dropped and *counted* by the generator rather than silently skipped.
    """
    if not str(row.get(QUESTION_FIELD, "")).strip():
        return False
    return final_answer(row) is not None


def reject_reason(row: dict[str, Any]) -> str:
    """Why a row is not usable, for the manifest's rejection counts."""
    if not str(row.get(QUESTION_FIELD, "")).strip():
        return "empty_question"
    solution = str(row.get(SOLUTION_FIELD, ""))
    if "####" not in solution:
        return "no_final_answer_marker"
    return "final_answer_is_not_a_number"
