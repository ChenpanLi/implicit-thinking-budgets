"""The StrategyQA pool: where it comes from, and what counts as usable.

The counterpart of :mod:`llm.core.shortmath.gsm8k` and of
:mod:`llm.core.mcq.gpqa`, for **StrategyQA**: yes/no questions whose
answer needs an implicit multi-step chain ("Was the ship that recovered Apollo 13
named after a World War II battle?" -- find the ship, find the battle, compare).
The 2290 annotated questions are published by ``ChilleD/StrategyQA`` as a
1603/687 train/test split; the pool is the **test** split by default, and
``--split train`` gives the other 1603.

Why this upload rather than the original: StrategyQA's own test set ships
without answers (it is a leaderboard split), so every public evaluation of the
benchmark scores the annotated questions. This upload is those questions, split
in two, with the annotation intact.

Three properties of this pool shape everything downstream.

**The answer is one of two words, so chance is 50%.** That is the single most
important thing to keep in mind reading a StrategyQA composition curve, and it is
not a defect: it is what makes the domain a useful contrast with the others. A
model guessing at random scores 0.5 per term and ``0.5 ** k`` on a whole k-term
prompt, so the all-correct accuracy of a *guessing* model already collapses with
k -- the curve this experiment is about. The per-term accuracy (``acc_terms``)
is the metric to read against that floor, because it is flat in k at 0.5 while
``acc``'s is not.

**``True`` counts as ``yes``.** The accepted set of a yes-question is
``["yes", "true"]`` and of a no-question ``["no", "false"]``, written into the
pool where it is visible and reviewable rather than hidden in a normalisation
rule (the stance :mod:`llm.core.oneword.normalization` sets out). A model that
answers a yes/no question with ``True`` has answered it; scoring that wrong would
measure our instruction's reach rather than the model.

**The supporting facts are not part of the question.** Each row carries
``facts``, the decomposition the annotators wrote, and ``term``/``description``,
the Wikipedia entity the question was seeded from. None of them is rendered: they
would hand the model the reasoning the question is asking for. They are not
copied onto the record either, so nothing downstream can leak them into a prompt
by accident.

There is one category, ``strategy``, and it is written onto every record for the
reason :mod:`llm.core.shortmath.gsm8k` gives for its own constant:
``component_categories`` is a field the shared composition builder fills and the
batching engine's category-balanced draws read, so leaving it out would need a
special case in every reader. What follows for analysis is that a StrategyQA
composition has nothing to balance *across*, and the ``random`` draw is
pool-uniform.
"""
from __future__ import annotations

from typing import Any

DATASET = "ChilleD/StrategyQA"
CONFIG = None
SPLIT = "test"

QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
RECORD_ID_FIELD = "qid"

#: The only category StrategyQA has, written onto every record.
CATEGORY = "strategy"
BROAD_CATEGORY = "commonsense"

#: The canonical answers, and the variants accepted for each. The canonical one
#: comes first: it is what ``reference_answer`` shows and what every table and
#: figure prints. See the module docstring for why ``true``/``false`` are here.
YES: tuple[str, ...] = ("yes", "true")
NO: tuple[str, ...] = ("no", "false")

#: What the raw ``answer`` column may hold. The parquet upload gives Python
#: booleans; the JSON files in the same repo give the strings ``True``/``False``,
#: and a future re-upload could give ``yes``/``no``. All three are read, and
#: anything else is a rejection rather than a guess.
_TRUE = {"true", "yes", "1"}
_FALSE = {"false", "no", "0"}

# Deliberately NOT copied onto a record: the annotators' decomposition
# (``facts``) and the seed entity (``term``, ``description``).


def answer_of(row: dict[str, Any]) -> str | None:
    """The canonical answer of one raw row -- ``"yes"``, ``"no"`` or ``None``."""
    value = row.get(ANSWER_FIELD)
    if isinstance(value, bool):
        return YES[0] if value else NO[0]
    text = str(value).strip().casefold()
    if text in _TRUE:
        return YES[0]
    if text in _FALSE:
        return NO[0]
    return None


def accepted_of(row: dict[str, Any]) -> list[str] | None:
    """Every answer that counts as right for this row, canonical one first."""
    answer = answer_of(row)
    if answer is None:
        return None
    return list(YES if answer == YES[0] else NO)


def is_clean(row: dict[str, Any]) -> bool:
    """Is this row usable as a question?

    Deliberately shallow -- StrategyQA ships no per-row quality annotation to
    defer to, so the only things checkable here are that a question was asked and
    that it was answered yes or no. A row failing this is dropped and *counted*
    by the generator rather than silently skipped.
    """
    if not str(row.get(QUESTION_FIELD, "")).strip():
        return False
    return answer_of(row) is not None


def reject_reason(row: dict[str, Any]) -> str:
    """Why a row is not usable, for the manifest's rejection counts."""
    if not str(row.get(QUESTION_FIELD, "")).strip():
        return "empty_question"
    if not str(row.get(ANSWER_FIELD, "")).strip():
        return "empty_answer"
    return "answer_is_not_yes_or_no"
