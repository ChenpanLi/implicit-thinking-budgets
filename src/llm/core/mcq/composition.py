"""The domain fields of one multiple-choice composition.

Shared by the composed-set builder (:mod:`llm.compose`, which draws
compositions from a solved pool) and the batching builder (:mod:`llm.batches`,
which partitions the whole pool), so the two cannot disagree about what a
composition record holds. They differ in *which*
questions end up together and in nothing else.

Identity -- the row's id, its stage, its k, the strategy that made it -- is
deliberately not built here; see the note in
:mod:`llm.core.oneword.composition`.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .answers import LETTERS
from .rendering import render_block


def answer_letter_of(question: Mapping[str, Any]) -> str:
    """One question's correct option letter, however the record spells it."""
    letter = question.get("answer_letter")
    if letter:
        return str(letter)
    if question.get("answer_index") is not None:
        return LETTERS[int(question["answer_index"])]
    raise KeyError(f"question {question.get('id')!r} carries no answer letter")


def composition_fields(chosen: Sequence[Mapping[str, Any]]) -> dict:
    """The problem, the provenance and the answer key of one composition.

    ``questions_block`` is the only problem-bearing field -- the prompt is built
    from it alone -- and it is rendered here by the one renderer every other
    reader of this experiment uses.
    """
    if not chosen:
        raise ValueError("a composition must hold at least one question")
    questions = [q["question"] for q in chosen]
    choices = [list(q["choices"]) for q in chosen]
    answers = [answer_letter_of(q) for q in chosen]
    block, _ = render_block(questions, choices)

    widths = sorted({len(c) for c in choices})
    if len(widths) != 1:
        # A composition whose questions offered different numbers of options
        # would have no single legal letter range, so the prompt could not state
        # one and the grader could not check against one. Fatal here rather than
        # resolved by picking a width, which is what an arbitrary `set.pop()`
        # would quietly do.
        raise ValueError(
            f"a composition mixes {widths} option counts; every question in one "
            "prompt must offer the same number of options, so that the prompt can "
            "name a single legal letter range"
        )

    return {
        "component_ids": [q["id"] for q in chosen],
        "component_questions": questions,
        "component_choices": choices,
        "component_categories": [q.get("category") for q in chosen],
        "component_broad_categories": [q.get("broad_category") for q in chosen],
        "component_answer_letters": answers,
        "category_counts": dict(Counter(q.get("category") for q in chosen)),
        # The prompt is built from this field alone.
        "questions_block": block,
        # The grader's ground truth, in both the forms anything downstream
        # wants: the list for comparing, the string for `doc_to_target` and for
        # identifying the problem in a trace.
        "reference_answers": answers,
        "reference_answer": ", ".join(answers),
        "n_choices": widths[0],
    }
