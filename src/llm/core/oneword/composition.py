"""The domain fields of one one-word QA composition.

Two builders need these fields and must not disagree about them: the k-sweep
builder (:mod:`llm.compose`, which draws compositions from a solved pool) and
the batching builder (:mod:`llm.batches`, which partitions the whole pool). They differ in
*which* questions end up together and in nothing else, so which questions end up
together is all that is duplicated between them.

Identity -- the row's id, its stage, its k, the strategy that made it -- is
deliberately *not* built here. Each caller owns its own id scheme (``comp-k3-0007``
for a drawn composition, ``eval-random-s0-k10-00042`` for a batched one), and a
shared builder that guessed at one would be the single place the two schemes
could silently collide.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .rendering import render_block

# What joins the k canonical answers into the single `reference_answer` string.
# Not a comma, which is what the multiple-choice builder uses: 39 of this pool's
# answers contain a space, and a reader splitting on commas would be right until
# the first `Washington, D.C.`-shaped answer is added to the pool.
ANSWER_JOIN = " | "


def accepted_of(question: Mapping[str, Any]) -> list[str]:
    """One question's accepted answers, however the pool record spells them.

    ``accepted_answers`` is a list of lists at k=1 (one entry, because an atomic
    document is a one-question composition); ``correct_answers`` is the flat form
    the base file uses. Both are read so a hand-written document is usable.
    """
    accepted = question.get("accepted_answers")
    if accepted:
        return [str(a).strip() for a in accepted[0]]
    if question.get("correct_answers"):
        return [str(a).strip() for a in question["correct_answers"]]
    if question.get("reference_answers"):
        return [str(question["reference_answers"][0]).strip()]
    raise KeyError(f"question {question.get('id')!r} carries no accepted answer")


def composition_fields(chosen: Sequence[Mapping[str, Any]]) -> dict:
    """The problem, the provenance and the answer key of one composition.

    ``questions_block`` is the only problem-bearing field -- the prompt is built
    from it alone -- and it is rendered here by the one renderer every other
    reader of this experiment uses, so the dataset, the prompt and the span
    locator cannot disagree about what a question looks like.

    ``component_answers`` holds every *accepted* answer of each question and
    ``reference_answers`` the canonical one. The two are kept apart rather than
    collapsed because they answer different questions: grading needs the set,
    while every join, table and figure needs one string per position.
    """
    if not chosen:
        raise ValueError("a composition must hold at least one question")
    questions = [q["question"] for q in chosen]
    accepted = [accepted_of(q) for q in chosen]
    canonical = [variants[0] for variants in accepted]
    block, _ = render_block(questions)
    return {
        "component_ids": [q["id"] for q in chosen],
        "component_questions": questions,
        "component_categories": [q.get("category") for q in chosen],
        "component_broad_categories": [q.get("broad_category") for q in chosen],
        "component_answers": accepted,
        "component_reference_answers": canonical,
        "category_counts": dict(Counter(q.get("category") for q in chosen)),
        # The prompt is built from this field alone.
        "questions_block": block,
        # The grader's ground truth, in both the forms anything downstream
        # wants: the list for comparing, the string for `doc_to_target` and for
        # identifying the problem in a trace.
        "reference_answers": canonical,
        "reference_answer": ANSWER_JOIN.join(canonical),
    }
