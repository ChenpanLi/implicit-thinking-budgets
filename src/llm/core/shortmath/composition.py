"""The domain fields of one short-answer maths composition.

Two builders need these fields and must not disagree about them: the k-sweep
builder (:mod:`llm.compose`, which draws compositions from a solved pool) and
the batching builder (:mod:`llm.batches`, which partitions the whole pool). They differ in *which* problems end up together and
in nothing else, so which problems end up together is all that is duplicated
between them.

Identity -- the row's id, its stage, its k, the strategy that made it -- is
deliberately *not* built here. Each caller owns its own id scheme
(``comp-k3-0007`` for a drawn composition, ``eval-random-s0-k10-00042`` for a
batched one), and a shared builder that guessed at one would be the single place
the two schemes could silently collide.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .rendering import render_block

# What joins the k canonical answers into the single ``reference_answer`` string.
# Not a comma, which is what the multiple-choice builder uses: a MATH-500 answer
# may itself be ``(6,31,-1)``, and a reader splitting that string on commas
# would be wrong on the first tuple-valued answer it met.
ANSWER_JOIN = " | "


def accepted_of(problem: Mapping[str, Any]) -> list[str]:
    """One problem's accepted answers, however the pool record spells them.

    ``accepted_answers`` is a list of lists at k=1 (one entry, because an atomic
    document is a one-problem composition); ``correct_answers`` is the flat form.
    Both are read so a hand-written document is usable.
    """
    accepted = problem.get("accepted_answers")
    if accepted:
        return [str(a).strip() for a in accepted[0]]
    if problem.get("correct_answers"):
        return [str(a).strip() for a in problem["correct_answers"]]
    if problem.get("reference_answers"):
        return [str(problem["reference_answers"][0]).strip()]
    raise KeyError(f"problem {problem.get('id')!r} carries no accepted answer")


def composition_fields(chosen: Sequence[Mapping[str, Any]]) -> dict:
    """The problem, the provenance and the answer key of one composition.

    ``questions_block`` is the only problem-bearing field -- the prompt is built
    from it alone -- and it is rendered here by the one renderer every other
    reader of these experiments uses, so the dataset, the prompt and the span
    locator cannot disagree about what a problem looks like.

    ``component_answers`` holds every *accepted* answer of each problem and
    ``reference_answers`` the canonical one. The two are kept apart rather than
    collapsed because they answer different questions: grading needs the set,
    while every join, table and figure needs one string per position.
    """
    if not chosen:
        raise ValueError("a composition must hold at least one problem")
    questions = [problem["question"] for problem in chosen]
    accepted = [accepted_of(problem) for problem in chosen]
    canonical = [variants[0] for variants in accepted]
    block, _ = render_block(questions)
    return {
        "component_ids": [problem["id"] for problem in chosen],
        "component_questions": questions,
        "component_categories": [problem.get("category") for problem in chosen],
        "component_broad_categories": [problem.get("broad_category") for problem in chosen],
        # The difficulty band, where the pool has one (MATH-500 levels 1-5;
        # GSM8K ships none and writes null). Parallel to `term_correct`, so a
        # wrong position can be attributed to a level without joining back.
        "component_levels": [problem.get("level") for problem in chosen],
        "component_answers": accepted,
        "component_reference_answers": canonical,
        "category_counts": dict(Counter(problem.get("category") for problem in chosen)),
        # The prompt is built from this field alone.
        "questions_block": block,
        # The grader's ground truth, in both the forms anything downstream
        # wants: the list for comparing, the string for `doc_to_target` and for
        # identifying the problem in a trace.
        "reference_answers": canonical,
        "reference_answer": ANSWER_JOIN.join(canonical),
    }
