"""Grading a list of multiple-choice answers against the reference letters.

The counterpart of :func:`llm.core.math.verification.check_antiderivative`, and
deliberately the same *shape*: it returns a :class:`CheckResult` with an ordered
``stages`` list whose last entry is, by construction, the stage that decided.
That is what lets the trace writer and every analysis loader read an MCQ run
and an integrals run with the same code.

The verdict types are reused rather than redefined for the same reason. There is
no TIMEOUT and no CHECKER_ERROR here -- comparing letters cannot hang and cannot
be subtly wrong -- but a reader that switches on ``Verdict`` keeps working.
"""
from __future__ import annotations

import time
from collections.abc import Sequence

from llm.core.math.verification import CheckResult, StageRecord, Verdict

from .answers import parse_answer_list


def check_answers(
    candidate: str | None,
    reference: Sequence[str],
    *,
    n_choices: int = 4,
) -> CheckResult:
    """Compare a response's extracted answer against the reference letters.

    ``reference`` is the composition's answer, one letter per question in prompt
    order. A parse failure is UNPARSEABLE and never reaches the comparison, for
    the same reason a checker timeout is not WRONG: it says something about the
    answer's form, not about whether the model knew the material.
    """
    reference = [str(letter).strip().upper() for letter in reference]
    if not reference:
        raise ValueError("reference must name at least one answer")

    started = time.perf_counter()
    parsed = parse_answer_list(candidate, expected=len(reference), n_choices=n_choices)
    parse_stage = StageRecord(
        "candidate_answer_parse",
        status="ok" if parsed.ok else "fail",
        ms=(time.perf_counter() - started) * 1000.0,
        data=parsed.to_data(),
    )

    if not parsed.ok:
        return CheckResult(
            Verdict.UNPARSEABLE,
            "candidate_answer_parse",
            [parse_stage],
            {
                "reason": parsed.status.value,
                "message": parsed.detail,
                "candidate_letters": list(parsed.letters),
                "reference_letters": list(reference),
                "n_terms": len(reference),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
            },
        )

    term_correct = [given == want for given, want in zip(parsed.letters, reference)]
    n_correct = sum(term_correct)
    all_correct = n_correct == len(reference)

    compare_stage = StageRecord(
        "compare",
        status="ok" if all_correct else "fail",
        ms=(time.perf_counter() - started) * 1000.0 - parse_stage.ms,
        data={
            "candidate_letters": list(parsed.letters),
            "reference_letters": list(reference),
            "term_correct": term_correct,
            "n_correct": n_correct,
            "n_terms": len(reference),
            # The positions a reader wants first; 1-based to match the prompt's
            # own numbering rather than the list index.
            "wrong_positions": [i + 1 for i, ok in enumerate(term_correct) if not ok],
        },
    )

    return CheckResult(
        Verdict.CORRECT if all_correct else Verdict.WRONG,
        "compare",
        [parse_stage, compare_stage],
        {
            "reason": "all_correct" if all_correct else "letter_mismatch",
            "candidate_letters": list(parsed.letters),
            "reference_letters": list(reference),
            "term_correct": term_correct,
            "n_correct": n_correct,
            "n_terms": len(reference),
            # Graded on the first k of a longer list. Carried up here, and not
            # left on the parse stage alone, so that a run can count these
            # attempts as a metric -- reading one is a decision this module
            # made, and a decision nobody can count is one nobody can check.
            "truncated": parsed.truncated,
            "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
    )
