"""Grading k written answers against each problem's accepted set.

The counterpart of :func:`llm.core.oneword.grading.check_answers` and of
:func:`llm.core.math.verification.check_antiderivative`, and deliberately the
same *shape*: it returns a :class:`CheckResult` whose ordered ``stages`` list
ends, by construction, with the stage that decided. That is what lets the trace
writer and every analysis loader read a GSM8K run, a MATH-500 run, a one-word
run and an integrals run with the same code.

The one argument this grader has that the others do not is ``comparator``: the
datasets sharing this package agree about everything except what it means for
a written answer to be the reference one (see
:mod:`llm.core.shortmath.comparison`). Passing it in, rather than branching on a
domain name in here, is what keeps "GSM8K compares numbers" a fact stated once,
in the task module that knows it.

The verdict types are reused rather than redefined. There is no TIMEOUT here --
a symbolic comparison that times out is reported as a mismatch by the
comparator, because the levels above it have already failed -- but a reader that
switches on ``Verdict`` keeps working.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from llm.core.math.verification import CheckResult, StageRecord, Verdict

from .answers import parse_answer_lines
from .comparison import Comparator


def _accepted_sets(reference: Sequence[Any]) -> list[list[str]]:
    """Normalise the reference to one list of accepted variants per problem."""
    out: list[list[str]] = []
    for entry in reference:
        if isinstance(entry, (list, tuple)):
            variants = [str(a).strip() for a in entry if str(a).strip()]
        else:
            variants = [str(entry).strip()] if str(entry).strip() else []
        if not variants:
            raise ValueError("every problem must carry at least one accepted answer")
        out.append(variants)
    return out


def check_answers(candidate: str | None, reference: Sequence[Any], *,
                  comparator: Comparator,
                  readable: Callable[[str], bool] | None = None) -> CheckResult:
    """Compare a response's extracted answer block against the reference answers.

    ``reference`` is the composition's answer key in prompt order: either one
    string per problem, or one list of accepted variants per problem.

    A parse failure is UNPARSEABLE and never reaches the comparison, for the
    same reason a checker timeout is not WRONG: it says something about the
    answer's form, not about whether the model solved the problem.

    ``readable`` is the second fact a task module states about its own domain,
    next to ``comparator``: whether a written line could be an answer here at
    all. It is passed straight to
    :func:`llm.core.shortmath.answers.parse_answer_lines`, which needs it to
    tell a glued answer index (``1.18``) from a decimal. A tree that does not
    pass one has its block read as written.
    """
    accepted = _accepted_sets(reference)
    if not accepted:
        raise ValueError("reference must name at least one answer")
    canonical = [variants[0] for variants in accepted]

    started = time.perf_counter()
    parsed = parse_answer_lines(candidate, expected=len(accepted), readable=readable)
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
                "candidate_answers": list(parsed.answers),
                "reference_answers": canonical,
                "accepted_answers": accepted,
                "term_correct": [],
                "n_correct": 0,
                "n_terms": len(accepted),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
            },
        )

    matches = [comparator(given, variants)
               for given, variants in zip(parsed.answers, accepted)]
    term_correct = [match.matched for match in matches]
    n_correct = sum(term_correct)
    all_correct = n_correct == len(accepted)

    compare_stage = StageRecord(
        "compare",
        status="ok" if all_correct else "fail",
        ms=(time.perf_counter() - started) * 1000.0 - parse_stage.ms,
        data={
            "candidate_answers": list(parsed.answers),
            "normalized_candidates": [match.normalized for match in matches],
            "reference_answers": canonical,
            "matched_variants": [match.variant for match in matches],
            # Which rung of the ladder decided each position. A run whose
            # correct answers are mostly `symbolic` is a run whose grader is
            # doing a lot of work, and that is worth being able to see.
            "match_levels": [match.level for match in matches],
            "term_correct": term_correct,
            "n_correct": n_correct,
            "n_terms": len(accepted),
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
            "reason": "all_correct" if all_correct else "answer_mismatch",
            "candidate_answers": list(parsed.answers),
            "normalized_candidates": [match.normalized for match in matches],
            "reference_answers": canonical,
            "accepted_answers": accepted,
            "matched_variants": [match.variant for match in matches],
            "match_levels": [match.level for match in matches],
            "term_correct": term_correct,
            "n_correct": n_correct,
            "n_terms": len(accepted),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
    )
