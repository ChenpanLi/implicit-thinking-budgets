"""Grading k written answers against each question's accepted set.

The counterpart of :func:`llm.core.mcq.grading.check_answers` and of
:func:`llm.core.math.verification.check_antiderivative`, and deliberately the
same *shape*: it returns a :class:`CheckResult` whose ordered ``stages`` list
ends, by construction, with the stage that decided. That is what lets the trace
writer and every analysis loader read a one-word run, a multiple-choice run and
an integrals run with the same code.

The verdict types are reused rather than redefined for the same reason. There is
no TIMEOUT and no CHECKER_ERROR here -- comparing normalised strings cannot hang
-- but a reader that switches on ``Verdict`` keeps working.
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from llm.core.math.verification import CheckResult, StageRecord, Verdict

from .answers import parse_answer_lines
from .normalization import answer_matches


def _accepted_sets(reference: Sequence[Any]) -> list[list[str]]:
    """Normalise the reference to one list of accepted variants per question."""
    out: list[list[str]] = []
    for entry in reference:
        if isinstance(entry, (list, tuple)):
            variants = [str(a).strip() for a in entry if str(a).strip()]
        else:
            variants = [str(entry).strip()] if str(entry).strip() else []
        if not variants:
            raise ValueError("every question must carry at least one accepted answer")
        out.append(variants)
    return out


def check_answers(candidate: str | None, reference: Sequence[Any]) -> CheckResult:
    """Compare a response's extracted answer block against the reference answers.

    ``reference`` is the composition's answer key in prompt order: either one
    string per question, or one list of accepted variants per question. Four
    questions in the pool genuinely have two right answers, so the list form is
    not an edge case.

    A parse failure is UNPARSEABLE and never reaches the comparison, for the
    same reason a checker timeout is not WRONG: it says something about the
    answer's form, not about whether the model knew the material.
    """
    accepted = _accepted_sets(reference)
    if not accepted:
        raise ValueError("reference must name at least one answer")
    canonical = [variants[0] for variants in accepted]

    started = time.perf_counter()
    parsed = parse_answer_lines(candidate, expected=len(accepted))
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

    matches = [answer_matches(given, variants)
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
            "term_correct": term_correct,
            "n_correct": n_correct,
            "n_terms": len(accepted),
            "total_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
    )
