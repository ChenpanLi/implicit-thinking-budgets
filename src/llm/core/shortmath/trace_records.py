"""Assembling one trace record for a graded short-answer maths attempt.

The sibling of :func:`llm.core.oneword.trace_records.build_oneword_record` and
of :func:`llm.core.trace.build_record`, which hardcodes the integrals document
schema. Rather than parameterise either, this module builds the record and
hands it to the same :class:`llm.core.trace.TraceWriter`.

The fields that matter downstream, and that nothing else records:

``term_correct``
    One boolean per problem, in prompt order. Comparing a written answer
    against its accepted set is exact and free, so the per-position figures
    read the trace directly.
``match_levels``
    Parallel to ``term_correct``: the rung of the comparison ladder that decided
    each position (``exact``, ``normalized``, ``numeric``, ``symbolic``, ...).
    A run whose correct answers are mostly ``symbolic`` is one whose grader is
    doing a lot of work, and that has to be visible without re-grading.
``component_categories`` / ``component_levels``
    Also parallel, so a wrong position can be attributed to a subject or a
    difficulty band without joining back to the dataset.
"""
from __future__ import annotations

from typing import Any

from llm.core.trace import copy_batching_identity, describe_raw


def build_shortmath_record(
    *,
    run_id: str,
    task: str,
    doc_id: int | str,
    filter_name: str,
    trial: int,
    doc: dict[str, Any],
    raw_text: str | None,
    result: Any,  # CheckResult, kept loose to match trace.build_record
    include_raw: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One trace record joining lm-eval's identity keys to our stages."""
    detail = getattr(result, "detail", None) or {}

    record: dict[str, Any] = {
        "run_id": run_id,
        "task": task,
        "doc_id": doc_id,
        "filter": filter_name,
        "trial": trial,
        "problem_id": doc.get("id"),
        "reference_answer": doc.get("reference_answer"),
        "reference_answers": doc.get("reference_answers"),
        "raw": describe_raw(raw_text, include_raw),
    }
    if (k := doc.get("k")) is not None:
        record["k"] = k
    if (components := doc.get("component_ids")) is not None:
        record["component_ids"] = components
    if (categories := doc.get("component_categories")) is not None:
        record["component_categories"] = categories
    if (levels := doc.get("component_levels")) is not None:
        record["component_levels"] = levels
    if (category := doc.get("category")) is not None:
        record["category"] = category
    if (level := doc.get("level")) is not None:
        record["level"] = level
    copy_batching_identity(record, doc)

    # Hoisted out of `detail` onto the record itself: every per-position figure
    # reads these, and burying them one level down would make every loader reach
    # into a dict whose shape depends on the verdict.
    record["candidate_answers"] = detail.get("candidate_answers", [])
    record["term_correct"] = detail.get("term_correct", [])
    record["match_levels"] = detail.get("match_levels", [])
    record["n_correct"] = detail.get("n_correct", 0)
    record["n_terms"] = detail.get("n_terms", doc.get("k", 1))

    record.update(result.to_dict())
    if extra:
        record.update(extra)
    return record
