"""Assembling one trace record for a graded multiple-choice attempt.

The sibling of :func:`llm.core.trace.build_record`, which hardcodes the
integrals document schema (``integrand``). Rather than parameterise that
function, this module builds the MCQ record and hands it to the same
:class:`llm.core.trace.TraceWriter`.

The fields that matter downstream, and that nothing else records:

``term_correct``
    One boolean per question, in prompt order. The per-question verdict falls
    out of the comparison for free, so the per-position figures read the trace
    directly.
``component_categories``
    Parallel to ``term_correct``, so a wrong position can be attributed to a
    subject without joining back to the dataset.
"""
from __future__ import annotations

from typing import Any

from llm.core.trace import copy_batching_identity, describe_raw


def build_mcq_record(
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
    if (category := doc.get("category")) is not None:
        record["category"] = category
    copy_batching_identity(record, doc)

    # The experiment's condition, carried by the run rather than left to be
    # re-derived from the dataset: `pattern_class` and `letter_pattern` for an
    # answer-letter-pattern set, and the `base_set_index` its
    # matched conditions share. A record whose dataset has moved still says what
    # condition it was, which is the whole reason these are hoisted.
    for name in ("pattern_class", "letter_pattern", "base_set_index"):
        if (value := doc.get(name)) is not None:
            record[name] = value

    # Hoisted out of `detail` onto the record itself: every per-position figure
    # reads these, and burying them one level down would make every loader
    # reach into a dict whose shape depends on the verdict.
    record["candidate_letters"] = detail.get("candidate_letters", [])
    record["term_correct"] = detail.get("term_correct", [])
    record["n_correct"] = detail.get("n_correct", 0)
    record["n_terms"] = detail.get("n_terms", doc.get("k", 1))

    record.update(result.to_dict())
    if extra:
        record.update(extra)
    return record
