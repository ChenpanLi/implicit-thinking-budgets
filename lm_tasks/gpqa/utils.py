"""lm-eval hooks for the GPQA-main multiple-choice tasks.

Every ``!function`` referenced by the YAML in this directory resolves here.

Division of labour, chosen so each artifact stays authoritative for one thing,
and deliberately identical to the MMLU and integrals task layers:

* **Filters** extract candidate answers and return *strings only*, so lm-eval's
  own ``filtered_resps`` shows exactly what was extracted. Running two
  extraction strategies as separate ensembles scores them side by side in a
  single pass (``acc,lenient`` vs ``acc,explicit``), which keeps the price of
  the lenient reading visible in the same units as the score.
* **process_results** grades, and emits the stage-by-stage trace that nothing
  else records -- including the per-question ``term_correct`` list. A letter
  comparison is exact and free, so the per-question analyses read that straight
  out of the trace rather than paying an extractor model for it.

Everything below the prompt matches the MMLU task layer:
the grading, the stash, the ensemble recovery and the metric list are shared
code in ``llm.core.mcq``, and this module is the thin hook layer over them. The
two task layers are kept as separate files rather than parameterised into one
for the reason stated across this repository: a change made for one domain must
not be able to move a byte of another domain's prompt.
"""
from __future__ import annotations

import functools
import os
from typing import Any

import datasets

from llm.core.math.verification import CheckResult, StageRecord, Verdict
from llm.core.mcq.answers import AnswerListStatus
from llm.core.mcq.extraction import extract_answer
from llm.core.mcq.grading import check_answers
from llm.core.mcq.trace_records import build_mcq_record
from llm.core.parser.extraction import ExtractResult, ExtractionStatus
from llm.core.serving import token_stash
from llm.core.trace import TraceWriter

# Sentinels surfaced in lm-eval's filtered_resps so a failed extraction is
# visible in its artifacts too, not only in ours.
_SENTINEL = {
    ExtractionStatus.EMPTY: "[[EMPTY_RESPONSE]]",
    ExtractionStatus.NO_ANSWER: "[[NO_ANSWER]]",
}

# process_results receives only (doc, results) -- neither the raw generations
# nor the name of the filter ensemble it is scoring. Filters see both, so they
# stash what the grader needs, keyed by the dataset's stable id. Raw text is
# stored once per doc; extraction outcomes are stored per ensemble, and the
# active ensemble is recovered by call order, validated against the values
# lm-eval hands back.
_STASH: dict[str, dict[str, Any]] = {}


# The prompt. One template for the atomic, compositional and evaluation stages
# alike: an atomic problem *is* a one-question composition, and giving it its
# own wording would mean the k=1 point of a curve was measured on a different
# instruction from the rest of it.
#
# The answer format is a plain `Answer:` line rather than a `\boxed{}`, for the
# reason the MMLU prompt records: a LaTeX box is an unreasonable demand for a
# capital letter and would measure the model's house style rather than its knowledge.
# `llm.core.mcq.extraction` accepts the variations models write anyway --
# `**Answer**:`, `ANSWER:`, a box -- so the format line is a nudge rather than a
# contract the score depends on.
#
# The closing re-check line is the other half of that. GPQA questions are long
# and hard, and a reasoning model that has argued for thousands of tokens
# routinely loses the answer format it was given at the start; the resulting
# responses are perfectly readable and score UNPARSEABLE. Asking it to re-read
# its own answer line costs one sentence.
_PROMPT = """
Answer the following multiple-choice question{{ "s" if n_questions > 1 }}.

{{ questions_block }}

For each question, give only the letter of the single best option.
{% if n_questions > 1 %}
End your response with a line of the following form, and write nothing after it: a comma-separated list of {{ n_questions }} letters, one per question, in the order the questions were asked.
Answer: {{ example }}
{% else %}
End your response with a line of the following form, and write nothing after it.
Answer: {{ example }}
{% endif %}
Before you finish, re-read that final line and check that it holds exactly {{ n_questions }} letter{% if n_questions > 1 %}s, in the order the questions were asked{% endif %}, with nothing written after {{ "them" if n_questions > 1 else "it" }}.
"""

ENV_DATA = "LMEVAL_GPQA_DATA"
ENV_RUN_ID = "LMEVAL_GPQA_RUN_ID"

# The section-marker prompt (`--section-markers`): `_PROMPT` word for word, with
# the questions labelled Q01, Q02, ... instead of 1., 2., ... and one sentence after
# the block asking the model to name a question by its label whenever it works on
# it, so the thinking can be cut per question at every label mention
# (llm.core.analysis.question_sections.parse_label_sections). Built from
# `_PROMPT` by inserting that one sentence, so the two cannot drift apart, and
# kept a separate template so that adding it cannot move a byte of the prompt
# every ordinary run is measured on. Version 3 is the MMLU task layer's
# wording; a run records it as `section_markers_prompt` in run.json.
MARKER_PROMPT_VERSION = 3
_MARKER_SENTENCE = ("While you think, refer to each question by its label ({{ labels }}) "
                    "whenever you start or return to working on it.")
_BLOCK_LINE = "{{ questions_block }}\n"
assert _PROMPT.count(_BLOCK_LINE) == 1
_MARKER_PROMPT = _PROMPT.replace(_BLOCK_LINE, _BLOCK_LINE + "\n" + _MARKER_SENTENCE + "\n", 1)
# Selects how questions are labelled. `"markers"` labels them Q01, Q02, ... and
# takes `_MARKER_PROMPT` (`--section-markers`); anything else, including unset,
# is the ordinary `1.`, `2.`, ... block written into the dataset.
ENV_LABEL_MODE = "LMEVAL_GPQA_LABEL_MODE"


@functools.lru_cache(maxsize=1)
def _trace() -> TraceWriter:
    return TraceWriter.from_env()


@functools.lru_cache(maxsize=1)
def _prompt_template():
    from jinja2 import Template

    return Template(_PROMPT)


def _run_id() -> str:
    return os.environ.get(ENV_RUN_ID, "adhoc")


def _include_raw() -> bool:
    return os.environ.get("LMEVAL_TRACE_RAW", "1") not in {"0", "false", "no"}


def _task_name(doc: dict) -> str:
    """Derive the task name from the document.

    process_results is not told which task it is scoring, and one run may span
    several, so the doc itself is the only reliable source.
    """
    stage = doc.get("stage", "atomic")
    if stage == "compositional" and doc.get("k") is not None:
        return f"gpqa_compositional_k{doc['k']}"
    return f"gpqa_{stage}"


def reference_answers(doc: dict) -> list[str]:
    """The document's answer, one letter per question, in prompt order.

    ``reference_answers`` is written by the pool generator, the composition
    builder and the batching engine alike, at k=1 and k>1, so this needs no
    special case. The fallbacks exist only so a hand-written document is usable.
    """
    answers = doc.get("reference_answers")
    if answers:
        return [str(a).strip().upper() for a in answers]
    if doc.get("reference_answer"):
        return [part.strip().upper()
                for part in str(doc["reference_answer"]).split(",") if part.strip()]
    if doc.get("answer_letter"):
        return [str(doc["answer_letter"]).strip().upper()]
    raise ValueError(f"document {doc.get('id')!r} carries no reference answer")


def n_choices(doc: dict) -> int:
    """How many options each of the document's questions offers."""
    if doc.get("n_choices"):
        return int(doc["n_choices"])
    if doc.get("choices"):
        return len(doc["choices"])
    if doc.get("component_choices"):
        return len(doc["component_choices"][0])
    return 4


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def load_gpqa_dataset(**meta: Any) -> datasets.DatasetDict:
    """Load a JSONL problem file, optionally narrowed to a single k.

    Used instead of ``dataset_path: json`` because it is the only hook that
    receives ``metadata`` -- which is how both the file path and ``k`` arrive --
    and because it sidesteps HF cache staleness when a JSONL is regenerated in
    place.
    """
    import json

    path = os.environ.get(ENV_DATA) or meta.get("dataset_path")
    if not path:
        raise ValueError(
            f"no dataset path: set {ENV_DATA} or pass "
            "TaskManager(metadata={'dataset_path': ...})"
        )
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))

    if (k := meta.get("k")) is not None:
        rows = [r for r in rows if r.get("k") == k]

    if not rows:
        # ConfigurableTask smoke-tests doc_to_text on docs[0] at construction
        # time, so an empty split fails later and far less legibly than here.
        raise ValueError(f"no rows for k={meta.get('k')!r} in {path}")

    return datasets.DatasetDict({"test": datasets.Dataset.from_list(rows)})


def _example_answer(count: int, choices: int) -> str:
    """A format example of the right length, and never the reference answer.

    Cycling the offered letters keeps the example from being a hint: it is the
    same list for every document of a given (k, n_choices), so it carries no
    information about the document it is shown with.
    """
    from llm.core.mcq.answers import LETTERS

    letters = LETTERS[:choices]
    return ", ".join(letters[i % len(letters)] for i in range(count))


@functools.lru_cache(maxsize=1)
def _marker_prompt_template():
    from jinja2 import Template

    return Template(_MARKER_PROMPT)


def _component_lists(doc: dict) -> tuple[list[str], list[list[str]]]:
    """The document's questions and choice lists in prompt order."""
    if doc.get("component_questions"):
        return list(doc["component_questions"]), [list(c) for c in doc["component_choices"]]
    if doc.get("question") is not None and doc.get("choices"):
        return [doc["question"]], [list(doc["choices"])]
    raise ValueError(
        f"document {doc.get('id')!r} carries neither component_questions nor "
        "question/choices, so its block cannot be re-labelled"
    )


def marker_block(doc: dict) -> tuple[str, list[str]]:
    """The document's question block labelled Q01, Q02, ..., and those labels.

    Re-rendered from the component list with the one renderer, never rewritten
    from `questions_block` by pattern. The numbered re-rendering is first
    checked against `questions_block` byte for byte, so the questions the model
    reads are exactly those every other run of this document was measured on;
    only the labels in front of them differ.
    """
    from llm.core.mcq.rendering import marker_label, render_block

    questions, choices = _component_lists(doc)
    numbered, _ = render_block(questions, choices)
    if numbered != doc["questions_block"]:
        raise ValueError(
            f"document {doc.get('id')!r}: re-rendering its component list does not "
            "reproduce questions_block, so a re-labelled block would not be the same "
            "questions"
        )
    width = max(2, len(str(len(questions))))
    block, _ = render_block(questions, choices,
                            label=lambda i: f"{marker_label(i, width)}.")
    return block, [marker_label(i, width) for i in range(1, len(questions) + 1)]


def _label_list(labels: list[str]) -> str:
    """`Q01`, `Q01, Q02`, `Q01, Q02, ...` -- the labels as the prompt's parenthesis names them.

    Two are written out and the rest elided, so the sentence stays one short
    line at k=20 as at k=3.
    """
    if len(labels) <= 2:
        return ", ".join(labels)
    return f"{labels[0]}, {labels[1]}, ..."


def doc_to_text(doc: dict) -> str:
    """Render the shared jinja2 prompt.

    ``questions_block`` is the only problem-bearing field, written by the
    dataset builder with :func:`llm.core.mcq.rendering.render_block` -- so the
    prompt, the dataset and any span locator cannot disagree about what a
    question looks like.
    """
    answers = reference_answers(doc)
    if os.environ.get(ENV_LABEL_MODE) == "markers":
        block, labels = marker_block(doc)
        return _marker_prompt_template().render(
            questions_block=block,
            labels=_label_list(labels),
            n_questions=len(answers),
            example=_example_answer(len(answers), n_choices(doc)),
        )
    return _prompt_template().render(
        questions_block=doc["questions_block"],
        n_questions=len(answers),
        example=_example_answer(len(answers), n_choices(doc)),
    )


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
def _rendered(extraction: ExtractResult) -> str:
    if extraction.status is ExtractionStatus.OK:
        return extraction.value
    # `.get` rather than a lookup: the sentinel table names the two outcomes the
    # MCQ extractor produces, and a future third one must surface as itself in
    # lm-eval's filtered_resps rather than as a KeyError in a filter.
    return _SENTINEL.get(extraction.status, f"[[{extraction.status.value.upper()}]]")


def _extract_one(text, doc: dict, *, tail_fallback: bool) -> ExtractResult:
    """Locate one response's answer, bounded by what the document asked for."""
    return extract_answer(
        text,
        expected=len(reference_answers(doc)),
        n_choices=n_choices(doc),
        tail_fallback=tail_fallback,
    )


def _stash_token_counts(entry: dict[str, Any], doc: dict, n: int) -> None:
    """Drain this doc's token-usage stash into ``entry``, once."""
    if "token_usage" not in entry:
        entry["token_usage"] = token_stash.by_doc.drain(doc.get("id"), n)


def _apply_extraction(resps, docs, *, name: str, tail_fallback: bool) -> list[list[str]]:
    """Shape-preserving: list[per-doc][per-repeat]."""
    out: list[list[str]] = []
    for per_doc, doc in zip(resps, docs):
        trials = list(per_doc) if isinstance(per_doc, (list, tuple)) else [per_doc]
        results = [_extract_one(r, doc, tail_fallback=tail_fallback) for r in trials]
        entry = _STASH.setdefault(str(doc.get("id")), {"raw": trials, "by_filter": {}})
        entry["raw"] = trials
        entry["by_filter"][name] = results
        _stash_token_counts(entry, doc, len(trials))
        out.append([_rendered(r) for r in results])
    return out


def extract_lenient(resps, docs):
    """Everything ``explicit`` reads, plus the letters closing an unmarked
    response -- the ensemble the figures read by default."""
    return _apply_extraction(resps, docs, name="lenient", tail_fallback=True)


def extract_explicit(resps, docs):
    """Answers written *as* answers: a marker, a box, or a bare letter line.

    The control for the lenient ensemble -- it drops only the last step, reading
    option letters out of the response's closing prose. Both read the same
    responses, so the gap between ``acc,lenient`` and ``acc,explicit`` is exactly
    what that guess bought, including the luck of one that happens to be right.
    """
    return _apply_extraction(resps, docs, name="explicit", tail_fallback=False)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _result_for_failed_extraction(extraction: ExtractResult,
                                  answers: list[str]) -> CheckResult:
    """Turn an extraction failure into a verdict without invoking the grader."""
    stage = StageRecord(
        "candidate_answer_extract", status="fail",
        data={"n_matches": extraction.n_matches, "reason": extraction.status.value,
              "detail": extraction.detail},
    )
    return CheckResult(
        Verdict.UNPARSEABLE, "candidate_answer_extract", [stage],
        {"reason": extraction.status.value, "message": extraction.detail,
         "candidate_letters": [], "reference_letters": list(answers),
         "term_correct": [], "n_correct": 0, "n_terms": len(answers)},
    )


def _grade_trial(doc: dict, extraction: ExtractResult) -> CheckResult:
    answers = reference_answers(doc)
    if extraction.status is not ExtractionStatus.OK:
        return _result_for_failed_extraction(extraction, answers)

    result = check_answers(extraction.value, answers, n_choices=n_choices(doc))
    # Prepend the extraction step so the trace tells the whole story from raw
    # text onward, even though extraction happened in the filter.
    result.stages.insert(0, StageRecord(
        "candidate_answer_extract",
        data={"n_matches": extraction.n_matches, "chosen_index": extraction.chosen_index,
              "chosen": extraction.value, "strategy": extraction.detail},
    ))
    return result


def process_results(doc: dict, results: list) -> dict[str, float]:
    """Grade each trial for one document and emit its trace.

    ``results`` holds one entry per Instance, and there is exactly one Instance
    per doc for generate_until; ``repeats`` clones the request so all repeats
    land in that entry as a list.
    """
    trials = results[0] if results and isinstance(results[0], (list, tuple)) else list(results)

    stash = _STASH.get(str(doc.get("id"))) or {}
    raw_texts = stash.get("raw") or [None] * len(trials)
    by_filter: dict[str, list[ExtractResult]] = stash.get("by_filter") or {}
    token_usage = stash.get("token_usage") or [None] * len(trials)

    # Identify the ensemble being scored. lm-eval applies every filter first and
    # only then calls process_results once per ensemble, iterating docs inside,
    # so the nth call for a given doc corresponds to the nth ensemble. Value
    # matching alone cannot do this: when both strategies find the same answer --
    # the common case -- their outputs are identical.
    names = list(by_filter)
    index = stash.get("consumed", 0)
    filter_name, extractions = "unknown", None
    if index < len(names):
        filter_name = names[index]
        extractions = by_filter[filter_name]
        stash["consumed"] = index + 1
        # Ordering is an assumption about lm-eval internals; verify it rather
        # than trust it, and fall back to matching if it ever drifts.
        if [_rendered(c) for c in extractions] != list(trials):
            filter_name, extractions = "unknown", None
            for name, candidates in by_filter.items():
                if [_rendered(c) for c in candidates] == list(trials):
                    filter_name, extractions = name, candidates
                    break
    if extractions is None:
        # A filter other than ours ran, or shapes drifted; recover from the
        # values we were handed rather than dropping the document.
        extractions = [_extract_one(t, doc, tail_fallback=True) for t in trials]

    verdicts: list[CheckResult] = []
    response_chars: list[int] = []
    # Identical across every trial/filter of this doc; hoisted out of the loop.
    prompt = doc_to_text(doc)

    for index, extraction in enumerate(extractions):
        result = _grade_trial(doc, extraction)
        verdicts.append(result)
        raw = raw_texts[index] if index < len(raw_texts) else None
        response_chars.append(len(raw) if raw else 0)
        extra: dict[str, Any] = {"prompt": prompt}
        if index < len(token_usage) and token_usage[index] is not None:
            extra["token_usage"] = token_usage[index]
        _trace().emit(build_mcq_record(
            run_id=_run_id(), task=_task_name(doc),
            doc_id=doc.get("id", index), filter_name=filter_name, trial=index,
            doc=doc, raw_text=raw, result=result, include_raw=_include_raw(),
            extra=extra,
        ))

    n = max(len(verdicts), 1)
    n_terms = len(reference_answers(doc))

    def fraction(predicate) -> float:
        return sum(bool(predicate(v)) for v in verdicts) / n

    def reason_is(name: str):
        return lambda v: v.detail.get("reason") == name

    correct = sum(v.verdict is Verdict.CORRECT for v in verdicts)
    # Partial credit. An unparseable attempt scores 0 here rather than being
    # excluded: the question was asked and no usable answer came back, and
    # dropping it would quietly raise the mean of every model that formats
    # badly.
    terms_correct = [int(v.detail.get("n_correct") or 0) for v in verdicts]

    return {
        "acc": correct / n,
        "acc_terms": sum(terms_correct) / (n * n_terms) if n_terms else 0.0,
        "mean_n_correct": sum(terms_correct) / n,
        # The criterion for "reliably answers": every trial correct, not one.
        "solved_all": float(correct == n and n > 0),
        "frac_wrong": fraction(lambda v: v.verdict is Verdict.WRONG),
        "frac_unparseable": fraction(lambda v: v.verdict is Verdict.UNPARSEABLE),
        "frac_empty": fraction(reason_is(ExtractionStatus.EMPTY.value)),
        "frac_no_answer": fraction(reason_is(ExtractionStatus.NO_ANSWER.value)),
        "frac_wrong_length": fraction(reason_is(AnswerListStatus.WRONG_LENGTH.value)),
        "frac_invalid_letter": fraction(reason_is(AnswerListStatus.INVALID_LETTER.value)),
        "frac_no_letters": fraction(reason_is(AnswerListStatus.NO_LETTERS.value)),
        # Attempts graded on the first k letters of a longer list. Not a failure
        # bucket -- these are scored like any other answer -- but a reading the
        # parser chose, so it is counted rather than assumed to be rare.
        "frac_truncated": fraction(lambda v: bool(v.detail.get("truncated"))),
        "mean_response_chars": sum(response_chars) / n,
    }
