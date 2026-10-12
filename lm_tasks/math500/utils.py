"""lm-eval hooks for the MATH-500 composition tasks.

Every ``!function`` referenced by the YAML in this directory resolves here.

Division of labour, chosen so each artifact stays authoritative for one thing,
and deliberately identical to the other task layers in this repo:

* **Filters** extract candidate answers and return *strings only*, so lm-eval's
  own ``filtered_resps`` shows exactly what was extracted. Running two
  extraction strategies as separate ensembles scores them side by side in a
  single pass (``acc,lenient`` vs ``acc,explicit``), which keeps the price of
  the lenient reading visible in the same units as the score.
* **process_results** grades, and emits the stage-by-stage trace that nothing
  else records -- including the per-problem ``term_correct`` list and the
  ``match_levels`` list that says which rung of the comparison ladder decided
  each position. Comparing two LaTeX answers is exact and local, so the
  per-position figures read that straight out of the trace instead of paying an
  extractor model for it.

Everything below the prompt is :mod:`llm.core.shortmath`, shared with the GSM8K
task layer, which differs from this file in three things and no others: the
prompt, the answer *shape* a line must have to be read as one, and the
comparator. The two are kept as separate files rather than parameterised
into one for the reason stated across this repository: a change made for one
domain must not be able to move a byte of another domain's prompt.
"""
from __future__ import annotations

import functools
import os
from typing import Any

import datasets

from llm.core.math.verification import CheckResult, StageRecord, Verdict
from llm.core.parser.extraction import ExtractResult, ExtractionStatus
from llm.core.serving import token_stash
from llm.core.shortmath.answers import AnswerLineStatus
from llm.core.shortmath.extraction import LATEX_SHAPE, extract_answers
from llm.core.shortmath.grading import check_answers
from llm.core.shortmath.latex import answer_matches
from llm.core.shortmath.trace_records import build_shortmath_record
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


# The prompt. One template for the atomic and compositional stages alike: an
# atomic problem *is* a one-problem composition, and giving it its own wording
# would mean the k=1 point of a curve was measured on a different instruction
# from the rest of it.
#
# The answer contract is k answers on k lines, numbered, each boxed. The box is
# asked for because a MATH-500 answer may contain spaces and commas -- `(6,31,-1)`,
# `\left( 3, \frac{\pi}{2} \right)`, `10,\!080` -- so the newline is the only
# separator available and a delimiter around each answer is what keeps a stray
# trailing word out of it. It is a request and not a contract: the extractor
# reads an unboxed line, a bare block and a marker just as happily (see
# llm.core.shortmath.extraction), and `llm.core.shortmath.answers.unwrap_boxed`
# takes the box off whether or not it was asked for.
#
# "in the form the problem asks for" is deliberate and load-bearing: many
# MATH-500 problems specify their own answer format ("Enter your answer in the
# form $(r,\theta)$", "Express your answer as a common fraction"), and the
# reference answer follows *that* instruction. A prompt that imposed a format of
# its own would be asking for an answer the answer key does not hold.
#
# The closing re-read instruction is there because a reasoning model that has
# argued for several thousand tokens routinely loses the format it was given at
# the start, and a readable response scored UNPARSEABLE measures us rather than
# the model.
#
# The example lines are angle-bracketed placeholders rather than plausible
# answers, deliberately: the MMLU prompt's `Answer: A, B, C` was copied verbatim
# by models often enough to need its own figure, and a placeholder cannot be
# mistaken for a real answer -- while still being detectable when it is echoed.
_PROMPT = r"""
Solve the following problem{{ "s" if n_questions > 1 }}.

{{ questions_block }}

Give the final answer to each problem only, with no explanation. Write each answer in LaTeX, in the form the problem asks for, wrapped in \boxed{...}.
{% if n_questions > 1 %}
End your response with an answer block: the line "Answer:" followed by exactly {{ n_questions }} lines, one answer per line, numbered, in the order the problems were asked. Write nothing after the block.

{{ example }}
{% else %}
End your response with an answer block: the line "Answer:" followed by one line holding your answer. Write nothing after it.

{{ example }}
{% endif %}
Before you finish, re-read your answer block and check that it holds exactly {{ "one" if n_questions == 1 else n_questions }} answer line{% if n_questions > 1 %}s, in the order the problems were asked{% endif %}, with nothing written after {{ "them" if n_questions > 1 else "it" }}.
"""

ENV_DATA = "LMEVAL_MATH500_DATA"
ENV_RUN_ID = "LMEVAL_MATH500_RUN_ID"

# The section-marker prompt (`--section-markers`): `_PROMPT` word for word, with
# the problems labelled Q01, Q02, ... instead of 1., 2., ... and one sentence after
# the block asking the model to name a problem by its label whenever it works on
# it, so the thinking can be cut per problem at every label mention
# (llm.core.analysis.question_sections.parse_label_sections). Built from
# `_PROMPT` by inserting that one sentence, so the two cannot drift apart, and
# kept a separate template so that adding it cannot move a byte of the prompt
# every ordinary run is measured on. Version 3 is the MMLU task layer's
# wording; a run records it as `section_markers_prompt` in run.json.
MARKER_PROMPT_VERSION = 3
_MARKER_SENTENCE = ("While you think, refer to each problem by its label ({{ labels }}) "
                    "whenever you start or return to working on it.")
_BLOCK_LINE = "{{ questions_block }}\n"
assert _PROMPT.count(_BLOCK_LINE) == 1
_MARKER_PROMPT = _PROMPT.replace(_BLOCK_LINE, _BLOCK_LINE + "\n" + _MARKER_SENTENCE + "\n", 1)
# Selects how problems are labelled. `"markers"` labels them Q01, Q02, ... and
# takes `_MARKER_PROMPT` (`--section-markers`); anything else, including unset,
# is the ordinary `1.`, `2.`, ... block written into the dataset.
ENV_LABEL_MODE = "LMEVAL_MATH500_LABEL_MODE"

# What a line has to look like to be read as one answer: generous in characters
# and in tokens, strict in English words, and with no digit required. A MATH-500
# answer may be `\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}` -- 45
# characters and nine whitespace tokens, not one of them a word -- or
# `\text{Evelyn}`, which holds no digit at all.
SHAPE = LATEX_SHAPE


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
        return f"math500_compositional_k{doc['k']}"
    return f"math500_{stage}"


def reference_answers(doc: dict) -> list[str]:
    """The document's canonical answer, one per problem, in prompt order.

    `reference_answers` is written by both the pool generator and the
    composition builder, at k=1 and k>1 alike, so this needs no special case.
    The fallbacks exist only so a hand-written document is usable.
    """
    answers = doc.get("reference_answers")
    if answers:
        return [str(a).strip() for a in answers]
    if doc.get("accepted_answers"):
        return [str(variants[0]).strip() for variants in doc["accepted_answers"]]
    if doc.get("correct_answers"):
        return [str(doc["correct_answers"][0]).strip()]
    raise ValueError(f"document {doc.get('id')!r} carries no reference answer")


def accepted_answers(doc: dict) -> list[list[str]]:
    """Every accepted variant of every answer, parallel to `reference_answers`.

    A list per problem rather than a string, because the shared schema is the
    one-word tree's, where a problem may genuinely have two right answers. MATH-500
    writes exactly one, so the inner lists have length one -- but the shape is
    what the grader and every join already read.
    """
    accepted = doc.get("accepted_answers")
    if accepted:
        return [[str(a).strip() for a in variants] for variants in accepted]
    if doc.get("component_answers"):
        return [[str(a).strip() for a in variants] for variants in doc["component_answers"]]
    if doc.get("correct_answers") and int(doc.get("k", 1)) == 1:
        return [[str(a).strip() for a in doc["correct_answers"]]]
    return [[answer] for answer in reference_answers(doc)]


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def load_math500_dataset(**meta: Any) -> datasets.DatasetDict:
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


def example_block(count: int) -> str:
    """The format example, of the right length and carrying no information.

    Placeholders, never plausible answers: the example is the same text for
    every document of a given k, so it cannot hint at the document it is shown
    with, and when a model copies it the copy is recognisable as a copy.
    """
    if count < 1:
        raise ValueError(f"a document must ask at least one problem, got {count}")
    lines = ["Answer:"]
    if count == 1:
        lines.append(r"1. \boxed{<your answer>}")
    elif count <= 3:
        lines += [rf"{i}. \boxed{{<your answer to problem {i}>}}"
                  for i in range(1, count + 1)]
    else:
        # Abbreviated for a long block: spelling out twenty placeholder lines
        # would take more of the prompt than the problems do.
        lines += [r"1. \boxed{<your answer to problem 1>}",
                  r"2. \boxed{<your answer to problem 2>}", "..."]
        lines.append(rf"{count}. \boxed{{<your answer to problem {count}>}}")
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def _marker_prompt_template():
    from jinja2 import Template

    return Template(_MARKER_PROMPT)


def _component_questions(doc: dict) -> list[str]:
    """The document's problems in prompt order."""
    if doc.get("component_questions"):
        return list(doc["component_questions"])
    if doc.get("question") is not None:
        return [doc["question"]]
    raise ValueError(
        f"document {doc.get('id')!r} carries neither component_questions nor "
        "question, so its block cannot be re-labelled"
    )


def marker_block(doc: dict) -> tuple[str, list[str]]:
    """The document's problem block labelled Q01, Q02, ..., and those labels.

    Re-rendered from the component list with the one renderer, never rewritten
    from `questions_block` by pattern. The numbered re-rendering is first
    checked against `questions_block` byte for byte, so the problems the model
    reads are exactly those every other run of this document was measured on;
    only the labels in front of them differ.
    """
    from llm.core.shortmath.rendering import marker_label, render_block

    questions = _component_questions(doc)
    numbered, _ = render_block(questions)
    if numbered != doc["questions_block"]:
        raise ValueError(
            f"document {doc.get('id')!r}: re-rendering its component list does not "
            "reproduce questions_block, so a re-labelled block would not be the same "
            "problems"
        )
    width = max(2, len(str(len(questions))))
    block, _ = render_block(questions,
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

    `questions_block` is the only problem-bearing field, written by the dataset
    builder with `llm.core.shortmath.rendering.render_block` -- so the prompt,
    the dataset and the span locator cannot disagree about what a problem looks
    like.
    """
    answers = reference_answers(doc)
    if os.environ.get(ENV_LABEL_MODE) == "markers":
        block, labels = marker_block(doc)
        return _marker_prompt_template().render(
            questions_block=block,
            labels=_label_list(labels),
            n_questions=len(answers),
            example=example_block(len(answers)),
        )
    return _prompt_template().render(
        questions_block=doc["questions_block"],
        n_questions=len(answers),
        example=example_block(len(answers)),
    )


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
def _rendered(extraction: ExtractResult) -> str:
    if extraction.status is ExtractionStatus.OK:
        return extraction.value
    # `.get` rather than a lookup: the sentinel table names the two outcomes the
    # extractor produces, and a future third one must surface as itself in
    # lm-eval's filtered_resps rather than as a KeyError in a filter.
    return _SENTINEL.get(extraction.status, f"[[{extraction.status.value.upper()}]]")


def _extract_one(text, doc: dict, *, tail_fallback: bool) -> ExtractResult:
    """Locate one response's answer block, bounded by what the document asked for."""
    return extract_answers(
        text,
        expected=len(reference_answers(doc)),
        shape=SHAPE,
        tail_fallback=tail_fallback,
    )


def _stash_token_counts(entry: dict[str, Any], doc: dict, n: int) -> None:
    """Drain this doc's token-usage stash into `entry`, once (see integrals/utils.py)."""
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
    """Everything `explicit` reads, plus the closing lines of a response that
    never announced an answer -- the ensemble the figures read by default."""
    return _apply_extraction(resps, docs, name="lenient", tail_fallback=True)


def extract_explicit(resps, docs):
    """Answer blocks written *as* answer blocks: a numbered block, a marker, a
    box, or a bare block of answer-shaped lines.

    The control for the lenient ensemble -- it drops only the last step, which
    takes the closing lines whatever they look like. Both read the same
    responses, so the gap between `acc,lenient` and `acc,explicit` is exactly
    what that guess bought, including the luck of one that happens to be right,
    which is why it is measured rather than assumed to be small.
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
         "candidate_answers": [], "reference_answers": list(answers),
         "term_correct": [], "n_correct": 0, "n_terms": len(answers)},
    )


def _grade_trial(doc: dict, extraction: ExtractResult) -> CheckResult:
    answers = reference_answers(doc)
    if extraction.status is not ExtractionStatus.OK:
        return _result_for_failed_extraction(extraction, answers)

    result = check_answers(extraction.value, accepted_answers(doc),
                           comparator=answer_matches)
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
    # matching alone cannot do this: when both strategies find the same block --
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
        _trace().emit(build_shortmath_record(
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
    # excluded: the problem was asked and no usable answer came back, and
    # dropping it would quietly raise the mean of every model that formats
    # badly.
    terms_correct = [int(v.detail.get("n_correct") or 0) for v in verdicts]

    return {
        "acc": correct / n,
        "acc_terms": sum(terms_correct) / (n * n_terms) if n_terms else 0.0,
        "mean_n_correct": sum(terms_correct) / n,
        # The criterion for "reliably solves": every trial correct, not one.
        "solved_all": float(correct == n and n > 0),
        "frac_wrong": fraction(lambda v: v.verdict is Verdict.WRONG),
        "frac_unparseable": fraction(lambda v: v.verdict is Verdict.UNPARSEABLE),
        "frac_empty": fraction(reason_is(ExtractionStatus.EMPTY.value)),
        "frac_no_answer": fraction(reason_is(ExtractionStatus.NO_ANSWER.value)),
        "frac_no_answer_lines": fraction(reason_is(AnswerLineStatus.NO_ANSWERS.value)),
        "frac_wrong_length": fraction(reason_is(AnswerLineStatus.WRONG_LENGTH.value)),
        "mean_response_chars": sum(response_chars) / n,
    }
