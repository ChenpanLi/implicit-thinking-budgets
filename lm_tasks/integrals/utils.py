"""lm-eval hooks for the indefinite-integral tasks.

Every ``!function`` referenced by the YAML in this directory resolves here. Every
task asks for one antiderivative per term of the sum (the per-term contract),
k=1 included. The one exception in the paper's data: its ``isolated-a`` integrals
runs were made with a single-answer prompt ("Find an antiderivative for
the following indefinite integral ... FINAL ANSWER: \\boxed{...}"); for a single
integral the two ask the same question, and a rerun of that stage uses this prompt.

Division of labour, chosen so each artifact stays authoritative for one thing:

* **Filters** extract candidate answers and return *strings only*, so lm-eval's
  own ``filtered_resps`` shows exactly what was extracted. Running two
  extraction strategies as separate ensembles scores them side by side in a
  single pass (``acc,boxed_last`` vs ``acc,final_answer``).
* **process_results** verifies, and emits the stage-by-stage trace that nothing
  else records.
"""
from __future__ import annotations

import functools
import os
from typing import Any

import datasets

from llm.core.math.per_term import (
    BoxedList,
    BoxedListStatus,
    check_term_answers,
    extract_boxed_list,
)
from llm.core.math.verification import CheckResult, Verdict
from llm.core.serving import token_stash
from llm.core.trace import TraceWriter, build_record

# process_results receives only (doc, results) -- neither the raw generations
# nor the name of the filter ensemble it is scoring. Filters see both, so they
# stash what the verifier needs, keyed by the dataset's stable id. Raw text is
# stored once per doc; extraction outcomes are stored per ensemble, and the
# active ensemble is recovered by matching the values lm-eval hands back.
_STASH: dict[str, dict[str, Any]] = {}


# The prompt: the k-term integral, k answers asked for, one boxed antiderivative
# per line. That is the contract the one-word QA and MMLU tasks use, and it is
# what makes `term_correct` exact here.
#
# Three choices in here are load-bearing:
#
# * **The whole integral is shown.** The problem is the k-term sum as SymPy
#   renders it; only what the model is asked to write back is per term.
# * **The terms are listed, numbered, in the order the integral shows them.**
#   The grader compares answer i with term i, and a position alone would ask the
#   model to agree with us about what "the third term" is -- SymPy chose that
#   order and the model has no reason to reproduce it.
# * **One `\boxed{}` per line, and the count is named.** The newline is the
#   delimiter, so a block of k lines is unambiguous whatever the answers are;
#   naming k is what lets the extractor insist on the count rather than guess at
#   which term a short block skipped.
_PER_TERM_PROMPT = """
Find an antiderivative for each term of the following indefinite integral: {{problem}}.

The integrand is a sum of {{n_terms}} term{{ "s" if n_terms > 1 }}, listed below in the order {{ "they appear" if n_terms > 1 else "it appears" }} in the integral:
{% for term in terms %}
{{loop.index}}. \\int {{term}} \\, d{{variable}}
{% endfor %}
Do not add the antiderivatives together, and do not give an antiderivative of the whole sum. Solve each term on its own and report {{n_terms}} separate antiderivative{{ "s" if n_terms > 1 }}, one per term, in the order above.

Give symbolic answers only in standard LaTeX format. Do not give a decimal approximation. The constant of integration may be omitted.
Return your answers in the following format in the last {{n_terms}} line{{ "s" if n_terms > 1 }}, with no additional text: one line per term, in the order above, each line holding exactly one \\boxed{}.

{{example}}

Before you finish, re-read your answer block and check that it holds exactly {{n_terms}} line{{ "s" if n_terms > 1 }}, each with exactly one \\boxed{}, in the order the terms are listed above.
"""


# The section-marker variant of the prompt (`python -m llm.run --section-markers`,
# the per-term-trace and budget-forcing runs). The terms are listed
# as Q01, Q02, ... instead of 1., 2., ..., one sentence after the list asks the
# model to name a term by its label whenever it works on it, so the thinking can
# be cut per term at every label mention (llm.core.analysis.question_sections.
# parse_label_sections). Built from
# `_PER_TERM_PROMPT` by three exact replacements, each asserted, so the two
# templates cannot drift apart.
#
# One thing differs beyond the labels, and it is load-bearing: **the list is in
# batch order, not in the order SymPy rendered the sum** (`prompt_terms`). A
# `dummy-tail` batching puts a throwaway item last so that the answer recap and
# format check, which a label-cut trace hands to whichever item is labelled last,
# land on it; in SymPy's order the dummy would sit anywhere in the list. So the
# list does not claim to follow the integral's order, and the grader, which
# reads the order from `prompt_terms` too, compares answer i with list item i.
MARKER_PROMPT_VERSION = 3
ENV_LABEL_MODE = "LMEVAL_INTEGRALS_LABEL_MODE"
_ORDER_CLAUSE = (', listed below in the order {{ "they appear" if n_terms > 1 else "it appears" }} '
                 'in the integral:')
_LIST_LINE = "{{loop.index}}. \\int {{term}} \\, d{{variable}}\n{% endfor %}\n"
assert _PER_TERM_PROMPT.count(_ORDER_CLAUSE) == 1
assert _PER_TERM_PROMPT.count(_LIST_LINE) == 1
_MARKER_PER_TERM_PROMPT = (
    _PER_TERM_PROMPT
    .replace(_ORDER_CLAUSE, ", listed below:", 1)
    .replace(_LIST_LINE,
             "{{ labels[loop.index0] }}. \\int {{term}} \\, d{{variable}}\n{% endfor %}\n"
             "While you think, refer to each term by its label ({{ label_list }}) whenever you "
             "start or return to working on it.\n\n", 1)
)


def marker_mode() -> bool:
    """Whether this process labels the per-term list Q01, Q02, ..., read fresh."""
    return os.environ.get(ENV_LABEL_MODE) == "markers"


def _label_list(labels: list[str]) -> str:
    """`Q01`, `Q01, Q02`, `Q01, Q02, ...` -- as the sentence's parenthesis names them."""
    return ", ".join(labels) if len(labels) <= 2 else f"{labels[0]}, {labels[1]}, ..."


@functools.lru_cache(maxsize=1)
def _trace() -> TraceWriter:
    return TraceWriter.from_env()


@functools.lru_cache(maxsize=1)
def _marker_per_term_prompt_template():
    from jinja2 import Template

    return Template(_MARKER_PER_TERM_PROMPT)


@functools.lru_cache(maxsize=1)
def _per_term_prompt_template():
    from jinja2 import Template

    return Template(_PER_TERM_PROMPT)


# --------------------------------------------------------------------------
# per-term answers
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=65536)
def _term_order(doc_id: str, integral: str, terms: tuple[str, ...]) -> tuple[int, ...]:
    """Component indices in the order the rendered integral shows them.

    Cached because it is sympy work done once per document and then wanted again
    on every trial and every filter ensemble; the key is everything it depends
    on, so two documents can never share an entry.
    """
    from llm.core.math.per_term import term_order

    return tuple(term_order({"id": doc_id, "integrand_latex_integral": integral,
                             "component_integrands": list(terms)}))


def prompt_terms(doc: dict) -> tuple[tuple[int, ...], list[str]]:
    """``(component indices, integrands)`` in prompt order.

    The single source of the numbering the per-term prompt shows and the per-term
    grader compares against, so the two cannot disagree. A composition whose
    terms cannot be located in its own integral is a hard error rather than a
    guessed order -- see `llm.core.analysis.prompt_spans`.
    """
    terms = tuple(str(term) for term in doc.get("component_integrands") or [])
    if not terms:
        raise ValueError(
            f"{doc.get('id')!r} carries no component_integrands, so it cannot be "
            "asked for one answer per term"
        )
    if marker_mode():
        # Batch order, so a dummy-tail's dummy is listed last (see
        # `_MARKER_PER_TERM_PROMPT`); the grader reads the same order.
        order = tuple(range(len(terms)))
    else:
        order = _term_order(str(doc.get("id")),
                            str(doc.get("integrand_latex_integral") or ""), terms)
    return order, [terms[index] for index in order]


def answer_example(count: int) -> str:
    r"""The format example: `count` numbered lines, each holding one box.

    Placeholders rather than plausible antiderivatives, for the reason the
    one-word QA prompt gives: an example that could be mistaken for an answer is
    copied, and a copy of a placeholder is at least recognisable as one. Long
    blocks are abbreviated -- spelling out twenty placeholder lines would take
    more of the prompt than the integral does.
    """
    if count < 1:
        raise ValueError(f"a document must ask for at least one answer, got {count}")
    line = "{index}. \\boxed{{<antiderivative of term {index}>}}"
    if count <= 3:
        return "\n".join(line.format(index=i) for i in range(1, count + 1))
    return "\n".join([line.format(index=1), line.format(index=2), "...",
                      line.format(index=count)])


def _run_id() -> str:
    return os.environ.get("LMEVAL_INTEGRALS_RUN_ID", "adhoc")


def _checker_timeout() -> float:
    return float(os.environ.get("LMEVAL_INTEGRALS_CHECKER_TIMEOUT", "10"))


def _include_raw() -> bool:
    return os.environ.get("LMEVAL_TRACE_RAW", "1") not in {"0", "false", "no"}


def _task_name(doc: dict) -> str:
    """Derive the task name from the document.

    process_results is not told which task it is scoring, and one run may span
    several, so the doc itself is the only reliable source.
    """
    stage = doc.get("stage", "evaluation")
    if stage == "compositional" and doc.get("k") is not None:
        return f"indefinite_perterm_k{doc['k']}"
    if stage == "evaluation":
        return "indefinite_perterm_evaluation"
    return f"indefinite_{stage}"


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def load_integral_dataset(**meta: Any) -> datasets.DatasetDict:
    """Load a JSONL problem file, optionally narrowed to a single k.

    Used instead of ``dataset_path: json`` because it is the only hook that
    receives ``metadata`` -- which is how both the file path and ``k`` arrive --
    and because it sidesteps HF cache staleness when a JSONL is regenerated in
    place.
    """
    import json

    path = os.environ.get("LMEVAL_INTEGRALS_DATA") or meta.get("dataset_path")
    if not path:
        raise ValueError(
            "no dataset path: set LMEVAL_INTEGRALS_DATA or pass "
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


def doc_to_text(doc: dict) -> str:
    """Render the prompt: the integral, its terms listed, one answer per term."""
    from llm.core.analysis.prompt_spans import term_latex

    _order, terms = prompt_terms(doc)
    if marker_mode():
        from llm.core.shortmath.rendering import marker_label

        width = max(2, len(str(len(terms))))
        labels = [marker_label(i, width) for i in range(1, len(terms) + 1)]
        return _marker_per_term_prompt_template().render(
            problem=doc["integrand_latex_integral"],
            n_terms=len(terms),
            terms=[term_latex(term) for term in terms],
            labels=labels,
            label_list=_label_list(labels),
            example=answer_example(len(terms)),
            variable=doc.get("variable", "x"),
        )
    return _per_term_prompt_template().render(
        problem=doc["integrand_latex_integral"],
        n_terms=len(terms),
        terms=[term_latex(term) for term in terms],
        example=answer_example(len(terms)),
        variable=doc.get("variable", "x"),
    )


# --------------------------------------------------------------------------
# filters
# --------------------------------------------------------------------------
def _rendered(extraction: BoxedList) -> str:
    """What lm-eval shows as this response's filtered extraction: the k answers,
    joined by newlines -- the format the prompt asked for."""
    return extraction.rendered()


def _stash_token_counts(entry: dict[str, Any], doc: dict, n: int) -> None:
    """Drain this doc's token-usage stash into `entry`, once.

    Filters run once per ensemble on the same `(resps, docs)`, so this would
    empty `token_stash.by_doc`'s queue on the first ensemble and leave the
    second with nothing -- guarded the same way `entry["raw"]` is effectively
    idempotent across those calls.
    """
    if "token_usage" not in entry:
        entry["token_usage"] = token_stash.by_doc.drain(doc.get("id"), n)


def _apply_list_extraction(resps, docs, *, name: str,
                           prefer_final_answer: bool) -> list[list[str]]:
    """Extract a block of k boxed answers from every trial, shape-preserving."""
    out: list[list[str]] = []
    for per_doc, doc in zip(resps, docs):
        trials = list(per_doc) if isinstance(per_doc, (list, tuple)) else [per_doc]
        expected = len(doc.get("component_integrands") or []) or 1
        results = [extract_boxed_list(response, expected=expected,
                                      prefer_final_answer=prefer_final_answer)
                   for response in trials]
        entry = _STASH.setdefault(str(doc.get("id")), {"raw": trials, "by_filter": {}})
        entry["raw"] = trials
        entry["by_filter"][name] = results
        _stash_token_counts(entry, doc, len(trials))
        out.append([_rendered(result) for result in results])
    return out


def extract_boxed_list_last(resps, docs):
    r"""Take the last block of k ``\boxed{}`` lines -- the per-term `boxed_last`."""
    return _apply_list_extraction(resps, docs, name="boxed_last", prefer_final_answer=False)


def extract_boxed_list_final_answer(resps, docs):
    """Prefer the block following a FINAL ANSWER marker, falling back to the last."""
    return _apply_list_extraction(resps, docs, name="final_answer", prefer_final_answer=True)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def _verify_per_term_trial(doc: dict, parsed: BoxedList) -> CheckResult:
    """Grade one trial's k boxed answers against the k terms, in prompt order.

    The terms come from `prompt_terms`, the same call that numbered them in the
    prompt, so answer i and term i are the same term by construction rather than
    by an assumption about `component_integrands` order -- which SymPy does not
    preserve when it renders the sum.
    """
    _order, terms = prompt_terms(doc)
    return check_term_answers(parsed, terms, variable=doc.get("variable", "x"),
                              timeout_s=_checker_timeout())


def process_results(doc: dict, results: list) -> dict[str, float]:
    """Verify each trial for one document and emit its trace.

    ``results`` holds one entry per Instance, and there is exactly one Instance
    per doc for generate_until; ``repeats`` clones the request so all repeats
    land in that entry as a list.
    """
    trials = results[0] if results and isinstance(results[0], (list, tuple)) else list(results)

    stash = _STASH.get(str(doc.get("id"))) or {}
    raw_texts = stash.get("raw") or [None] * len(trials)
    by_filter: dict[str, list[BoxedList]] = stash.get("by_filter") or {}
    token_usage = stash.get("token_usage") or [None] * len(trials)

    # Identify the ensemble being scored. lm-eval applies every filter first
    # and only then calls process_results once per ensemble, iterating docs
    # inside, so the nth call for a given doc corresponds to the nth ensemble.
    # Value-matching alone cannot do this: when both strategies find the same
    # box -- the common case -- their outputs are identical.
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
        expected = len(doc.get("component_integrands") or []) or 1
        extractions = [extract_boxed_list(t, expected=expected) for t in trials]

    verdicts: list[CheckResult] = []
    response_chars: list[int] = []
    # Identical across every trial/filter of this doc; hoisted out of the loop.
    prompt = doc_to_text(doc)

    # The per-term identity of every position, in prompt order and recorded on
    # every record beside `term_correct`, so a wrong position can be attributed
    # to an atom and a category without joining back to the dataset -- the same
    # service `component_categories` performs in the one-word QA trace.
    order, terms = prompt_terms(doc)
    term_context: dict[str, Any] = {
        "term_positions": list(order),
        "term_integrands": terms,
        "term_ids": [(doc.get("component_ids") or [None] * len(order))[i] for i in order],
        "term_categories": [(doc.get("component_categories")
                             or [None] * len(order))[i] for i in order],
    }

    for index, extraction in enumerate(extractions):
        result = _verify_per_term_trial(doc, extraction)
        verdicts.append(result)
        raw = raw_texts[index] if index < len(raw_texts) else None
        response_chars.append(len(raw) if raw else 0)
        extra: dict[str, Any] = {"prompt": prompt}
        if index < len(token_usage) and token_usage[index] is not None:
            extra["token_usage"] = token_usage[index]
        # Hoisted out of `detail` onto the record itself: every per-position
        # reading wants these.
        extra.update(term_context)
        extra.update({
            "candidate_answers": result.detail.get("candidate_answers", []),
            "term_correct": result.detail.get("term_correct", []),
            "term_verdicts": result.detail.get("term_verdicts", []),
            "n_correct": result.detail.get("n_correct", 0),
            "n_terms": result.detail.get("n_terms", len(terms)),
            "per_term": True,
        })
        _trace().emit(build_record(
            run_id=_run_id(), task=_task_name(doc),
            doc_id=doc.get("id", index), filter_name=filter_name, trial=index,
            doc=doc, raw_text=raw, result=result, include_raw=_include_raw(),
            extra=extra,
        ))

    n = max(len(verdicts), 1)

    def fraction(predicate) -> float:
        return sum(bool(predicate(v)) for v in verdicts) / n

    def reason_is(name: str):
        return lambda v: v.detail.get("reason") == name

    correct = sum(v.verdict is Verdict.CORRECT for v in verdicts)
    metrics = {
        "acc": correct / n,
        # The criterion for "reliably solves": every trial correct, not one.
        "solved_all": float(correct == n and n > 0),
        "frac_wrong": fraction(lambda v: v.verdict is Verdict.WRONG),
        "frac_unparseable": fraction(lambda v: v.verdict is Verdict.UNPARSEABLE),
        "frac_empty": fraction(reason_is("empty")),
        "frac_no_boxed": fraction(reason_is("no_boxed")),
        "frac_unbalanced": fraction(reason_is("unbalanced")),
        "frac_echoed_integral": fraction(reason_is("echoed_integral")),
        "frac_latex_error": fraction(reason_is("latex_error")),
        "frac_timeout": fraction(lambda v: v.verdict is Verdict.TIMEOUT),
        "frac_checker_error": fraction(lambda v: v.verdict is Verdict.CHECKER_ERROR),
        "frac_numeric_only": fraction(lambda v: v.confidence == "numeric_only"),
        "mean_response_chars": sum(response_chars) / n,
    }
    # Partial credit, first-class rather than derived. `acc` above means "every
    # term right", so it and `acc_terms` disagree by construction past k=1.
    #
    # An unreadable answer block scores 0 here rather than being excluded: the
    # terms were asked for and no usable answer came back. It contributes no
    # per-term outcomes at all, though -- k-1 boxes cannot be attributed to k terms.
    n_terms = len(terms)
    term_counts = [int(v.detail.get("n_correct") or 0) for v in verdicts]
    unparseable_terms = [int(v.detail.get("n_unparseable_terms") or 0) for v in verdicts]
    metrics.update({
        "acc_terms": sum(term_counts) / (n * n_terms) if n_terms else 0.0,
        "mean_n_correct": sum(term_counts) / n,
        "frac_wrong_length": fraction(reason_is(BoxedListStatus.WRONG_LENGTH.value)),
        # Of the *terms*, not of the attempts: a single box that would not
        # parse costs one term here and the whole attempt in `acc`.
        "frac_terms_unparseable": (sum(unparseable_terms) / (n * n_terms)
                                   if n_terms else 0.0),
    })
    return metrics
