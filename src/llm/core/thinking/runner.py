"""The domain-free half of a budget-forced run.

One model, one set of documents, each capped at its own thinking budget (the
``budget_tokens`` field ``python -m llm.budget_forcing build`` writes). The run
directory is laid out exactly like an ``lm_eval`` run, so that every loader reads
a budget-forced run without knowing it was made differently.

    <output root>/<run_id>/
        run.json          -- what was asked, plus `thinking_budget`/`budget_mode`
        trace.jsonl       -- written by the task's own process_results
        samples_<task>_<stamp>.jsonl
        generations.jsonl -- how each response was forced (no lm_eval counterpart)
        summary.json      -- realized thinking length and the headline metrics

The budget is a floor, a cap, or both, as :class:`llm.core.thinking.budget.BudgetMode`
says; one invocation is one mode, and the mode is in the run id as well as in
``run.json``.

What makes this honest rather than a re-implementation
------------------------------------------------------
The prompt, the extraction ensembles, the grading and the trace records all come
from the experiment's own ``lm_tasks/<exp>/utils.py`` -- the same module lm-eval
loads -- through the same three-step contract lm-eval uses: the filter functions
extract and stash, then ``process_results`` is called once per ensemble and both
grades and traces. Nothing about scoring is written twice, so a budget-forced
run and a normal run of the same documents differ in how the text was generated
and in nothing else.

Two consequences of reusing that module deserve naming, because both are silent
when got wrong:

* ``utils._STASH`` is keyed by document id and its ``consumed`` counter decides
  which ensemble a ``process_results`` call is scoring. It is therefore filled
  and then cleared per document here, exactly as one lm-eval pass would leave
  it -- an entry left behind would make the *next* document's second ensemble
  read the previous document's extractions.
* ``utils._trace()`` is ``lru_cache``d, so it holds the first trace path it ever
  saw. It is cleared before the run starts, so the trace lands in this run's
  directory and nowhere else.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from llm.core.serving.providers import require_local
from llm.core.serving.vllm_server import (
    ServerError, build_inprocess_model_args, load_model_config,
)
from llm.core.thinking.budget import (
    BudgetError, BudgetForcer, BudgetMode, ForcedGeneration,
)
from llm.core.thinking.delimiters import DelimiterError, resolve_delimiters
from llm.core.thinking_text import strip_thinking

# The budget label of a run whose documents carry their own caps: what `run.json`
# records as `thinking_budget`, and, shortened, what the run id says
# (`thinkperdocupper`).
PER_DOCUMENT = "per-document"


@dataclass(frozen=True)
class ExperimentSpec:
    """What one experiment's task layer is called, and where its runs go.

    `ensembles` names the extraction filters in the order the task YAML lists
    them -- the order is load-bearing, because ``process_results`` recovers which
    ensemble it is scoring from the order the filters ran in. The labels are
    asserted against what the module actually stashed rather than trusted, so a
    renamed ensemble stops the run instead of mislabelling every score.
    """

    name: str                                   # "onewordqa"
    module_path: str                            # "lm_tasks.onewordqa.utils"
    ensembles: tuple[tuple[str, str], ...]      # ((label, filter function name), ...)
    env_run_id: str                             # "LMEVAL_ONEWORDQA_RUN_ID"
    stage: str = "compositional"

    def load_module(self):
        import importlib

        return importlib.import_module(self.module_path)


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
def group_by_task(module, docs: list[dict]) -> dict[str, list[dict]]:
    """Documents grouped by the task name their own fields imply.

    Derived by the task module, never by us: the hint variant of the integrals
    tasks is named from a field of the document, and a second opinion about that
    is a second source of truth.
    """
    grouped: dict[str, list[dict]] = {}
    for doc in docs:
        grouped.setdefault(module._task_name(doc), []).append(doc)
    return grouped


# --------------------------------------------------------------------------- #
# grading (through the task module, never beside it)
# --------------------------------------------------------------------------- #
def grade_document(spec: ExperimentSpec, module, doc: dict,
                   texts: list[str]) -> dict[str, dict]:
    """Extract, grade and trace one document's trials, once per ensemble.

    Returns ``{ensemble: {"metrics": ..., "filtered": [...]}}``. The trace
    records are emitted as a side effect by ``process_results`` itself, which is
    the point: they are then byte-identical in shape to a normal run's.
    """
    doc_key = str(doc.get("id"))
    rendered: list[list[str]] = []
    for _, function_name in spec.ensembles:
        extract = getattr(module, function_name)
        rendered.append(list(extract([list(texts)], [doc])[0]))

    stash = module._STASH.get(doc_key) or {}
    labels = list(stash.get("by_filter") or {})
    expected = [label for label, _ in spec.ensembles]
    if labels != expected:
        raise SystemExit(
            f"{spec.module_path} stashed ensembles {labels}, but this runner was "
            f"told they are {expected}. Fix the spec rather than the labels: the "
            "ensemble name travels into every trace record and every figure."
        )

    results: dict[str, dict] = {}
    # One call per ensemble, in the order the filters ran -- the same order
    # lm-eval calls in, which is how `process_results` knows which ensemble's
    # extractions it is scoring.
    for label, filtered in zip(labels, rendered):
        metrics = module.process_results(doc, [filtered])
        results[label] = {"metrics": metrics, "filtered": filtered}
    module._STASH.pop(doc_key, None)
    return results


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def prepare_prompts(module, forcer: BudgetForcer, tasks: dict[str, list[dict]],
                    system_prompt: str | None) -> dict[str, list[list[int]]]:
    """Chat-template every document once, through the task module's own ``doc_to_text``.

    Never a re-implementation of it: a drifted prompt would make every number in
    the run describe a text nothing was ever asked.
    """
    prepared: dict[str, list[list[int]]] = {}
    for task, docs in tasks.items():
        prompts = []
        for doc in docs:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": module.doc_to_text(doc)})
            prompts.append(forcer.prepare_prompt(messages))
        prepared[task] = prompts
    return prepared


def run_budget(spec: ExperimentSpec, module, forcer: BudgetForcer, *,
               tasks: dict[str, list[dict]],
               prompts: dict[str, list[list[int]]], run_dir: Path,
               run_id: str, repeats: int, include_raw: bool,
               keep_thinking: bool, progress: bool,
               budget_key: str) -> dict[str, Any]:
    """Generate, grade and write the run directory.

    Every task's prompts are checked against the context window *before* the
    first one is generated, so a budget that does not fit costs nothing and
    leaves no half-written run directory behind. Each document is capped at its
    own ``doc[budget_key]``.
    """
    budget = PER_DOCUMENT

    def task_budgets(docs: list[dict], per_trial: int) -> list[int]:
        return [int(doc[budget_key]) for doc in docs for _ in range(per_trial)]

    for task, docs in tasks.items():
        forcer.check_fits(prompts[task], task_budgets(docs, 1),
                          [doc.get("id") for doc in docs])

    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LMEVAL_TRACE"] = str(run_dir / "trace.jsonl")
    os.environ["LMEVAL_TRACE_RAW"] = "1" if include_raw else "0"
    os.environ["LMEVAL_TRACE_THINKING"] = "1" if keep_thinking else "0"
    os.environ[spec.env_run_id] = run_id
    # The writer caches the path it was built with.
    module._trace.cache_clear()

    stamp = time.strftime("%Y-%m-%dT%H-%M-%S.000000")
    summary: dict[str, Any] = {"run_id": run_id, "thinking_budget": budget,
                               "budget_mode": forcer.mode.value,
                               "budget_margin": forcer.margin, "tasks": {}}
    # Per-document metric values, keyed the way lm-eval keys them
    # (`<metric>,<ensemble>`).
    scores: dict[str, dict[str, list[float]]] = {}
    generations_path = run_dir / "generations.jsonl"
    generations = generations_path.open("w", encoding="utf-8")

    try:
        for task, docs in tasks.items():
            # One request per trial: `repeats` independent forced generations of
            # the same prompt, which is what `solved_all` is defined over. They
            # go into one batch so the engine schedules across trials too.
            requests, labels = [], []
            for doc, token_ids in zip(docs, prompts[task]):
                for trial in range(repeats):
                    requests.append(token_ids)
                    labels.append((doc.get("id"), trial))

            print(f"[budget {budget}] {task}: {len(docs)} docs x {repeats} trials", flush=True)
            forced = forcer.run(
                requests, task_budgets(docs, repeats), labels=labels,
                progress=f"[budget {budget}] {task}" if progress else "",
            )

            samples_path = run_dir / f"samples_{task}_{stamp}.jsonl"
            with samples_path.open("w", encoding="utf-8") as samples:
                for index, doc in enumerate(docs):
                    trials = forced[index * repeats:(index + 1) * repeats]
                    graded = grade_document(spec, module, doc, [t.text for t in trials])
                    for label, entry in graded.items():
                        samples.write(json.dumps(_sample_record(
                            index, doc, label, trials, entry,
                            keep_thinking=keep_thinking), ensure_ascii=False,
                            default=str) + "\n")
                        for metric, value in entry["metrics"].items():
                            (scores.setdefault(task, {})
                                   .setdefault(f"{metric},{label}", []).append(value))
                    for trial, generation in enumerate(trials):
                        generations.write(json.dumps({
                            "run_id": run_id, "task": task, "doc_id": doc.get("id"),
                            "trial": trial, "k": doc.get("k"),
                            **generation.to_dict(),
                        }, default=str) + "\n")

            summary["tasks"][task] = _task_summary(docs, forced, repeats)
            print(f"[budget {budget}] {task}: {_format_summary(summary['tasks'][task])}",
                  flush=True)
    finally:
        generations.close()

    summary["scores"] = {
        task: {metric: (sum(values) / len(values)) if values else 0.0
               for metric, values in metrics.items()}
        for task, metrics in scores.items()
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary, scores


def _sample_record(index: int, doc: dict, ensemble: str,
                   trials: list[ForcedGeneration], entry: dict,
                   keep_thinking: bool = False) -> dict:
    """One `samples_*.jsonl` row, in the shape lm-eval writes and loaders read.

    ``filtered_resps`` is a list *of one list* of per-trial strings, matching
    what lm-eval produces for a `generate_until` task with repeats -- the doc
    loader reads its trial count off exactly that nesting.
    """
    texts = [trial.text for trial in trials]
    if not keep_thinking:
        # The same removal `llm.core.trace` performs on the trace: the thinking
        # is the budget -- by construction the largest part of every response.
        texts = [strip_thinking(text)[0] for text in texts]
    return {
        "doc_id": index,
        "doc": doc,
        "target": doc.get("reference_answer") or doc.get("integrand"),
        "filter": ensemble,
        "resps": [texts],
        "filtered_resps": [entry["filtered"]],
        # Absent from lm-eval's own samples, and the reason a budget-forced
        # samples file is not merely a normal one: what the generation cost in
        # thinking, per trial.
        "thinking_tokens": [trial.thinking_tokens for trial in trials],
        "thinking_budget": trials[0].thinking_budget if trials else None,
        **entry["metrics"],
    }


def _task_summary(docs: list[dict], forced: list[ForcedGeneration],
                  repeats: int) -> dict[str, Any]:
    lengths = [generation.thinking_tokens for generation in forced]
    return {
        "n_docs": len(docs),
        "repeats": repeats,
        "n_generations": len(forced),
        "thinking_tokens_mean": statistics.fmean(lengths) if lengths else 0.0,
        "thinking_tokens_median": statistics.median(lengths) if lengths else 0.0,
        "thinking_tokens_min": min(lengths, default=0),
        "thinking_tokens_max": max(lengths, default=0),
        "continuations_mean": (statistics.fmean(g.continuations for g in forced)
                               if forced else 0.0),
        "answer_tokens_mean": (statistics.fmean(g.answer_tokens for g in forced)
                               if forced else 0.0),
        # Generations whose thinking fell short of the budget because the
        # continuation cap was reached. Not an error, but the x-axis is only
        # honest while this is zero, so it is reported every time.
        "n_below_budget": sum(not g.budget_reached for g in forced),
        "n_answer_truncated": sum(g.answer_finish_reason == "length" for g in forced),
        "n_cap_hit": sum(g.cap_hit for g in forced),
        "free_tokens_mean": (statistics.fmean(g.free_tokens for g in forced)
                             if forced else 0.0),
        "n_count_never_started": sum(g.count_started is False for g in forced),
    }


def _format_summary(entry: dict) -> str:
    return (f"thinking {entry['thinking_tokens_mean']:.0f} tokens mean "
            f"({entry['thinking_tokens_min']}-{entry['thinking_tokens_max']}), "
            f"{entry['continuations_mean']:.1f} continuations, "
            f"{entry['n_below_budget']} below budget, "
            f"{entry['n_answer_truncated']} answers truncated")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser(spec: ExperimentSpec, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="model alias under src/llm/config/models. Local weights "
                             "only: budget forcing needs token-level control of the "
                             "generation, which no hosted API offers")
    parser.add_argument("--model-variant", default=None,
                        help="a named block under `variants` in the model YAML, merged "
                             "over its settings")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="the documents to run (a JSONL)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="generations per document; solved_all needs all of them correct")
    parser.add_argument("--limit", type=int, default=None,
                        help="the first N documents only (smoke tests)")
    parser.add_argument("--seed", type=int, default=42, help="engine seed")
    parser.add_argument("--tag", "-T", default=None, help="extra tag in the run id")

    forcing = parser.add_argument_group("budget forcing")
    forcing.add_argument("--budget-mode", default=BudgetMode.EXACT.value,
                         choices=[m.value for m in BudgetMode],
                         help="whether the budget is a floor (`lower`), a cap "
                              "(`upper`) or both (`exact`, the default)")
    forcing.add_argument("--budget-margin", type=int, default=0,
                         help="tokens of tolerance on the *floor*: a model that "
                              "closes its thinking within this many tokens of the "
                              "budget is taken at its word rather than pushed over "
                              "the line. There is no matching tolerance on the cap")
    forcing.add_argument("--continue-cue", default="\nWait",
                         help="appended when the model tries to stop thinking early "
                              "(default: a newline and 'Wait', as in the s1 paper)")
    forcing.add_argument("--answer-cue", default="\n\nAnswer:",
                         help="appended after the closing delimiter once the budget is "
                              "spent (default: an 'Answer:' line)")
    forcing.add_argument("--answer-tokens", type=int, default=2048,
                         help="cap on the answer generated after the forced close")
    forcing.add_argument("--continue-min-tokens", type=int, default=32,
                         help="tokens a continuation must produce before it may close "
                              "the block again; 0 is the unconstrained method, at the "
                              "risk of spending a large budget one 'Wait' at a time")
    forcing.add_argument("--max-continuations", type=int, default=512,
                         help="give up forcing after this many continuations; such "
                              "generations are counted in summary.json as below budget")
    forcing.add_argument("--think-open", default=None,
                         help="override the detected thinking-open delimiter")
    forcing.add_argument("--think-close", default=None,
                         help="override the detected thinking-close delimiter")

    parser.add_argument("--output-root", type=Path, required=True,
                        help="where the run directories go")
    parser.add_argument("--no-trace-raw", action="store_true",
                        help="store hashes and head/tail instead of full generations")
    parser.add_argument("--keep-thinking", action="store_true",
                        help="record the model's thinking block verbatim instead of "
                             "replacing it with a placeholder. Off by default: the block "
                             "is most of every response")
    parser.add_argument("--progress", action="store_true",
                        help="print how many requests are still under budget each round")
    parser.add_argument("--tqdm", action="store_true", help="vLLM's own progress bars")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve everything, render one forced prompt and print the "
                             "plan, without loading the model")
    return parser


def build_run_id(spec: ExperimentSpec, args: argparse.Namespace, dataset_name: str,
                 stamp: str, *, extra: Sequence[str] = ()) -> str:
    """The run id, which is also the run directory's name.

    The budget sits next to the model, with the *mode* in the same component
    (`thinkperdocupper`): a floor and a cap are different conditions over the
    same documents, so ids that differed only by timestamp would be two runs
    nothing downstream could tell apart.

    ``extra`` are further condition tags from the caller (the budget variant),
    placed before the tag so the id reads condition-first.
    """
    mode = BudgetMode(args.budget_mode).value
    parts = [spec.name, spec.stage, dataset_name, args.model, f"thinkperdoc{mode}"]
    parts += [str(part) for part in extra if part]
    if args.tag:
        parts.append(args.tag)
    parts += [f"seed{args.seed}", stamp]
    return "-".join(parts)


def run_per_document(spec: ExperimentSpec, args: argparse.Namespace, *,
                     documents: list[dict],
                     budget_key: str,
                     output_root: Path,
                     dataset_name: str,
                     run_id_extra: Sequence[str] = (),
                     run_config_extra: dict[str, Any] | None = None) -> int:
    """Load the engine and run every document, capped at its own ``doc[budget_key]``.

    ``args`` is the namespace :func:`build_parser` returns. ``args.dataset`` is
    recorded as the provenance of the run; the documents themselves come from
    the caller. ``run_config_extra`` is merged into ``run.json``, which is where
    the caller says how it derived them.

    The run is labelled :data:`PER_DOCUMENT` rather than with a number, since no
    one number describes it; each trial's own cap is on its `generations.jsonl` row.
    """
    budget = PER_DOCUMENT
    mode = BudgetMode(args.budget_mode)
    margin = int(args.budget_margin or 0)

    variant = args.model_variant
    config, config_path = load_model_config(args.model, variant=variant)
    try:
        require_local(config, config_path, "budget forcing")
    except ServerError as exc:
        print(f"[model] {exc}", file=sys.stderr)
        print("[model] budget forcing suppresses and injects tokens inside the model's "
              "own thinking block, which a hosted chat API does not expose.", file=sys.stderr)
        return 2

    if not args.dataset.exists():
        raise SystemExit(f"dataset not found: {args.dataset}")

    module = spec.load_module()
    docs = list(documents)
    if not docs:
        raise SystemExit("no documents to run")
    bad = [doc.get("id") for doc in docs
           if not isinstance(doc.get(budget_key), int) or doc[budget_key] < 0]
    if bad:
        raise SystemExit(f"{len(bad)} documents have no integer {budget_key!r} >= 0 "
                         f"(first: {bad[:3]})")
    tasks = group_by_task(module, docs)
    system_prompt = str(config["system_prompt"]) if config.get("system_prompt") else None

    print(f"[run] experiment : {spec.name}")
    print(f"[run] model      : {config['model_id']}  (config {config_path})")
    print(f"[run] dataset    : {args.dataset}")
    print(f"[run] tasks      : {', '.join(f'{t} ({len(d)} docs)' for t, d in tasks.items())}")
    caps = sorted(doc[budget_key] for doc in docs)
    print(f"[run] budgets    : per document from {budget_key!r}: min {caps[0]}, "
          f"median {caps[len(caps) // 2]}, max {caps[-1]}, "
          f"{sum(c == 0 for c in caps)} zero  ({mode.value})")
    print(f"[run] repeats    : {args.repeats}   limit: {args.limit}")
    print(f"[run] output     : {output_root}")
    if system_prompt is not None:
        print(f"[run] system     : {system_prompt!r}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model_id"]), trust_remote_code=bool(config.get("trust_remote_code")))
    try:
        delimiters = resolve_delimiters(
            tokenizer, open_text=args.think_open, close_text=args.think_close)
    except DelimiterError as exc:
        print(f"[thinking] {exc}", file=sys.stderr)
        return 2
    print(f"[run] thinking   : {delimiters.family} "
          f"{delimiters.open_text!r} .. {delimiters.close_text!r}")

    engine_args = build_inprocess_model_args(config, seed=args.seed)
    # `build_inprocess_model_args` is the single place a model YAML is mapped
    # onto an engine, and it speaks lm-eval's parameter names. Two of them are
    # lm-eval's own rather than vLLM's, so they are translated here instead of
    # a second mapping being written next to it.
    llm_kwargs = {key: value for key, value in engine_args.items()
                  if key not in {"pretrained", "max_gen_toks"}}
    llm_kwargs["model"] = engine_args["pretrained"]

    forcer_kwargs = dict(
        temperature=float(config.get("temperature", 0.6)),
        top_p=float(config.get("top_p", 0.95)),
        max_model_len=int(config.get("max_model_len", 32768)),
        answer_tokens=args.answer_tokens,
        continue_cue=args.continue_cue,
        answer_cue=args.answer_cue,
        continue_min_tokens=args.continue_min_tokens,
        max_continuations=args.max_continuations,
        mode=mode,
        margin=margin,
        use_tqdm=args.tqdm,
    )

    if args.dry_run:
        return _dry_run(module, tokenizer, delimiters, tasks, system_prompt,
                        llm_kwargs, forcer_kwargs, args, budget_key=budget_key)

    from vllm import LLM

    llm = LLM(**llm_kwargs)
    try:
        forcer = BudgetForcer(llm, tokenizer, delimiters, **forcer_kwargs)
        prompts = prepare_prompts(module, forcer, tasks, system_prompt)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        run_id = build_run_id(spec, args, dataset_name, stamp, extra=run_id_extra)
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_config: dict[str, Any] = {
            "run_id": run_id,
            "experiment": spec.name,
            "stage": spec.stage,
            "tasks": sorted(tasks),
            "dataset": str(args.dataset),
            "model_config": str(config_path),
            "model": dict(config),
            "model_variant": variant,
            "model_args": llm_kwargs,
            "system_prompt": system_prompt,
            "in_process": True,
            "repeats": args.repeats,
            "limit": args.limit,
            "seed": args.seed,
            # The condition. Everything else in this file is context for it.
            "thinking_budget": budget,
            # Top-level as well as inside `budget_forcing`, so a reader of the
            # condition need not reach into a nested block for its other half.
            "budget_mode": mode.value,
            "budget_margin": margin,
            "budget_forcing": {
                "method": "s1 budget forcing (Wait continuation, forced close)",
                "mode": mode.value,
                "margin": margin,
                "enforces_floor": mode.has_floor,
                "enforces_cap": mode.has_cap,
                "family": delimiters.family,
                "think_open": delimiters.open_text,
                "think_close": delimiters.close_text,
                "continue_cue": args.continue_cue,
                "answer_cue": args.answer_cue,
                "answer_tokens": args.answer_tokens,
                "continue_min_tokens": args.continue_min_tokens,
                "max_continuations": args.max_continuations,
                "budgets": [budget],
                "budget_key": budget_key,
                # The budget counts from the first thinking token.
                "count_from": None,
            },
            "doc_counts": {task: len(entries) for task, entries in tasks.items()},
            "argv": sys.argv,
        }
        run_config.update(run_config_extra or {})
        (run_dir / "run.json").write_text(
            json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8")

        print(f"\n[budget {budget}] -> {run_dir}", flush=True)
        try:
            run_budget(
                spec, module, forcer, tasks=tasks,
                prompts=prompts, run_dir=run_dir, run_id=run_id,
                repeats=args.repeats,
                include_raw=not args.no_trace_raw,
                keep_thinking=args.keep_thinking,
                progress=args.progress, budget_key=budget_key)
        except BudgetError as exc:
            # A budget that does not fit is said loudly, and the directory goes
            # with it: a run holding a run.json and no samples is not a smaller
            # result, and a loader would stop on it.
            print(f"[budget {budget}] skipped: {exc}", file=sys.stderr)
            _discard_unstarted(run_dir)
    finally:
        _free_engine(llm)
    return 0


def _dry_run(module, tokenizer, delimiters, tasks, system_prompt,
             llm_kwargs, forcer_kwargs, args, budget_key: str) -> int:
    """Print the plan and one fully forced prompt, without loading the model."""
    forcer = BudgetForcer(None, tokenizer, delimiters, **forcer_kwargs)
    task = sorted(tasks)[0]
    doc = tasks[task][0]
    messages = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    messages.append({"role": "user", "content": module.doc_to_text(doc)})
    prompt_ids = forcer.prepare_prompt(messages)

    print("\n[dry-run] vLLM would be built with:")
    for key, value in llm_kwargs.items():
        print(f"    {key} = {value!r}")
    generations = sum(len(entries) for entries in tasks.values()) * args.repeats
    print(f"\n[dry-run] {generations} forced responses ({forcer.mode.value} budget)")
    print(f"[dry-run] prompt for {task} doc {doc.get('id')}: {len(prompt_ids)} tokens, "
          f"ending:\n---\n{tokenizer.decode(prompt_ids[-200:], skip_special_tokens=False)}\n---")
    print("[dry-run] once the thinking block ends, generation continues from:\n---\n"
          f"...{delimiters.close_text}{args.answer_cue}\n---")
    mode = forcer.mode
    if mode.has_floor:
        print(f"[dry-run] a model that stops before the floor is continued with: "
              f"{args.continue_cue!r}"
              + (f" (floor relaxed by {forcer.margin} tokens)" if forcer.margin else ""))
    else:
        print("[dry-run] nothing is injected mid-thought: this mode has no floor, "
              "so a model that stops early is taken at its word")
    if mode.has_cap:
        print("[dry-run] a model still thinking at the budget is cut off there")
    else:
        print("[dry-run] there is no cap: past the floor the model thinks until it "
              "stops, or until the context window leaves only room for the answer")
    # Every document, not the first: the caps differ, so one fitting says
    # nothing about the others. Tokenizing is cheap next to generating.
    for name, entries in sorted(tasks.items()):
        ids = [forcer.prepare_prompt(
                   ([{"role": "system", "content": system_prompt}] if system_prompt else [])
                   + [{"role": "user", "content": module.doc_to_text(entry)}])
               for entry in entries]
        try:
            forcer.check_fits(ids, [int(entry[budget_key]) for entry in entries],
                              [entry.get("id") for entry in entries])
            print(f"[dry-run] {name}: all {len(entries)} per-document budgets fit")
        except BudgetError as exc:
            print(f"[dry-run] {name}: {exc}")
    return 0


def _discard_unstarted(run_dir: Path) -> None:
    """Remove a run directory that never generated anything.

    Deliberately narrow: only a directory holding nothing but the ``run.json``
    written before generation begins. A trace, a generations log or a samples
    file means text was generated, and generated text is never deleted here --
    the budget check that raises runs before any of the three exists.
    """
    present = {path.name for path in run_dir.iterdir()} if run_dir.is_dir() else set()
    if present != {"run.json"}:
        return
    (run_dir / "run.json").unlink()
    run_dir.rmdir()
    print(f"[budget] removed the empty run directory {run_dir}", file=sys.stderr)


class _EngineHolder:
    """The one attribute `free_engine` reaches for, around a bare `vllm.LLM`."""

    def __init__(self, model: Any) -> None:
        self.model = model


def _free_engine(engine: Any) -> None:
    """Release the GPU through the same teardown the lm-eval path uses.

    Reused rather than reimplemented: `free_engine` is where the knowledge lives
    that vLLM's distributed state has to be cleaned up as well as the object
    dropped, and it already promises never to raise -- teardown must not turn a
    finished run into a failure.
    """
    from llm.core.serving.lm_eval_vllm import free_engine

    free_engine(_EngineHolder(engine))
