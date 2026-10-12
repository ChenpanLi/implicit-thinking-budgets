"""Run one stage of one (dataset, model) through lm-eval.

    python -m llm.run --stage isolated-a --dataset gsm8k --model qwen3-8b
    python -m llm.run --stage composed --dataset gsm8k --model qwen3-8b --dry-run
    python -m llm.run --stage per-term-trace --dataset mmlu --model qwen3-8b \\
        --section-markers --keep-thinking --run-id k6-s0-9

The batch is read from ``paths.batch(stage, dataset, model)`` (for per-term-trace
the part named by ``--run-id``) unless ``--batch`` names another file, and the run
is written to ``paths.model_runs_dir(stage, dataset, model)/<run-id>/``:

    run.json        what was asked: model config, batch, tasks, batching manifest
    trace.jsonl     one record per (document, extraction filter), written by the task
    <hf-model>/results_*.json, samples_*.jsonl    lm-eval's own output

``--stage composed`` runs one ``<prefix>_compositional_k<k>`` task per k; every other
stage runs the dataset's ``<prefix>_evaluation`` task over the whole batch, whatever k
its rows hold. Integrals asks for, and grades, one answer per term at every k
(the paper's isolated-a integrals runs used a single-answer prompt; see
``lm_tasks/integrals/utils.py``).

Local weights are loaded into this process (in-process vLLM); a model YAML with
``provider: openrouter`` is sent to the hosted API instead, billed per request.
``--dry-run`` loads the batch, builds the lm-eval tasks, prints the first prompts,
the output directory and the model arguments, and never loads a model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm import paths
from llm.specs import DATASETS, Dataset, dataset as dataset_spec

CHECKER_TIMEOUT = "10.0"


@dataclass
class Plan:
    """Everything a run resolves before it touches a model."""

    stage: str
    ds: Dataset
    model: str
    batch: Path
    run_id: str
    run_dir: Path
    tasks: list[str]
    section_markers: bool
    keep_thinking: bool
    seed: int
    limit: int | None
    model_variant: str | None


def task_names(ds: Dataset, stage: str, ks: list[int]) -> list[str]:
    # Integrals' tasks are named for the per-term contract they all use.
    prefix = f"{ds.task_prefix}_perterm" if ds.per_term else ds.task_prefix
    if stage == "composed":
        middle = "" if ds.per_term else "_compositional"
        return [f"{prefix}{middle}_k{k}" for k in sorted(ks)]
    return [f"{prefix}_evaluation"]


def task_env(ds: Dataset, *, section_markers: bool, batch: Path | None = None,
             run_id: str | None = None) -> dict[str, str]:
    """The environment the dataset's task module renders and grades under.

    It is the only channel into lm-eval's ``!function`` code, and it must be in
    place before the tasks are built: lm-eval renders docs[0] at construction.
    """
    module = ds.task_utils()
    env = {module.ENV_LABEL_MODE: "markers" if section_markers else "numbers"}
    if ds.name == "integrals":
        env["LMEVAL_INTEGRALS_CHECKER_TIMEOUT"] = CHECKER_TIMEOUT
        data_var, run_id_var = "LMEVAL_INTEGRALS_DATA", "LMEVAL_INTEGRALS_RUN_ID"
    else:
        data_var, run_id_var = module.ENV_DATA, module.ENV_RUN_ID
    if batch is not None:
        env[data_var] = str(batch)
    if run_id is not None:
        env[run_id_var] = run_id
    return env


def make_plan(args: argparse.Namespace) -> Plan:
    ds = dataset_spec(args.dataset)
    if args.stage == "budget-forcing":
        raise SystemExit("budget-forcing runs through `python -m llm.budget_forcing run`")
    if args.stage == "per-term-trace" and not (args.section_markers and args.keep_thinking):
        raise SystemExit("per-term-trace needs --section-markers and --keep-thinking")
    if args.stage == "per-term-trace" and not args.run_id:
        raise SystemExit("per-term-trace needs --run-id k<K>-s<A>-<B>, which also names its batch")
    run_id = args.run_id or f"seed{args.seed}-{time.strftime('%Y%m%dT%H%M%S')}"
    part = run_id if args.stage == "per-term-trace" else None
    batch = args.batch or paths.batch(args.stage, ds.name, args.model, part=part)
    if not batch.exists():
        raise SystemExit(f"batch not found: {batch}")
    return Plan(
        stage=args.stage, ds=ds, model=args.model, batch=batch, run_id=run_id,
        run_dir=paths.model_runs_dir(args.stage, ds.name, args.model) / run_id,
        tasks=task_names(ds, args.stage, args.k or ds.composed_ks),
        section_markers=args.section_markers,
        keep_thinking=args.keep_thinking, seed=args.seed, limit=args.limit,
        model_variant=args.model_variant)


def load_tasks(plan: Plan) -> tuple[Any, dict[str, Any]]:
    """Set the task environment and build the lm-eval tasks: (task manager, tasks)."""
    os.environ.update(task_env(plan.ds, section_markers=plan.section_markers,
                               batch=plan.batch, run_id=plan.run_id))
    os.environ["LMEVAL_TRACE"] = str(plan.run_dir / "trace.jsonl")
    os.environ["LMEVAL_TRACE_RAW"] = "1"
    os.environ["LMEVAL_TRACE_THINKING"] = "1" if plan.keep_thinking else "0"

    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=str(paths.REPO_ROOT / "lm_tasks"),
                               metadata={"dataset_path": str(plan.batch)})
    tasks = task_manager.load(plan.tasks)["tasks"]
    return task_manager, {name: tasks[name] for name in plan.tasks}


def render_prompts(task: Any, n: int) -> list[tuple[str, str]]:
    """(doc id, user message) of the task's first `n` documents."""
    return [(str(doc.get("id")), task.doc_to_text(doc))
            for doc in list(task.test_docs())[:n]]


def model_args_of(config: dict, endpoint: Any) -> dict:
    """lm-eval model arguments: the engine for local weights, the request for a hosted API."""
    if endpoint.in_process:
        return dict(endpoint.model_args)
    model_args = {
        "model": endpoint.model_id,
        "base_url": endpoint.chat_url,
        "num_concurrent": 16,
        "max_retries": 3,
        "timeout": 36000,
        # A chat endpoint takes text, so no local tokenizer is needed.
        "tokenized_requests": False,
        "tokenizer_backend": None,
        "max_length": int(config.get("max_model_len", 32768)),
        "max_gen_toks": int(config.get("max_tokens", 8192)),
    }
    if endpoint.headers:
        model_args["extra_headers"] = dict(endpoint.headers)
    return model_args


def gen_kwargs_of(config: dict, *, in_process: bool) -> dict:
    gen_kwargs = {
        "temperature": float(config.get("temperature", 0.6)),
        "top_p": float(config.get("top_p", 0.95)),
        "max_gen_toks": int(config.get("max_tokens", 8192)),
        "until": [],
        # Overrides lm-eval's hard-coded request seed, which would make every
        # trial of a document return the same text.
        "seed": None,
    }
    # `extra_body` (OpenRouter's `reasoning` block) is merged into the request body.
    extra_body = dict(config.get("extra_body") or {})
    if extra_body and in_process:
        raise SystemExit("extra_body only means something for a hosted API's request body")
    gen_kwargs.update(extra_body)
    return gen_kwargs


def finished(run_dir: Path) -> bool:
    return any(run_dir.glob("*/results_*.json"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", required=True, choices=sorted(paths.STAGES))
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--model", required=True, help="a YAML name under src/llm/config/models")
    parser.add_argument("--model-variant", default=None,
                        help="a named block under `variants` in the model YAML, merged over "
                             "its settings (the few runs the paper made with other settings)")
    parser.add_argument("--batch", type=Path, default=None,
                        help="the batch JSONL (default: paths.batch(stage, dataset, model))")
    parser.add_argument("--k", type=int, nargs="+", default=None,
                        help="composed only: which k to run (default: the dataset's composed_ks)")
    parser.add_argument("--section-markers", action="store_true",
                        help="label the items Q01, Q02, ... and ask the model to name an item "
                             "by its label whenever it works on it (per-term-trace)")
    parser.add_argument("--keep-thinking", action="store_true",
                        help="keep the thinking block in trace.jsonl and the samples")
    parser.add_argument("--seed", type=int, default=42, help="the engine seed")
    parser.add_argument("--run-id", default=None,
                        help="the run directory's name (default: seed<seed>-<timestamp>)")
    parser.add_argument("--limit", type=int, default=None,
                        help="documents per task, to check the wiring; not a result")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the tasks and print the first prompts; load no model")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = make_plan(args)

    from llm.core.batching import manifest_beside
    from llm.core.serving.providers import (api_key_env, is_hosted, provider_of,
                                            read_api_key, register_backend,
                                            resolve_endpoint)
    from llm.core.serving.vllm_server import (ServerError, build_inprocess_model_args,
                                              load_model_config)

    config, config_path = load_model_config(args.model, variant=args.model_variant)
    hosted = is_hosted(config)
    batching = manifest_beside(plan.batch) if plan.stage != "composed" else None

    print(f"[run] stage    : {plan.stage}")
    print(f"[run] dataset  : {plan.ds.name}  batch {plan.batch}")
    print(f"[run] model    : {config['model_id']}  ({config_path.name}"
          f"{', variant ' + args.model_variant if args.model_variant else ''})")
    print(f"[run] tasks    : {plan.tasks}")
    if plan.section_markers or plan.keep_thinking:
        print(f"[run] prompt   : {'section markers' if plan.section_markers else ''}"
              f"{', thinking kept' if plan.keep_thinking else ''}")
    if batching is not None:
        print(f"[run] batching : {batching.get('strategy')}, seeds {batching.get('seeds')}")
    print(f"[run] output   : {plan.run_dir}")

    done = finished(plan.run_dir)
    if done and not args.dry_run:
        print(f"[run] {plan.run_dir} is finished already; nothing to do")
        return 0
    if (plan.run_dir / "run.json").exists() and not done and not args.dry_run:
        raise SystemExit(f"{plan.run_dir} holds an unfinished run; delete it to run again")

    task_manager, tasks = load_tasks(plan)
    doc_counts = {name: len(task.test_docs()) for name, task in tasks.items()}
    for name, count in doc_counts.items():
        print(f"[task] {name}: {count} docs")

    if args.dry_run:
        for name, task in tasks.items():
            for doc_id, text in render_prompts(task, 2):
                print(f"\n[prompt] {name} / {doc_id}\n{text}")
        if hosted:
            provider = provider_of(config)
            try:
                read_api_key(provider)
                print(f"\n[dry-run] {provider} API; {api_key_env(provider)} is set")
            except ServerError as exc:
                print(f"\n[dry-run] {exc}")
        else:
            print("\n[dry-run] vLLM would be built in this process with:")
            for key, value in build_inprocess_model_args(config, seed=plan.seed).items():
                print(f"    {key} = {value!r}")
        return 0

    try:
        endpoint = resolve_endpoint(config, config_path=config_path, seed=plan.seed)
    except ServerError as exc:
        print(f"[run] {exc}", file=sys.stderr)
        return 2
    register_backend(endpoint)
    model_args = model_args_of(config, endpoint)
    gen_kwargs = gen_kwargs_of(config, in_process=endpoint.in_process)
    batch_size: Any = "auto" if endpoint.in_process else 1
    if hosted:
        n = sum(min(c, plan.limit) if plan.limit else c for c in doc_counts.values())
        print(f"[cost] about {n} requests will be billed to {api_key_env(provider_of(config))}")

    plan.run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "run_id": plan.run_id, "experiment": plan.ds.tree, "stage": plan.stage,
        "dataset_name": plan.ds.name, "tasks": plan.tasks,
        "dataset": str(plan.batch), "model_config": str(config_path),
        "model": config, "model_variant": args.model_variant,
        "model_args": model_args, "gen_kwargs": gen_kwargs, "system_prompt": None,
        "in_process": endpoint.in_process, "batch_size": batch_size,
        "repeats": 1, "limit": plan.limit, "seed": plan.seed,
        "endpoint": endpoint.to_dict(),
        "per_term": plan.ds.per_term,
        "section_markers": plan.section_markers,
        "section_markers_prompt": (plan.ds.task_utils().MARKER_PROMPT_VERSION
                                   if plan.section_markers else None),
        "keep_thinking": plan.keep_thinking,
        "batching_manifest": batching,
        "argv": sys.argv, "doc_counts": doc_counts,
    }
    (plan.run_dir / "run.json").write_text(
        json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8")

    from lm_eval import simple_evaluate
    from lm_eval.loggers import EvaluationTracker
    from lm_eval.utils import make_table

    from llm.core.thinking_text import strip_samples
    from llm.core.trace import keep_thinking

    lm: Any = endpoint.lm_eval_backend
    try:
        if endpoint.in_process:
            from llm.core.serving.lm_eval_vllm import InProcessVLLM
            lm = InProcessVLLM(**model_args, batch_size=batch_size)
        tracker = EvaluationTracker(output_path=str(plan.run_dir))
        results = simple_evaluate(
            model=lm, model_args=model_args, tasks=list(tasks.values()),
            task_manager=task_manager, apply_chat_template=True, system_instruction=None,
            log_samples=True, evaluation_tracker=tracker, limit=plan.limit,
            batch_size=batch_size, gen_kwargs=gen_kwargs, bootstrap_iters=0,
            random_seed=plan.seed, numpy_random_seed=plan.seed,
            torch_random_seed=plan.seed, fewshot_random_seed=plan.seed,
        )
        if results is None:
            return 0
        samples = results.pop("samples", {})
        # lm-eval writes samples_*.jsonl from this structure, so this is the one
        # place the thinking can be kept out of it.
        if not keep_thinking():
            strip_samples(samples)
        tracker.save_results_aggregated(results=results, samples=samples)
        for task_name, task_samples in samples.items():
            tracker.save_results_samples(task_name=task_name, samples=task_samples)
        print("\n" + make_table(results))
        print(f"[run] done -> {plan.run_dir}")
        return 0
    finally:
        if endpoint.in_process and not isinstance(lm, str):
            from llm.core.serving.lm_eval_vllm import free_engine
            free_engine(lm)


if __name__ == "__main__":
    raise SystemExit(main())
