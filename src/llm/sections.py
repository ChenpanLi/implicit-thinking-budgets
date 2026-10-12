"""Cut every trace of a per-term-trace run per item and count each item's thinking tokens.

    python -m llm.sections <run dir> [<run dir> ...] [--force] [--dry-run]
    python -m llm.sections --dataset gsm8k --model qwen3-8b [--run-id k6-s0-9]

Writes ``<run>/sections.jsonl``, one row per attempt:

* `tokens.per_question` -- thinking tokens per prompt position ("1".."k"). A position's
  section starts at a mention of its label (Q03, Question 3, ...) and runs until another
  position's label is mentioned or the thinking ends
  (`llm.core.analysis.question_sections.parse_label_sections`). A token belongs to the
  region holding its first character; the label token counts inside its own section.
* `tokens.outside` -- thinking before the first label; `tokens.answer` -- after `</think>`.
* `component_actual` / `component_ids` -- copied from the trace, never applied: dropping
  the dummy and the fillers is the reader's decision, made on this field.
* `term_correct`, `verdict`, the candidate answers -- copied, for accuracy by position.

Tokens are counted with the run's own tokenizer. Only the rows of the dataset's
extraction filter are read (every filter scores the same text): `boxed_last` for
integrals, `lenient` elsewhere. A trace written without `--keep-thinking` holds a
placeholder instead of the thinking and is refused.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from llm import paths
from llm.specs import DATASETS, dataset as dataset_spec


def _field(value):
    """A trace field written as a list/dict or as its repr."""
    if isinstance(value, (list, dict)) or value is None:
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def filter_of(run_dir: Path) -> str:
    """The extraction filter of the run's dataset (`runs/<stage>/<dataset>/<model>/<run>`)."""
    name = run_dir.resolve().parent.parent.name
    if name not in DATASETS:
        raise SystemExit(f"{run_dir}: cannot tell its dataset from the path; pass --filter")
    return dataset_spec(name).filter


def cut(run_dir: Path, filter_name: str) -> int:
    from transformers import AutoTokenizer

    from llm.core.analysis.question_sections import parse_label_sections, summary, token_breakdown
    from llm.core.parser.math_parser import strip_bpe_sentinels
    from llm.core.serving.token_stash import normalize_usage
    from llm.core.thinking_text import has_placeholder

    run = json.loads((run_dir / "run.json").read_text())
    if not run.get("section_markers"):
        raise SystemExit(f"{run_dir}: run.json does not record --section-markers")
    if (run.get("section_markers_prompt") or 1) < 3:
        raise SystemExit(f"{run_dir}: prompt v{run.get('section_markers_prompt')} predates "
                         "the label sentence this cutter reads")
    model_id = run["model"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    rows = []
    with open(run_dir / "trace.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                r = json.loads(line)
                if r.get("filter") == filter_name:
                    rows.append(r)
    rows.sort(key=lambda r: (str(r["doc_id"]), int(r.get("trial") or 0)))
    tmp = run_dir / "sections.jsonl.tmp"
    with open(tmp, "w", encoding="utf-8") as out:
        for r in rows:
            text = (r.get("raw") or {}).get("text") or ""
            if has_placeholder(text):
                raise SystemExit(f"{run_dir}: the thinking was stripped from the trace; "
                                 "re-run with --keep-thinking")
            text, _ = strip_bpe_sentinels(text)
            k = int(r["k"])
            # An integrals record lists its terms in prompt order (`term_positions`);
            # marker runs list them in batch order, so the two must coincide, or each
            # section would be attached to the wrong item.
            positions = _field(r.get("term_positions"))
            if positions is not None and list(positions) != list(range(k)):
                raise SystemExit(f"{run_dir}: {r['doc_id']} lists its terms in order "
                                 f"{positions}, not batch order")
            parse = parse_label_sections(text, k)
            offsets = tokenizer(text, add_special_tokens=False,
                                return_offsets_mapping=True)["offset_mapping"]
            out.write(json.dumps({
                "run_id": run["run_id"], "model_id": model_id,
                "prompt_version": run.get("section_markers_prompt"), "parser": "labels",
                "doc_id": r["doc_id"], "trial": int(r.get("trial") or 0), "k": k,
                "verdict": r.get("verdict"),
                "term_correct": _field(r.get("term_correct")),
                "candidate_letters": _field(r.get("candidate_letters")),
                "candidate_answers": _field(r.get("candidate_answers")),
                "reference_answers": _field(r.get("reference_answers")),
                "component_actual": _field(r.get("component_actual")),
                "component_ids": _field(r.get("component_ids")),
                "n_actual": r.get("n_actual"),
                "batch_seed": r.get("batch_seed"),
                "tokens": token_breakdown(parse, offsets),
                "completion_tokens": normalize_usage(r.get("token_usage"))["completion"],
                "parse": summary(parse),
            }) + "\n")
    tmp.replace(run_dir / "sections.jsonl")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="*", type=Path, help="run directories to cut")
    parser.add_argument("--dataset", choices=sorted(DATASETS),
                        help="instead of run directories: the per-term-trace runs of a dataset")
    parser.add_argument("--model", help="with --dataset: whose runs")
    parser.add_argument("--run-id", help="with --dataset: only this run (k<K>-s<A>-<B>)")
    parser.add_argument("--filter", default=None,
                        help="which filter's rows to cut (default: the dataset's)")
    parser.add_argument("--force", action="store_true", help="re-cut runs that have sections.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="list what would be cut")
    args = parser.parse_args(argv)

    run_dirs = list(args.runs)
    if args.dataset:
        if not args.model:
            parser.error("--dataset needs --model")
        root = paths.model_runs_dir("per-term-trace", args.dataset, args.model)
        run_dirs += ([root / args.run_id] if args.run_id
                     else paths.runs("per-term-trace", args.dataset, args.model))
    if not run_dirs:
        parser.error("name run directories, or --dataset and --model")

    for run_dir in run_dirs:
        filter_name = args.filter or filter_of(run_dir)
        if (run_dir / "sections.jsonl").exists() and not args.force:
            print(f"[skip] {run_dir}: sections.jsonl exists (--force to re-cut)")
            continue
        if args.dry_run:
            state = "" if (run_dir / "trace.jsonl").exists() else " (not run yet)"
            print(f"[dry-run] would cut {run_dir} on the {filter_name!r} rows{state}")
            continue
        if not (run_dir / "trace.jsonl").exists():
            raise SystemExit(f"{run_dir}: no trace.jsonl")
        n = cut(run_dir, filter_name)
        print(f"[cut] {run_dir.name}: {n} attempts -> sections.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
