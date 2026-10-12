"""Load one per-term-trace experiment -- one model, every k, one dataset -- as DataFrames.

    from llm.core.analysis.per_term_trace import load_experiment, load_solo
    attempts, positions = load_experiment("gsm8k", model="qwen3-8b")

Every finished run of the model under `$RUN_DIR/runs/per-term-trace/<dataset>/<model>/`
(one directory `k<K>-s<A>-<B>` per prompt size and seed chunk) is read and pooled from its
`sections.jsonl`, which `python -m llm.sections` writes.

* `attempts` -- one row per response: k, seed, doc_id, verdict, the thinking tokens
  outside any section, after `</think>` and in total, the backend's completion tokens,
  the number of sections, whether the thinking closed.
* `positions` -- one row per (response, actual prompt position): item id, the thinking
  tokens of its section, its visits, `correct` (NA when the response's answer list
  could not be read) and `echo_read` (multiple-choice datasets: the graded letters are
  the prompt's format example the model quoted, not its own list; the figures leave
  these out of accuracy -- do the same with
  `positions.loc[positions.echo_read, "correct"] = pd.NA`).
  **Only actual positions by default.** The dummy (always the last position) and the
  fillers (padding of the one short batch per seed) are not measurements: the dummy's
  section also holds the answer recap and format check, so at k=2 it runs to ~1.3x the
  item's solo thinking. `include_non_actual=True` returns them too, with `role` saying
  which is which (`actual`, `dummy`, `filler`).

`load_solo(dataset, model)` gives, per item, the isolated thinking tokens (output tokens
minus the re-tokenized answer minus one for `</think>`) and solo accuracy from the
`isolated-a` run, and the mean solo output tokens from the independent `isolated-b` run,
which is what the token-ratio figures bin on. It re-tokenizes every answer, so it takes a
few tens of seconds.

    python -m llm.core.analysis.per_term_trace gsm8k --model qwen3-8b            # summary
    python -m llm.core.analysis.per_term_trace gsm8k --model qwen3-8b --csv out/  # + CSVs
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from llm import paths
from llm.specs import dataset as dataset_spec

PLACEHOLDER_END = "chars]"
_RUN_RE = re.compile(r"^k(\d+)-s(\d+)-(\d+)$")
MCQ_DATASETS = ("mmlu", "gpqa-diamond")


def _field(value):
    """A trace field written as a list/dict or as its repr."""
    if isinstance(value, (list, dict)) or value is None:
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _finished(run_dir: Path) -> bool:
    return bool(glob.glob(str(run_dir / "*" / "results_*.json")))


def _runs(dataset: str, model: str) -> list[Path]:
    run_dirs = [d for d in paths.runs("per-term-trace", dataset, model) if _RUN_RE.match(d.name)]
    missing = [d.name for d in run_dirs if _finished(d) and not (d / "sections.jsonl").exists()]
    if missing:
        print(f"[warn] finished but not cut yet (run python -m llm.sections): {missing}",
              file=sys.stderr)
    run_dirs = [d for d in run_dirs if _finished(d) and (d / "sections.jsonl").exists()]
    if not run_dirs:
        raise FileNotFoundError(f"no finished, cut per-term-trace run of {model} on {dataset}")
    return run_dirs


# --------------------------------------------------------------------------
# the solo baseline
# --------------------------------------------------------------------------
def reference_run(dataset: str, stage: str, model: str) -> Path | None:
    """The one run of `model` in an isolated stage (`isolated-a`, `isolated-b`, ...)."""
    runs = [d for d in paths.runs(stage, dataset, model) if (d / "trace.jsonl").exists()]
    if len(runs) > 1:
        raise SystemExit(f"{stage}/{dataset}/{model} holds {len(runs)} runs; expected one")
    return runs[0] if runs else None


def _lenient(trace: Path, filter_name: str = "lenient"):
    with open(trace, encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("filter") == filter_name:
                yield r


def load_solo(dataset: str, model: str, stages: tuple[str, str] = ("isolated-a", "isolated-b")):
    """Per item: solo thinking tokens, solo accuracy, isolated-b mean output tokens.

    `stages` names the two isolated passes to read; a pair rerun after its first pass
    reads its first pass with ("isolated-a-v1", "isolated-b-v1").
    """
    import numpy as np
    import pandas as pd
    from transformers import AutoTokenizer

    filter_name = dataset_spec(dataset).filter
    stage_a, stage_b = stages
    control = reference_run(dataset, stage_a, model)
    if control is None:
        raise FileNotFoundError(f"no {stage_a} run of {model} on {dataset}")
    model_id = json.loads((control / "run.json").read_text())["model"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    think, acc = defaultdict(list), defaultdict(list)
    for r in _lenient(control / "trace.jsonl", filter_name):
        item = _field(r["component_ids"])[0]
        # A whole-prompt dataset (integrals) grades its one item by the verdict alone.
        correct = _field(r.get("term_correct")) or ([r["verdict"] == "CORRECT"]
                                                   if r.get("verdict") else [])
        if correct:
            acc[item].append(bool(correct[0]))
        out = (_field(r.get("token_usage")) or {}).get("vllm_output_tokens")
        if out is None:
            continue
        text = (r.get("raw") or {}).get("text") or ""
        cut = text.find(PLACEHOLDER_END)
        if cut == -1:
            # Thinking never closed (cut off at the length cap): the whole output is
            # thinking, there is no answer. Reading its text as the answer would leave
            # ~0 thinking and send a bin's mean ratio into the hundreds.
            think[item].append(out)
            continue
        tail = text[cut + len(PLACEHOLDER_END):]
        think[item].append(max(out - len(tokenizer(tail, add_special_tokens=False)["input_ids"]) - 1, 0))

    split1 = reference_run(dataset, stage_b, model)
    if split1 is None:
        print(f"[warn] no {stage_b} run of {model} on {dataset}; "
              f"split1_output_tokens come from {stage_a}", file=sys.stderr)
    lengths = defaultdict(list)
    for r in _lenient((split1 or control) / "trace.jsonl", filter_name):
        out = (_field(r.get("token_usage")) or {}).get("vllm_output_tokens")
        if out is not None:
            lengths[_field(r["component_ids"])[0]].append(out)

    items = sorted(set(think) | set(acc))
    mean = lambda d, i: float(np.mean(d[i])) if d.get(i) else None  # noqa: E731
    return pd.DataFrame({
        "item_id": items,
        "solo_thinking_tokens": [mean(think, i) for i in items],
        "solo_accuracy": [mean(acc, i) for i in items],
        "split1_output_tokens": [mean(lengths, i) for i in items],
    })


# --------------------------------------------------------------------------
# the composed runs
# --------------------------------------------------------------------------
def _echo_reads(run_dir: Path, k: int) -> set[str]:
    """Doc ids whose graded letters are the prompt's format example the model quoted.

    A response whose own last line does not hold k letters, but whose extracted answer
    is exactly the format example (A, B, C, D, A, ...), is a wrong-length list the
    extractor filled from the example -- UNPARSEABLE by the dataset's contract, not
    WRONG. Multiple-choice datasets only.
    """
    out = set()
    with open(run_dir / "trace.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if '"lenient"' not in line:
                continue
            r = json.loads(line)
            if r.get("filter") != "lenient":
                continue
            letters = list(_field(r.get("candidate_letters")) or [])
            if len(letters) != k:
                continue
            n_choices = int(r.get("n_choices") or 4)
            example = [chr(ord("A") + i % n_choices) for i in range(k)]
            if letters != example:
                continue
            text = (r.get("raw") or {}).get("text") or ""
            close = text.find("</think>")
            lines = (text[close:] if close != -1 else text).strip().splitlines() or [""]
            if len(re.findall(r"\b[A-J]\b", lines[-1])) != k:
                out.add(r["doc_id"])
    return out


def load_experiment(dataset: str, model: str, *, include_non_actual: bool = False,
                    solo: bool = False):
    """(attempts, positions[, solo]) DataFrames for one model's per-term-trace runs."""
    import pandas as pd

    run_dirs = _runs(dataset, model)
    attempts, positions = [], []
    for run_dir in run_dirs:
        k_run = int(_RUN_RE.match(run_dir.name).group(1))
        echoes = _echo_reads(run_dir, k_run) if dataset in MCQ_DATASETS else set()
        with open(run_dir / "sections.jsonl", encoding="utf-8") as handle:
            for line in handle:
                r = json.loads(line)
                k, t, p = int(r["k"]), r["tokens"], r["parse"]
                seed = int(r["batch_seed"])
                correct = r.get("term_correct") or []
                attempts.append({
                    "tree": dataset, "model": model, "k": k, "seed": seed,
                    "doc_id": r["doc_id"], "trial": r["trial"], "n_actual": int(r["n_actual"]),
                    "verdict": r["verdict"], "readable": bool(correct),
                    "echo_read": r["doc_id"] in echoes,
                    "thinking_tokens": t["total"] - t["answer"], "outside_tokens": t["outside"],
                    "answer_tokens": t["answer"], "total_tokens": t["total"],
                    "completion_tokens": r["completion_tokens"],
                    "n_sections": p["n_sections"], "think_closed": p["think_closed"],
                    "run": run_dir.name,
                })
                for pos, (item, actual) in enumerate(zip(r["component_ids"], r["component_actual"]), 1):
                    role = "actual" if actual else ("dummy" if pos == k else "filler")
                    if role != "actual" and not include_non_actual:
                        continue
                    positions.append({
                        "tree": dataset, "model": model, "k": k, "seed": seed,
                        "doc_id": r["doc_id"], "trial": r["trial"], "position": pos,
                        "item_id": item, "role": role,
                        "tokens": t["per_question"][str(pos)],
                        "visits": t["visits"][str(pos)],
                        "correct": bool(correct[pos - 1]) if correct else pd.NA,
                        "echo_read": r["doc_id"] in echoes,
                    })
    attempts = pd.DataFrame(attempts).sort_values(["k", "seed", "doc_id"], ignore_index=True)
    positions = pd.DataFrame(positions).sort_values(["k", "seed", "doc_id", "position"],
                                                    ignore_index=True)
    positions["correct"] = positions["correct"].astype("boolean")
    if not solo:
        return attempts, positions
    return attempts, positions, load_solo(dataset, model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--solo", action="store_true", help="also load the solo baseline")
    parser.add_argument("--include-non-actual", action="store_true",
                        help="keep the dummy and filler positions (role column) in positions")
    parser.add_argument("--csv", type=Path, default=None, help="write the frames here as CSV")
    args = parser.parse_args()
    frames = load_experiment(args.dataset, args.model, solo=args.solo,
                             include_non_actual=args.include_non_actual)
    attempts, positions = frames[:2]
    actual = positions[positions.role == "actual"]
    print(f"{args.dataset} / {args.model}: {len(attempts)} responses, "
          f"{len(actual)} actual positions, {actual.item_id.nunique()} items")
    summary = actual.groupby("k").agg(seeds=("seed", "nunique"), responses=("doc_id", "nunique"),
                                      mean_tokens=("tokens", "mean"), accuracy=("correct", "mean"))
    print(summary.round(3).to_string())
    if args.csv:
        args.csv.mkdir(parents=True, exist_ok=True)
        for name, frame in zip(["attempts", "positions", "solo"], frames):
            frame.to_csv(args.csv / f"{name}.csv", index=False)
            print(f"  -> {args.csv / f'{name}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
