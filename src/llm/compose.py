"""Build the composed sets: k-item compositions of the items a model reliably solves.

    python -m llm.compose --dataset gsm8k --model qwen3-8b

writes ``paths.batch("composed", dataset, model)`` and a manifest beside it.

**The filter.** An item is *reliably solved* when the model answered it correctly on
all three of batching seeds 0, 1 and 2 of its ``isolated-a`` run (every item asked
alone; the run holds five seeds and the other two decide nothing). Attempts are
read from the run's ``trace.jsonl`` under the dataset's extraction filter, keyed on
the item id (``component_ids[0]``) and the batching seed, so the three attempts are
a named set rather than whichever three were written first. An unparseable answer
is an attempt and it is wrong.

**The draw.** For each k, ``n`` compositions are drawn with an RNG keyed on
``"<seed>:<k>"`` (so adding a k never changes another k's draw). Each position
picks a category uniformly, then a solved item of that category uniformly, so a
composition's category mix does not follow the model's strengths. The same group
in the same order is never drawn twice (for integrals, whose sum does not depend
on order, the same set of terms is never drawn twice). For GSM8K, MATH-500, AIME
and StrategyQA a composition never holds one item twice; for integrals, MMLU,
GPQA and OneWordQA that rule is off, so their stored sets rebuild exactly.

``n`` and the k values come from ``llm.specs`` (n is 100 for claude-opus-5, whose
runs are billed).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from llm import paths, specs
from llm.core.batching.engine import read_jsonl, sha256_of

FILTER_SEEDS = (0, 1, 2)
OPUS_COMPOSED_N = 100
# Trees whose compositions never hold one item twice.
DISTINCT_ITEMS = {"gsm8k", "math500", "aime", "strategyqa"}
MAX_ATTEMPTS_FACTOR = 200


def read_attempts(run_dirs: list[Path], filter_name: str) -> tuple[dict, Counter]:
    """``{item id: {batching seed: [correct, ...]}}`` over every isolated-a run."""
    attempts: dict[str, dict[int, list[bool]]] = {}
    stats: Counter = Counter()
    for run_dir in run_dirs:
        seen: set = set()
        with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("filter") != filter_name:
                    continue
                key = (record.get("doc_id"), record.get("trial"))
                if key in seen:  # the same generation logged twice
                    stats["duplicate"] += 1
                    continue
                seen.add(key)
                components = record.get("component_ids") or []
                if len(components) != 1 or int(record.get("k") or 1) != 1:
                    stats["not_k1"] += 1
                    continue
                seed = record.get("batch_seed")
                if seed is None:
                    raise SystemExit(f"{run_dir}: a record carries no batch_seed")
                correct = str(record.get("verdict")).upper() == "CORRECT"
                attempts.setdefault(str(components[0]), {}).setdefault(
                    int(seed), []).append(correct)
                stats["attempts"] += 1
    return attempts, stats


def select_solved(attempts: dict, pool_ids: list[str]) -> list[str]:
    """The pool items correct under every one of ``FILTER_SEEDS``, in pool order."""
    missing = [i for i in pool_ids
               if any(s not in attempts.get(i, {}) for s in FILTER_SEEDS)]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(pool_ids)} pool items have no attempt under one of "
            f"the batching seeds {list(FILTER_SEEDS)} in the isolated-a run (first: "
            f"{missing[0]}); the filter cannot be applied.")
    return [i for i in pool_ids
            if all(all(attempts[i][s]) for s in FILTER_SEEDS)]


def draw(solved: list[dict], k: int, n: int, seed: str, *, distinct: bool,
         unordered: bool, build_fields) -> tuple[list[dict], dict]:
    """``n`` compositions of ``k`` solved items, category-balanced per position."""
    by_category: dict[str, list[dict]] = {}
    for item in solved:
        by_category.setdefault(item.get("category") or "uncategorized", []).append(item)
    categories = sorted(by_category)

    rng = random.Random(f"{seed}:{k}")
    rejected: Counter = Counter()
    seen: set = set()
    records: list[dict] = []
    attempts = 0
    max_attempts = max(n * MAX_ATTEMPTS_FACTOR, 100)
    while len(records) < n and attempts < max_attempts:
        attempts += 1
        chosen = [rng.choice(by_category[rng.choice(categories)]) for _ in range(k)]
        ids = [item["id"] for item in chosen]
        if distinct and len(set(ids)) != len(ids):
            rejected["repeated_item"] += 1
            continue
        key = frozenset(ids) if unordered else tuple(ids)
        if key in seen:
            rejected["duplicate_composition"] += 1
            continue
        seen.add(key)
        records.append({
            "id": f"comp-k{k}-{len(records):04d}",
            "stage": "compositional",
            "k": k,
            "strategy": "random",
            **build_fields(chosen),
            "rng_seed": f"{seed}:{k}",
        })
    if len(records) < n:
        print(f"[compose] k={k}: only {len(records)} of {n} compositions in {attempts} "
              f"draws ({dict(rejected)}); the solved pool is too small for more")
    rejected["attempts"] = attempts
    return records, dict(rejected)


def fields_builder(dataset: specs.Dataset):
    if dataset.family == "math":
        import sympy as sp

        from llm.core.math.composition import composition_fields

        variable, cache = sp.Symbol("x"), {}
        return lambda items: composition_fields(items, variable=variable, cache=cache)
    return dataset.composition_fields()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(specs.DATASETS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, help="compositions per k (default: from specs)")
    parser.add_argument("--k", type=int, nargs="+", help="default: the dataset's composed_ks")
    parser.add_argument("--seed", default="42", help="the per-k RNG key is '<seed>:<k>'")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing composed set that differs")
    parser.add_argument("--dry-run", action="store_true", help="build, write nothing")
    args = parser.parse_args(argv)

    dataset = specs.dataset(args.dataset)
    n = args.n or (OPUS_COMPOSED_N if args.model == specs.OPUS else dataset.composed_n)
    ks = sorted(args.k or dataset.composed_ks)
    pool_path = paths.pool(args.dataset)
    pool = read_jsonl(pool_path)
    by_id = {item["id"]: item for item in pool}

    run_dirs = paths.runs("isolated-a", args.dataset, args.model)
    if not run_dirs:
        raise SystemExit(f"no isolated-a run under "
                         f"{paths.model_runs_dir('isolated-a', args.dataset, args.model)}")
    attempts, read_stats = read_attempts(run_dirs, dataset.filter)
    solved_ids = select_solved(attempts, list(by_id))
    solved = [by_id[i] for i in solved_ids]
    print(f"[compose] {args.dataset} / {args.model}: {len(solved)} of {len(pool)} items "
          f"correct on seeds {list(FILTER_SEEDS)} ({', '.join(d.name for d in run_dirs)})")

    build_fields = fields_builder(dataset)
    rows: list[dict] = []
    rejections: dict[str, dict] = {}
    skipped: list[int] = []
    for k in ks:
        if k > len(solved):
            print(f"[compose] k={k}: skipped, only {len(solved)} solved items")
            skipped.append(k)
            continue
        records, rejections[str(k)] = draw(
            solved, k, n, args.seed, distinct=args.dataset in DISTINCT_ITEMS,
            unordered=dataset.family == "math", build_fields=build_fields)
        rows.extend(records)
        print(f"[compose] k={k}: {len(records)} compositions")

    manifest = {
        "dataset": args.dataset,
        "model": args.model,
        "pool": str(pool_path),
        "pool_sha256": sha256_of(pool_path),
        "pool_size": len(pool),
        "isolated_runs": [str(d) for d in run_dirs],
        "filter_name": dataset.filter,
        "criterion": f"all_correct_of_{len(FILTER_SEEDS)}",
        "filter_seeds": list(FILTER_SEEDS),
        "read": dict(read_stats),
        "solved_pool_size": len(solved),
        "solved_ids": solved_ids,
        "solved_by_category": dict(Counter(item.get("category") for item in solved)),
        "k_values": ks,
        "k_skipped": skipped,
        "n_per_k": n,
        "seed": args.seed,
        "counts_per_k": {str(k): sum(r["k"] == k for r in rows) for k in ks},
        "rejections": rejections,
        "command": " ".join(["python -m llm.compose",
                             *(argv if argv is not None else sys.argv[1:])]),
    }
    out = paths.batch("composed", args.dataset, args.model)
    if args.dry_run:
        print(f"[compose] dry run: would write {len(rows)} compositions to {out}")
        return 0
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if out.exists():
        if out.read_text(encoding="utf-8") == text:
            print(f"[compose] unchanged: {out}")
            return 0
        if not args.force:
            raise SystemExit(f"{out} exists and differs from this build; pass --force "
                             "to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[compose] wrote {len(rows)} compositions -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
