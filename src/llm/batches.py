"""Build one batch file: the prompts a run of one (stage, dataset, model) is asked.

    python -m llm.batches --stage isolated-a --dataset gsm8k --model qwen3-8b \\
        --strategy control --seeds 5
    python -m llm.batches --stage random-a --dataset gsm8k --model qwen3-8b \\
        --strategy random --seeds 5                       # k from specs.random_ks
    python -m llm.batches --stage per-term-trace --dataset gsm8k --model qwen3-8b \\
        --strategy dummy-tail --k 3 --seed-list 0 1 2 3 4 5 6 7 8 9
    python -m llm.batches --stage fragility-probe --dataset gsm8k --model qwen3-8b \\
        --strategy predicted-calibration --fallback-k 1 --seeds 5 \\
        --stats predicted-largest-k=<selection.csv> --param column=<k column>

writes ``paths.batch(stage, dataset, model)`` and a ``.manifest.json`` beside it
recording the strategy, seeds, parameters, the pool's SHA-256 and the command. A
per-term-trace batch is one file per prompt size and seed range,
``batches/per-term-trace/<dataset>/<model>/k<K>-s<A>-<B>.jsonl``.

Every pass of a strategy partitions the whole pool (see ``llm.core.batching``), and
the batching depends only on the strategy, the seed and k, never on the model: two
models get the same prompts at the same seed. The rows of a file are deterministic,
so rebuilding one gives the same bytes; an existing file that differs is only
replaced with ``--force``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm import paths, specs
from llm.core.batching import StatsBundle, engine, registry

DEFAULT_SEEDS = 5


def record_builder(dataset: specs.Dataset):
    """The function turning one group of pool items into a row's domain fields."""
    if dataset.family == "math":
        import sympy as sp

        from llm.core.math.composition import composition_fields

        variable = sp.Symbol("x")
        cache: dict = {}  # each atom is integrated once per build, not once per pass

        def build(items):
            return composition_fields(items, variable=variable, cache=cache)

        return build
    return dataset.composition_fields()


def key_value(text: str, what: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not name.strip():
        raise SystemExit(f"{what} must be written NAME=VALUE, got {text!r}")
    return name.strip(), value.strip()


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    seeds = list(args.seed_list) if args.seed_list else list(range(args.seeds))
    if not seeds:
        raise SystemExit("--seeds must be at least 1")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"repeated seed in {seeds}; each seed is one batching of the pool")
    return seeds


def resolve_params(args: argparse.Namespace, dataset: specs.Dataset) -> dict:
    params: dict = {}
    k = args.k
    if k is None and args.strategy in ("random", "predicted-calibration"):
        k = dataset.random_ks
    if k is not None:
        params["k"] = list(k)
    if args.fallback_k is not None:
        params["fallback_k"] = args.fallback_k
    for entry in args.param:
        name, value = key_value(entry, "--param")
        params[name] = value
    return params


def output_path(args: argparse.Namespace, params: dict, seeds: list[int]) -> Path:
    """The batch file; per-term-trace files are named by prompt size and seed range."""
    if args.stage != "per-term-trace":
        return paths.batch(args.stage, args.dataset, args.model)
    ks = params.get("k") or []
    if len(ks) != 1:
        raise SystemExit("a per-term-trace batch is one prompt size: pass exactly one --k")
    if sorted(seeds) != list(range(min(seeds), max(seeds) + 1)):
        raise SystemExit(f"a per-term-trace batch covers a contiguous seed range, got {seeds}")
    part = f"k{ks[0]}-s{min(seeds)}-{max(seeds)}"
    return paths.batch(args.stage, args.dataset, args.model, part=part)


def check_model(model: str) -> None:
    config_dir = paths.REPO_ROOT / "src" / "llm" / "config" / "models"
    if config_dir.is_dir() and not (config_dir / f"{model}.yaml").exists():
        known = ", ".join(sorted(p.stem for p in config_dir.glob("*.yaml")))
        raise SystemExit(f"unknown model {model!r}; known: {known}")


def summarise(result: engine.BuildResult) -> None:
    manifest = result.manifest
    print(f"[batch] rows     : {manifest['n_rows']} across {manifest['n_passes']} "
          f"pass(es) of {manifest['n_pool_items']} pool items")
    for entry in manifest["passes"]:
        sizes = ", ".join(f"k={k}x{n}" for k, n in entry["k_counts"].items())
        scope = ("" if entry.get("covers_whole_pool", True)
                 else f" over {entry['n_items_covered']} item(s)")
        print(f"[batch]   seed {entry['seed']} pass {entry['label']:<8} "
              f"{entry['n_batches']:>5} batches{scope}  ({sizes})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=sorted(paths.STAGES))
    parser.add_argument("--dataset", required=True, choices=sorted(specs.DATASETS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--strategy", required=True,
                        choices=["control", "random", "dummy-tail", "predicted-calibration"])
    parser.add_argument("--k", type=int, nargs="+",
                        help="batch sizes, one full pass of the pool each (default for "
                             "random and predicted-calibration: the dataset's random_ks)")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS,
                        help=f"batching seeds 0..N-1 (default {DEFAULT_SEEDS})")
    parser.add_argument("--seed-list", type=int, nargs="+",
                        help="explicit batching seeds instead of 0..N-1")
    parser.add_argument("--fallback-k", type=int,
                        help="predicted-calibration: batch size of the items the "
                             "selection cannot speak for (1 asks them alone)")
    parser.add_argument("--stats", action="append", default=[], metavar="NAME=CSV",
                        help="a statistics table the strategy reads, repeatable")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                        help="any other strategy parameter, repeatable")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing batch file that differs")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and check everything, write nothing")
    args = parser.parse_args(argv)

    dataset = specs.dataset(args.dataset)
    check_model(args.model)
    registry.load_builtin()
    strategy = registry.get(args.strategy)
    seeds = resolve_seeds(args)
    params = resolve_params(args, dataset)
    out = output_path(args, params, seeds)

    pool_path = paths.pool(args.dataset)
    if not pool_path.exists():
        raise SystemExit(f"item pool not found: {pool_path}; build it with "
                         f"`python -m llm.pools {args.dataset}`")
    pool = engine.read_jsonl(pool_path)
    tables = {}
    for entry in args.stats:
        name, path = key_value(entry, "--stats")
        tables[name] = Path(path)

    print(f"[batch] {args.stage} / {args.dataset} / {args.model}: {strategy.name}, "
          f"seeds {seeds}, params {params or '(none)'}")
    print(f"[batch] pool     : {pool_path} ({len(pool)} items)")
    result = engine.build(
        strategy=strategy, pool=pool, pool_path=pool_path, model=args.model,
        seeds=seeds, params=params, stats=StatsBundle(tables=tables),
        record_builder=record_builder(dataset), experiment=dataset.tree,
        command=" ".join(["python -m llm.batches", *(argv if argv is not None
                                                     else sys.argv[1:])]),
    )
    result.manifest.update({"dataset": args.dataset, "run_stage": args.stage})
    summarise(result)
    if args.dry_run:
        print(f"[batch] dry run  : would write {out}")
        return 0

    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result.rows)
    if out.exists():
        if out.read_text(encoding="utf-8") == text:
            print(f"[batch] unchanged: {out}")
            return 0
        if not args.force:
            raise SystemExit(
                f"{out} exists and differs from this build. Runs are keyed on it, so "
                "rebuilding it in place would change what an existing run measured; "
                "pass --force if that is what you mean.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(result.manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[batch] wrote    : {out}")
    print(f"[batch] manifest : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
