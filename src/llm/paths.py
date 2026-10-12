"""Every path the pipeline reads or writes. Nothing else in the repository builds one.

All data lives under one directory, ``RUN_DIR``, set in ``.env`` at the repository root
(or in the environment). Its layout:

    $RUN_DIR/datasets/<dataset>.jsonl                      item pools
    $RUN_DIR/batches/<stage>/<dataset>/<model>.jsonl         what a run is asked
    $RUN_DIR/runs/<stage>/<dataset>/<model>/<run-id>/        what a run answered
    $RUN_DIR/probe/                                          fragility probe store
    $RUN_DIR/results/<experiment>/                           figures, tables, CSVs

A stage is one kind of run (``isolated-a``, ``random-b``, ``composed``, ...); see
``STAGES``. Analyses read every run under ``runs/<stage>/<dataset>/<model>/`` and pool
them, so that directory must hold exactly the runs that belong to the result.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    """Read ``KEY=VALUE`` lines from ``<repo>/.env`` without overriding the environment."""
    env = REPO_ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

STAGES = {
    "isolated-a": "every item alone (k=1), batching seeds 0-4, generation seed 42",
    "isolated-b": "the same batches again, generation seed 43",
    "random-a": "the pool split into random batches of each k, batching seeds 0-4, generation seed 42",
    "random-b": "the same batches again, generation seed 43",
    # The first isolated and random passes of qwen3-8-27b on gpqa-diamond. The model was
    # later rerun from scratch (the regular stages hold the rerun), but its fragility-aware
    # runs, and so the batching table and probe labels, were built from these passes.
    # See specs.baseline_stages.
    "isolated-a-v1": "first pass of isolated-a (qwen3-8-27b on gpqa-diamond only)",
    "isolated-b-v1": "first pass of isolated-b (qwen3-8-27b on gpqa-diamond only)",
    "random-a-v1": "first pass of random-a (qwen3-8-27b on gpqa-diamond only)",
    "random-b-v1": "first pass of random-b (qwen3-8-27b on gpqa-diamond only)",
    "composed": "compositions of confidently solved items (3/3 in isolated-a)",
    "per-term-trace": "dummy-tail batches with Q-tagged items, thinking kept",
    "budget-forcing": "every item alone, thinking capped at its per-term budget",
    "fragility-measured": "batches sized by measured fragility",
    "fragility-probe": "batches sized by the probe's predicted fragility",
}


def run_dir() -> Path:
    value = os.environ.get("RUN_DIR")
    if not value:
        raise SystemExit("RUN_DIR is not set: copy .env.example to .env and set it")
    return Path(value).expanduser()


def pool(dataset: str) -> Path:
    return run_dir() / "datasets" / f"{dataset}.jsonl"


def shipped_pool(dataset: str) -> Path:
    """The pools built by us and committed (integrals, onewordqa)."""
    return REPO_ROOT / "datasets" / dataset / "atomic.jsonl"


def _check_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; one of {sorted(STAGES)}")


def batch(stage: str, dataset: str, model: str, part: str | None = None) -> Path:
    """A batch file. ``part`` names one of several files of a (stage, dataset, model)."""
    _check_stage(stage)
    base = run_dir() / "batches" / stage / dataset
    return base / model / f"{part}.jsonl" if part else base / f"{model}.jsonl"


def model_runs_dir(stage: str, dataset: str, model: str) -> Path:
    _check_stage(stage)
    return run_dir() / "runs" / stage / dataset / model


def runs(stage: str, dataset: str, model: str) -> list[Path]:
    """Every finished run of a (stage, dataset, model), oldest first."""
    root = model_runs_dir(stage, dataset, model)
    if not root.is_dir():
        return []
    return sorted(d for d in root.iterdir() if (d / "run.json").exists())


def runs_glob(stage: str, dataset: str, model: str = "*") -> str:
    """A glob over every run of a (stage, dataset), for loaders that take patterns."""
    _check_stage(stage)
    return str(run_dir() / "runs" / stage / dataset / model / "*")


def models_with_runs(stage: str, dataset: str) -> list[str]:
    root = run_dir() / "runs" / stage / dataset
    return sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []


def probe(*parts: str) -> Path:
    return run_dir().joinpath("probe", *parts)


def results(experiment: str) -> Path:
    out = run_dir() / "results" / experiment
    out.mkdir(parents=True, exist_ok=True)
    return out
