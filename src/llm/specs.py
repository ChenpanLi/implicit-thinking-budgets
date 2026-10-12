"""The eight datasets and the models of the paper, in one table.

Everything that differs between datasets is here: which task layer under ``lm_tasks/``
renders and grades it, which family builds a composition's fields, the extraction
filter the analyses read, and the k values each stage uses.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field

MODELS = ["qwen3-4b-thinking", "qwen3-8b", "qwen3-14b", "qwen3-8-27b",
          "r1-qwen-distill-7b", "r1-llama-distill-8b", "r1-qwen3-8b"]
OPUS = "claude-opus-5"
OLMO = ["olmo3-7b-think-sft", "olmo3-7b-think-dpo", "olmo3-7b-think-rlvr"]
OLMO_DATASETS = ["gpqa-diamond", "math500", "mmlu"]

MODEL_NAMES = {
    "qwen3-4b-thinking": "Qwen3 4B Thinking", "qwen3-8b": "Qwen3 8B",
    "qwen3-14b": "Qwen3 14B", "qwen3-8-27b": "Qwen3.8 27B",
    "r1-qwen-distill-7b": "R1 Qwen 7B", "r1-llama-distill-8b": "R1 Llama 8B",
    "r1-qwen3-8b": "R1 Qwen3 8B", "claude-opus-5": "Claude Opus 5",
    "olmo3-7b-think-sft": "OLMo 3 SFT", "olmo3-7b-think-dpo": "OLMo 3 DPO",
    "olmo3-7b-think-rlvr": "OLMo 3 RLVR",
}

# Per-term trace: k counts the trailing dummy item; 10 batching seeds per k.
TRACE_KS = [2, 3, 4, 6, 8, 10, 12, 15, 18, 20]
TRACE_SEEDS = 10
# The main token-ratio figure (Qwen3 4B Thinking on MMLU-Redux) was traced at every k.
TRACE_KS_BY_PAIR = {("mmlu", "qwen3-4b-thinking"): list(range(2, 21))}


def trace_ks(dataset_name: str, model: str) -> list[int]:
    """The prompt sizes k the per-term trace of a (dataset, model) pair runs at."""
    return TRACE_KS_BY_PAIR.get((dataset_name, model)) or dataset(dataset_name).trace_ks


@dataclass(frozen=True)
class Dataset:
    name: str
    tree: str                 # directory under lm_tasks/
    task_prefix: str          # lm-eval task names: <prefix>_evaluation, <prefix>_compositional_k<k>
    family: str               # llm.core.<family>.composition builds a composition's fields
    title: str                # as printed in the paper
    composed_n: int           # compositions per k in the composed sets
    composed_ks: list[int] = field(default_factory=lambda: [3, 6, 10, 20])
    random_ks: list[int] = field(default_factory=lambda: [3, 6, 10, 20])
    trace_max_k: int = 20
    filter: str = "lenient"   # extraction ensemble the analyses read
    per_term: bool = False    # integrals: grade one answer per term

    @property
    def trace_ks(self) -> list[int]:
        return [k for k in TRACE_KS if k <= self.trace_max_k]

    def composition_fields(self):
        return importlib.import_module(f"llm.core.{self.family}.composition").composition_fields

    def task_utils(self):
        """The task layer module lm-eval loads for this dataset."""
        import sys

        from llm.paths import REPO_ROOT
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        return importlib.import_module(f"lm_tasks.{self.tree}.utils")


DATASETS: dict[str, Dataset] = {d.name: d for d in [
    Dataset("integrals", "integrals", "indefinite", "math", "Integrals", 100,
            filter="boxed_last", per_term=True),
    Dataset("onewordqa", "onewordqa", "onewordqa", "oneword", "OneWordQA", 1000),
    Dataset("gpqa-diamond", "gpqa", "gpqa", "mcq", "GPQA-Diamond", 300,
            composed_ks=[3, 6, 10], random_ks=[3, 6, 10], trace_max_k=10),
    Dataset("gsm8k", "gsm8k", "gsm8k", "shortmath", "GSM8K", 300),
    Dataset("math500", "math500", "math500", "shortmath", "MATH-500", 300),
    Dataset("mmlu", "mmlu", "mmlu", "mcq", "MMLU-Redux", 300),
    Dataset("strategyqa", "strategyqa", "strategyqa", "oneword", "StrategyQA", 300),
    Dataset("aime", "aime", "aime", "shortmath", "AIME", 100, composed_ks=[3, 6, 10]),
]}

# The datasets the per-term trace and budget forcing were run on.
TRACE_DATASETS = ["integrals", "mmlu", "onewordqa", "gpqa-diamond", "gsm8k", "math500"]
# The datasets of the fragility-aware batching tables (AIME falls back to k=1 throughout).
BATCHING_DATASETS = ["gpqa-diamond", "gsm8k", "math500", "integrals", "onewordqa",
                     "strategyqa", "mmlu"]

# Pairs whose fragility-aware runs were built from an earlier isolated/random pass than the
# one in the regular stages (the model was rerun afterwards). The probe labels and the
# batching table read these stages for the pair, so that both arms compare like with like.
FIRST_PASS = {("gpqa-diamond", "qwen3-8-27b")}


def baseline_stages(dataset_name: str, model: str) -> dict[str, str]:
    """The isolated/random stages the fragility-aware batching of a pair was built from."""
    suffix = "-v1" if (dataset_name, model) in FIRST_PASS else ""
    return {s: s + suffix for s in ("isolated-a", "isolated-b", "random-a", "random-b")}


def dataset(name: str) -> Dataset:
    try:
        return DATASETS[name]
    except KeyError:
        raise SystemExit(f"unknown dataset {name!r}; one of {sorted(DATASETS)}") from None
