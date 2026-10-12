"""How much of its output budget every composed response used, and whether it parsed.

One row per composed attempt (a composition at one k on one trial) with its output tokens,
its budget and two flags, written to context-records.csv for context_usage.py and
context_table.py. The budget is what was left for the response:

    budget = min(max_gen_toks, max_model_len - prompt tokens)

Most runs record exact token counts. The integrals, MMLU-Redux and OneWordQA composed runs
recorded only characters, so their tokens are estimated from the characters with the
model's own token-per-character ratio, measured on that dataset's random-a runs (which
recorded both), and their prompt tokens are counted with the model's tokenizer. Such a
response counts as truncated when the 95th-percentile ratio puts it within 10% of its
budget; an exact count counts when it reached the budget.
"""
import json

import pandas as pd

from llm import paths, specs
from llm.core.analysis.item_accuracy import _records, model_label, run_dirs
from llm.core.serving.token_stash import normalize_usage
from llm.plotting import MODEL_COLORS

KS = (3, 6, 10, 20)
MIN_CHARS = 2000      # the ratio is measured on responses at least this long
_TOKENIZERS = {}
_PROMPT_TOKENS = {}


def token_ratio(dataset: str, model: str) -> tuple[float, float]:
    """Median and 95th percentile of tokens per character in the random-a runs."""
    filter_name = specs.dataset(dataset).filter
    ratios = []
    for run_dir in paths.runs("random-a", dataset, model):
        for record in _records(run_dir, filter_name):
            tokens = normalize_usage(record.get("token_usage"))["completion"]
            chars = (record.get("raw") or {}).get("chars")
            if tokens and chars and chars > MIN_CHARS:
                ratios.append(tokens / chars)
    if not ratios:
        raise SystemExit(f"no random-a run of {dataset}/{model} records both tokens and chars")
    median, p95 = pd.Series(ratios).quantile([0.5, 0.95])
    return median, p95


def prompt_tokens(model_id: str, prompt: str) -> int:
    """The prompt's length under the model's chat template, as the server would see it."""
    key = (model_id, prompt)
    if key not in _PROMPT_TOKENS:
        if model_id not in _TOKENIZERS:
            from transformers import AutoTokenizer
            _TOKENIZERS[model_id] = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        ids = _TOKENIZERS[model_id].apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
        if isinstance(ids, dict) or hasattr(ids, "keys"):
            ids = ids["input_ids"]
        _PROMPT_TOKENS[key] = len(ids)
    return _PROMPT_TOKENS[key]


def scan(dataset: str) -> list[dict]:
    """One row per composed attempt of every model of ``dataset``."""
    filter_name = specs.dataset(dataset).filter
    rows, ratios = [], {}
    for run_dir in run_dirs(paths.runs_glob("composed", dataset)):
        meta = json.loads((run_dir / "run.json").read_text())
        model = model_label(meta)
        if model not in MODEL_COLORS:
            continue
        spec = meta.get("model", {})
        max_len = spec.get("max_model_len")
        max_gen = meta.get("gen_kwargs", {}).get("max_gen_toks", spec.get("max_tokens"))
        for record in _records(run_dir, filter_name):
            k = record.get("k")
            if k not in KS:
                continue
            usage = normalize_usage(record.get("token_usage"))
            exact = usage["completion"] is not None
            if exact:
                tokens = tokens_hi = usage["completion"]
                prompt = usage["prompt"]
            else:
                if model not in ratios:
                    ratios[model] = token_ratio(dataset, model)
                chars = (record.get("raw") or {}).get("chars", 0)
                tokens, tokens_hi = chars * ratios[model][0], chars * ratios[model][1]
                prompt = prompt_tokens(spec["model_id"], record["prompt"])
            budget = min(max_gen, max_len - prompt)
            truncated = tokens >= budget - 1 if exact else tokens_hi / budget >= 0.90
            rows.append({
                "dataset": dataset, "model": model, "run_id": meta.get("run_id", run_dir.name),
                "k": k, "composition_id": record.get("problem_id") or record.get("doc_id"),
                "trial": record.get("trial"), "exact_tokens": exact,
                "out_tokens": tokens, "out_tokens_hi": tokens_hi, "prompt_tokens": prompt,
                "max_model_len": max_len, "max_gen_toks": max_gen, "budget": budget,
                "budget_share": tokens / budget, "truncated": truncated,
                "unparseable": record.get("verdict") == "UNPARSEABLE",
            })
        _TOKENIZERS.clear()  # one model per run; keeps the memory of the scan small
        _PROMPT_TOKENS.clear()
    print(f"{dataset}: {len(rows)} composed attempts", flush=True)
    return rows


def main() -> None:
    out = paths.results("I-success-conditioned-drop") / "context-records.csv"
    rows = [row for dataset in specs.DATASETS for row in scan(dataset)]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
