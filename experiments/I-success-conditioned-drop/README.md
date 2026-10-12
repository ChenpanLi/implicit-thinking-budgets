# I. Success-conditioned drop

Does a model still solve a task when the task shares a prompt with others, if it solved
that task in all three of its isolated attempts? It does not. Accuracy falls on every
dataset except OneWordQA, and the fall grows with the number k of tasks in a
composition. This folder draws the evidence and checks that neither truncation nor
answer parsing explains it:

- `gpqa-composed-sets-drop-k10.pdf`: the accuracy drop per model at k=10 on GPQA-Diamond
  (main text, Figure "success-conditioned-analysis", left). It also prints the number the
  text quotes: Claude Opus 5 loses 6.8 pp (95% CI 5.2 to 8.3), "about 7 pp".
- `gpqa-composed-sets-drop.pdf`: the same drop against k (same figure, right).
- `all-datasets-composed-sets-drop.pdf`: one panel per dataset (appendix, "Compositional
  degradation is observable over a wide range of datasets and models").
- `context-budget-usage.pdf`: the share of composed responses by how much of their output
  budget they used (main text, Figure "context-usage").
- `context-parsing.tex`: the appendix table "Context limits and parsing failures"
  (`tab:context-parsing`).

```bash
bash experiments/I-success-conditioned-drop/analyze.sh
```

The analysis needs no GPU. The composed runs it reads come from the GPU stage described
under "GPU runs" below, and the isolated runs from `experiments/1-isolated/`.

## The estimand

Each composed slot (one task at one position of one composition, on one trial) gives

    d = isolated-b accuracy of the task - 1/0 outcome of the task in the composition

and a point of the figures is the mean of d over the slots of one (dataset, model, k).
Tasks were selected on `isolated-a`, so the baseline is `isolated-b`, the independent
second pass; using `isolated-a` would score every selected task at about 1.0 by
construction. A composed answer that cannot be parsed counts as wrong, because the
isolated accuracies also divide by every trial. The k=1 point is `isolated-a` minus
`isolated-b` over the tasks that appear in the model's compositions, which shows how much
of the drop selection noise alone could produce.

The error bars are 95% percentile bootstrap intervals, 2000 resamples, seed 0. They
resample single slots, not tasks, as the paper's error bars do; clustering by task would
widen them. The draws are taken in the paper's order (models in legend order, per model
the composed k ascending, then k=1), with a fresh generator per figure.

## Inputs

| what | where |
| --- | --- |
| composed outcomes, tokens, verdicts | `runs/composed/<dataset>/<model>/` (all runs of a model are pooled) |
| isolated accuracy, selection pass | `runs/isolated-a/<dataset>/<model>/` |
| isolated accuracy, independent pass | `runs/isolated-b/<dataset>/<model>/` |
| tokens per character, for runs that recorded only characters | `runs/random-a/{integrals,mmlu,onewordqa}/<model>/` |
| prompt tokens for those runs | the models' tokenizers in the Hugging Face cache (never downloaded) |

Claude Opus 5 has composed runs on GPQA-Diamond only. Two pairs have none: MATH-500 with
Qwen3 4B Thinking and OneWordQA with R1 Qwen3 8B. The extraction filter is `boxed_last` for
Integrals and `lenient` elsewhere (`specs.DATASETS[...].filter`).

## Steps

1. `gpqa_drop_k10.py`, `gpqa_drop_by_k.py` and `all_datasets_drop.py` each load the
   composed and isolated runs through `drop.py` and draw one figure. Each also writes the
   points it plotted.
2. `context_records.py` reads every composed response once and writes one row per
   response: output tokens, the budget `min(max_gen_toks, max_model_len - prompt tokens)`,
   whether the response was truncated and whether it parsed. Most runs recorded exact
   token counts. The Integrals, MMLU-Redux and OneWordQA composed runs recorded only
   characters, so their output tokens are characters times the model's median
   tokens-per-character ratio, measured on responses of more than 2000 characters in the
   same dataset's `random-a` runs, and their prompt tokens are counted with the model's
   chat template. An exact response is truncated when it reached its budget; an estimated
   one when the 95th-percentile ratio puts it within 10% of its budget. It loads one
   tokenizer at a time; run it through the memory-slot wrapper on a shared login node.
3. `context_usage.py` bins the responses by budget share and draws the context figure.
4. `context_table.py` writes the appendix table: per dataset, the median response at the
   largest k and its median budget share, and the truncation and parsing-failure rates
   pooled over models and k, with the worst (model, k) cell in brackets.

## Outputs (`$RUN_DIR/results/I-success-conditioned-drop/`)

| file | content |
| --- | --- |
| `gpqa-composed-sets-drop-k10.pdf`, `.csv` | bar figure; one row per model at k=10 |
| `gpqa-composed-sets-drop.pdf`, `.csv` | line figure; one row per (model, k) |
| `all-datasets-composed-sets-drop.pdf`, `.csv` | appendix figure; one row per (dataset, model, k) |
| `context-records.csv` | one row per composed response: `out_tokens`, `out_tokens_hi`, `prompt_tokens`, `budget`, `budget_share`, `truncated`, `unparseable` |
| `context_budget_usage.csv` | per dataset and pooled: shares of responses per budget bin, truncated share, median tokens and budget share |
| `context-budget-usage.pdf` | the context figure |
| `context-parsing.tex` | the appendix table |

The drop CSVs have the columns `dataset, model, k, n, drop_pp, lo_pp, hi_pp`: n slots
(tasks for k=1), the mean drop and its interval in percentage points, positive when
accuracy falls. The figures plot their negative. The bar figure and the line figure draw
from separate generators, so the interval at GPQA-Diamond k=10 differs between their CSVs
in the second decimal.

The appendix table's AIME row reads the Qwen3 8B and Qwen3 14B runs at 131,072 tokens
(YaRN) at k=3, 6 and 10.

## GPU runs

`gpu.sh` builds and runs the composed sets. Stage 1 (`experiments/1-isolated`) has to
be done first, because the filter reads `isolated-a`.

```bash
bash experiments/I-success-conditioned-drop/gpu.sh --list
bash experiments/I-success-conditioned-drop/gpu.sh gsm8k qwen3-8b
DRY_RUN=1 bash experiments/I-success-conditioned-drop/gpu.sh gsm8k qwen3-8b
```

Each job runs two commands:

```bash
python -m llm.compose --dataset D --model M
python -m llm.run --stage composed --dataset D --model M --seed 42 [--model-variant V]
```

`llm.compose` keeps the items the model answered correctly under all three of batching
seeds 0, 1 and 2 of its `isolated-a` run. It then draws n compositions per k: each
position picks a category uniformly, then a solved item from that category. The k
values are `specs.composed_ks`: 3, 6, 10 and 20, or 3, 6 and 10 for GPQA-Diamond and
AIME. n is `specs.composed_n` (integrals 100, OneWordQA 1000, AIME 100, the rest 300;
claude-opus-5 100). The draw is keyed on `"42:<k>"`, so it is deterministic. The
manifest beside the set lists the solved ids.

`llm.run` then runs one lm-eval task per k. Integrals are graded one answer per term.

The third JOBS column names the context window that some of the paper's composed runs
used, from `variants:` in the model YAML:

- `ctx-49k`: Qwen3 4B Thinking, Qwen3.8 27B, R1 Qwen 7B and R1 Qwen3 8B on integrals,
  MMLU and OneWordQA.
- `ctx-64k`: R1 Llama 8B on the same three datasets.
- `yarn-131k`: Qwen3 8B and Qwen3 14B on AIME.

Two pairs have no composed run in the paper and are commented out: MATH-500 with
Qwen3 4B Thinking, and OneWordQA with R1 Qwen3 8B. claude-opus-5 is commented out
because it is billed.

One job needs one H100 (80 GB). Most composed runs took under an hour. MATH-500 took
about 5 hours, and AIME about 13 hours (up to 32 hours with the 131k window).

Output:

```
$RUN_DIR/batches/composed/<dataset>/<model>.jsonl (+ .manifest.json)
$RUN_DIR/runs/composed/<dataset>/<model>/<run-id>/
```
