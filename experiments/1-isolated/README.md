# 1 · Isolated

Asks every pool item alone (k=1) to measure what each model gets right without
composition. Every later stage is measured against this baseline. It also decides
which items count as reliably solved for the composed sets of experiment I.

There are two passes, five trials each:

| Stage | Batch | Engine seed |
| --- | --- | --- |
| `isolated-a` | the whole pool once per batching seed 0–4 | 42 |
| `isolated-b` | the same batch | 43 |

The batches of the two passes are identical. Only the generation seed differs, so the
two passes are independent samples of the same questions. An item is *reliably solved*
when it is correct under batching seeds 0, 1 and 2 of `isolated-a`.

## Run

```bash
bash experiments/1-isolated/gpu.sh --list                 # the jobs
bash experiments/1-isolated/gpu.sh gsm8k qwen3-8b         # one job, on one GPU
DRY_RUN=1 bash experiments/1-isolated/gpu.sh gsm8k qwen3-8b
```

Each job builds and runs both passes:

```bash
python -m llm.batches --stage isolated-a --dataset D --model M --strategy control --seeds 5
python -m llm.run     --stage isolated-a --dataset D --model M --seed 42
python -m llm.batches --stage isolated-b --dataset D --model M --strategy control --seeds 5
python -m llm.run     --stage isolated-b --dataset D --model M --seed 43
```

The jobs are 7 models × 8 datasets. The three OLMo 3 checkpoints run pass a only, on
GPQA-Diamond, MATH-500 and MMLU (experiment VII). The claude-opus-5 line is commented
out because it is billed per request through OpenRouter. Integrals are graded one
answer per term, as in every stage.

One job needs one H100 (80 GB) for both passes. In the paper's runs one pass took
minutes for StrategyQA and OneWordQA, and about 1–3 hours for MMLU, MATH-500, AIME and
integrals. The longest pass took 7.6 hours (MMLU, R1 Llama 8B).

## Output

```
$RUN_DIR/batches/isolated-{a,b}/<dataset>/<model>.jsonl (+ .manifest.json)
$RUN_DIR/runs/isolated-{a,b}/<dataset>/<model>/<run-id>/
```

## Differences from the paper's runs

Some of the paper's runs were five trials assembled from a first run plus a top-up run
with other batching seeds. Their batch files hold only the first run's seeds. A clean
rerun of this stage uses seeds 0–4 throughout, so it reproduces the design but not those
exact trials:

- `isolated-a`, gsm8k, all 7 models: seeds 0, 1, 2 plus 5, 6.
- `isolated-a`, math500, 2 models: seeds 0, 1, 2 plus 5, 6.
- `isolated-a`, gpqa-diamond, 6 models: seeds 0, 1, 2 plus 6, 7.
- `isolated-b`, gpqa-diamond, 6 models: seeds 3, 4, 5 plus 8, 9.

qwen3-8-27b on GPQA-Diamond was rerun later (engine seeds 44 and 45, batching seeds 0–4
and 5–9). The first passes, from which its fragility-aware batching was built, are kept
in the `isolated-*-v1` stages. This script does not rebuild those.
