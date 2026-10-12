# 2 · Random batching

Cuts the whole pool into random batches of k items and asks each batch as one prompt.
Every item lands in exactly one batch per k and seed, so each item is measured at every
k without any selection on whether the model can solve it. Combined with stage 1, this
gives each item's accuracy drop at each k. The fragility analyses (II, V) and the probe
labels (VI) read it.

k is `specs.random_ks`: 3, 6, 10 and 20, or 3, 6 and 10 on GPQA-Diamond. The last batch
of a pass is shorter when the pool does not divide by k. There are two passes:

| Stage | Batch | Engine seed |
| --- | --- | --- |
| `random-a` | one shuffle per (batching seed 0–4, k) | 42 |
| `random-b` | the same batch | 43 |

The shuffle depends only on the batching seed and k, never on the model, so every model
sees the same prompts.

## Run

```bash
bash experiments/2-random-batching/gpu.sh --list
bash experiments/2-random-batching/gpu.sh gsm8k qwen3-8b
DRY_RUN=1 bash experiments/2-random-batching/gpu.sh gsm8k qwen3-8b
```

Each job runs:

```bash
python -m llm.batches --stage random-a --dataset D --model M --strategy random --seeds 5
python -m llm.run     --stage random-a --dataset D --model M --seed 42
python -m llm.batches --stage random-b --dataset D --model M --strategy random --seeds 5
python -m llm.run     --stage random-b --dataset D --model M --seed 43
```

The jobs are 7 models × 8 datasets, each on one H100 (80 GB). Integrals are graded one
answer per term. In the paper's runs one pass took under an hour for StrategyQA and
GSM8K, and 3–5 hours for MATH-500 and AIME. The longest pass took about 15 hours
(MATH-500).

## Output

```
$RUN_DIR/batches/random-{a,b}/<dataset>/<model>.jsonl (+ .manifest.json)
$RUN_DIR/runs/random-{a,b}/<dataset>/<model>/<run-id>/
```

## Differences from the paper's runs

- `random-a`, gsm8k (7 models) and math500 (1 model): five trials assembled from
  batching seeds 0, 1, 2 plus a top-up with seeds 5 and 6. The batch file holds seeds
  0–2 only.
- `random-a`, gpqa-diamond (6 models): merged from a first run and a top-up.
- qwen3-8-27b on GPQA-Diamond: the paper has `random-a` of the rerun only, so that job
  is marked `a-only`. The first passes are kept in `random-{a,b}-v1`, and this script
  does not rebuild them.
- claude-opus-5 ran `random-a` only (commented out; billed).
