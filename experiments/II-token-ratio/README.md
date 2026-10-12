# II. Token ratio

How much thinking does a task get when it shares a prompt with k-1 others? The model
names a tag whenever it turns to a task, so each task's share of the thinking trace can
be measured. This folder turns those per-task counts into the token ratio
r_i(k) = T_i(k) / T_i(1) and produces:

- `token-ratio-integrals-plus-benchmarks.pdf`: the main-text figure (Section 4, Qwen3 4B
  Thinking, MMLU-Redux large plus GPQA-Diamond, Integrals, MATH-500 and GSM8K).
- `<model>-token-ratio-all-datasets.pdf`: one appendix figure per open-weight model (7).
- The power-law numbers: the median and range of the slope m and the mean R^2 in each
  appendix caption, and in the main text the median m = -0.54 (range [-0.88, -0.04]),
  the median R^2 of 0.91 and the per-model mean R^2 of 0.55 to 0.94.
- The tagging coverage in the appendix section "Measuring per-task thinking length":
  99.5% of tagged traces name every task, 92.9% on GPQA-Diamond, and 61.5% for R1 Qwen 7B
  on GPQA-Diamond at k ≥ 3.

```bash
bash experiments/II-token-ratio/analyze.sh
```

The analysis needs no GPU. The per-term-trace runs it reads are produced by the GPU
stage described under "GPU runs" below.

## Inputs

| what | where |
| --- | --- |
| per-task section tokens, visits and correctness, per response | `runs/per-term-trace/<dataset>/<model>/k<K>-s<A>-<B>/sections.jsonl` |
| isolated thinking tokens (output minus re-tokenized answer minus one) and accuracy, 5 trials | `runs/isolated-a/<dataset>/<model>/` |
| isolated output tokens from the independent pass, 5 trials | `runs/isolated-b/<dataset>/<model>/` |

The datasets are the six per-term-trace ones: Integrals, MMLU-Redux, OneWordQA,
GPQA-Diamond, GSM8K and MATH-500. k counts the dummy task appended to each prompt (k=2 is
one task plus the dummy). Only actual positions count; the dummy and the fillers that pad
one batch per seed are left out.

## Two definitions of the ratio

The figures and the fits use different denominators. Both are the ones the paper uses.

- **Figures:** a task's mean section tokens at k over its mean isolated *output* tokens
  in `isolated-b`. The length bins are quintiles of the same `isolated-b` output tokens,
  taken over the (k, task) rows: Low is the first quintile, Med. the middle three, High
  the last. Each bin gets a power law through (1, 1), labelled with its law, ∝ k^m,
  written horizontally beside the line in the line's colour (every panel of the
  per-model figures, and the large MMLU-Redux panel of the main figure). These m belong to the drawn lines, so they
  differ from the per-dataset fits in the captions. Only the ten k of the standard
  ladder (2, 3, 4, 6, 8, 10, 12, 15, 18, 20) are drawn.
- **Fits behind the numbers:** a task's mean section tokens at k over its mean isolated
  *thinking* tokens in `isolated-a`. Tasks with no isolated thinking are dropped. A
  point is the mean ratio over all tasks at that k, and every k that was run enters,
  including the 19 values of k of Qwen3 4B Thinking on MMLU-Redux. Each (model, dataset)
  gets one fit, log r = m log k, with no intercept. R^2 is the centred one, so it goes
  negative when the line through (1, 1) fits worse than a flat line.

One pair reads two sets of isolated runs. For Qwen3.8 27B on GPQA-Diamond, the paper's
figure uses the rerun (`isolated-a`, `isolated-b`), while its caption numbers come from
the first pass in `isolated-a-v1` and `isolated-b-v1`. `power_law_fits.py` reads the
first pass for that pair through `trace_data.fit_solo` (cached as
`solo-first-pass.csv`); the figures read the rerun. With the rerun, that caption would read median m = -0.14 and mean R^2 = 0.71.

## Steps

1. `trace_data.py` loads each (dataset, model) once and caches three CSVs under
   `$RUN_DIR/results/II-token-ratio/cache/<dataset>/<model>/`: per (k, task) section
   tokens and accuracy (`items_k.csv`), per response the counts of named positions
   (`attempts.csv`), and `load_solo` (`solo.csv`). Re-tokenizing the isolated answers
   takes minutes per pair, so every later step reads the cache. `--refresh` rebuilds it.
   It loads one pair at a time, so on a shared machine run it through a memory-slot
   wrapper rather than in parallel.
2. `token_ratio_main.py` draws the main-text figure (`--model` picks another model).
3. `token_ratio_per_model.py` draws the seven appendix figures and writes every plotted
   point to `token-ratio-points.csv`.
4. `power_law_fits.py` fits all 7 models × 6 datasets and prints the per-model and
   overall summaries of the captions and the main text.
5. `tag_coverage.py` counts, among tagged traces (thinking closed, at least one section),
   those in which every actual position was named at least once, and prints the three
   shares of the appendix.

## Outputs (`$RUN_DIR/results/II-token-ratio/`)

| file | content |
| --- | --- |
| `token-ratio-integrals-plus-benchmarks.pdf` | main-text figure |
| `<model>-token-ratio-all-datasets.pdf` | appendix figures, one per model |
| `token-ratio-points.csv` | every plotted point: dataset, model, bin (Low, Med., High, all), k, ratio, tasks |
| `power-law-fits.csv` | one fit per (dataset, model): number and range of k, slope, R^2 |
| `tag-coverage.csv` | per (dataset, model, k): tagged traces, share naming every task |
| `cache/` | the per-pair CSVs of step 1 |

## GPU runs

`gpu.sh` reproduces the per-term-trace runs: 42 jobs, one per (dataset, model), each
needing one GPU.

```bash
bash experiments/II-token-ratio/gpu.sh --list                 # the jobs
bash experiments/II-token-ratio/gpu.sh mmlu qwen3-8b          # one job
DRY_RUN=1 bash experiments/II-token-ratio/gpu.sh mmlu qwen3-8b
```

A job loops over the dataset's prompt sizes (k = 2 3 4 6 8 10 12 15 18 20, up to 10 on
gpqa-diamond). For each k it runs three steps:

1. `llm.batches --strategy dummy-tail` builds batching seeds 0-9 into
   `batches/per-term-trace/<dataset>/<model>/k<K>-s0-9.jsonl`. Each batch holds k-1 pool
   items plus a dummy item in the last position.
2. `llm.run --stage per-term-trace --section-markers --keep-thinking` asks them. It labels
   the items Q01, Q02, ..., asks the model to name an item whenever it works on it, and
   keeps the thinking in the trace.
3. `llm.sections` cuts every trace at those labels and writes `sections.jsonl` beside it.

A finished run, or a run that already has its `sections.jsonl`, is skipped. That makes a
job safe to restart.

Some of the paper's runs were split into seed chunks (`k2-s0-4`, `k2-s5-9`) so they fit
on short allocations. The analyses pool every run of a (dataset, model, k), so one run
over seeds 0-9 gives the same data. r1-llama-distill-8b ran with generation capped at
32768 tokens on every dataset except mmlu, so its jobs pass `--model-variant gen-32k`.
