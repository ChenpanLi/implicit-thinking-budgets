# VII. OLMo 3 post-training stages

Does post-training change implicit budgeting? OLMo 3 7B Think is released after each of
its three post-training stages (SFT, DPO, RLVR), so the same model can be compared before
and after preference tuning and reinforcement learning. This folder produces Section 5's
figure pair and its three correlations:

- `olmo3-gpqa-diamond-token-ratio-by-k.pdf`: token ratio against composition size.
- `olmo3-gpqa-diamond-token-ratio-vs-accuracy-drop.pdf`: accuracy drop against the
  token ratio.
- The disattenuated Pearson rho between the stages' accuracy drops on GPQA-Diamond,
  MATH-500 and MMLU-Redux: 1.03 ± 0.35 (SFT, DPO), 0.69 ± 0.23 (SFT, RLVR) and
  0.89 ± 0.26 (DPO, RLVR).

```bash
bash experiments/VII-olmo-post-training/analyze.sh
```

The analysis needs no GPU. It reads the OLMo runs of two stages: `isolated-a` and
`per-term-trace`, on GPQA-Diamond, MATH-500 and MMLU-Redux.

## Inputs

| what | where |
| --- | --- |
| per-term thinking tokens and per-position correctness | `runs/per-term-trace/<dataset>/olmo3-7b-think-<stage>/k<K>-s<A>-<B>/` |
| isolated thinking tokens, accuracy and per-trial outcomes (5 trials) | `runs/isolated-a/<dataset>/olmo3-7b-think-<stage>/` |

OLMo has no `isolated-b` run, so `load_solo` warns that its `split1_output_tokens` come
from `isolated-a`. Nothing here reads that column.

## Steps

1. `olmo_data.py` loads each (dataset, stage) once and caches three tables under
   `$RUN_DIR/results/VII-olmo-post-training/cache/<dataset>/<model>/`: the actual
   positions of the per-term-trace runs, `load_solo`, and every isolated trial with its
   outcome. Building them re-tokenizes the isolated answers, which takes minutes per
   pair; later steps read the cache. `--refresh` rebuilds it.
2. `token_ratio_figures.py` computes, per task and k, the ratio of its mean section
   tokens to its mean isolated thinking tokens, and its accuracy change in points.
   Echo reads and unreadable answer lists are left out of accuracy, tasks with no
   isolated thinking are dropped, and a k enters only once all 10 seeds finished. A
   point is the mean over tasks. Figure A fits a power law through (1, 1) per stage and
   writes its law, ∝ k^m, beside the line as the token-ratio figures do (DPO's line is
   dashed, since it meets RLVR's); figure B plots the accuracy change against the token reduction.
   `--dataset math500` or `--dataset mmlu` draws the same pair for another dataset.
3. `fragility_rho.py` correlates the stages' per-task accuracy drops. With one pass per
   stage, the two "passes" a disattenuation needs are random halves of each task's
   trials (isolated 2|3, composed 5|5). Each correlation is Fisher-z averaged over 200
   splits, r* = r / sqrt(r_AA r_BB) is averaged over every (dataset, k), and the ± is half
   the width of a 95% interval from 5000 bootstrap resamples of each dataset's tasks.
   Unparsed answers count as wrong. With the default `--boot 5000`, the run takes
   about 20 minutes; `--boot 20` gives the point estimates quickly (written to
   `fragility-rho-boot20.json`).

## Outputs (`$RUN_DIR/results/VII-olmo-post-training/`)

| file | content |
| --- | --- |
| `olmo3-gpqa-diamond-token-ratio-by-k.pdf` | Section 5, left |
| `olmo3-gpqa-diamond-token-ratio-vs-accuracy-drop.pdf` | Section 5, right |
| `olmo3-gpqa-diamond-token-ratio-points.csv` | the points of both figures: k, ratio, accuracy change (pp), tasks, seeds, stage |
| `fragility-rho.json` | rho, the 95% interval and its half width per stage pair, plus per-dataset rho and reliabilities per k |

The bootstrap of `fragility_rho.py` draws from one random generator (seed 0) in a fixed
order, so the result depends on the order of tasks and of isolated trials; the cache
keeps trials in trace order.

## GPU runs

`gpu.sh` reproduces the OLMo runs: 9 jobs, one per (dataset, model) over gpqa-diamond,
math500 and mmlu and the SFT, DPO and RLVR checkpoints.

```bash
bash experiments/VII-olmo-post-training/gpu.sh --list
bash experiments/VII-olmo-post-training/gpu.sh math500 olmo3-7b-think-dpo
```

A job runs two existing stage jobs back to back:

- `1-isolated/gpu.sh <dataset> <model> a-only` asks every item alone, five trials with
  engine seed 42 (isolated-a).
- `II-token-ratio/gpu.sh <dataset> <model>` runs the per-term traces at every k and cuts
  them into sections.

The OLMo checkpoints have no isolated-b pass.
