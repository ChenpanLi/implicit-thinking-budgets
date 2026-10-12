# IV. Position effect

Does a task get less thinking, and lose more accuracy, the later it sits in a
composition? This folder draws the appendix figure `position-effect-k20.pdf`
(section "Budgeting is slightly position dependent"): MATH-500 at k=20, every
open-weight model, thinking tokens and accuracy drop for each of the 20 prompt positions.

```bash
bash experiments/IV-position-effect/analyze.sh
```

It needs no GPU and runs no model. It reads the per-term-trace runs that
`experiments/II-token-ratio/` also reads; this folder has no GPU stage of its own.

## Inputs

| what | where |
| --- | --- |
| per-term thinking tokens and per-position correctness | `runs/per-term-trace/<dataset>/<model>/k<K>-s<A>-<B>/sections.jsonl` (+ `trace.jsonl` for echo reads) |
| isolated accuracy of each task | `runs/isolated-a/<dataset>/<model>/` |

## Steps

1. `position_effect_csv.py` loads every per-term-trace run of each model with
   `load_experiment(..., include_non_actual=True)`. It drops the fillers, keeps the dummy
   (the last position) with `role=dummy`, and leaves echo reads and unreadable answer
   lists out of accuracy. One row per (k, position, seed) holds the mean section tokens
   and the mean accuracy drop (composed correctness minus the task's isolated accuracy).
   It covers MMLU-Redux and MATH-500 by default; `--datasets` picks other per-term-trace
   datasets.
2. `position_effect.py` keeps MATH-500 at k=20 and draws two box rows per model: thinking
   tokens and accuracy drop by position. A box spans the 10 batching seeds. The dummy is
   drawn grey, because its section also holds the answer recap. `--k` draws another k and
   `--datasets` another dataset.

Both scripts load one model at a time; run them through the memory-slot wrapper on a
shared login node.

## Outputs (`$RUN_DIR/results/IV-position-effect/`)

| file | content |
| --- | --- |
| `position_effect_seeds.csv` | one row per (dataset, model, k, position, seed): `n`, `mean_thinking_tokens`, `composed_accuracy`, `solo_accuracy`, `accuracy_drop` |
| `position-effect-k20.pdf` | the appendix figure |

## Why MATH-500

The MMLU-Redux prompt ends with a format example, `Answer: A, B, C, D, A, B, ...`, which
some models copy into their answer list, so its first positions drop for reasons unrelated
to the thinking budget. MATH-500's answer block shows placeholders
(`\boxed{<your answer to problem 1>}`), which no model copies, so its positions measure
budgeting alone. `--datasets mmlu` draws the MMLU-Redux version.
