# III. Budget forcing

Composing k tasks shrinks the thinking each one gets. This experiment asks whether that
shrinkage alone explains the accuracy drop. Each item is asked again on its own, with its
thinking cut off (`</think>` injected) once it reaches the budget the same model gave it
inside a k-item prompt. If composition acts like a token cut, accuracy under the forced
budget should fall with k the way it falls in composition.

The figures put both conditions on the same axes. The x axis is the token reduction
`1 - r_i(k)`, where `r_i(k)` is the item's thinking at k over its thinking alone; the
y axis is the change in accuracy against the item alone. Items are binned by their
isolated trace length: Low (lowest 20%), Medium (20-80%) and High (top 20%). Composed points
are circles joined in k order; forced points are pale diamonds. When the forced points
reach more than three times past the composed ones, the y axis turns symmetric-log:
linear down to a dashed line, logarithmic beyond (`draw.symlog_y`, used by the appendix
grids and the main figure alike). Where even x ticks would crowd at the left end of
a grid panel, the labels that would touch their right neighbour are left blank
(`draw.thin_xticklabels`). Forcing hurts more than
composing, but both follow the same pattern: accuracy falls as the reduction grows, and
long-trace items fall furthest.

## Paper artefacts

| Artefact | Script |
| --- | --- |
| Main text, Figure "token-ratio-vs-accuracy" (Qwen3 4B Thinking): `mmlu-token-ratio-vs-accuracy-drop.pdf`, `math500-token-ratio-vs-accuracy-drop.pdf` | `plot_main.py` |
| Appendix "Token ratio influences accuracy degradation", one grid per model (all seven): `<model>-composed-vs-forced-token-ratio-vs-accuracy-drop-grid.pdf` | `plot_grids.py` |
| Appendix paragraph "Budget forcing causes a degradation similar to composition" | the same figures; the paragraph prints no numbers |

## Inputs

All read through `llm.paths` from `$RUN_DIR`, for the six per-term-trace datasets
(integrals, mmlu, onewordqa, gpqa-diamond, gsm8k, math500) and the seven models of
`llm.specs.MODELS`:

- `runs/per-term-trace/<dataset>/<model>/k<K>-s<A>-<B>/sections.jsonl`: each item's
  thinking tokens and outcome inside a k-item prompt (`load_experiment`).
- `runs/isolated-a/<dataset>/<model>/`: each item alone, five trials. Gives the isolated
  thinking tokens `T_i(1)` and the isolated accuracy (`load_solo`, plus `solo_table` in
  `build_points.py` for the forced side).
- `runs/isolated-b/<dataset>/<model>/`: an independent set of five trials, whose mean
  output length sets the length bins. Binning on the same trials that define `T_i(1)`
  would correlate the bin with the noise in the ratio.
- `runs/budget-forcing/<dataset>/<model>/<run-id>/` (`generations.jsonl`, `trace.jsonl`):
  the forced runs, one per (dataset, model). `build_points.py` refuses a run whose
  `run.json` names another model or a budget variant other than `section-outside`.
- `batches/budget-forcing/<dataset>/<model>.budgets.jsonl`: the drawn budget of every
  (k, item). Each run's cap is checked against it.

k counts the trailing dummy item of the per-term trace, so k=2 is one real item plus the
dummy. The budget (`section-outside`) is the item's own section plus all thinking in that
response not attributed to any item, mostly the read-through before the first label. That
overhead was spent in the composed prompt too, so the forced item gets it as well.

## Outputs

In `$RUN_DIR/results/III-budget-forcing/`:

- `composed-bins.csv`: one row per (dataset, model, length group, k) with the mean token
  reduction (`saved_tokens`), the mean accuracy change in pp (`acc_change_pp`) and the
  number of items. Groups are quintiles of the isolated-b output length (lowest quintile
  `low`, middle three `medium`, top one `high`).
- `forced-bins.csv`: one row per (dataset, model, k, length bin) with the mean token ratio,
  the forced and isolated accuracy, the drop in pp and the bin edges in tokens. Bins are cut
  at the 20th and 80th percentiles of the isolated-b output length. An unreadable forced
  answer counts as wrong.
- `parts/`: the same rows per (dataset, model) cell, cached by `build_points.py`.
- The nine figures listed above.

Items with zero isolated thinking tokens are left out of both tables. The grids also drop
points whose token reduction is not positive.

## How to run

```bash
bash experiments/III-budget-forcing/analyze.sh
```

`build_points.py` re-tokenizes every isolated answer to count its thinking tokens, so a
full build takes a while. It writes each (dataset, model) cell as it finishes and skips
cells already in `parts/`, so it can be split across processes and resumed:

```bash
python experiments/III-budget-forcing/build_points.py --model qwen3-8b qwen3-14b
python experiments/III-budget-forcing/build_points.py --dataset mmlu --refresh
```

The two tables are written once every cell is present. The plot scripts read only the
tables and take seconds; `plot_grids.py --model <model>` draws a single grid.

## GPU runs

`gpu.sh` reproduces the forced runs: 42 jobs, one per (dataset, model), each needing one
GPU. Each job reads that pair's per-term-trace runs (experiment II), cut into sections.

```bash
bash experiments/III-budget-forcing/gpu.sh --list
bash experiments/III-budget-forcing/gpu.sh gsm8k qwen3-8b
DRY_RUN=1 bash experiments/III-budget-forcing/gpu.sh gsm8k qwen3-8b
```

A job has two steps:

1. `python -m llm.budget_forcing build` (CPU) draws one budget per (k, item) from the
   per-term-trace sections. It writes `batches/budget-forcing/<dataset>/<model>.jsonl`
   (the k=1 documents), `.budgets.jsonl` and `.manifest.json`. The draw is seeded by
   dataset, model, k and item, so a rebuild gives the same files. A finished build is
   skipped unless `--force` is passed.
2. `python -m llm.budget_forcing run` asks every document alone in one vLLM engine and
   caps its thinking at its own budget. 512 tokens are left for the answer after the
   forced `</think>`. The run lands in `runs/budget-forcing/<dataset>/<model>/`. In a
   dry run the step loads only the tokenizer, renders one forced prompt and checks that
   every budget fits the context window.

qwen3-8b and qwen3-14b ran these with the 131072-token YaRN window
(`--model-variant yarn-131k`). Their other stages used the 40960-token default.
