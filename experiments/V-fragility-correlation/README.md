# V. Fragility correlation

Do models, and one model at different composition sizes, lose accuracy on the same
items? This folder answers it for Section 3.2 ("the mean correlation across models is
only 0.24±0.03", "Qwen3 8B and 14B correlate at 0.56±0.09", "within a single model,
the correlation across composition sizes reaches 0.81±0.04") and draws the appendix
figure `fragility-correlation-cross-k-and-model.pdf`. The method is in the appendix
section "Correlating accuracy degradation across conditions".

```bash
bash experiments/V-fragility-correlation/analyze.sh
```

It needs no GPU and runs no model: it reads finished runs only (about 5 minutes and
0.6 GB of memory).

## Inputs

An item's drop at k is its isolated accuracy minus its accuracy in random batches of
nominal size k. Each item is measured twice, independently:

| pass | isolated (k=1) | random batches |
| --- | --- | --- |
| A | `runs/isolated-a/<dataset>/<model>/` | `runs/random-a/<dataset>/<model>/` |
| B | `runs/isolated-b/<dataset>/<model>/` | `runs/random-b/<dataset>/<model>/` |

The datasets are the eight of the paper; GPQA enters as GPQA-Diamond only.

A model is used on a dataset only if it has all four stages there. Qwen3.8 27B has no
`random-b` run on GPQA-Diamond (its rerun, which these stages hold, made one pass), so it
is left out on that dataset and printed as dropped, as in the paper.

## Steps

1. `fragility_correlation.py` builds the per-item drop tables (pass A, pass B, and both
   pooled) and computes three kinds of correlation per dataset: test-retest (pass A
   against pass B, one model and k), across k (k from pass A against k' from pass B, so
   the two sides never share the isolated accuracy they subtract), and across models
   (pooled drops, one k). Each is disattenuated by the test-retest reliabilities and
   bootstrapped over items (5000 resamples). The `all` row of the summary and the
   heatmap cells average over datasets, pairing replicate b of every dataset.
2. `plot_fragility_correlation.py` averages `r_disatt` per k pair and per model pair and
   draws the two heatmaps.

## Outputs (`$RUN_DIR/results/V-fragility-correlation/`)

| file | content |
| --- | --- |
| `fragility_correlation_same.csv` | one row per (dataset, model, k): test-retest r, rho, phi, Spearman-Brown reliability of the pooled drop, fragile rate, trial counts |
| `fragility_correlation_crossk.csv` | one row per (dataset, model, k1, k2): r, rho, phi, `r_disatt`, and the confounded same-pass `r_same_pass` |
| `fragility_correlation_crossmodel.csv` | one row per (dataset, model1, model2, k) |
| `fragility_correlation_summary.csv` | means per dataset, plus the pooled `all` row |
| `fragility_correlation_cells.csv` | one row per heatmap cell |
| `fragility_correlation_paper.csv` | the three numbers of Section 3.2; the ± is half the width of the 95% bootstrap interval |
| `fragility-correlation-cross-k-and-model.pdf` | the appendix figure |

Every statistic carries a 95% percentile interval (`_lo`, `_hi`). The bootstrap seed of
a dataset is `--seed` plus a fixed offset per dataset.
