# VI. Fragility-aware batching

## Results table

`analyze.sh` computes every cell of Table 2 (`tab:batching_results`) and of the appendix
table (`tab:batching_results_full`) from finished runs, together with the averages the
text of Section 4 quotes. It needs no GPU.

```bash
bash experiments/VI-fragility-aware-batching/analyze.sh
```

Each cell is an item accuracy or an output-token count per item (thinking and answer,
never the prompt), computed per batching seed and then averaged over seeds:

| column | runs | seeds |
| --- | --- | --- |
| Isolated, k=1 | `isolated-a` + `isolated-b` | 10 |
| Random batching, k=6 | `random-a` + `random-b`, prompts holding exactly 6 items | 10 (Opus 5, `random-a` only) |
| Fragility-aware (measured) | `fragility-measured` | 5 |
| Fragility-aware (probe) | `fragility-probe` | 5 |

A seed's accuracy is the share of items answered correctly; an unparseable prompt
counts as wrong for each of its items. Its tokens are the completion tokens of all its
prompts divided by the items they held. The changes are against k=1: percentage points
for accuracy, relative % for tokens. An Average row is the mean over the models shown
of the k=1 values and of each change. On OneWordQA it leaves out R1 Qwen3 8B, the grey
row, whose answers in composition break the answer format.

The k=6 column counts prompts by the number of items they held. In a random pass at
k=10 the short tail batch can hold 6 items (GPQA-Diamond: 196 = 19 × 10 + 6), and it
then enters the k=6 column, as it did in the published table.

AIME is left out: fragility-aware batching keeps nearly every AIME item alone.

Qwen3.8 27B on GPQA-Diamond was rerun after its fragility-aware runs were built. Its
isolated and random columns therefore read the first passes, stored as `isolated-a-v1`,
`isolated-b-v1`, `random-a-v1` and `random-b-v1` (see `llm.specs.baseline_stages`), so
that every column of that row is measured against the passes its batching came from.

Outputs, in `$RUN_DIR/results/VI-fragility-aware-batching/`:

| file | content |
| --- | --- |
| `batching_results_<dataset>.csv` | per model, every condition: k=1 pooled and per pass, random k=3/6/10(/20), measured, probe; accuracy, tokens per item, changes, `mean_k` and `frac_alone` of the fragility-aware passes |
| `batching_table_main.tex` | the tabular of Table 2 (GPQA-Diamond, MATH-500, MMLU-Redux) |
| `batching_table_full.tex` | the tabular of the appendix table (all seven datasets) |
| `batching_text_numbers.csv` | the averages quoted in Section 4, next to the printed value |

The text averages are means of the dataset Average rows over the seven datasets:
measured batching saves 43% of tokens at +0.6 pp, the probe 31.0%; "at most 0.5 pp" is
the worst probe Average; "roughly 32%" is the mean over the other five. The "68–74%" of
OneWordQA and StrategyQA are their measured Averages, 67.7% (StrategyQA) and 73.5%
(OneWordQA), rounded to whole percents.

## Probe and batching runs

Each fragility-aware run needs a composition size per item. `probe.sh` computes it in
two arms with `llm.probe`, and `gpu.sh` turns each arm's sizes into batches and runs
them.

```bash
bash experiments/VI-fragility-aware-batching/probe.sh embed gsm8k      # GPU, once per dataset
bash experiments/VI-fragility-aware-batching/probe.sh splits gsm8k
bash experiments/VI-fragility-aware-batching/probe.sh pair gsm8k qwen3-8b
bash experiments/VI-fragility-aware-batching/gpu.sh measured gsm8k qwen3-8b
bash experiments/VI-fragility-aware-batching/gpu.sh probe gsm8k qwen3-8b
```

`bash probe.sh --list` and `bash gpu.sh --list` print every job. Apart from the
embeddings, `probe.sh` runs on the CPU: a pair's 20 heads train in a few minutes, and
reading the traces for the labels and the measured arm takes about a minute.

| step | module | writes, under `$RUN_DIR/probe/` |
| --- | --- | --- |
| item embeddings | `llm.probe.embed_items` | `embeddings/<ds>/` (Qwen3-Embedding-8B, EOS pooling) |
| labels | `llm.probe.build_labels` | `labels/<ds>/<model>.csv`: per item and k, accuracy over 10 isolated and 10 composed trials, the drop, and `label_k = drop_k <= 0` |
| outer folds | `llm.probe.make_splits` | `splits/<ds>/kfold5-seed0-by-category.json` |
| heads | `llm.probe.train` | `heads/kb-emb{,v1,v2,v3}-<ds>-<model>-fold<F>/`: per outer fold, four heads, each holding out one inner fold |
| token prices | `llm.probe.tokens` | `token-cost.csv`: output tokens per item at each k |
| sizes | `llm.probe.assign_k --arm measured` / `--arm probe` | `assignments/{measured,probe}/<ds>/<model>.csv` |

The two arms run the same rule: the Lagrangian assignment, an isotonic map from a score to the
expected drop, and λ chosen by repeated cross-fitting at budget 0.005 and margin 1.0.
They differ only in the score. The measured arm splits each item's trials in half at
random (seed 0) and maps the drop on one half onto the drop on the other. The probe arm
uses the survival probabilities of the four cross-fitted heads. Both read the isolated
and random stages through `llm.specs.baseline_stages`, so Qwen3.8 27B on GPQA-Diamond
reads its first passes (`*-v1`).

`gpu.sh` builds the `fragility-measured` and `fragility-probe` batches with
`predicted-calibration` (five batching seeds; items assigned no k are asked alone) and
runs them with engine seed 42. Claude Opus 5 is billed through OpenRouter, so its lines
are commented out. Its sizes come from `probe.sh opus`: it has one random pass, its
measured arm was built from 5 isolated trials and its probe arm from 10, and each arm
gets its own token prices (`--token-cost`).

### Reproducing the paper's sizes

Two stored artefacts cannot be rebuilt from the stored runs alone:

- **The token table.** The stored `token-cost.csv` is the one the paper's sizes were
  computed with. It pools a different number of trials per pair, for example 3 isolated
  and 3 random seeds for most GSM8K models, and 10 isolated but 5 random trials for
  Integrals, MMLU and OneWordQA. `llm.probe.tokens` pools all 10 + 10
  trials, which moves the cost ratios by up to about 0.01, enough to change some
  assignments. `assign_k` reads the stored table, so skip the `tokens` job to reproduce
  the paper's sizes. Opus has no row there; `probe.sh opus` writes its two tables.
- **StrategyQA, R1 Qwen3 8B.** Its traces were regraded after its labels
  and both arms' sizes were computed, and the regrade raised its isolated accuracy from
  0.574 to 0.700. The stored labels, heads and sizes come from the earlier grading.
  Rebuilt from the stored runs, its labels and measured sizes change (607 of 687 items
  get another k). The pre-regrade traces are in `$RUN_DIR/regrade-backups/strategyqa/`.
