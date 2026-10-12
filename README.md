<div align="center">

# On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models


**Chenpan Li**<sup>1</sup> · **Thies Brockmöller**<sup>1</sup> · **Stefanie Jegelka**<sup>1,2</sup> · **Sharut Gupta**<sup>2,◇</sup> · **Nikolaos Karalias**<sup>2,◇</sup>

<sup>1</sup>Technical University of Munich &nbsp;·&nbsp; <sup>2</sup>Massachusetts Institute of Technology

<sup>◇</sup>Denotes equal advisory roles

[Setup](#setup) · [Reproduce the results](#reproduce-the-paper) · [Run from scratch](#run-from-scratch) · [FAQ](#faq) · [Citation](#citation)


</div>


## Setup

Developed on Python 3.12, vLLM 0.26.0 and lm-eval 0.4.13.dev0. Install with
[uv](https://docs.astral.sh/uv/):

```bash
uv sync                    # analysis only (CPU)
uv sync --extra gpu        # also vLLM, to run the models
cp .env.example .env       # then set RUN_DIR
```

`.env` holds the following:

| Variable | Needed for |
| --- | --- |
| `RUN_DIR` | everything: the one directory that holds all data, runs and results |
| `OPENROUTER_API_KEY` | only the Claude Opus 5 runs (billed per request) |
| `HF_TOKEN` | only if your Hugging Face setup needs a token to download models or datasets |

Run every command from the repository root. The stage scripts use `.venv/bin/python`;
set `PY` to use another interpreter.

## Reproduce the paper

**1. Point to the runs.** Set `RUN_DIR` in `.env` to the paper's run directory. No GPU
is needed from here on.

**2. Rebuild every result.**

```bash
bash reproduce_results.sh
```

This runs every `analyze.sh` in paper order and writes all figures (.pdf), tables (.tex)
and CSVs to `$RUN_DIR/results/`. The first pass takes a few hours on one CPU, mostly
re-tokenizing traces; later passes reuse cached parts.

## Run from scratch

**1. Build the datasets.** This builds the six Hugging Face Hub pools and copies the two
shipped ones into `$RUN_DIR/datasets/`.

```bash
bash experiments/0-datasets/build.sh
```

**2. Run the GPU stages.** Each GPU stage has a `gpu.sh` with one job per (dataset,
model), and each job fits on one 80 GB GPU. Spread the jobs over your GPUs by hand:

```bash
bash experiments/1-isolated/gpu.sh --list              # print the jobs
bash experiments/1-isolated/gpu.sh gsm8k qwen3-8b      # run one job
bash experiments/1-isolated/gpu.sh                     # run every job, one after another
DRY_RUN=1 bash experiments/1-isolated/gpu.sh gsm8k qwen3-8b
```

`DRY_RUN=1` builds the batches, renders the first prompts and prints the output paths,
but loads no model.


**3. Analyse.** Each analysis folder has an `analyze.sh`; `reproduce_results.sh` runs
them all.

Run the stages in this order. Each folder has a README with its inputs, outputs,
commands, cost, and how the paper's runs differ from a clean rerun.

| Folder | GPU | Paper |
| --- | :---: | --- |
| [`0-datasets`](experiments/0-datasets) | | the eight item pools |
| [`1-isolated`](experiments/1-isolated) | ✓ | k=1 baseline; the "reliably solved" filter |
| [`2-random-batching`](experiments/2-random-batching) | ✓ | random batching at each k; fragility labels |
| [`I-success-conditioned-drop`](experiments/I-success-conditioned-drop) | ✓ | accuracy drop of solved items with k; context-limit figure and table |
| [`II-token-ratio`](experiments/II-token-ratio) | ✓ | per-task thinking shrinks with k; power-law fits; tag coverage |
| [`III-budget-forcing`](experiments/III-budget-forcing) | ✓ | forcing the composed thinking length onto isolated items |
| [`IV-position-effect`](experiments/IV-position-effect) | | accuracy by position in the prompt |
| [`V-fragility-correlation`](experiments/V-fragility-correlation) | | fragility correlates across models and k |
| [`VI-fragility-aware-batching`](experiments/VI-fragility-aware-batching) | ✓ | the batching tables and their averages |
| [`VII-olmo-post-training`](experiments/VII-olmo-post-training) | ✓ | OLMo 3 SFT, DPO and RLVR |

## FAQ

<details>
<summary><b>Where does the data live?</b></summary>
<br>

Every input and output lives under `$RUN_DIR`, and nothing is written inside the
repository. The code builds all paths in `src/llm/paths.py`.

```
$RUN_DIR/
  datasets/<dataset>.jsonl                       item pools                      (5 MB)
  batches/<stage>/<dataset>/<model>.jsonl        what each run is asked           (7 GB)
  runs/<stage>/<dataset>/<model>/<run-id>/       what each run answered         (108 GB)
  probe/                                         embeddings, labels, heads, k    (0.5 GB)
  results/<experiment>/                          figures (.pdf), tables (.tex), CSVs
```

A run directory holds `run.json` (model, settings, command), `trace.jsonl` (one graded
record per response) and lm-eval's `results_*.json` and `samples_*.jsonl`. Analyses pool
every run they find under `runs/<stage>/<dataset>/<model>/`, so a directory must hold
only the runs that belong to the result.

| Stage | What is asked |
| --- | --- |
| `isolated-a`, `isolated-b` | every item alone, 5 batching seeds; pass a uses generation seed 42, pass b seed 43 |
| `random-a`, `random-b` | the pool split into random prompts of k items, same seeds |
| `composed` | prompts of k reliably solved items (correct 3 of 3 times in `isolated-a`) |
| `per-term-trace` | prompts whose items are tagged `Q01`, `Q02`, …, so thinking can be cut per item |
| `budget-forcing` | every item alone, its thinking capped at the length it got inside a composition |
| `fragility-measured`, `fragility-probe` | prompts sized per item by measured or predicted fragility |

</details>

<details>
<summary><b>Which datasets and models are included?</b></summary>
<br>

| Dataset | Items | Source |
| --- | --- | --- |
| `integrals` | 1,000 indefinite integrals | ours, shipped in `datasets/integrals/` (Git LFS) |
| `onewordqa` | 1,548 questions with a one-word answer | ours, shipped in `datasets/onewordqa/` (Git LFS) |
| `gpqa-diamond` | 196 | Hugging Face Hub, pinned revision |
| `gsm8k` | 1,319 (test split) | Hugging Face Hub, pinned revision |
| `math500` | 500 | Hugging Face Hub, pinned revision |
| `mmlu` | 1,406 (MMLU-Redux) | Hugging Face Hub, pinned revision |
| `strategyqa` | 687 | Hugging Face Hub, pinned revision |
| `aime` | 60 (AIME 2024 and 2025) | Hugging Face Hub, pinned revisions |

The models are Qwen3 4B Thinking, Qwen3 8B, Qwen3 14B, Qwen3.8 27B and three
DeepSeek-R1-distilled models (R1 Qwen 7B, R1 Llama 8B, R1 Qwen3 8B). Claude Opus 5 runs
on GPQA-Diamond only, and OLMo 3 7B Think SFT, DPO and RLVR run on GPQA-Diamond,
MATH-500 and MMLU-Redux.

Their serving settings are in `src/llm/config/models/`. A few of the paper's runs used
other context lengths. These are named `variants:` in the model's YAML, and a stage
script passes the variant as a third job column.

</details>

<details>
<summary><b>Can I run my own model?</b></summary>
<br>

Add a YAML for it under `src/llm/config/models/` (copy one of the existing files), then
add a `<dataset> <model>` line to the `JOBS` list of each stage's `gpu.sh`. The figures
draw the models listed in `src/llm/specs.py`; add yours to `MODELS` and `MODEL_NAMES`
there to include it.

</details>

<details>
<summary><b>Will a clean rerun match the paper exactly?</b></summary>
<br>

Not in every case. The stage scripts encode the paper's design, but a few of the paper's
runs departed from it. Those runs are kept as they are, so every published number still
reproduces from them:

- **Top-up runs:** some isolated and random passes on GSM8K, MATH-500 and GPQA-Diamond
  merged a first run with a top-up that used other batching seeds. A rerun uses seeds
  0–4.
- **Qwen3.8 27B on GPQA-Diamond** was rerun after its fragility-aware runs were built.
  The regular stages hold the rerun, and `isolated-*-v1` and `random-*-v1` hold the first
  passes. The batching table and the probe read the first passes for this pair
  (`llm.specs.baseline_stages`).
- **Integrals in `isolated-a`:** the paper's runs used a single-answer prompt; every
  other stage, and every rerun, grades one answer per term.
- **Composed sets of integrals, MMLU-Redux and OneWordQA** were drawn from an earlier
  filter run. `llm.compose` filters on `isolated-a` for every dataset, so it draws
  different compositions for these three. The other five rebuild exactly.
- **Composed MMLU-Redux, Qwen3 8B:** its composed-set file was redrawn after the run.
  The batch file in `$RUN_DIR` was rebuilt from the documents stored with the run.
- **StrategyQA, R1 Qwen3 8B** was regraded after its probe labels were written. The
  stored labels and composition sizes are used.
- **Per-term traces** were run in seed chunks (`k3-s0-4`, `k3-s5-9`, …); a rerun makes
  one run per k.

</details>

<details>
<summary><b>How is the code organised?</b></summary>
<br>

```
src/llm/
  paths.py  specs.py       every path; every dataset, model and stage setting
  run.py                   python -m llm.run: one run of one stage (lm-eval + vLLM or OpenRouter)
  pools/  batches.py       python -m llm.pools, python -m llm.batches
  compose.py               python -m llm.compose: composed sets of reliably solved items
  sections.py              python -m llm.sections: cut a per-term trace into per-item sections
  budget_forcing.py        python -m llm.budget_forcing build|run
  probe/                   python -m llm.probe.<step>: embeddings, labels, heads, composition sizes
  plotting.py              the figure style
  core/                    task families (math, mcq, shortmath, oneword), grading, serving,
                           batching strategies, thinking-budget driver, analysis loaders
lm_tasks/<dataset>/        lm-eval task definitions and prompt templates
datasets/                  the two pools we built (Git LFS)
assets/                    Figure 1 of the paper
experiments/               one folder per stage, in paper order
reproduce_results.sh       every analysis, CPU only
```

</details>

## Citation

If you use this code or the two datasets, please cite the paper:

```bibtex
@inproceedings{li2026implicitthinkingbudgets,
  title     = {On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models},
  author    = {Li, Chenpan and Brockm{\"o}ller, Thies and Jegelka, Stefanie and
               Gupta, Sharut and Karalias, Nikolaos},
  year      = {2026},
}
```
