<div align="center">

<h1>On the Fragility of Tasks under Composition:<br>Implicit Budgeting in Language Models</h1>

<p>
<b>Chenpan Li</b><sup>1</sup>&emsp;
<b>Thies Brockmoeller</b><sup>1</sup>&emsp;
<b>Stefanie Jegelka</b><sup>1,2</sup>&emsp;
<b>Sharut Gupta</b><sup>2,◇</sup>&emsp;
<b>Nikolaos Karalias</b><sup>2,◇</sup>
</p>

<p>
<sup>1</sup>Technical University of Munich&emsp;&emsp;<sup>2</sup>Massachusetts Institute of Technology<br>
<sub><sup>◇</sup>Equal advisory role</sub>
</p>

<p>
<a href="https://chenpanli.github.io/fragility/"><img alt="Paper" src="https://img.shields.io/badge/Paper-C8413F?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxwYXRoIGQ9Ik0xNCAySDZhMiAyIDAgMCAwLTIgMnYxNmEyIDIgMCAwIDAgMiAyaDEyYTIgMiAwIDAgMCAyLTJWOHoiLz48cG9seWxpbmUgcG9pbnRzPSIxNCAyIDE0IDggMjAgOCIvPjxsaW5lIHgxPSIxNiIgeTE9IjEzIiB4Mj0iOCIgeTI9IjEzIi8%2BPGxpbmUgeDE9IjE2IiB5MT0iMTciIHgyPSI4IiB5Mj0iMTciLz48L3N2Zz4%3D"></a>&nbsp;
<a href="https://chenpanli.github.io/fragility/"><img alt="Project page" src="https://img.shields.io/badge/Project_page-2E6DB4?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiPjxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjEwIi8%2BPGxpbmUgeDE9IjIiIHkxPSIxMiIgeDI9IjIyIiB5Mj0iMTIiLz48cGF0aCBkPSJNMTIgMmExNS4zIDE1LjMgMCAwIDEgNCAxMCAxNS4zIDE1LjMgMCAwIDEtNCAxMCAxNS4zIDE1LjMgMCAwIDEtNC0xMCAxNS4zIDE1LjMgMCAwIDEgNC0xMHoiLz48L3N2Zz4%3D"></a>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/figure1-dark.png">
  <img src="assets/figure1.png" alt="Three tasks solved alone each get more reasoning than they need. Composed into one prompt, every task's reasoning shrinks; tasks 1 and 3 still clear their requirement, while task 2, which needs long reasoning, falls below it and fails." width="90%">
</picture>


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
| [`0-datasets`](experiments/0-datasets) | | the eight task pools |
| [`1-isolated`](experiments/1-isolated) | ✓ | k=1 baseline; the "reliably solved" filter |
| [`2-random-batching`](experiments/2-random-batching) | ✓ | random batching at each k; fragility labels |
| [`I-success-conditioned-drop`](experiments/I-success-conditioned-drop) | ✓ | accuracy drop of solved tasks with k; context-limit figure and table |
| [`II-token-ratio`](experiments/II-token-ratio) | ✓ | per-task thinking shrinks with k; power-law fits; tag coverage |
| [`III-budget-forcing`](experiments/III-budget-forcing) | ✓ | forcing the composed thinking length onto isolated tasks |
| [`IV-position-effect`](experiments/IV-position-effect) | | accuracy by position in the prompt |
| [`V-fragility-correlation`](experiments/V-fragility-correlation) | | fragility correlates across models and k |
| [`VI-fragility-aware-batching`](experiments/VI-fragility-aware-batching) | ✓ | the batching tables and their averages |
| [`VII-olmo-post-training`](experiments/VII-olmo-post-training) | ✓ | OLMo 3 SFT, DPO and RLVR |

## FAQ

<details>
<summary><b>Which datasets and models are included?</b></summary>
<br>

| Dataset | tasks | Source |
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
<summary><b>How is the code organised?</b></summary>
<br>

```
src/llm/
  paths.py  specs.py       every path; every dataset, model and stage setting
  run.py                   python -m llm.run: one run of one stage (lm-eval + vLLM or OpenRouter)
  pools/  batches.py       python -m llm.pools, python -m llm.batches
  compose.py               python -m llm.compose: composed sets of reliably solved tasks
  sections.py              python -m llm.sections: cut a per-term trace into per-task sections
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
@inproceedings{li2026fragility,
  title     = {On the Fragility of Tasks under Composition: Implicit Budgeting in Language Models},
  author    = {Li, Chenpan and Brockmoeller, Thies and Jegelka, Stefanie and
               Gupta, Sharut and Karalias, Nikolaos},
  year      = {2026},
}
```
