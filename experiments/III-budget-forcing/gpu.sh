#!/usr/bin/env bash
# Budget forcing: every item asked alone, its thinking capped at the tokens the same model
# gave it inside a k-item per-term-trace prompt (its own section plus the response's
# unlabelled thinking), one drawn budget per (k, item). Needs the per-term-trace runs of
# experiment II, cut into sections. The build is CPU-only and skips a finished build.
#   job <dataset> <model> [model variant]
source "$(dirname "$0")/../common.sh"

job() {
    local ds=$1 model=$2 variant=${3:-}
    cpu $PY -m llm.budget_forcing build --dataset "$ds" --model "$model"
    gpu $PY -m llm.budget_forcing run --dataset "$ds" --model "$model" --progress \
        ${variant:+--model-variant "$variant"}
}

# qwen3-8b and qwen3-14b ran these with the 131k YaRN window.
JOBS="
integrals qwen3-4b-thinking
integrals qwen3-8b yarn-131k
integrals qwen3-14b yarn-131k
integrals qwen3-8-27b
integrals r1-qwen-distill-7b
integrals r1-llama-distill-8b
integrals r1-qwen3-8b
mmlu qwen3-4b-thinking
mmlu qwen3-8b yarn-131k
mmlu qwen3-14b yarn-131k
mmlu qwen3-8-27b
mmlu r1-qwen-distill-7b
mmlu r1-llama-distill-8b
mmlu r1-qwen3-8b
onewordqa qwen3-4b-thinking
onewordqa qwen3-8b yarn-131k
onewordqa qwen3-14b yarn-131k
onewordqa qwen3-8-27b
onewordqa r1-qwen-distill-7b
onewordqa r1-llama-distill-8b
onewordqa r1-qwen3-8b
gpqa-diamond qwen3-4b-thinking
gpqa-diamond qwen3-8b yarn-131k
gpqa-diamond qwen3-14b yarn-131k
gpqa-diamond qwen3-8-27b
gpqa-diamond r1-qwen-distill-7b
gpqa-diamond r1-llama-distill-8b
gpqa-diamond r1-qwen3-8b
gsm8k qwen3-4b-thinking
gsm8k qwen3-8b yarn-131k
gsm8k qwen3-14b yarn-131k
gsm8k qwen3-8-27b
gsm8k r1-qwen-distill-7b
gsm8k r1-llama-distill-8b
gsm8k r1-qwen3-8b
math500 qwen3-4b-thinking
math500 qwen3-8b yarn-131k
math500 qwen3-14b yarn-131k
math500 qwen3-8-27b
math500 r1-qwen-distill-7b
math500 r1-llama-distill-8b
math500 r1-qwen3-8b
"
main "$@"
