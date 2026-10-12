#!/usr/bin/env bash
# Stage 2: the whole pool cut into random batches of k items, every item in exactly one
# batch per k (k from specs.random_ks: 3 6 10 20, or 3 6 10 on GPQA-Diamond). Two passes:
# the same batch (batching seeds 0-4) run with engine seed 42 (random-a) and again with
# engine seed 43 (random-b). Together with stage 1 this gives each item's accuracy drop
# at every k, which the fragility analyses and the probe labels read.
#   job <dataset> <model> [a-only]
source "$(dirname "$0")/../common.sh"

job() {
    local ds=$1 model=$2 only=${3:-}
    cpu $PY -m llm.batches --stage random-a --dataset "$ds" --model "$model" --strategy random --seeds 5
    gpu $PY -m llm.run --stage random-a --dataset "$ds" --model "$model" --seed 42
    [ "$only" = a-only ] && return 0
    cpu $PY -m llm.batches --stage random-b --dataset "$ds" --model "$model" --strategy random --seeds 5
    gpu $PY -m llm.run --stage random-b --dataset "$ds" --model "$model" --seed 43
}

JOBS="
integrals qwen3-4b-thinking
integrals qwen3-8b
integrals qwen3-14b
integrals qwen3-8-27b
integrals r1-qwen-distill-7b
integrals r1-llama-distill-8b
integrals r1-qwen3-8b
onewordqa qwen3-4b-thinking
onewordqa qwen3-8b
onewordqa qwen3-14b
onewordqa qwen3-8-27b
onewordqa r1-qwen-distill-7b
onewordqa r1-llama-distill-8b
onewordqa r1-qwen3-8b
gpqa-diamond qwen3-4b-thinking
gpqa-diamond qwen3-8b
gpqa-diamond qwen3-14b
# The paper has no random-b pass of the rerun of this pair (see README).
gpqa-diamond qwen3-8-27b a-only
gpqa-diamond r1-qwen-distill-7b
gpqa-diamond r1-llama-distill-8b
gpqa-diamond r1-qwen3-8b
gsm8k qwen3-4b-thinking
gsm8k qwen3-8b
gsm8k qwen3-14b
gsm8k qwen3-8-27b
gsm8k r1-qwen-distill-7b
gsm8k r1-llama-distill-8b
gsm8k r1-qwen3-8b
math500 qwen3-4b-thinking
math500 qwen3-8b
math500 qwen3-14b
math500 qwen3-8-27b
math500 r1-qwen-distill-7b
math500 r1-llama-distill-8b
math500 r1-qwen3-8b
mmlu qwen3-4b-thinking
mmlu qwen3-8b
mmlu qwen3-14b
mmlu qwen3-8-27b
mmlu r1-qwen-distill-7b
mmlu r1-llama-distill-8b
mmlu r1-qwen3-8b
strategyqa qwen3-4b-thinking
strategyqa qwen3-8b
strategyqa qwen3-14b
strategyqa qwen3-8-27b
strategyqa r1-qwen-distill-7b
strategyqa r1-llama-distill-8b
strategyqa r1-qwen3-8b
aime qwen3-4b-thinking
aime qwen3-8b
aime qwen3-14b
aime qwen3-8-27b
aime r1-qwen-distill-7b
aime r1-llama-distill-8b
aime r1-qwen3-8b
# Hosted through OpenRouter and billed per request; uncomment deliberately.
# gpqa-diamond claude-opus-5 a-only
"
main "$@"
