#!/usr/bin/env bash
# Experiment I, GPU part: the composed sets. Per (dataset, model), draw compositions of k
# items from the items the model solved on all three of batching seeds 0-2 of its
# isolated-a run (stage 1 must be done), then run them, one lm-eval task per k
# (specs.composed_ks; n compositions per k from specs.composed_n, 100 for claude-opus-5).
#   job <dataset> <model> [model-variant]
# The variant is the context window a few of the paper's composed runs used; it names a
# block under `variants:` in src/llm/config/models/<model>.yaml.
source "$(dirname "$0")/../common.sh"

job() {
    local ds=$1 model=$2 variant=${3:-}
    cpu $PY -m llm.compose --dataset "$ds" --model "$model"
    gpu $PY -m llm.run --stage composed --dataset "$ds" --model "$model" --seed 42 \
        ${variant:+--model-variant "$variant"}
}

JOBS="
integrals qwen3-4b-thinking ctx-49k
integrals qwen3-8b
integrals qwen3-14b
integrals qwen3-8-27b ctx-49k
integrals r1-qwen-distill-7b ctx-49k
integrals r1-llama-distill-8b ctx-64k
integrals r1-qwen3-8b ctx-49k
onewordqa qwen3-4b-thinking ctx-49k
onewordqa qwen3-8b
onewordqa qwen3-14b
onewordqa qwen3-8-27b ctx-49k
onewordqa r1-qwen-distill-7b ctx-49k
onewordqa r1-llama-distill-8b ctx-64k
# not in the paper:
# onewordqa r1-qwen3-8b
gpqa-diamond qwen3-4b-thinking
gpqa-diamond qwen3-8b
gpqa-diamond qwen3-14b
gpqa-diamond qwen3-8-27b
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
# not in the paper:
# math500 qwen3-4b-thinking
math500 qwen3-8b
math500 qwen3-14b
math500 qwen3-8-27b
math500 r1-qwen-distill-7b
math500 r1-llama-distill-8b
math500 r1-qwen3-8b
mmlu qwen3-4b-thinking ctx-49k
mmlu qwen3-8b
mmlu qwen3-14b
mmlu qwen3-8-27b ctx-49k
mmlu r1-qwen-distill-7b ctx-49k
mmlu r1-llama-distill-8b ctx-64k
mmlu r1-qwen3-8b ctx-49k
strategyqa qwen3-4b-thinking
strategyqa qwen3-8b
strategyqa qwen3-14b
strategyqa qwen3-8-27b
strategyqa r1-qwen-distill-7b
strategyqa r1-llama-distill-8b
strategyqa r1-qwen3-8b
aime qwen3-4b-thinking
aime qwen3-8b yarn-131k
aime qwen3-14b yarn-131k
aime qwen3-8-27b
aime r1-qwen-distill-7b
aime r1-llama-distill-8b
aime r1-qwen3-8b
# Hosted through OpenRouter and billed per request; uncomment deliberately.
# gpqa-diamond claude-opus-5
"
main "$@"
