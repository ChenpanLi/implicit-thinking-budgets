#!/usr/bin/env bash
# Fragility-aware batching runs: per (dataset, model), the pool of each outer fold
# grouped greedily by the per-item sizes of one arm (probe.sh writes them), never mixing
# folds; items assigned no k are asked alone (--fallback-k 1). Five batching seeds,
# engine seed 42.
#   job <measured|probe> <dataset> <model>
source "$(dirname "$0")/../common.sh"

job() {
    local arm=$1 ds=$2 model=$3
    # --k defaults to the dataset's random_ks: 3 6 10 20, or 3 6 10 on GPQA-Diamond.
    cpu $PY -m llm.batches --stage fragility-$arm --dataset "$ds" --model "$model" \
        --strategy predicted-calibration --fallback-k 1 --seeds 5 \
        --stats predicted-largest-k="$RUN_DIR/probe/assignments/$arm/$ds/$model.csv" \
        --param column=pred_largest_k_lagrangian
    gpu $PY -m llm.run --stage fragility-$arm --dataset "$ds" --model "$model" --seed 42
}

JOBS="
measured integrals qwen3-4b-thinking
measured integrals qwen3-8b
measured integrals qwen3-14b
measured integrals qwen3-8-27b
measured integrals r1-qwen-distill-7b
measured integrals r1-llama-distill-8b
measured integrals r1-qwen3-8b
measured onewordqa qwen3-4b-thinking
measured onewordqa qwen3-8b
measured onewordqa qwen3-14b
measured onewordqa qwen3-8-27b
measured onewordqa r1-qwen-distill-7b
measured onewordqa r1-llama-distill-8b
measured onewordqa r1-qwen3-8b
measured gpqa-diamond qwen3-4b-thinking
measured gpqa-diamond qwen3-8b
measured gpqa-diamond qwen3-14b
measured gpqa-diamond qwen3-8-27b
measured gpqa-diamond r1-qwen-distill-7b
measured gpqa-diamond r1-llama-distill-8b
measured gpqa-diamond r1-qwen3-8b
measured gsm8k qwen3-4b-thinking
measured gsm8k qwen3-8b
measured gsm8k qwen3-14b
measured gsm8k qwen3-8-27b
measured gsm8k r1-qwen-distill-7b
measured gsm8k r1-llama-distill-8b
measured gsm8k r1-qwen3-8b
measured math500 qwen3-4b-thinking
measured math500 qwen3-8b
measured math500 qwen3-14b
measured math500 qwen3-8-27b
measured math500 r1-qwen-distill-7b
measured math500 r1-llama-distill-8b
measured math500 r1-qwen3-8b
measured mmlu qwen3-4b-thinking
measured mmlu qwen3-8b
measured mmlu qwen3-14b
measured mmlu qwen3-8-27b
measured mmlu r1-qwen-distill-7b
measured mmlu r1-llama-distill-8b
measured mmlu r1-qwen3-8b
measured strategyqa qwen3-4b-thinking
measured strategyqa qwen3-8b
measured strategyqa qwen3-14b
measured strategyqa qwen3-8-27b
measured strategyqa r1-qwen-distill-7b
measured strategyqa r1-llama-distill-8b
measured strategyqa r1-qwen3-8b
# BILLED (OpenRouter): never run without asking.
# measured gpqa-diamond claude-opus-5
probe integrals qwen3-4b-thinking
probe integrals qwen3-8b
probe integrals qwen3-14b
probe integrals qwen3-8-27b
probe integrals r1-qwen-distill-7b
probe integrals r1-llama-distill-8b
probe integrals r1-qwen3-8b
probe onewordqa qwen3-4b-thinking
probe onewordqa qwen3-8b
probe onewordqa qwen3-14b
probe onewordqa qwen3-8-27b
probe onewordqa r1-qwen-distill-7b
probe onewordqa r1-llama-distill-8b
probe onewordqa r1-qwen3-8b
probe gpqa-diamond qwen3-4b-thinking
probe gpqa-diamond qwen3-8b
probe gpqa-diamond qwen3-14b
probe gpqa-diamond qwen3-8-27b
probe gpqa-diamond r1-qwen-distill-7b
probe gpqa-diamond r1-llama-distill-8b
probe gpqa-diamond r1-qwen3-8b
probe gsm8k qwen3-4b-thinking
probe gsm8k qwen3-8b
probe gsm8k qwen3-14b
probe gsm8k qwen3-8-27b
probe gsm8k r1-qwen-distill-7b
probe gsm8k r1-llama-distill-8b
probe gsm8k r1-qwen3-8b
probe math500 qwen3-4b-thinking
probe math500 qwen3-8b
probe math500 qwen3-14b
probe math500 qwen3-8-27b
probe math500 r1-qwen-distill-7b
probe math500 r1-llama-distill-8b
probe math500 r1-qwen3-8b
probe mmlu qwen3-4b-thinking
probe mmlu qwen3-8b
probe mmlu qwen3-14b
probe mmlu qwen3-8-27b
probe mmlu r1-qwen-distill-7b
probe mmlu r1-llama-distill-8b
probe mmlu r1-qwen3-8b
probe strategyqa qwen3-4b-thinking
probe strategyqa qwen3-8b
probe strategyqa qwen3-14b
probe strategyqa qwen3-8-27b
probe strategyqa r1-qwen-distill-7b
probe strategyqa r1-llama-distill-8b
probe strategyqa r1-qwen3-8b
# BILLED (OpenRouter): never run without asking.
# probe gpqa-diamond claude-opus-5
"
main "$@"
