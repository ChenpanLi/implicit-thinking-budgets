#!/usr/bin/env bash
# Per-term thinking traces: every pool item inside dummy-tail batches of each prompt size
# k (k counts the trailing dummy; specs.trace_ks: TRACE_KS, up to 10 on gpqa-diamond,
# every k from 2 to 20 for qwen3-4b-thinking on mmlu), 10 batching
# seeds per k. Items are labelled Q01, Q02, ... and the model is asked to name an item
# whenever it works on it, so each kept thinking trace can be cut per item
# (python -m llm.sections writes sections.jsonl beside the run).
#   job <dataset> <model> [model variant]
source "$(dirname "$0")/../common.sh"

trace_ks() { $PY -c "from llm.specs import trace_ks; print(*trace_ks('$1', '$2'))"; }

job() {
    local ds=$1 model=$2 variant=${3:-}
    for k in $(trace_ks "$ds" "$model"); do
        cpu $PY -m llm.batches --stage per-term-trace --dataset "$ds" --model "$model" \
            --strategy dummy-tail --k "$k" --seed-list 0 1 2 3 4 5 6 7 8 9
        gpu $PY -m llm.run --stage per-term-trace --dataset "$ds" --model "$model" \
            --section-markers --keep-thinking --run-id "k$k-s0-9" ${variant:+--model-variant "$variant"}
        gpu $PY -m llm.sections --dataset "$ds" --model "$model" --run-id "k$k-s0-9"
    done
}

# r1-llama-distill-8b generated at most 32768 tokens in these runs, except on mmlu (where
# only four top-up chunks, k3-s9-9, k4-s8-9, k6-s5-9 and k8-s0-4, used the 32768 cap).
JOBS="
integrals qwen3-4b-thinking
integrals qwen3-8b
integrals qwen3-14b
integrals qwen3-8-27b
integrals r1-qwen-distill-7b
integrals r1-llama-distill-8b gen-32k
integrals r1-qwen3-8b
mmlu qwen3-4b-thinking
mmlu qwen3-8b
mmlu qwen3-14b
mmlu qwen3-8-27b
mmlu r1-qwen-distill-7b
mmlu r1-llama-distill-8b
mmlu r1-qwen3-8b
onewordqa qwen3-4b-thinking
onewordqa qwen3-8b
onewordqa qwen3-14b
onewordqa qwen3-8-27b
onewordqa r1-qwen-distill-7b
onewordqa r1-llama-distill-8b gen-32k
onewordqa r1-qwen3-8b
gpqa-diamond qwen3-4b-thinking
gpqa-diamond qwen3-8b
gpqa-diamond qwen3-14b
gpqa-diamond qwen3-8-27b
gpqa-diamond r1-qwen-distill-7b
gpqa-diamond r1-llama-distill-8b gen-32k
gpqa-diamond r1-qwen3-8b
gsm8k qwen3-4b-thinking
gsm8k qwen3-8b
gsm8k qwen3-14b
gsm8k qwen3-8-27b
gsm8k r1-qwen-distill-7b
gsm8k r1-llama-distill-8b gen-32k
gsm8k r1-qwen3-8b
math500 qwen3-4b-thinking
math500 qwen3-8b
math500 qwen3-14b
math500 qwen3-8-27b
math500 r1-qwen-distill-7b
math500 r1-llama-distill-8b gen-32k
math500 r1-qwen3-8b
"
main "$@"
