#!/usr/bin/env bash
# OLMo 3 7B Think at three post-training stages (SFT, DPO, RLVR), on gpqa-diamond,
# math500 and mmlu: every item alone (isolated-a, five trials, the same job as stage 1's
# pass a), then the per-term thinking traces of experiment II.
#   job <dataset> <model>
source "$(dirname "$0")/../common.sh"

job() {
    local ds=$1 model=$2
    bash experiments/1-isolated/gpu.sh "$ds" "$model" a-only
    bash experiments/II-token-ratio/gpu.sh "$ds" "$model"
}

JOBS="
gpqa-diamond olmo3-7b-think-sft
gpqa-diamond olmo3-7b-think-dpo
gpqa-diamond olmo3-7b-think-rlvr
math500 olmo3-7b-think-sft
math500 olmo3-7b-think-dpo
math500 olmo3-7b-think-rlvr
mmlu olmo3-7b-think-sft
mmlu olmo3-7b-think-dpo
mmlu olmo3-7b-think-rlvr
"
main "$@"
