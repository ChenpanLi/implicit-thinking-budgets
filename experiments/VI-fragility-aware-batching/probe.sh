#!/usr/bin/env bash
# The per-item composition sizes of fragility-aware batching, both arms (llm.probe).
# Reads the isolated and random runs of stages 1 and 2; writes $RUN_DIR/probe/.
#   bash probe.sh embed <dataset>           item embeddings (GPU, once per dataset)
#   bash probe.sh splits <dataset>          the five outer folds (CPU)
#   bash probe.sh pair <dataset> <model>    labels, 20 heads, both assignments (CPU)
#   bash probe.sh tokens                    token-cost.csv for every pair (CPU)
# Order: embed and splits for every dataset, then tokens, then every pair.
# The 20 heads of a pair train in a few minutes on CPU (2000 steps each).
source "$(dirname "$0")/../common.sh"

job() {
    local what=$1; shift
    case $what in
    embed)  gpu $PY -m llm.probe.embed_items --dataset "$1" ;;
    splits) cpu $PY -m llm.probe.make_splits --dataset "$1" ;;
    tokens) cpu $PY -m llm.probe.tokens ;;
    pair)
        cpu $PY -m llm.probe.build_labels --dataset "$1" --model "$2"
        cpu $PY -m llm.probe.train --dataset "$1" --model "$2"
        cpu $PY -m llm.probe.assign_k --arm measured --dataset "$1" --model "$2"
        cpu $PY -m llm.probe.assign_k --arm probe --dataset "$1" --model "$2" ;;
    opus)
        # Claude Opus 5 on GPQA-Diamond has one random pass. Its measured arm reads
        # isolated-a only (5 + 5 trials), its probe arm both isolated passes (10 + 5),
        # so each arm gets its own token prices.
        local m=claude-opus-5 p=$RUN_DIR/probe
        cpu $PY -m llm.probe.build_labels --dataset gpqa-diamond --model $m --random random-a
        cpu $PY -m llm.probe.train --dataset gpqa-diamond --model $m
        cpu $PY -m llm.probe.tokens --dataset gpqa-diamond --model $m --isolated isolated-a \
            --random random-a --out $p/token-cost-opus-measured.csv
        cpu $PY -m llm.probe.tokens --dataset gpqa-diamond --model $m --random random-a \
            --out $p/token-cost-opus-probe.csv
        cpu $PY -m llm.probe.assign_k --arm measured --dataset gpqa-diamond --model $m \
            --isolated isolated-a --random random-a --token-cost $p/token-cost-opus-measured.csv
        cpu $PY -m llm.probe.assign_k --arm probe --dataset gpqa-diamond --model $m \
            --token-cost $p/token-cost-opus-probe.csv ;;
    *) echo "unknown job $what" >&2; return 1 ;;
    esac
}

JOBS="
embed integrals
embed onewordqa
embed gpqa-diamond
embed gsm8k
embed math500
embed mmlu
embed strategyqa
splits integrals
splits onewordqa
splits gpqa-diamond
splits gsm8k
splits math500
splits mmlu
splits strategyqa
# Rebuilds token-cost.csv from all 10 + 10 trials. The paper's assignments read the
# stored table, which pooled fewer trials for some pairs (see README); skip this job to
# reproduce them.
tokens
pair integrals qwen3-4b-thinking
pair integrals qwen3-8b
pair integrals qwen3-14b
pair integrals qwen3-8-27b
pair integrals r1-qwen-distill-7b
pair integrals r1-llama-distill-8b
pair integrals r1-qwen3-8b
pair onewordqa qwen3-4b-thinking
pair onewordqa qwen3-8b
pair onewordqa qwen3-14b
pair onewordqa qwen3-8-27b
pair onewordqa r1-qwen-distill-7b
pair onewordqa r1-llama-distill-8b
pair onewordqa r1-qwen3-8b
pair gpqa-diamond qwen3-4b-thinking
pair gpqa-diamond qwen3-8b
pair gpqa-diamond qwen3-14b
pair gpqa-diamond qwen3-8-27b
pair gpqa-diamond r1-qwen-distill-7b
pair gpqa-diamond r1-llama-distill-8b
pair gpqa-diamond r1-qwen3-8b
pair gsm8k qwen3-4b-thinking
pair gsm8k qwen3-8b
pair gsm8k qwen3-14b
pair gsm8k qwen3-8-27b
pair gsm8k r1-qwen-distill-7b
pair gsm8k r1-llama-distill-8b
pair gsm8k r1-qwen3-8b
pair math500 qwen3-4b-thinking
pair math500 qwen3-8b
pair math500 qwen3-14b
pair math500 qwen3-8-27b
pair math500 r1-qwen-distill-7b
pair math500 r1-llama-distill-8b
pair math500 r1-qwen3-8b
pair mmlu qwen3-4b-thinking
pair mmlu qwen3-8b
pair mmlu qwen3-14b
pair mmlu qwen3-8-27b
pair mmlu r1-qwen-distill-7b
pair mmlu r1-llama-distill-8b
pair mmlu r1-qwen3-8b
pair strategyqa qwen3-4b-thinking
pair strategyqa qwen3-8b
pair strategyqa qwen3-14b
pair strategyqa qwen3-8-27b
pair strategyqa r1-qwen-distill-7b
pair strategyqa r1-llama-distill-8b
pair strategyqa r1-qwen3-8b
opus
"
main "$@"
