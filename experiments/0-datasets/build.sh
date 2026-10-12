#!/usr/bin/env bash
# Stage 0: the eight item pools. Six are downloaded from the Hugging Face Hub at pinned
# revisions (GPQA needs HF_TOKEN with the gated dataset's terms accepted); integrals and
# onewordqa are copied from datasets/<dataset>/. Writes $RUN_DIR/datasets/<dataset>.jsonl
# and a manifest; a pool already on disk is left alone when it is identical.
source "$(dirname "$0")/../common.sh"

job() { local ds=$1; cpu $PY -m llm.pools "$ds"; }

JOBS="
integrals
onewordqa
gpqa-diamond
gsm8k
math500
mmlu
strategyqa
aime
"
main "$@"
