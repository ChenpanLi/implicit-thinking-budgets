#!/usr/bin/env bash
# The batching tables: accuracy and output tokens per task of isolated solving, random
# batching at k=6 and fragility-aware batching (measured and probe), per dataset and
# model, plus the averages the text quotes. CPU only; reads runs/, writes
# results/VI-fragility-aware-batching/.
source "$(dirname "$0")/../common.sh"
cpu $PY experiments/VI-fragility-aware-batching/batching_table.py
