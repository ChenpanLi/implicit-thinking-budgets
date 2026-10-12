#!/usr/bin/env bash
# Rebuild every figure, table and number of the paper from the runs in $RUN_DIR.
# CPU only: it reads finished runs and never starts a model. Outputs go to
# $RUN_DIR/results/<experiment>/. The first run takes about an hour, mostly spent
# re-tokenizing the per-term traces (II and III); the cached parts make reruns fast.
set -euo pipefail
cd "$(dirname "$0")"

for experiment in \
    I-success-conditioned-drop \
    II-token-ratio \
    III-budget-forcing \
    IV-position-effect \
    V-fragility-correlation \
    VI-fragility-aware-batching \
    VII-olmo-post-training
do
    echo "=== $experiment"
    bash "experiments/$experiment/analyze.sh"
done
