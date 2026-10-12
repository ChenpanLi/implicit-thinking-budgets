#!/usr/bin/env bash
# OLMo 3 7B Think after SFT, DPO and RLVR: token ratio by k, accuracy drop against the
# token ratio (GPQA-Diamond), and how much the stages share their fragile tasks.
# The first step caches the per-term-trace and isolated runs under
# $RUN_DIR/results/VII-olmo-post-training/cache/.
source "$(dirname "$0")/../common.sh"
D=experiments/VII-olmo-post-training
cpu $PY $D/olmo_data.py
cpu $PY $D/token_ratio_figures.py --dataset gpqa-diamond
cpu $PY $D/fragility_rho.py
