#!/usr/bin/env bash
# Correlate the per-item accuracy drop across models, across k and between the two
# passes (pass A = isolated-a + random-a, pass B = isolated-b + random-b), then draw the
# appendix heatmaps. CPU only; reads runs/, writes results/V-fragility-correlation/.
source "$(dirname "$0")/../common.sh"
cpu $PY experiments/V-fragility-correlation/fragility_correlation.py
cpu $PY experiments/V-fragility-correlation/plot_fragility_correlation.py
