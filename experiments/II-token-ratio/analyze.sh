#!/usr/bin/env bash
# Token ratio against composition size: the main-text figure, one appendix figure per
# model, the power-law fits behind the captions, and how often traces tag every task.
# The first step loads every (dataset, model) once and caches it under
# $RUN_DIR/results/II-token-ratio/cache/ (re-tokenizing the isolated answers is slow).
source "$(dirname "$0")/../common.sh"
D=experiments/II-token-ratio
cpu $PY $D/trace_data.py
cpu $PY $D/token_ratio_main.py
cpu $PY $D/token_ratio_per_model.py
cpu $PY $D/power_law_fits.py
cpu $PY $D/tag_coverage.py
