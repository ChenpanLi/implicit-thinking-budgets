#!/usr/bin/env bash
# Thinking tokens and accuracy drop by prompt position, MATH-500 at k=20.
source "$(dirname "$0")/../common.sh"
D=experiments/IV-position-effect
cpu $PY $D/position_effect_csv.py
cpu $PY $D/position_effect.py
