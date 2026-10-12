#!/usr/bin/env bash
# Budget forcing against composition: the token-ratio figures of the main text and the
# appendix. CPU only. build_points.py is the slow step (it re-tokenizes every isolated
# answer); its cells are cached under results/III-budget-forcing/parts/, so a second run
# only redraws. Pass --refresh to build_points.py to recompute them.
source "$(dirname "$0")/../common.sh"

cpu $PY experiments/III-budget-forcing/build_points.py
cpu $PY experiments/III-budget-forcing/plot_main.py
cpu $PY experiments/III-budget-forcing/plot_grids.py
