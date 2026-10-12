#!/usr/bin/env bash
# Experiment I, CPU part: the accuracy drop of confidently solved items under composition
# (two GPQA-Diamond figures and the all-datasets appendix figure), and the context and
# parsing audit of the composed responses (the context-usage figure and the appendix
# table). Reads runs/{composed,isolated-a,isolated-b,random-a}/, writes
# results/I-success-conditioned-drop/. context_records.py loads tokenizers (cached in the
# Hugging Face cache, never downloaded) and must run before the two context scripts.
source "$(dirname "$0")/../common.sh"
D=experiments/I-success-conditioned-drop
cpu $PY $D/gpqa_drop_k10.py
cpu $PY $D/gpqa_drop_by_k.py
cpu $PY $D/all_datasets_drop.py
cpu $PY $D/context_records.py
cpu $PY $D/context_usage.py
cpu $PY $D/context_table.py
