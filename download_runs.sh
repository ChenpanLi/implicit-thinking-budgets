#!/usr/bin/env bash
# Download the paper's run directory from the Hugging Face Hub into $RUN_DIR: the item
# pools, batches, runs, probe store and results (61 GB). The dataset is gated because it
# holds GPQA questions in plain text: accept its terms at
# https://huggingface.co/datasets/chenpanli/fragility and run `hf auth login` first.
# A rerun resumes and skips the files already downloaded.
source "$(dirname "$0")/experiments/common.sh"
mkdir -p "$RUN_DIR"
cpu $PY -c "import os; from huggingface_hub import snapshot_download; snapshot_download('chenpanli/fragility', repo_type='dataset', local_dir=os.environ['RUN_DIR'])"
