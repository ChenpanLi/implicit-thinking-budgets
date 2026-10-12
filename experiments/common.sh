# Sourced by every stage script. Not meant to be run on its own.
#
#   bash experiments/<stage>/gpu.sh                    every job of the stage, one after another
#   bash experiments/<stage>/gpu.sh gsm8k qwen3-8b     one job (to spread jobs over GPUs)
#   bash experiments/<stage>/gpu.sh --list             print the jobs
#   DRY_RUN=1 bash experiments/<stage>/gpu.sh ...      CPU steps run, GPU steps only check
#                                                      their inputs and render prompts
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
# .env fills in RUN_DIR, but a RUN_DIR already in the environment wins (as in paths.py),
# so a dry run can point at a scratch directory.
_env_run_dir="${RUN_DIR:-}"
if [ -f .env ]; then set -a; source .env; set +a; fi
if [ -n "$_env_run_dir" ]; then RUN_DIR="$_env_run_dir"; fi
: "${RUN_DIR:?RUN_DIR is not set: copy .env.example to .env and set it}"
export RUN_DIR
PY="${PY:-$REPO/.venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"

# A CPU step: always executed, also in a dry run.
cpu() { echo "+ $*"; "$@"; }

# A step that needs a GPU (or a billed API): in a dry run it gets --dry-run, which
# resolves every input and output path and renders prompts without loading a model.
gpu() {
    echo "+ $*"
    if [ "$DRY_RUN" = 1 ]; then "$@" --dry-run; else "$@"; fi
}

# The stage script defines `job <args...>` and a JOBS list (one job per line, `#`
# comments allowed), then ends with `main "$@"`.
main() {
    if [ "${1:-}" = "--list" ]; then
        grep -v '^\s*\(#\|$\)' <<< "$JOBS"
    elif [ $# -gt 0 ]; then
        job "$@"
    else
        while read -r line; do
            [[ -z "${line// }" || "$line" =~ ^[[:space:]]*# ]] && continue
            # shellcheck disable=SC2086
            job $line
        done <<< "$JOBS"
    fi
}
