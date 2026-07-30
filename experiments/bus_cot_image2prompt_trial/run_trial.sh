#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
EXPERIMENT_DIR="$ROOT_DIR/experiments/bus_cot_image2prompt_trial"
PYTHON_BIN="$ROOT_DIR/bioclip2/.venv/bin/python"

# Keep all downloaded model assets inside the project and use the experiment's
# additive dependencies without altering either pre-existing virtualenv.
export HF_HOME="$EXPERIMENT_DIR/.hf_cache"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$EXPERIMENT_DIR/pydeps${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" "$EXPERIMENT_DIR/generate_trial.py" "$@"
