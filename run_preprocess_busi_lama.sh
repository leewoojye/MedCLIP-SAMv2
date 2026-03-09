#!/usr/bin/env bash
set -euo pipefail

ENV_PATH="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/envs/busi_lama"
PROJECT_ROOT="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"

cd "$PROJECT_ROOT"
conda run -p "$ENV_PATH" python preprocess_busi.py "$@"
