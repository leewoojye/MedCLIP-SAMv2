#!/usr/bin/env bash
# CSV -> embeddings -> cosine scores -> unchanged Clip4Retrofit metrics.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/bioclip2/.venv/bin/python"
PYDEPS_DIR="${SCRIPT_DIR}/pydeps"
CACHE_DIR="${SCRIPT_DIR}/.hf_cache"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected evaluation Python was not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${PYDEPS_DIR}" ]]; then
  echo "Missing ${PYDEPS_DIR}. Run a1_inference/setup_csv_classwise_eval.sh first." >&2
  exit 1
fi

mkdir -p "${CACHE_DIR}/hub" "${CACHE_DIR}/transformers"
export PYTHONPATH="${PYDEPS_DIR}:${PYTHONPATH:-}"
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
export HF_HUB_DISABLE_XET=1

# This PyTorch build supports the local A6000 GPUs (physical indices 1, 2, 3)
# but not the Blackwell GPU at physical index 0. Respect an explicit user
# setting; otherwise keep evaluation on the three compatible A6000 devices.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/csv_classwise_eval.py" "$@"
