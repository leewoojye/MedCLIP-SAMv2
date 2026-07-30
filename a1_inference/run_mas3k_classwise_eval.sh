#!/usr/bin/env bash
# Run one MAS3K model with all cache writes confined to a1_inference.
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
export PYTHONPATH="${PYDEPS_DIR}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_HOME="${CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${CACHE_DIR}/transformers"
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/mas3k_classwise_eval.py" "$@"
