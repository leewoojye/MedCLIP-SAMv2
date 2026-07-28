#!/usr/bin/env bash
# Install the only missing BioMedCLIP dependency without changing any existing
# project virtual environment.  The target directory is deliberately local to
# a1_inference and is added by run_csv_classwise_eval.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/bioclip2/.venv/bin/python"
PYDEPS_DIR="${SCRIPT_DIR}/pydeps"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Expected evaluation Python was not found: ${PYTHON_BIN}" >&2
  exit 1
fi

mkdir -p "${PYDEPS_DIR}"
CURRENT_PACKAGES="$(PYTHONPATH="${PYDEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" - <<'PY'
try:
    from importlib.metadata import version
    print(f'{version("open-clip-torch")},{version("sentencepiece")}')
except Exception:
    print("")
PY
)"

if [[ "${CURRENT_PACKAGES}" != "2.23.0,0.2.0" ]]; then
  "${PYTHON_BIN}" -m pip install --upgrade --no-deps --target "${PYDEPS_DIR}" \
    -r "${SCRIPT_DIR}/requirements-csv-eval.txt"
else
  echo "CSV evaluator packages already installed in ${PYDEPS_DIR}"
fi

PYTHONPATH="${PYDEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -c 'import open_clip; print("open_clip import: OK")'
