#!/usr/bin/env bash
# Evaluate the 4-class Chest CT regional-noise CSV with the shared model runner.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_csv_classwise_eval.sh" \
  --csv "${SCRIPT_DIR}/chest_ct_200x4_classwise_eval.csv" \
  --output-dir "${SCRIPT_DIR}/csv_classwise_results/chest_ct_200x4" \
  --class-order bronchus_noise heart_noise lung_noise normal \
  --models openai_clip siglip medsiglip biomedclip \
  "$@"
