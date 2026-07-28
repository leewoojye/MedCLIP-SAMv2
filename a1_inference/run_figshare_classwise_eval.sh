#!/usr/bin/env bash
# Figshare 3 tumor types + DDPM dilation=9 normal samples, using the shared
# raw-cosine and unchanged Clip4Retrofit OVR metric implementation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${SCRIPT_DIR}/run_csv_classwise_eval.sh" \
  --csv "${SCRIPT_DIR}/figshare3064x2_classwise_eval.csv" \
  --output-dir "${SCRIPT_DIR}/csv_classwise_results/figshare3064x2_dilation9" \
  --class-order meningioma glioma pituitary normal \
  --models openai_clip siglip medsiglip biomedclip \
  "$@"
