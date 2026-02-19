#!/usr/bin/env bash
# source /opt/conda/etc/profile.d/conda.sh
# conda activate medclipsamv2
# set -euo pipefail

# Run BiomedCLIP-guided negative sample generation for brain tumor test images.
# Limit to 50 images for experimental comparison.

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
IMG_DIR="$ROOT_DIR/data/brain_tumors/test_images"
MASK_DIR="$ROOT_DIR/data/brain_tumors/test_masks"
OUT_DIR="$ROOT_DIR/generated_neg_output/brain_tumors_guided"

# Ensure output directory exists
mkdir -p "$OUT_DIR"

echo "Starting guided generation..."
echo "Input: $IMG_DIR"
echo "Output: $OUT_DIR"

conda run -n medclipsamv2 python -u generate_neg_guided.py \
    --image "$IMG_DIR" \
    --mask "$MASK_DIR" \
    --out "$OUT_DIR" \
    --limit 30 \
    --scale 200.0

echo "Done. Outputs in $OUT_DIR"
