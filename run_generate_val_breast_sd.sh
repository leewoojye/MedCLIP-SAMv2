#!/bin/bash
# Generate Healthy Samples (Validation Set) using Stable Diffusion
# Source: data/breast_tumors/val_images
# Target: generated_neg_output/breast_val_sd

INPUT_DIR="data/breast_tumors/val_images"
MASK_DIR="data/breast_tumors/val_masks"
OUTPUT_DIR="generated_neg_output/breast_val_sd"
PROMPT="healthy breast tissue, no tumor, normal mammogram"

mkdir -p $OUTPUT_DIR

echo "Generating Healthy Samples for 10 Validation Images using Stable Diffusion..."

/home/woojye2020/.conda/envs/medclipsamv2/bin/python generate_neg_guided.py \
    --image "$INPUT_DIR" \
    --mask "$MASK_DIR" \
    --out "$OUTPUT_DIR" \
    --target "$PROMPT" \
    --limit 10 \
    --scale 200.0

echo "Done. Results saved in $OUTPUT_DIR"
