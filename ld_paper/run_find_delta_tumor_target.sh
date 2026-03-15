#!/bin/bash
#
# Grid Search for Optimal δ
# Paper Section 3.4: δ* = argmin_δ L1(D_θ(·|δ), D_GT)
#

set -e
cd "$(dirname "$0")"

MODEL_PATH="CompVis/stable-diffusion-v1-4"
SOURCE_DIR="./data/train/tumor"
TARGET_DIR="./data/train/healthy"
OUTPUT_FILE="optimal_delta_tumor_target.txt"
GENERATED_DIR="./generated_delta_search_tumor"

python find_delta_tumor_target.py \
    --model_path "$MODEL_PATH" \
    --source_data_dir "$SOURCE_DIR" \
    --reference_data_dir "$TARGET_DIR" \
    --source_prompt "a brain MRI with brain tumor" \
    --target_prompt "a healthy brain MRI" \
    --num_samples 10 \
    --delta_min -0.2 \
    --delta_max 0.2 \
    --delta_steps 9 \
    --num_inversion_steps 50 \
    --num_denoise_steps 50 \
    --guidance_scale 7.5 \
    --tau 0.1 \
    --seed 42 \
    --output_file "$OUTPUT_FILE" \
    --device cuda \
    --save_generated \
    --output_dir "$GENERATED_DIR"

