#!/bin/bash

# UDIAT3 Test Split DHN Inference Script (High-Fidelity Prompts)
# Target: 49 test images in UDIAT3/test_images
# Uses DHN configuration with complex prompts from breast_tumors_testing.json

# Robustly find root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( dirname "$SCRIPT_DIR" )"
cd "$ROOT_DIR" || exit

DATASET="$ROOT_DIR/UDIAT4"
TEST_CSV="$DATASET/udiat_test_augmented.csv"
JSON_PROMPTS="$ROOT_DIR/saliency_maps/text_prompts/udiat4_test_prompts.json"
SAL_PATH="$ROOT_DIR/saliency_map_outputs/UDIAT4_DHN_HF/test_masks"
COARSE_PATH="$ROOT_DIR/coarse_outputs/UDIAT4_DHN_HF/test_masks"
SAM_PATH="$ROOT_DIR/sam_outputs/UDIAT4_DHN_HF/test_masks"

# Clear previous outputs
rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

echo "1. Generating Saliency Maps..."
python saliency_maps/generate_saliency_maps.py \
    --input-path "${DATASET}/test_images" \
    --output-path "${SAL_PATH}" \
    --model-name BiomedCLIP \
    --finetuned \
    --json-path "${JSON_PROMPTS}" \
    --reproduce \
    --vvar 1.0 \
    --vbeta 1.0 \
    --vlayer 9 \
    --seed 12

echo "2. Postprocessing..."
python postprocessing/postprocess_saliency_maps.py \
    --input-path "${DATASET}/test_images" \
    --output-path "${COARSE_PATH}" \
    --sal-path "${SAL_PATH}" \
    --postprocess kmeans \
    --filter

echo "3. Prompting SAM..."
python segment-anything/prompt_sam.py \
    --input "${DATASET}/test_images" \
    --mask-input "${COARSE_PATH}" \
    --output "${SAM_PATH}" \
    --model-type vit_h \
    --checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
    --prompts boxes

echo "4. Evaluation..."
python evaluation/eval.py \
    --gt_path "${DATASET}/test_masks" \
    --seg_path "${SAM_PATH}"

echo "UDIAT3 Test DHN Inference Complete."
