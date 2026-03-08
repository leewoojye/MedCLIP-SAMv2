#!/bin/bash

# BUSI3 DHN Inference Script
# Target: 195 test images in BUSI3/test_images
# Uses DHN configuration with High-Fidelity prompts.

GPU=${CUDA_VISIBLE_DEVICES:-0}
PYTHON="/home/woojye2020/.conda/envs/medclipsamv2/bin/python"
ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
DATASET="$ROOT_DIR/BUSI3"
JSON_PROMPTS="$ROOT_DIR/saliency_maps/text_prompts/busi3_test_prompts.json"

# Generate prompts JSON for BUSI3 test set
if [ ! -f "$JSON_PROMPTS" ]; then
    echo "Creating BUSI3 Test Prompts JSON..."
    $PYTHON - <<EOF
import json, os

master_json = '$ROOT_DIR/saliency_maps/text_prompts/busi_test_prompts.json'
test_dir = '$DATASET/test_images'
output_json = '$JSON_PROMPTS'

with open(master_json, 'r') as f:
    master_prompts = json.load(f)

test_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
prompts = {}
benign_prompt = "A medical breast mammogram showing a well-defined, round mass suggestive of a benign breast tumor."
malignant_prompt = "A medical breast mammogram showing an irregularly shaped, spiculated mass suggestive of a malignant breast tumor."

for f in test_files:
    if f in master_prompts:
        prompts[f] = master_prompts[f]
    elif f.startswith('benign_'):
        prompts[f] = benign_prompt
    else:
        prompts[f] = malignant_prompt

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, 'w') as f:
    json.dump(prompts, f, indent=2)
print(f"Saved {len(prompts)} prompts to {output_json}")
EOF
fi

SAL_PATH="saliency_map_outputs/BUSI3_DHN_HF/test_masks"
COARSE_PATH="coarse_outputs/BUSI3_DHN_HF/test_masks"
SAM_PATH="sam_outputs/BUSI3_DHN_HF/test_masks"

rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

echo "Using GPU: $GPU"

echo "1. Generating Saliency Maps using DHN Configuration..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON saliency_maps/generate_saliency_maps.py \
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
CUDA_VISIBLE_DEVICES=$GPU $PYTHON postprocessing/postprocess_saliency_maps.py \
    --input-path "${DATASET}/test_images" \
    --output-path "${COARSE_PATH}" \
    --sal-path "${SAL_PATH}" \
    --postprocess kmeans \
    --filter

echo "3. Prompting SAM..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON segment-anything/prompt_sam.py \
    --input "${DATASET}/test_images" \
    --mask-input "${COARSE_PATH}" \
    --output "${SAM_PATH}" \
    --model-type vit_h \
    --checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
    --prompts boxes

echo "4. Evaluation..."
$PYTHON evaluation/eval.py \
    --gt_path "${DATASET}/test_masks" \
    --seg_path "${SAM_PATH}"

echo "BUSI3 DHN Inference Complete."
