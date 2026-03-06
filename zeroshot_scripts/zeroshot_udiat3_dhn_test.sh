#!/bin/bash

# UDIAT3 Test Split DHN Inference Script (High-Fidelity Prompts)
# Target: 49 test images in UDIAT3/test_images
# Uses DHN configuration with complex prompts from breast_tumors_testing.json

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
DATASET="$ROOT_DIR/UDIAT3"
JSON_PROMPTS="$ROOT_DIR/saliency_maps/text_prompts/udiat3_test_prompts.json"

SAL_PATH="saliency_map_outputs/UDIAT3_DHN_HF/test_masks"
COARSE_PATH="coarse_outputs/UDIAT3_DHN_HF/test_masks"
SAM_PATH="sam_outputs/UDIAT3_DHN_HF/test_masks"

echo "Creating High-Fidelity UDIAT3 Test Prompts JSON..."
python3 - <<EOF
import json
import os

master_json = '$ROOT_DIR/saliency_maps/text_prompts/breast_tumors_testing.json'
udiat3_test_dir = '$DATASET/test_images'
output_json = '$JSON_PROMPTS'

with open(master_json, 'r') as f:
    master_prompts = json.load(f)

test_files = [f for f in os.listdir(udiat3_test_dir) if f.endswith('.png')]
udiat3_prompts = {}
missing_count = 0

# Fallback prompts
benign_prompt = "A medical breast mammogram showing a well-defined, round mass suggestive of a benign breast tumor."
malignant_prompt = "A medical breast mammogram showing an irregularly shaped, spiculated mass suggestive of a malignant breast tumor."

for f in test_files:
    if f in master_prompts:
        udiat3_prompts[f] = master_prompts[f]
    else:
        missing_count += 1
        # Determine category for fallback
        if os.path.exists(os.path.join('$ROOT_DIR/UDIAT/Benign', f)):
            udiat3_prompts[f] = benign_prompt
        else:
            udiat3_prompts[f] = malignant_prompt

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, 'w') as f:
    json.dump(udiat3_prompts, f, indent=2)

print(f"Total test files: {len(test_files)}")
print(f"Used complex prompts: {len(test_files) - missing_count}")
print(f"Used simple fallbacks: {missing_count}")
print(f"Saved to {output_json}")
EOF

# Clear previous outputs
rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

echo "1. Generating Saliency Maps using DHN Configuration..."
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
