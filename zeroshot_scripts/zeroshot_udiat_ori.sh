#!/bin/bash

# custom config
JSON_PATH="saliency_maps/text_prompts/breast_tumors_testing.json"

echo "================================================="
echo "Preparing UDIAT text prompts..."
echo "================================================="
python3 -c "
import json
import os

json_path_in = 'saliency_maps/text_prompts/breast_tumors_testing.json'
json_path_out = 'saliency_maps/text_prompts/udiat_temp_testing.json'

with open(json_path_in, 'r') as f:
    data = json.load(f)

default_tumor_prompt = \"A medical breast mammogram revealing an area of concern suggestive of a breast tumor.\"
default_normal_prompt = \"A medical breast mammogram showing healthy breast tissue with no signs of tumors or abnormalities.\"

for folder in ['Benign', 'Malignant']:
    folder_path = os.path.join('UDIAT', folder)
    if not os.path.exists(folder_path): continue
    for f in os.listdir(folder_path):
        if f not in data and f.endswith('.png'):
            if folder == 'normal':
                data[f] = default_normal_prompt
            else:
                data[f] = default_tumor_prompt

with open(json_path_out, 'w') as f:
    json.dump(data, f, indent=4)
"
JSON_PATH="saliency_maps/text_prompts/udiat_temp_testing.json"

for FOLDER in "Benign" "Malignant"; do
    echo "================================================="
    echo "Processing UDIAT ${FOLDER}..."
    echo "================================================="
    
    INPUT_PATH="UDIAT/${FOLDER}"
    GT_PATH="UDIAT/${FOLDER}_mask"
    
    # Clear previous outputs to avoid mixing runs
    SAL_PATH="saliency_map_outputs/UDIAT/${FOLDER}"
    COARSE_PATH="coarse_outputs/UDIAT/${FOLDER}"
    SAM_PATH="sam_outputs/UDIAT/${FOLDER}"
    
    rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
    mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
    
    echo "1. Generating Saliency Maps..."
    python saliency_maps/generate_saliency_maps.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${SAL_PATH}" \
    --model-name BiomedCLIP \
    --finetuned \
    --json-path "${JSON_PATH}" \
    --reproduce \
    --vvar 1.0 \
    --vbeta 1.0 \
    --vlayer 9 \
    --seed 12
    
    echo "2. Postprocessing Saliency Maps..."
    python postprocessing/postprocess_saliency_maps.py \
    --input-path "${INPUT_PATH}" \
    --output-path "${COARSE_PATH}" \
    --sal-path "${SAL_PATH}" \
    --postprocess kmeans \
    --filter
    
    echo "3. Prompting SAM..."
    python segment-anything/prompt_sam.py \
    --input "${INPUT_PATH}" \
    --mask-input "${COARSE_PATH}" \
    --output "${SAM_PATH}" \
    --model-type vit_h \
    --checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
    --prompts boxes
    
    echo "4. Evaluation..."
    python evaluation/eval.py \
    --gt_path "${GT_PATH}" \
    --seg_path "${SAM_PATH}"

done
