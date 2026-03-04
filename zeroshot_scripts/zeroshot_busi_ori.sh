#!/bin/bash

# custom config
JSON_PATH="saliency_maps/text_prompts/breast_tumors_testing.json"

echo "================================================="
echo "Preparing BUSI dataset (separating images and combining masks)..."
echo "================================================="

# Create a separated dataset layout suitable for MedCLIP-SAMv2 evaluation
python3 -c "
import os, cv2, glob, shutil, numpy as np

src_dir = 'Dataset_BUSI_with_GT'
dst_dir = 'Dataset_BUSI_eval'

for folder in ['normal']:
    src_folder = os.path.join(src_dir, folder)
    if not os.path.exists(src_folder): continue
    
    img_dst = os.path.join(dst_dir, folder, 'images')
    mask_dst = os.path.join(dst_dir, folder, 'masks')
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)
    
    for f in os.listdir(src_folder):
        if not f.endswith('.png'): continue
        if '_mask' in f: continue
        
        img_path = os.path.join(src_folder, f)
        base_name = f.replace('.png', '')
        
        masks = glob.glob(os.path.join(src_folder, f'{base_name}_mask*.png'))
        
        if len(masks) == 0:
            if folder == 'normal':
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                combined_mask = np.zeros_like(img)
            else:
                continue
        else:    
            combined_mask = None
            for m in masks:
                mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
                if mask is None: continue
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = np.maximum(combined_mask, mask)
        
        if combined_mask is not None:
            # Copy original image
            if not os.path.exists(os.path.join(img_dst, f)):
                shutil.copy(img_path, os.path.join(img_dst, f))
            # Save combined mask with the SAME name as the image (for eval.py)
            cv2.imwrite(os.path.join(mask_dst, f), combined_mask)

print('Dataset preparation complete. Preparing text prompts...')

import json

json_path_in = 'saliency_maps/text_prompts/breast_tumors_testing.json'
json_path_out = 'saliency_maps/text_prompts/busi_temp_testing.json'

with open(json_path_in, 'r') as f:
    data = json.load(f)

default_tumor_prompt = \"A medical breast mammogram revealing an area of concern suggestive of a breast tumor.\"
default_normal_prompt = \"A medical breast mammogram showing healthy breast tissue with no signs of tumors or abnormalities.\"

# Add entries for all files in dst_dir
# 'benign', 'malignant', 
for folder in ['normal']:
    folder_path = os.path.join(dst_dir, folder, 'images')
    if not os.path.exists(folder_path): continue
    for f in os.listdir(folder_path):
        if f not in data:
            if folder == 'normal':
                data[f] = default_normal_prompt
            else:
                data[f] = default_tumor_prompt

with open(json_path_out, 'w') as f:
    json.dump(data, f, indent=4)
print('Text prompts prepared.')
"

# Use the temporary JSON file
JSON_PATH="saliency_maps/text_prompts/busi_temp_testing.json"

# "benign" "malignant" 
for FOLDER in "normal"; do
    echo "================================================="
    echo "Processing BUSI ${FOLDER}..."
    echo "================================================="
    
    INPUT_PATH="Dataset_BUSI_eval/${FOLDER}/images"
    GT_PATH="Dataset_BUSI_eval/${FOLDER}/masks"
    
    if [ ! -d "${INPUT_PATH}" ] || [ -z "$(ls -A ${INPUT_PATH})" ]; then
        echo "No images found in ${INPUT_PATH}. Skipping ${FOLDER}."
        continue
    fi
    
    # Clear previous outputs to avoid mixing runs
    SAL_PATH="saliency_map_outputs/BUSI/${FOLDER}"
    COARSE_PATH="coarse_outputs/BUSI/${FOLDER}"
    SAM_PATH="sam_outputs/BUSI/${FOLDER}"
    
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
