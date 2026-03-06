#!/bin/bash

# Default to benign, but can be overridden (e.g., ./zeroshot_breast_tumors_dhn.sh malignant)
FOLDER=${1:-benign}

# Prepare BUSI dataset (separating images and combining masks) for target folder
python3 -c "
import os, cv2, glob, shutil, numpy as np
import json
import sys

src_dir = 'Dataset_BUSI_with_GT'
dst_dir = 'Dataset_BUSI_eval'
folder = sys.argv[1]

src_folder = os.path.join(src_dir, folder)
if os.path.exists(src_folder):
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
        
        combined_mask = None
        if len(masks) == 0:
            if folder == 'normal':
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                combined_mask = np.zeros_like(img)
        else:    
            for m in masks:
                mask = cv2.imread(m, cv2.IMREAD_GRAYSCALE)
                if mask is None: continue
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = np.maximum(combined_mask, mask)
        
        if combined_mask is not None:
            if not os.path.exists(os.path.join(img_dst, f)):
                shutil.copy(img_path, os.path.join(img_dst, f))
            cv2.imwrite(os.path.join(mask_dst, f), combined_mask)

print(f'Dataset preparation complete. Preparing text prompts for {folder}...')

json_path_in = 'saliency_maps/text_prompts/breast_tumors_testing.json'
json_path_out = f'saliency_maps/text_prompts/busi_{folder}_testing.json'

with open(json_path_in, 'r') as f:
    data = json.load(f)

default_tumor_prompt = \"A medical breast mammogram revealing an area of concern suggestive of a breast tumor.\"

folder_path = os.path.join(dst_dir, folder, 'images')
if os.path.exists(folder_path):
    for f in os.listdir(folder_path):
        if f not in data:
            data[f] = default_tumor_prompt

with open(json_path_out, 'w') as f:
    json.dump(data, f, indent=4)
print('Text prompts prepared.')
" "$FOLDER"

# custom config
DATASET="BUSI_${FOLDER}_DHN"
INPUT_PATH="Dataset_BUSI_eval/${FOLDER}/images"
GT_PATH="Dataset_BUSI_eval/${FOLDER}/masks"
JSON_PATH="saliency_maps/text_prompts/busi_${FOLDER}_testing.json"

SAL_PATH="saliency_map_outputs/${DATASET}/test_masks"
COARSE_PATH="coarse_outputs/${DATASET}/test_masks" 
SAM_PATH="sam_outputs/${DATASET}/test_masks"

# Clear previous outputs to avoid mixing runs
rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}" 
mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

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
# --checkpoint-path "logs_finetuning/..." # Add checkpoint path if needed

python postprocessing/postprocess_saliency_maps.py \
--input-path "${INPUT_PATH}" \
--output-path "${COARSE_PATH}" \
--sal-path "${SAL_PATH}" \
--postprocess kmeans \
--filter

python segment-anything/prompt_sam.py \
--input "${INPUT_PATH}" \
--mask-input "${COARSE_PATH}" \
--output "${SAM_PATH}" \
--model-type vit_h \
--checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
--prompts boxes

python evaluation/eval.py \
--gt_path "${GT_PATH}" \
--seg_path "${SAM_PATH}"
