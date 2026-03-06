#!/bin/bash

# Dedicated script for BUSI 'normal' dataset inference
FOLDER="normal"

# Prepare BUSI dataset (separating images and creating zero-masks) for normal folder
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
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # For normal case, GT mask is always zero
        combined_mask = np.zeros_like(img)
        
        if not os.path.exists(os.path.join(img_dst, f)):
            shutil.copy(img_path, os.path.join(img_dst, f))
        cv2.imwrite(os.path.join(mask_dst, f), combined_mask)

print(f'Dataset preparation complete. Preparing text prompts for {folder}...')

json_path_out = f'saliency_maps/text_prompts/busi_{folder}_testing.json'

# Use a specific prompt for 'normal' cases
normal_prompt = \"A medical breast mammogram with no visible tumors or abnormalities.\"

data = {}
folder_path = os.path.join(dst_dir, folder, 'images')
if os.path.exists(folder_path):
    for f in os.listdir(folder_path):
        data[f] = normal_prompt

with open(json_path_out, 'w') as f:
    json.dump(data, f, indent=4)
print('Text prompts prepared.')
" "$FOLDER"

# custom config
DATASET="BUSI_${FOLDER}_DPO"
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
--checkpoint-path "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v13/hf_model" \
--json-path "${JSON_PATH}" \
--reproduce \
--vvar 1.0 \
--vbeta 1.0 \
--vlayer 9 \
--seed 12

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
