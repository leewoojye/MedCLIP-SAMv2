#!/bin/bash

# Experiment on data/breast_tumors using DPO model
DATASET="data/breast_tumors"
INPUT_PATH="${DATASET}/test_images"
GT_PATH="${DATASET}/test_masks"
JSON_PATH="saliency_maps/text_prompts/breast_tumors_testing.json"

SAL_PATH="saliency_map_outputs/${DATASET}_DPO/test_masks"
COARSE_PATH="coarse_outputs/${DATASET}_DPO/test_masks" 
SAM_PATH="sam_outputs/${DATASET}_DPO/test_masks"

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
