#!/bin/bash

# custom config
# Best Hyperparameter Combination:
# vbeta           0.100000
# vvar            1.000000
# vlayer          7.000000
# average_dice    0.327873
# Name: 3, dtype: float64
# >>>>>>>>>>>>>>>>>>>>
# Average DSC for masks: 0.3822098
# Average NSD for masks: 0.399592
# <<<<<<<<<<<<<<<<<<<<

# Enter the path to your dataset
DATASET="data/polyp"

SAL_PATH="saliency_map_outputs/${DATASET}/masks"
COARSE_PATH="coarse_outputs/${DATASET}/masks"
SAM_PATH="sam_outputs/${DATASET}/masks"

# Clear previous outputs to avoid mixing runs
rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

python saliency_maps/generate_saliency_maps.py \
--input-path ${DATASET}/test_images \
--output-path saliency_map_outputs/${DATASET}/masks \
--model-name BiomedCLIP \
--finetuned \
--json-path saliency_maps/text_prompts/polyp_testing.json \
--reproduce \
--vvar 0.1 \
--vbeta 1.0 \
--vlayer 7 \
--seed 12

python postprocessing/postprocess_saliency_maps.py \
--input-path ${DATASET}/test_images \
--output-path coarse_outputs/${DATASET}/masks \
--sal-path saliency_map_outputs/${DATASET}/masks \
--postprocess kmeans \
--filter \
# --num-contours 2 # number of contours to extract, for lungs, use 2 contours

python segment-anything/prompt_sam.py \
--input ${DATASET}/test_images \
--mask-input coarse_outputs/${DATASET}/masks \
--output sam_outputs/${DATASET}/masks \
--model-type vit_h \
--checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
--prompts boxes \
# --multicontour # for lungs, use this flag

python evaluation/eval.py \
--gt_path ${DATASET}/test_masks \
--seg_path sam_outputs/${DATASET}/masks