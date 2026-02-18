#!/bin/bash

# custom config
# Best Hyperparameter Combination:
# vbeta           1.000000
# vvar            2.000000
# vlayer          8.000000
# average_dice    0.039843
# Name: 16, dtype: float64

# Enter the path to your dataset
DATASET="data/pancreas"

SAL_PATH="saliency_map_outputs/${DATASET}/masks"
COARSE_PATH="coarse_outputs/${DATASET}/masks"
SAM_PATH="sam_outputs/${DATASET}/masks"

# Clear previous outputs to avoid mixing runs
rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

python saliency_maps/generate_saliency_maps.py \
--input-path ${DATASET}/test_images \
--output-path saliency_map_outputs/${DATASET}/masks \
--val-path ${DATASET}/val_images \
--model-name BiomedCLIP \
--hyper-opt \
--val-path ${DATASET}/val_images
# --finetuned \
# --json-path saliency_maps/text_prompts/pancreas_testing.json \

python postprocessing/postprocess_saliency_maps.py \
--input-path ${DATASET}/test_images \
--output-path coarse_outputs/${DATASET}/masks \
--sal-path saliency_map_outputs/${DATASET}/masks \
--postprocess kmeans \
--filter
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