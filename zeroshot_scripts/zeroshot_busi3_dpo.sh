#!/bin/bash

# BUSI3 DPO Inference Script
# Target: 195 test images in BUSI3/test_images
# Uses DPO fine-tuned model (v21 - trained on BUSI3 train set) with High-Fidelity prompts.

GPU=${CUDA_VISIBLE_DEVICES:-0}
PYTHON="/home/woojye2020/.conda/envs/medclipsamv2/bin/python"
ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
DATASET="$ROOT_DIR/BUSI_aug"
JSON_PROMPTS="$ROOT_DIR/saliency_maps/text_prompts/busi_aug_test_prompts.json"
MODEL_PATH="$ROOT_DIR/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v30/best_params"

SAL_PATH="saliency_map_outputs/BUSI_aug_DPO_v30_HF/test_masks"
COARSE_PATH="coarse_outputs/BUSI_aug_DPO_v30_HF/test_masks"
SAM_PATH="sam_outputs/BUSI_aug_DPO_v30_HF/test_masks"

rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

echo "Using GPU: $GPU"

echo "1. Generating Saliency Maps using DPO v21 Model..."
CUDA_VISIBLE_DEVICES=$GPU $PYTHON saliency_maps/generate_saliency_maps.py \
    --input-path "${DATASET}/test_images" \
    --output-path "${SAL_PATH}" \
    --model-name BiomedCLIP \
    --finetuned \
    --checkpoint-path "${MODEL_PATH}" \
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

echo "BUSI3 DPO Inference Complete."
