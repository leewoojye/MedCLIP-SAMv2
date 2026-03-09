#!/bin/bash

# UDIATnBUSI_TEST DPO Inference Script
# Uses DPO finetuned model with High-Fidelity prompts.
# v19 : udiatNbusi

GPU=${CUDA_VISIBLE_DEVICES:-1}
PYTHON="/home/woojye2020/.conda/envs/medclipsamv2/bin/python"
ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
DATASET="$ROOT_DIR/UDIAT7nBUSI7_aug"
JSON_PROMPTS="$ROOT_DIR/saliency_maps/text_prompts/udiat7nbusi7_aug_test_prompts.json"
CHECKPOINT="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v29/best_params"

SAL_PATH="saliency_map_outputs/UDIAT7nBUSI7_aug_TEST_DPO_v29_HF/test_masks"
COARSE_PATH="coarse_outputs/UDIAT7nBUSI7_aug_TEST_DPO_v29_HF/test_masks"
SAM_PATH="sam_outputs/UDIAT7nBUSI7_aug_TEST_DPO_v29_HF/test_masks"

mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

echo "Using GPU: $GPU"

CUDA_VISIBLE_DEVICES=$GPU $PYTHON saliency_maps/generate_saliency_maps.py \
    --input-path "${DATASET}/test_images" \
    --output-path "${SAL_PATH}" \
    --model-name BiomedCLIP \
    --checkpoint-path "${CHECKPOINT}" \
    --finetuned \
    --json-path "${JSON_PROMPTS}" \
    --reproduce \
    --vvar 1.0 \
    --vbeta 1.0 \
    --vlayer 9 \
    --seed 12

CUDA_VISIBLE_DEVICES=$GPU $PYTHON postprocessing/postprocess_saliency_maps.py \
    --input-path "${DATASET}/test_images" \
    --output-path "${COARSE_PATH}" \
    --sal-path "${SAL_PATH}" \
    --postprocess kmeans \
    --filter

CUDA_VISIBLE_DEVICES=$GPU $PYTHON segment-anything/prompt_sam.py \
    --input "${DATASET}/test_images" \
    --mask-input "${COARSE_PATH}" \
    --output "${SAM_PATH}" \
    --model-type vit_h \
    --checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
    --prompts boxes

$PYTHON evaluation/eval.py \
    --gt_path "${DATASET}/test_masks" \
    --seg_path "${SAM_PATH}"
