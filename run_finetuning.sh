#!/bin/bash

# Configuration
DATA_PATH="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT"
MODEL_NAME="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

PYTHON_EXEC="/home/woojye2020/.conda/envs/medclipsamv2/bin/python"

# =================================================================================================
# Stage 1: Positive-Positive Alignment (Tumor Alignment via Sinkhorn)
# =================================================================================================
echo "================================================================================================="
echo "Starting Stage 1: Positive-Positive Alignment..."
echo "================================================================================================="
# Generate a unique timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Variables
STAGE1_NAME="egobridge_stage1_pos_pos_${TIMESTAMP}"
STAGE1_OUTPUT="./logs_finetuning/${STAGE1_NAME}"
STAGE1_EPOCHS=15
STAGE1_BATCH_SIZE=32
STAGE1_LR=1e-5
STAGE1_SINKHORN_EPS=0.05
STAGE1_CONTRASTIVE_LAMBDA=1.0

$PYTHON_EXEC train_hf.py \
    --data-dir "$DATA_PATH" \
    --output-dir "$STAGE1_OUTPUT" \
    --stage stage1 \
    --batch-size "$STAGE1_BATCH_SIZE" \
    --epochs "$STAGE1_EPOCHS" \
    --lr "$STAGE1_LR" \
    --sinkhorn-eps "$STAGE1_SINKHORN_EPS" \
    --contrastive-lambda "$STAGE1_CONTRASTIVE_LAMBDA" \
    --num-workers 4

echo "Stage 1 Completed."

# =================================================================================================
# Stage 2: Positive-Negative Contrast (Tumor vs Normal)
# =================================================================================================
echo "================================================================================================="
echo "Starting Stage 2: Positive-Negative Contrast..."
echo "================================================================================================="

# Find the checkpoint from Stage 1
# Assuming train_hf.py saves "epoch_N" directory with config.json and model.safetensors/bin
STAGE1_CHECKPOINT="$STAGE1_OUTPUT/epoch_${STAGE1_EPOCHS}"

if [ ! -d "$STAGE1_CHECKPOINT" ]; then
    echo "Error: Stage 1 checkpoint directory not found at $STAGE1_CHECKPOINT"
    echo "Exiting."
    exit 1
else
    echo "Found Stage 1 checkpoint: $STAGE1_CHECKPOINT"
fi

# Variables
STAGE2_NAME="egobridge_stage2_pos_neg_${TIMESTAMP}"
STAGE2_OUTPUT="./logs_finetuning/${STAGE2_NAME}"
STAGE2_EPOCHS=10
STAGE2_BATCH_SIZE=32
STAGE2_LR=1e-6 # Lower LR for Stage 2
STAGE2_CONTRASTIVE_LAMBDA=1.0

$PYTHON_EXEC train_hf.py \
    --data-dir "$DATA_PATH" \
    --output-dir "$STAGE2_OUTPUT" \
    --stage stage2 \
    --pretrained-checkpoint "$STAGE1_CHECKPOINT" \
    --batch-size "$STAGE2_BATCH_SIZE" \
    --epochs "$STAGE2_EPOCHS" \
    --lr "$STAGE2_LR" \
    --contrastive-lambda "$STAGE2_CONTRASTIVE_LAMBDA" \
    --num-workers 4

echo "================================================================================================="
echo "Fine-tuning Pipeline Completed!"
echo "Final Model Key Checkpoint: $STAGE2_OUTPUT/epoch_${STAGE2_EPOCHS}"
echo "================================================================================================="

