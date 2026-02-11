#!/bin/bash
# run_evaluation.sh
# Evalutes the fine-tuned BioMedCLIP model by generating Saliency Maps on the test set.

# 1. Environment Setup
export PYTHONPATH="$PYTHONPATH:$(pwd)/biomedclip_finetuning/open_clip/src"
# Use the same python environment as training
PYTHON_PATH="/home/woojye2020/.conda/envs/medclipsamv2/bin/python"

# 2. Model & Data Paths
# Use the best checkpoint from Stage 2 (or latest if preferred)
CHECKPOINT_PATH="./logs_finetuning/egobridge_stage2_pos_neg_20260211_115516/checkpoints/epoch_10.pt"
TEST_DATA_PATH="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/data/breast_tumors/test_images"
OUTPUT_PATH="./saliency_map_results"

# 3. Execution (Assuming modifications to generate_saliency_maps.py to load .pt directly)
echo "================================================================================================="
echo "Starting Evaluation: Generating Saliency Maps..."
echo "Model Checkpoint: $CHECKPOINT_PATH"
echo "Test Data: $TEST_DATA_PATH"
echo "Output Directory: $OUTPUT_PATH"
echo "================================================================================================="

$PYTHON_PATH saliency_maps/generate_saliency_maps.py \
    --input-path "$TEST_DATA_PATH" \
    --output-path "$OUTPUT_PATH" \
    --model-name "BiomedCLIP" \
    --finetuned \
    --reproduce \
    --json-path "saliency_maps/busi.json" 

echo "Evaluation Completed. Check results in $OUTPUT_PATH"
