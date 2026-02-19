#!/bin/bash
# Generate Healthy Samples (Zero-Shot)
# Using trained BioMedDDBM on Breast Cancer to generate Healthy Brain/Lung

CHECKPOINT="biomed_ddbm_output/biomed_ddbm_epoch_50.pt"

# 1. Brain (Tumor -> Healthy)
echo "Generating Healthy Brain Samples..."
INPUT_DIR="data/brain_tumor/test_images" # Verify this path
OUTPUT_DIR="generated_neg_output/brain_healthy_ddbm"
PROMPT="healthy brain MRI, no tumor, normal brain structure"
mkdir -p $OUTPUT_DIR

# Find images
if [ -d "$INPUT_DIR" ]; then
    for img in "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg; do
        [ -e "$img" ] || continue
        filename=$(basename "$img")
        echo "Processing $filename..."
        
        python -m biomed_ddbm.generate \
            --checkpoint "$CHECKPOINT" \
            --input_image "$img" \
            --output "$OUTPUT_DIR/$filename" \
            --prompt "$PROMPT" \
            --strength 0.65
    done
else
    echo "Warning: Brain input directory $INPUT_DIR not found."
fi

# 2. Lung (Nodule -> Healthy)
echo "Generating Healthy Lung Samples..."
INPUT_DIR="data/lung_nodule/test_images" # Verify this path
OUTPUT_DIR="generated_neg_output/lung_healthy_ddbm"
PROMPT="healthy chest X-ray, clear lungs, no nodule, normal lung tissue"
mkdir -p $OUTPUT_DIR

if [ -d "$INPUT_DIR" ]; then
    for img in "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg; do
        [ -e "$img" ] || continue
        filename=$(basename "$img")
        echo "Processing $filename..."
        
        python -m biomed_ddbm.generate \
            --checkpoint "$CHECKPOINT" \
            --input_image "$img" \
            --output "$OUTPUT_DIR/$filename" \
            --prompt "$PROMPT" \
            --strength 0.65
    done
else
    echo "Warning: Lung input directory $INPUT_DIR not found."
fi
