#!/bin/bash
# Generate Healthy Samples (Validation Set) using Un-Finetuned Model
# Source: data/breast_tumors/val_images
# Target: generated_neg_output/breast_val_ddbm_unfinetuned

# Use a dummy checkpoint name to trigger the exception block in generate.py
# that catches missing/incompatible checkpoints and starts fresh (un-finetuned)
CHECKPOINT="dummy_unfinetuned_checkpoint.pt"
INPUT_DIR="data/breast_tumors/val_images"
MASK_DIR="data/breast_tumors/val_masks"
OUTPUT_DIR="generated_neg_output/breast_val_ddbm_unfinetuned"
PROMPT="healthy breast tissue, no tumor, normal mammogram"

mkdir -p $OUTPUT_DIR

echo "Generating Healthy Samples for 10 Validation Images using Un-finetuned Model..."

# Limit to 10 images
count=0
for img in "$INPUT_DIR"/*.png "$INPUT_DIR"/*.jpg; do
    [ -e "$img" ] || continue
    
    if [ "$count" -ge 10 ]; then
        break
    fi
    
    filename=$(basename "$img")
    
    # Find Mask
    mask=""
    if [ -f "$MASK_DIR/$filename" ]; then
        mask="$MASK_DIR/$filename"
    elif [ -f "$MASK_DIR/${filename%.*}_mask.png" ]; then
        mask="$MASK_DIR/${filename%.*}_mask.png"
    fi
    
    echo "Processing ($((count+1))/10) $filename..."
    if [ -n "$mask" ]; then
        echo "  Using mask: $mask"
    else
        echo "  Warning: No mask found!"
    fi
    
    /home/woojye2020/.conda/envs/medclipsamv2/bin/python -m biomed_ddbm.generate \
        --checkpoint "$CHECKPOINT" \
        --input_image "$img" \
        --mask_image "$mask" \
        --output "$OUTPUT_DIR/$filename" \
        --prompt "$PROMPT" \

        
    count=$((count + 1))
done

echo "Done. Results saved in $OUTPUT_DIR"
