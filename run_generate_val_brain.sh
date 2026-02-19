#!/bin/bash
# Generate Healthy Samples (Validation Set) - Brain
# Source: data/brain_tumors/val_images
# Masks: data/brain_tumors/val_masks
# Target: generated_neg_output/brain_val_ddbm

CHECKPOINT="biomed_ddbm_output/biomed_ddbm_epoch_50.pt"
INPUT_DIR="data/brain_tumors/val_images"
MASK_DIR="data/brain_tumors/val_masks"
OUTPUT_DIR="generated_neg_output/brain_val_ddbm"
PROMPT="healthy brain MRI, no tumor, normal brain tissue"

mkdir -p $OUTPUT_DIR

echo "Generating Healthy Samples for 10 Validation Images (Brain)..."

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
    # Try exact match or with _mask suffix or just same name in mask dir
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
