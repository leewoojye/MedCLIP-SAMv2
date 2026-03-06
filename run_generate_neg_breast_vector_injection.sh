#!/usr/bin/env bash

# Run BiomedCLIP vector injection zero-shot translation on UDIAT dataset.
# Uses the breast-specific vector_injection_breast.py script.

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
UDIAT_DIR="$ROOT_DIR/UDIAT3"
OUT_BASE_DIR="$ROOT_DIR/generated_neg_output/busi_vector_injection_breast3"

echo "Starting UDIAT Vector Injection (Breast Anchor) generation..."
echo "Input Directory: $UDIAT_DIR"
echo "Output Base Directory: $OUT_BASE_DIR"

# Process UDIAT3 splits (train)
for split in "train"; do
    IMG_DIR="$UDIAT_DIR/${split}_images"
    MASK_DIR="$UDIAT_DIR/${split}_masks"

    echo "Processing Split: $split"

    if [[ ! -d "$IMG_DIR" ]]; then
        echo "Directory not found: $IMG_DIR. Skipping."
        continue
    fi

    for img_path in "$IMG_DIR"/*.png; do
        # Check if the file exists to handle the case where the directory might be empty
        if [[ ! -f "$img_path" ]]; then
            echo "No .png images found in $IMG_DIR. Skipping."
            break
        fi

        filename=$(basename "$img_path")
        mask_path="$MASK_DIR/$filename"
        
        # Determine original category to save in the correct output folder (for CSV compatibility)
        if [[ -f "$ROOT_DIR/UDIAT/Benign/$filename" ]]; then
            category="Benign"
        else
            category="Malignant"
        fi
        
        OUT_DIR="$OUT_BASE_DIR/$category"
        mkdir -p "$OUT_DIR"
        out_path="$OUT_DIR/$filename"

        # Check if the corresponding mask exists
        if [[ ! -f "$mask_path" ]]; then
            echo "Warning: Mask not found for $filename. Expected at: $mask_path. Skipping."
            continue
        fi

        echo "  [$split -> $category] Generating for: $filename"
        
        # Run the specialized breast vector injection python script
        conda run -n medclipsamv2 python -u "$ROOT_DIR/zero_shot_translation/vector_injection_breast.py" \
            --image "$img_path" \
            --mask "$mask_path" \
            --out "$out_path" \
            --alpha 1.0 \
            --tumor_anchor "tumor breast" \
            --healthy_anchor "healthy breast" \
            --dilate_mask 3 \
            --sd_model "runwayml/stable-diffusion-inpainting"
    done
done

echo "Done. All outputs saved in $OUT_BASE_DIR"
