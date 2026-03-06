#!/usr/bin/env bash

# Run BiomedCLIP vector injection zero-shot translation on BUSI_TEST dataset.
# Uses the breast-specific vector_injection_breast.py script.

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
BUSI_DIR="$ROOT_DIR/BUSI_TEST"
OUT_BASE_DIR="$ROOT_DIR/generated_neg_output/busi_test_vector_injection"

echo "Starting BUSI_TEST Vector Injection (Breast Anchor) generation..."
echo "Input Directory: $BUSI_DIR"
echo "Output Base Directory: $OUT_BASE_DIR"

IMG_DIR="$BUSI_DIR/test_images"
MASK_DIR="$BUSI_DIR/test_masks"

if [[ ! -d "$IMG_DIR" ]]; then
    echo "Directory not found: $IMG_DIR. Exiting."
    exit 1
fi

for img_path in "$IMG_DIR"/*.png; do
    if [[ ! -f "$img_path" ]]; then
        echo "No .png images found in $IMG_DIR. Exiting."
        break
    fi

    filename=$(basename "$img_path")
    mask_path="$MASK_DIR/$filename"

    # Determine category from filename prefix (benign_ or malignant_)
    if [[ "$filename" == benign_* ]]; then
        category="benign"
    else
        category="malignant"
    fi

    OUT_DIR="$OUT_BASE_DIR/$category"
    mkdir -p "$OUT_DIR"
    out_path="$OUT_DIR/$filename"

    # Skip if already generated
    if [[ -f "$out_path" ]]; then
        echo "  [skip] Already exists: $filename"
        continue
    fi

    if [[ ! -f "$mask_path" ]]; then
        echo "Warning: Mask not found for $filename. Expected at: $mask_path. Skipping."
        continue
    fi

    echo "  [$category] Generating for: $filename"

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

echo "Done. All outputs saved in $OUT_BASE_DIR"
