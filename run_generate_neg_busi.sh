#!/usr/bin/env bash
set -euo pipefail

# Run negative sample generation for BUSI dataset (benign and malignant).
# Handles multiple masks per image (e.g., _mask.png, _mask_1.png, _mask_2.png).

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
BUSI_DIR="$ROOT_DIR/Dataset_BUSI_with_GT"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="$ROOT_DIR/generated_neg_output/BUSI_$TIMESTAMP"

mkdir -p "$OUT_DIR"

python - <<PY
import pathlib
import tempfile

import cv2
import numpy as np
from PIL import Image
from generate_neg import generate_healthy_ultrasound

root = pathlib.Path("/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2")
busi_dir = root / "Dataset_BUSI_with_GT"
out_dir = pathlib.Path("$OUT_DIR")

# Detect and remove color overlays (arrows/lines) before inpainting by fusing their mask
# with the lesion mask. Handles red/yellow/white markings commonly used in BUSI.


def detect_overlay_mask(img_path: pathlib.Path) -> np.ndarray:
    """Return uint8 mask (0/255) of colored overlays (red/yellow/white)."""
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((1, 1), np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Color ranges (tunable)
    mask_r1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    mask_r2 = cv2.inRange(hsv, (170, 70, 70), (180, 255, 255))
    mask_y = cv2.inRange(hsv, (20, 80, 80), (35, 255, 255))
    mask_w = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))

    mask = mask_r1 | mask_r2 | mask_y | mask_w
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return mask


def load_mask_as_uint8(mask_path: pathlib.Path) -> np.ndarray:
    arr = np.array(Image.open(mask_path).convert("L"))
    return np.where(arr > 0, 255, 0).astype(np.uint8)


# Process benign and malignant folders
# for category in ["benign", "malignant"]:
for category in ["malignant"]:
    cat_dir = busi_dir / category
    if not cat_dir.exists():
        print(f"[skip] {category} folder not found")
        continue
    
    # Create output subdirectories
    cat_out = out_dir / category
    cmp_out = cat_out / "comparisons"
    cat_out.mkdir(parents=True, exist_ok=True)
    cmp_out.mkdir(parents=True, exist_ok=True)
    
    # Find all original images (not mask files)
    all_files = sorted(cat_dir.glob("*.png"))
    images = [f for f in all_files if "_mask" not in f.name]
    
    print(f"\n=== Processing {category}: {len(images)} images ===")
    
    for img_path in images:
        # Find all associated mask files
        base_name = img_path.stem  # e.g., "benign (1)"
        # Pattern: base_name + "_mask" + optional "_N" + ".png"
        # Note: glob uses wildcards, not regex, so don't escape parentheses
        mask_pattern = f"{base_name}_mask*.png"
        masks = sorted(cat_dir.glob(mask_pattern))
        
        if not masks:
            print(f"[skip] {img_path.name} - no mask found")
            continue
        
        print(f"[run] {img_path.name} ({len(masks)} mask(s)) - merging all masks")
        
        # Merge ALL lesion masks for this image
        first_mask = load_mask_as_uint8(masks[0])
        h, w = first_mask.shape
        combined_lesion_mask = np.zeros((h, w), dtype=np.uint8)
        
        for mask_path in masks:
            lesion_mask = load_mask_as_uint8(mask_path)
            # Resize if needed
            if lesion_mask.shape != (h, w):
                lesion_mask = cv2.resize(lesion_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            combined_lesion_mask = cv2.bitwise_or(combined_lesion_mask, lesion_mask)
        
        # Detect and merge overlay mask (arrows, text, etc.)
        overlay_mask = detect_overlay_mask(img_path)
        if overlay_mask.shape != (h, w):
            overlay_mask = cv2.resize(overlay_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Merge lesion + overlay masks
        final_mask = cv2.bitwise_or(combined_lesion_mask, overlay_mask)
        
        # Apply stronger dilation to cover edge artifacts (3~5 pixels)
        final_mask = cv2.dilate(final_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        # Single inpainting pass with merged mask
        out_name = img_path.name
        out_path = cat_out / out_name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_mask_file:
            Image.fromarray(final_mask).save(tmp_mask_file.name)
            # Generate healthy tissue version using merged mask
            generate_healthy_ultrasound(str(img_path), tmp_mask_file.name, str(out_path))
        
        # Create comparison image (original | generated)
        if out_path.exists():
            src = Image.open(img_path).convert("RGB")
            gen = Image.open(out_path).convert("RGB")
            
            # Resize to same height
            h = max(src.height, gen.height)
            def resize_keep_height(im, target_h):
                if im.height != target_h:
                    return im.resize((int(im.width * target_h / im.height), target_h))
                return im
            
            src_r = resize_keep_height(src, h)
            gen_r = resize_keep_height(gen, h)
            
            # Concatenate side by side
            combo = Image.new("RGB", (src_r.width + gen_r.width, h))
            combo.paste(src_r, (0, 0))
            combo.paste(gen_r, (src_r.width, 0))
            
            cmp_name = f"{img_path.stem}_cmp{img_path.suffix}"
            combo.save(cmp_out / cmp_name)

print(f"\nDone. Outputs in {out_dir}")
PY

echo "Done. Check outputs in $OUT_DIR"
