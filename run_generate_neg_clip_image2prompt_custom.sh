#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# SPECIFY YOUR DESIRED IMAGES HERE
# Add the exact filenames you want to process, separated by spaces.
# Example: TARGET_IMAGES="000001.png 000005.png 000010.png"
# ==============================================================================
TARGET_IMAGES="000077.png 000062.png 000085.png 000086.png 000089.png 000095.png 000125.png 000121.png 000049.png"

echo "Will process the following images: $TARGET_IMAGES"

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"

# Use UDIAT2 as testing directory
IMG_DIR="$ROOT_DIR/UDIAT2/test_images"
MASK_DIR="$ROOT_DIR/UDIAT2/test_masks"

# Normal image reference (passed as image prompt)
NORMAL_IMAGE="$ROOT_DIR/Dataset_BUSI_with_GT/normal/normal (1).png"

OUT_DIR="$ROOT_DIR/generated_neg_output/udiat_clip_image2prompt_specific"
COMPARE_DIR="$OUT_DIR/comparisons"
PROJECTOR_PATH="$ROOT_DIR/zero_shot_translation/closed_form_projector.pt"
DILATE_MASK="3"

mkdir -p "$OUT_DIR"
mkdir -p "$COMPARE_DIR"

python - <<EOF
import pathlib
import os
import subprocess
from PIL import Image
import sys

root = pathlib.Path("$ROOT_DIR")
img_dir = pathlib.Path("$IMG_DIR")
mask_dir = pathlib.Path("$MASK_DIR")
out_dir = pathlib.Path("$OUT_DIR")
cmp_dir = pathlib.Path("$COMPARE_DIR")
normal_image = "$NORMAL_IMAGE"
dilate_mask_val = "$DILATE_MASK"
projector_path = "$PROJECTOR_PATH"
target_images_str = "$TARGET_IMAGES"

target_images = target_images_str.split()

if not target_images:
    print("Warning: No target images specified in TARGET_IMAGES variable in the bash script.")
    sys.exit(0)

for target_img_name in target_images:
    img_path = img_dir / target_img_name
    if not img_path.exists():
        print(f"Error: Target image {img_path} not found. Skipping.")
        continue

    mask_path = mask_dir / target_img_name
    if not mask_path.exists():
        print(f"Error: mask missing for {target_img_name}. Skipping.")
        continue

    out_path = out_dir / target_img_name
    print(f"[run] Generating for {target_img_name}")

    cmd = [
        "python", "zero_shot_translation/vector_injection_clip_image2prompt.py",
        "--image", str(img_path),
        "--mask", str(mask_path),
        "--normal_image", normal_image,
        "--projector_path", projector_path,
        "--out", str(out_path),
        "--dilate_mask", dilate_mask_val
    ]

    subprocess.run(cmd, check=True)

    # Create Side-by-Side Comparison
    if out_path.exists():
        src = Image.open(img_path).convert("RGB")
        gen = Image.open(out_path).convert("RGB")
        h = max(src.height, gen.height)
        def resize_keep_height(im, target_h):
            return im.resize((int(im.width * target_h / im.height), target_h)) if im.height != target_h else im
        src_r = resize_keep_height(src, h)
        gen_r = resize_keep_height(gen, h)
        combo = Image.new("RGB", (src_r.width + gen_r.width, h))
        combo.paste(src_r, (0, 0))
        combo.paste(gen_r, (src_r.width, 0))
        combo.save(cmp_dir / target_img_name)

    print(f"Done processing {target_img_name}")

EOF

echo "Finished script execution."
