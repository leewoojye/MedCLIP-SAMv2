#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"

# Use UDIAT2 as testing directory
IMG_DIR="$ROOT_DIR/UDIAT2/test_images"
MASK_DIR="$ROOT_DIR/UDIAT2/test_masks"

# Normal image reference (passed as image prompt)
NORMAL_IMAGE="$ROOT_DIR/Dataset_BUSI_with_GT/normal/normal (1).png"

OUT_DIR="$ROOT_DIR/generated_neg_output/udiat_clip_image2prompt_closedform"
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

root = pathlib.Path("$ROOT_DIR")
img_dir = pathlib.Path("$IMG_DIR")
mask_dir = pathlib.Path("$MASK_DIR")
out_dir = pathlib.Path("$OUT_DIR")
cmp_dir = pathlib.Path("$COMPARE_DIR")
normal_image = "$NORMAL_IMAGE"
dilate_mask_val = "$DILATE_MASK"
projector_path = "$PROJECTOR_PATH"

images = sorted(img_dir.glob("*.png"))
if not images:
    raise SystemExit(f"No images found in {img_dir}")

for img_path in images:
    mask_path = mask_dir / img_path.name
    if not mask_path.exists():
        print(f"[skip] mask missing for {img_path.name}")
        continue
    
    out_path = out_dir / img_path.name
    print(f"[run] Generating for {img_path.name}")
    
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
        combo.save(cmp_dir / img_path.name)

