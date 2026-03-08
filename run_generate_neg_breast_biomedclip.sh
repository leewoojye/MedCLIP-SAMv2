#!/usr/bin/env bash
set -euo pipefail

# Run negative sample generation using the fine-tuned SD + BiomedCLIP Cross-Attention weights
# SD clip을 biomedclip으로 교체 후 접합부 크로스 어텐션만 medpix로 미세조정함
# alpha 0.0으로 설정시 vector shift를 사용하지 않고 normal vector injection만 사용

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"

# 음성샘플 생성대상 경로지정
# IMG_DIR="$ROOT_DIR/BUSI_TEST/test_images"
# MASK_DIR="$ROOT_DIR/BUSI_TEST/test_masks"
IMG_DIR="$ROOT_DIR/UDIAT2/test_images"
MASK_DIR="$ROOT_DIR/UDIAT2/test_masks"

OUT_DIR="$ROOT_DIR/generated_neg_output/udiat_breast_biomedclip0.0v2"
COMPARE_DIR="$OUT_DIR/comparisons"
CKPT_PATH="$ROOT_DIR/zero_shot_translation/sd_biomedclip_ckpt/best.pt"
DILATE_MASK="3"
TUMOR_ANCHOR="breast ultrasound with tumor"
# A medical breast mammogram revealing an area of concern suggestive of a breast tumor.

mkdir -p "$OUT_DIR"
mkdir -p "$COMPARE_DIR"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_PATH"
    echo "Please ensure the training script has completed at least 1 epoch."
    exit 1
fi

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
ckpt_path = "$CKPT_PATH"
dilate_mask_val = "$DILATE_MASK"
tumor_anchor_val = "$TUMOR_ANCHOR"

# Run for breast tumor test images
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
    
    # Call the python script we just created
    cmd = [
        "python", "zero_shot_translation/vector_injection_biomedclip.py",
        "--image", str(img_path),
        "--mask", str(mask_path),
        "--out", str(out_path),
        "--ckpt_path", ckpt_path,
        "--tumor_anchor", tumor_anchor_val,
        "--healthy_anchor", "A medical breast mammogram with no visible tumors or abnormalities.",
        "--alpha", "0", # Direct generation without shift vector is fully supported now
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

EOF

echo "Done. Outputs in $OUT_DIR"
