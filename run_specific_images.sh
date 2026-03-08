#!/usr/bin/env bash
set -euo pipefail

# 특정 이미지 번호(파일명)만 골라서 음성 샘플을 생성하는 스크립트
# /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/zero_shot_translation/vector_injection_biomedclip.py

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"

# === ⭐️ 여기에 원하는 이미지 파일명들을 나열하세요 ===
# 예시: TARGET_IMAGES=("000006.png" "000007.png" "000015.png")
TARGET_IMAGES=("000077.png" "000062.png" "000085.png" "000086.png" "000089.png" "000095.png" "000125.png" "000121.png")
# =======================================================

IMG_DIR="$ROOT_DIR/UDIAT2/test_images"
MASK_DIR="$ROOT_DIR/UDIAT2/test_masks"

OUT_DIR="$ROOT_DIR/generated_neg_output/udiat_breast_biomedclip_specific"
COMPARE_DIR="$OUT_DIR/comparisons"
CKPT_PATH="$ROOT_DIR/zero_shot_translation/sd_biomedclip_ckpt/best.pt"
DILATE_MASK="3"
TUMOR_ANCHOR="breast ultrasound with tumor"

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

# Bash 배열을 Python 리스트로 가져오기
target_files = """${TARGET_IMAGES[@]}""".split()

if not target_files:
    print("No target images specified.")
    exit(0)

print(f"Target Images specifically selected: {target_files}")

for filename in target_files:
    img_path = img_dir / filename
    mask_path = mask_dir / filename
    out_path = out_dir / filename
    
    if not img_path.exists():
        print(f"[skip] image missing: {filename}")
        continue
    if not mask_path.exists():
        print(f"[skip] mask missing: {filename}")
        continue
    
    print(f"\n[run] Generating for {filename}")
    
    cmd = [
        "python", "zero_shot_translation/vector_injection_biomedclip.py",
        "--image", str(img_path),
        "--mask", str(mask_path),
        "--out", str(out_path),
        "--ckpt_path", ckpt_path,
        "--tumor_anchor", tumor_anchor_val,
        "--healthy_anchor", "A medical breast mammogram with no visible tumors or abnormalities.",
        "--alpha", "0.0",
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
        combo.save(cmp_dir / filename)

EOF

echo "Done. Outputs in $OUT_DIR"
