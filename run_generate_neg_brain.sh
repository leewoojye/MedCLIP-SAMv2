#!/usr/bin/env bash
set -euo pipefail

# Run negative sample generation for all brain tumor test images.

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
IMG_DIR="$ROOT_DIR/data/brain_tumors/test_images"
MASK_DIR="$ROOT_DIR/data/brain_tumors/test_masks"
OUT_DIR="$ROOT_DIR/generated_neg_output/brain_tumors"
COMPARE_DIR="$OUT_DIR/comparisons"

mkdir -p "$OUT_DIR"
mkdir -p "$COMPARE_DIR"

python - <<'PY'
import pathlib
from PIL import Image
from generate_neg import generate_healthy_brain

# Root path might differ in python env, explicit or use ROOT_DIR from bash via env or just hardcode as in original
root = pathlib.Path("/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2")
img_dir = root / "data/brain_tumors/test_images"
mask_dir = root / "data/brain_tumors/test_masks"
out_dir = root / "generated_neg_output/brain_tumors"
cmp_dir = out_dir / "comparisons"

# Ensure output directories exist in python too if needed, but bash did it
out_dir.mkdir(parents=True, exist_ok=True)
cmp_dir.mkdir(parents=True, exist_ok=True)

# Brain tumor images are also .png as verified
images = sorted(img_dir.glob("*.png"))[:50]

if not images:
    raise SystemExit(f"No images found in {img_dir}")

for img_path in images:
    mask_path = mask_dir / img_path.name
    if not mask_path.exists():
        print(f"[skip] mask missing for {img_path.name}")
        continue
    print(f"[run] {img_path.name}")
    
    # Use the brain-specific function
    generate_healthy_brain(str(img_path), str(mask_path), str(out_dir))

    gen_path = out_dir / img_path.name
    if gen_path.exists():
        src = Image.open(img_path).convert("RGB")
        gen = Image.open(gen_path).convert("RGB")
        # Keep side-by-side with matching height.
        h = max(src.height, gen.height)
        def resize_keep_height(im, target_h):
            return im.resize((int(im.width * target_h / im.height), target_h)) if im.height != target_h else im
        src_r = resize_keep_height(src, h)
        gen_r = resize_keep_height(gen, h)
        combo = Image.new("RGB", (src_r.width + gen_r.width, h))
        combo.paste(src_r, (0, 0))
        combo.paste(gen_r, (src_r.width, 0))
        combo.save(cmp_dir / img_path.name)
PY

echo "Done. Outputs in $OUT_DIR"
