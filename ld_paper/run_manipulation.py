"""
Disease-conditioned Manipulation — Main Inference Script.

Implements Section 4.2.2 of "Latent Drifting in Diffusion Models for
Counterfactual Medical Image Synthesis", adapted for brain tumors:

  Brain Tumor (positive) ──Pix2Pix Zero + LD──► Healthy Brain (negative)
  Healthy Brain (negative) ──Pix2Pix Zero + LD──► Brain Tumor (positive)

Usage
-----
# Tumor → Healthy  (primary use case)
python run_manipulation.py \
    --model_path ./checkpoints \
    --source_data_dir ./data/test/tumor \
    --source_label tumor \
    --target_label healthy \
    --output_dir ./generated/tumor2healthy \
    --delta 0.1

# Healthy → Tumor  (reverse direction)
python run_manipulation.py \
    --model_path ./checkpoints \
    --source_data_dir ./data/test/healthy \
    --source_label healthy \
    --target_label tumor \
    --output_dir ./generated/healthy2tumor \
    --delta 0.1

# Use base SD-v1.4 without fine-tuning (inference-only LD)
python run_manipulation.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --source_data_dir ./data/test/tumor \
    --output_dir ./generated/no_ft_tumor2healthy \
    --delta 0.1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from tqdm.auto import tqdm

from pix2pix_zero_ld import Pix2PixZeroWithLD, load_pipeline
from prompts import get_manipulation_prompt_pair, get_prompt


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Disease-conditioned Manipulation via Pix2Pix Zero + Latent Drifting.\n"
            "Brain Tumor (positive) → Healthy (negative) counterfactual synthesis."
        )
    )
    # ---- Model -----------------------------------------------------------
    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="HuggingFace model ID or path to fine-tuned directory.",
    )
    # ---- Data ------------------------------------------------------------
    parser.add_argument(
        "--source_data_dir",
        type=str,
        required=True,
        help="Directory containing source images (expects flat folder of images).",
    )
    parser.add_argument(
        "--source_label",
        type=str,
        default="tumor",
        choices=["tumor", "healthy"],
        help="Label of the source images.",
    )
    parser.add_argument(
        "--target_label",
        type=str,
        default="healthy",
        choices=["tumor", "healthy"],
        help="Desired target label for the generated counterfactuals.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of images to process (None = all).",
    )
    # ---- Latent Drifting -------------------------------------------------
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Latent drift δ (paper Section 3.3).",
    )
    # ---- Pix2Pix Zero hyperparameters ------------------------------------
    parser.add_argument("--num_inversion_steps", type=int,   default=50)
    parser.add_argument("--num_denoise_steps",   type=int,   default=50)
    parser.add_argument("--guidance_scale",      type=float, default=7.5)
    parser.add_argument(
        "--tau",
        type=float,
        default=0.1,
        help="Cross-attention guidance weight τ (Pix2Pix Zero).",
    )
    # ---- Prompting -------------------------------------------------------
    parser.add_argument(
        "--source_prompt",
        type=str,
        default=None,
        help="Fixed source prompt. If None, auto-generated from --source_label.",
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        default=None,
        help="Fixed target prompt. If None, auto-generated from --target_label.",
    )
    parser.add_argument("--use_diverse_prompts", action="store_true", default=True)
    # ---- Misc ------------------------------------------------------------
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--save_diff",  action="store_true",
                        help="Save pixel-difference image (source - counterfactual).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source_images(directory: str, max_n: Optional[int]) -> List[Path]:
    """Return sorted list of image paths from `directory`."""
    paths = [
        p for p in sorted(Path(directory).iterdir())
        if p.suffix.lower() in SUPPORTED_EXTS
    ]
    if max_n is not None:
        paths = paths[:max_n]
    return paths


def save_diff_image(source: Image.Image, edited: Image.Image, path: Path) -> None:
    """Save absolute pixel-difference image (× 5 for visibility)."""
    import numpy as np
    src_arr = np.array(source.convert("RGB"), dtype=np.float32)
    edt_arr = np.array(edited.convert("RGB"), dtype=np.float32)
    diff = np.abs(src_arr - edt_arr) * 5
    diff = diff.clip(0, 255).astype(np.uint8)
    Image.fromarray(diff).save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Output dirs -----------------------------------------------------
    out_dir = Path(args.output_dir)
    (out_dir / "counterfactual").mkdir(parents=True, exist_ok=True)
    if args.save_diff:
        (out_dir / "diff").mkdir(parents=True, exist_ok=True)

    # ---- Load pipeline ---------------------------------------------------
    print(f"Loading pipeline from: {args.model_path}")
    dtype    = torch.float16 if str(device) == "cuda" else torch.float32
    pipeline = load_pipeline(args.model_path, dtype=dtype, device=str(device))

    # ---- Build editor ----------------------------------------------------
    editor = Pix2PixZeroWithLD(
        pipeline=pipeline,
        delta=args.delta,
        num_ddim_inversion_steps=args.num_inversion_steps,
        num_ddim_denoising_steps=args.num_denoise_steps,
        guidance_scale=args.guidance_scale,
        cross_attention_guidance_amount=args.tau,
        dtype=dtype,
    )

    # ---- Resolve prompts -------------------------------------------------
    if args.source_prompt is None or args.target_prompt is None:
        auto_src, auto_tgt = get_manipulation_prompt_pair(
            source_label=args.source_label,
            target_label=args.target_label,
            use_diverse=args.use_diverse_prompts,
        )
    source_prompt = args.source_prompt or auto_src
    target_prompt = args.target_prompt or auto_tgt

    print(f"\nSource label   : {args.source_label}  →  '{source_prompt}'")
    print(f"Target label   : {args.target_label}  →  '{target_prompt}'")
    print(f"Latent Drift δ : {args.delta:+.3f}")
    print(f"Guidance τ     : {args.tau}")
    print(f"CFG scale      : {args.guidance_scale}")

    # ---- Load source images ----------------------------------------------
    source_paths = load_source_images(args.source_data_dir, args.max_samples)
    if not source_paths:
        raise RuntimeError(f"No images found in {args.source_data_dir}")

    print(f"\nProcessing {len(source_paths)} images …\n")

    # ---- Run manipulation ------------------------------------------------
    metadata = []

    for idx, src_path in enumerate(tqdm(source_paths, desc="Editing")):
        source_image = Image.open(src_path).convert("RGB").resize((512, 512), Image.BICUBIC)

        # Optionally re-sample diverse prompt per image
        if args.use_diverse_prompts and args.source_prompt is None:
            src_p, tgt_p = get_manipulation_prompt_pair(
                source_label=args.source_label,
                target_label=args.target_label,
                use_diverse=True,
            )
        else:
            src_p, tgt_p = source_prompt, target_prompt

        print(f"\n[{idx+1}/{len(source_paths)}] {src_path.name}")

        edited_image, _ = editor.edit(
            source_image=source_image,
            source_prompt=src_p,
            target_prompt=tgt_p,
            seed=args.seed + idx,
            verbose=True,
        )

        # Save outputs
        stem = src_path.stem
        save_path  = out_dir / "counterfactual" / f"{stem}_counterfactual.png"
        edited_image.save(save_path)

        if args.save_diff:
            diff_path = out_dir / "diff" / f"{stem}_diff.png"
            save_diff_image(source_image, edited_image, diff_path)

        metadata.append({
            "source_path"   : str(src_path),
            "output_path"   : str(save_path),
            "source_label"  : args.source_label,
            "target_label"  : args.target_label,
            "source_prompt" : src_p,
            "target_prompt" : tgt_p,
            "delta"         : args.delta,
            "seed"          : args.seed + idx,
        })

    # ---- Save metadata ---------------------------------------------------
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Generated {len(metadata)} counterfactual images.")
    print(f"  Counterfactuals : {out_dir / 'counterfactual'}")
    if args.save_diff:
        print(f"  Difference maps : {out_dir / 'diff'}")
    print(f"  Metadata        : {meta_path}")


if __name__ == "__main__":
    main()
