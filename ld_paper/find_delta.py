"""
Grid Search for Optimal Latent Drift δ.

Paper (Section 3.4, Supplement):
  "In Eq. (5), if (λ > 0), the minimum value of δ is found via grid search
   to minimise the distance function.  Here, we employed the L1norm as the
   distance function and tune hyperparameter δ."

Algorithm
---------
For each δ ∈ {−0.2, −0.15, …, 0.0, …, 0.15, 0.2}:
  1. Generate a small batch of images using the model conditioned on δ
     (source images → candidate edits).
  2. Compute L1 distance between the generated images and the reference
     healthy-brain images (D_GT).
Select  δ* = argmin_δ  L1(Dθ(·|δ), D_GT).

Usage
-----
python find_delta.py \
    --model_path ./checkpoints \
    --source_data_dir ./data/test/tumor \
    --reference_data_dir ./data/test/healthy \
    --source_prompt "a brain MRI with brain tumor" \
    --target_prompt "a healthy brain MRI" \
    --num_samples 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm.auto import tqdm

from latent_drifting import compute_l1_distribution_distance, find_optimal_delta
from pix2pix_zero_ld import Pix2PixZeroWithLD, load_pipeline
from prompts import get_manipulation_prompt_pair


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for optimal Latent Drift δ"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Path to fine-tuned model or HuggingFace model ID.",
    )
    parser.add_argument(
        "--source_data_dir",
        type=str,
        required=True,
        help="Directory with source (tumor) images.",
    )
    parser.add_argument(
        "--reference_data_dir",
        type=str,
        required=True,
        help="Directory with reference (healthy) images from D_GT.",
    )
    parser.add_argument(
        "--source_prompt",
        type=str,
        default="a brain MRI with brain tumor",
    )
    parser.add_argument(
        "--target_prompt",
        type=str,
        default="a healthy brain MRI",
    )
    parser.add_argument("--num_samples",        type=int,   default=5)
    parser.add_argument("--delta_min",          type=float, default=-0.2)
    parser.add_argument("--delta_max",          type=float, default=0.2)
    parser.add_argument("--delta_steps",        type=int,   default=9)
    parser.add_argument("--num_inversion_steps",type=int,   default=50)
    parser.add_argument("--num_denoise_steps",  type=int,   default=50)
    parser.add_argument("--guidance_scale",     type=float, default=7.5)
    parser.add_argument("--tau",                type=float, default=0.1)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--output_file",        type=str,   default="optimal_delta.txt")
    parser.add_argument("--device",             type=str,   default="cuda")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load images
# ---------------------------------------------------------------------------

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def load_images(directory: str, max_n: int) -> List[Image.Image]:
    paths = [
        p for p in sorted(Path(directory).iterdir())
        if p.suffix.lower() in SUPPORTED_EXTS
    ][:max_n]
    return [Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC) for p in paths]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load pipeline
    print(f"Loading pipeline from {args.model_path} …")
    pipeline = load_pipeline(args.model_path, device=str(device))

    # Load images
    print(f"Loading source images from {args.source_data_dir} …")
    source_images = load_images(args.source_data_dir, args.num_samples)

    print(f"Loading reference images from {args.reference_data_dir} …")
    reference_images = load_images(args.reference_data_dir, args.num_samples * 3)

    if not source_images:
        raise RuntimeError(f"No source images found in {args.source_data_dir}")
    if not reference_images:
        raise RuntimeError(f"No reference images found in {args.reference_data_dir}")

    print(f"Source images  : {len(source_images)}")
    print(f"Reference images: {len(reference_images)}")

    # ---------------------------------------------------------------------------
    # Generate function factory — closure over pipeline & fixed delta
    # ---------------------------------------------------------------------------
    def make_generate_fn(delta_value: float):
        """Return a function that generates images for one fixed δ."""
        editor = Pix2PixZeroWithLD(
            pipeline=pipeline,
            delta=delta_value,
            num_ddim_inversion_steps=args.num_inversion_steps,
            num_ddim_denoising_steps=args.num_denoise_steps,
            guidance_scale=args.guidance_scale,
            cross_attention_guidance_amount=args.tau,
        )
        generated = []
        for i, src in enumerate(source_images):
            edited_img, _ = editor.edit(
                source_image=src,
                source_prompt=args.source_prompt,
                target_prompt=args.target_prompt,
                seed=args.seed + i,
                verbose=False,
            )
            generated.append(edited_img)
        return generated

    # ---------------------------------------------------------------------------
    # Grid search
    # ---------------------------------------------------------------------------
    print(
        f"\n=== Grid search: δ ∈ [{args.delta_min}, {args.delta_max}]  "
        f"({args.delta_steps} steps) ===\n"
    )

    import torch as _torch
    delta_grid = _torch.linspace(args.delta_min, args.delta_max, args.delta_steps).tolist()

    best_delta = 0.0
    best_dist  = float("inf")
    results    = []

    for delta in delta_grid:
        print(f"  Generating with δ = {delta:+.3f} …")
        generated = make_generate_fn(delta)
        dist = compute_l1_distribution_distance(generated, reference_images, device)
        results.append((delta, dist))
        print(f"    L1 distance = {dist:.5f}")

        if dist < best_dist:
            best_dist  = dist
            best_delta = delta

    # ---------------------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------------------
    print("\n===== Grid Search Results =====")
    print(f"{'δ':>8}  |  {'L1 distance':>12}")
    print("-" * 26)
    for delta, dist in results:
        marker = " ← BEST" if abs(delta - best_delta) < 1e-6 else ""
        print(f"{delta:+8.3f}  |  {dist:12.5f}{marker}")

    print(f"\nOptimal δ* = {best_delta:+.3f}  (L1 = {best_dist:.5f})")

    # Save result
    with open(args.output_file, "w") as f:
        f.write(f"optimal_delta={best_delta}\n")
        f.write(f"l1_distance={best_dist}\n")
        f.write("\n--- Full grid ---\n")
        for d, dist in results:
            f.write(f"delta={d:+.3f}  l1={dist:.5f}\n")

    print(f"\nResult written to {args.output_file}")


if __name__ == "__main__":
    main()
