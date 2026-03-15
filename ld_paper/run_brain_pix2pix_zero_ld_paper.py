from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Add root directory to python path to access disease_conditioned_manipulation.pipeline
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from disease_conditioned_manipulation.pipeline import DiseaseConditionedLDPEditor
from disease_conditioned_manipulation.presets import get_preset_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-reproduction style brain MRI counterfactual editing with Pix2Pix Zero + latent drift."
    )
    # Target Delta configuration
    parser.add_argument(
        "--deltas",
        type=float,
        nargs="+",
        default=[-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2],
        help="List of delta values to grid search over for optimal latent drifting."
    )
    # Test dataset details
    parser.add_argument(
        "--test-image-dir",
        type=str,
        default="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/data/brain_tumors/test_images",
    )
    parser.add_argument(
        "--test-mask-dir",
        type=str,
        default="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/data/brain_tumors/test_masks",
    )
    parser.add_argument("--image-id", type=str, default=None, help="Process a specific image ID.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/ld_paper/outputs/brain_pix2pix_zero_ld",
    )
    parser.add_argument("--output-name", type=str, default=None)
    
    # Prompt Setup (Diverse Prompts with PI as per Sec 4.2.2 Paper)
    parser.add_argument("--source-prompt", type=str, default="brain MRI scan with tumor, clinical diagnosis")
    parser.add_argument("--target-prompt", type=str, default="healthy brain MRI scan, clinical diagnosis")
    
    # Checkpoints
    parser.add_argument(
        "--model-id",
        type=str,
        default="/home/woojye2020/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
    )
    parser.add_argument("--finetuned-unet", type=str, default=None, help="Path to fully fine-tuned UNet model")
    
    # Operational configuration
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--inversion-guidance-scale", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--cross-attention-guidance-amount", type=float, default=0.1)
    parser.add_argument("--attention-preservation-weight", type=float, default=1.0)
    parser.add_argument("--num-opt-steps", type=int, default=1)
    parser.add_argument("--delta-scale", type=float, default=0.5)
    parser.add_argument(
        "--delta-application",
        type=str,
        default="per_step",
        choices=["cumulative", "per_step"],
    )
    parser.add_argument(
        "--delta-schedule",
        type=str,
        default="linear_decay",
        choices=["constant", "linear_decay", "first_step"],
    )
    parser.add_argument("--mask-edit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-background", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--match-input-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


import numpy as np
from PIL import Image

def compute_background_l1(source_img: Image.Image, target_img: Image.Image) -> float:
    source_np = np.array(source_img.convert("RGB")).astype(np.float32) / 255.0
    target_np = np.array(target_img.convert("RGB")).astype(np.float32) / 255.0
    
    # In MRIs, the background is naturally black.
    # We define the background where the source image is very dark (e.g. mean RGB < 0.05)
    source_gray = np.mean(source_np, axis=-1)
    bg_mask_2d = (source_gray < 0.05)
    
    if not np.any(bg_mask_2d):
        return float('inf')
        
    diff = np.abs(source_np - target_np)
    # Apply 2D mask to each of the 3 channels
    l1_dist = np.mean(diff[bg_mask_2d])
    return float(l1_dist)


def select_test_pair(image_dir: str, mask_dir: str, image_id: str | None) -> tuple[Path, Path]:
    image_root = Path(image_dir)
    mask_root = Path(mask_dir)
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    if image_id is not None:
        for ext in exts:
            image_path = image_root / f"{image_id}{ext}"
            mask_path = mask_root / f"{image_id}{ext}"
            if image_path.exists() and mask_path.exists():
                return image_path, mask_path
        raise FileNotFoundError(f"No matched test pair found for image_id='{image_id}'")

    images = sorted(path for path in image_root.iterdir() if path.suffix.lower() in exts)
    for image_path in images:
        mask_path = mask_root / image_path.name
        if mask_path.exists():
            return image_path, mask_path
    raise FileNotFoundError(f"No matched image/mask pair found in {image_dir} and {mask_dir}")


def main() -> None:
    args = parse_args()
    image_path, mask_path = select_test_pair(args.test_image_dir, args.test_mask_dir, args.image_id)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_prompts = get_preset_prompts("brain_mri", "tumor")
    target_prompts = get_preset_prompts("brain_mri", "normal")

    generator = None
    if args.seed is not None:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        generator = torch.Generator(device=device).manual_seed(args.seed)

    editor = DiseaseConditionedLDPEditor.from_pretrained(
        model_id=args.model_id or "CompVis/stable-diffusion-v1-4",
        device=args.device,
        dtype=args.dtype,
        unet_path=args.finetuned_unet,
    )
    # Automatically apply memory optimizations to prevent OOM
    if hasattr(editor.pipe, "enable_model_cpu_offload"):
        editor.pipe.enable_model_cpu_offload()
    if hasattr(editor.pipe, "enable_xformers_memory_efficient_attention"):
        try:
            editor.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    
    # Load original image and mask for L1 computation
    original_img = Image.open(image_path)
    mask_img = Image.open(mask_path)
    
    best_delta = None
    best_loss = float('inf')
    best_image = None
    
    print(f"Starting grid search over deltas: {args.deltas}")
    for current_delta in args.deltas:
        print(f"Evaluating delta = {current_delta} ...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        current_img = editor.edit_with_delta(
            image=image_path,
            source_prompt=args.source_prompt,
            target_prompt=args.target_prompt,
            latent_drift_delta=current_delta,
            output_path=None,  # don't save intermediate
            mask=mask_path if args.mask_edit else None,
            source_concept_prompts=source_prompts,
            target_concept_prompts=target_prompts,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            inversion_guidance_scale=args.inversion_guidance_scale,
            guidance_scale=args.guidance_scale,
            cross_attention_guidance_amount=args.cross_attention_guidance_amount,
            attention_preservation_weight=args.attention_preservation_weight,
            num_opt_steps=args.num_opt_steps,
            delta_scale=args.delta_scale,
            delta_application=args.delta_application,
            delta_schedule=args.delta_schedule,
            preserve_background=args.preserve_background,
            match_input_mode=args.match_input_mode,
            generator=generator,
        )
        
        l1_loss = compute_background_l1(original_img, current_img)
        print(f"Delta: {current_delta} -> Background L1 Loss: {l1_loss:.5f}")
        
        if l1_loss < best_loss:
            best_loss = l1_loss
            best_delta = current_delta
            best_image = current_img

    print(f"Grid search complete. Selected Best Delta = {best_delta} with L1 loss = {best_loss:.5f}")
    
    output_name = args.output_name or f"{image_path.stem}_pix2pix_zero_ld.png"
    output_path = output_dir / output_name
    best_image.save(output_path)
    editor.clear_attention_cache()

    metadata = {
        "image": str(image_path),
        "mask": str(mask_path),
        "output": str(output_path),
        "deltas_searched": args.deltas,
        "best_delta": best_delta,
        "best_background_l1": best_loss,
        "source_prompt": args.source_prompt,
        "target_prompt": args.target_prompt,
        "source_concept_prompts": source_prompts,
        "target_concept_prompts": target_prompts,
        "model_id": args.model_id or "CompVis/stable-diffusion-v1-4",
        "finetuned_unet": args.finetuned_unet,
        "seed": args.seed,
        "mask_edit": args.mask_edit,
        "delta_scale": args.delta_scale,
        "delta_application": args.delta_application,
        "delta_schedule": args.delta_schedule,
        "preserve_background": args.preserve_background,
        "match_input_mode": args.match_input_mode,
        "mode": "disease_conditioned_manipulation",
    }
    with open(output_dir / f"{output_path.stem}.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved edited image to {output_path}")
    print(f"Used test pair: {image_path.name}")
    print(f"Used best delta: {best_delta}")


if __name__ == "__main__":
    main()
