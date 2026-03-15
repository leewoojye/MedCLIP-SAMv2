#!/usr/bin/env python3
"""
Grid Search for Optimal Latent Drift δ

Paper Section 3.4:
  δ* = argmin_δ L1(D_θ(·|δ), D_GT)
  
Reverse process with latent drift:
  pθ(xt-1|xt) = N(xt-1; µθ(xt,t) + δ, Σθ(xt,t))
  
기존 SD가 생성한 이미지 분포 -> BraTS MRI 분포
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

from latent_drifting import compute_l1_distribution_distance


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def load_images(directory: str, max_n: int) -> List[Image.Image]:
    """Load images from directory."""
    paths = [
        p for p in sorted(Path(directory).iterdir())
        if p.suffix.lower() in SUPPORTED_EXTS
    ][:max_n]
    print(f"    Loading {len(paths)} images…")
    images = []
    for p in tqdm(paths, desc="    ", leave=False):
        try:
            img = Image.open(p).convert("RGB").resize((512, 512), Image.BICUBIC)
            images.append(img)
        except Exception as e:
            print(f"      Warning: {p}: {e}")
    return images


def generate_with_latent_drift(
    pipeline,
    prompt: str,
    delta: float,
    num_samples: int,
    seed: int = 42,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> List[Image.Image]:
    """
    Generate images with latent drift δ applied in reverse process.
    
    Applies: pθ(xt-1|xt) = N(xt-1; µθ(xt,t) + δ, Σθ(xt,t))
    """
    generator = torch.Generator(device=pipeline.device)
    images = []
    
    for sample_idx in range(num_samples):
        # Text encoding
        text_input = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
        
        # Classifier-free guidance: unconditional embeddings
        if guidance_scale > 1.0:
            uncond_input = pipeline.tokenizer(
                "",
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                return_tensors="pt",
            )
            uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Initialize latents
        generator.manual_seed(seed + sample_idx)
        latents = torch.randn(
            (1, pipeline.unet.config.in_channels, 64, 64),
            generator=generator,
            device=pipeline.device,
            dtype=text_embeddings.dtype,
        )
        
        # Scheduler
        pipeline.scheduler.set_timesteps(num_inference_steps, device=pipeline.device)
        latents = latents * pipeline.scheduler.init_noise_sigma
        
        # Denoising loop with latent drift
        for t in tqdm(pipeline.scheduler.timesteps, desc=f"    δ={delta:+.3f} [{sample_idx+1}/{num_samples}]", leave=False):
            # Prepare latent model input
            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents
            
            # UNet prediction
            with torch.no_grad():
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                ).sample
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # ============================================================
            # APPLY LATENT DRIFT TO PREDICTED MEAN
            # Paper: pθ(xt-1|xt) = N(xt-1; µθ(xt,t) + δ, Σθ(xt,t))
            # ============================================================
            if abs(delta) > 1e-9:
                noise_pred = noise_pred + delta
            
            # Scheduler step
            latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode to image
        with torch.no_grad():
            latents = 1 / pipeline.vae.config.scaling_factor * latents
            image = pipeline.vae.decode(latents).sample
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image[0])
        
        images.append(image)
    
    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid search for optimal δ: argmin_δ L1(D_θ(δ), D_GT)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    parser.add_argument(
        "--reference_data_dir",
        type=str,
        required=True,
        help="Target domain D_GT",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a brain MRI with brain tumor",
    )
    parser.add_argument("--num_samples",          type=int,   default=3)
    parser.add_argument("--delta_min",            type=float, default=-0.2)
    parser.add_argument("--delta_max",            type=float, default=0.2)
    parser.add_argument("--delta_steps",          type=int,   default=9)
    parser.add_argument("--guidance_scale",       type=float, default=7.5)
    parser.add_argument("--num_inference_steps",  type=int,   default=50)
    parser.add_argument("--seed",                 type=int,   default=42)
    parser.add_argument("--output_file",          type=str,   default="optimal_delta.txt")
    parser.add_argument("--device",               type=str,   default="cuda")
    parser.add_argument("--save_generated",       action="store_true")
    parser.add_argument("--output_dir",           type=str,   default="./generated_delta_search")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("GRID SEARCH FOR OPTIMAL LATENT DRIFT δ")
    print("Paper Section 3.4: δ* = argmin_δ L1(D_θ(δ), D_GT)")
    print("=" * 80)
    print()
    print(f"Model (Dθ):    {args.model_path}")
    print(f"Target (D_GT): {args.reference_data_dir}")
    print(f"Prompt:        '{args.prompt}'")
    print(f"Device:        {device}")
    print()

    # Load model
    print("Loading model…")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32,
    )
    pipeline = pipeline.to(device)
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    print()

    # Load target domain
    print("Loading target domain D_GT…")
    reference_images = load_images(args.reference_data_dir, args.num_samples * 5)

    if not reference_images:
        raise RuntimeError(f"No target images in {args.reference_data_dir}")

    print(f"  Target images: {len(reference_images)}")
    print()

    if args.save_generated:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Grid search
    print("=" * 80)
    print(f"GRID SEARCH: δ ∈ [{args.delta_min}, {args.delta_max}] ({args.delta_steps} steps)")
    print("Applying latent drift in reverse process:")
    print("  pθ(xt-1|xt) = N(xt-1; µθ(xt,t) + δ, Σθ(xt,t))")
    print("=" * 80)
    print()

    import torch as _torch
    delta_grid = _torch.linspace(args.delta_min, args.delta_max, args.delta_steps).tolist()

    best_delta = 0.0
    best_dist  = float("inf")
    results    = []

    for idx, delta in enumerate(delta_grid, 1):
        print(f"[{idx}/{len(delta_grid)}] δ = {delta:+.3f}")
        
        generated = generate_with_latent_drift(
            pipeline,
            prompt=args.prompt,
            delta=delta,
            num_samples=args.num_samples,
            seed=args.seed,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
        )
        
        dist = compute_l1_distribution_distance(generated, reference_images, device)
        results.append((delta, dist))
        print(f"        L1(D_θ(δ), D_GT) = {dist:.6f}")

        if args.save_generated:
            delta_dir = Path(args.output_dir) / f"delta_{delta:+.3f}"
            delta_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(generated):
                img.save(delta_dir / f"{i:03d}.png")

        if dist < best_dist:
            best_dist  = dist
            best_delta = delta

        print()

    # Results
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"{'δ':>8}  |  {'L1':>12}")
    print("-" * 23)
    for delta, dist in results:
        marker = " ← OPTIMAL" if abs(delta - best_delta) < 1e-6 else ""
        print(f"{delta:+8.3f}  |  {dist:12.6f}{marker}")

    print()
    print("=" * 80)
    print(f"δ* = {best_delta:+.3f}  (L1 = {best_dist:.6f})")
    print("=" * 80)
    print()

    # Save result
    with open(args.output_file, "w") as f:
        f.write("Grid Search for Optimal Latent Drift δ\n")
        f.write(f"δ* = {best_delta:+.3f}\n")
        f.write(f"L1(D_θ(δ*), D_GT) = {best_dist:.6f}\n\n")
        for d, dist in results:
            f.write(f"δ={d:+.3f}  L1={dist:.6f}\n")

    print(f"Results saved: {args.output_file}")


if __name__ == "__main__":
    main()
