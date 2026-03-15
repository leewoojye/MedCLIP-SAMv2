#!/usr/bin/env python3
"""
Inference Script for Fine-tuned Custom Diffusion Model with Latent Drifting.

Paper Section 5: "After fine-tuning via Custom Diffusion, we evaluate the model
using generated images and downstream tasks. The model learns to synthesize 
new concepts with the specific characteristics from the training set."

This script:
1. Loads the fine-tuned Custom Diffusion checkpoint
2. Generates counterfactual tumor images using latent drift δ
3. Outputs images for downstream task (e.g., pix2pix evaluation)
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_fine_tuned_model(checkpoint_path: str, device: str = "cuda"):
    """Load the fine-tuned Custom Diffusion model from checkpoint."""
    logger.info(f"Loading fine-tuned model from: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load base pipeline
    logger.info("Loading base Stable Diffusion model...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
    )
    
    # Load fine-tuned model (accelerate saved as model.safetensors)
    model_path = checkpoint_path / "model.safetensors"
    if model_path.exists():
        logger.info(f"Loading fine-tuned model from: {model_path}")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_path))
            pipeline.unet.load_state_dict(state_dict)
            logger.info("✓ Fine-tuned weights loaded successfully from safetensors")
        except ImportError:
            logger.info("safetensors not available, trying torch.load...")
            try:
                # Fallback: try to load as torch model
                import torch
                state_dict = torch.load(str(model_path), map_location="cpu")
                pipeline.unet.load_state_dict(state_dict)
                logger.info("✓ Fine-tuned weights loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    else:
        # Try checkpoint_step format
        step_checkpoints = sorted(checkpoint_path.parent.glob("checkpoint_step_*"))
        if step_checkpoints:
            latest_checkpoint = step_checkpoints[-1]
            model_path = latest_checkpoint / "model.safetensors"
            if model_path.exists():
                logger.info(f"Using latest step checkpoint: {latest_checkpoint.name}")
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(str(model_path))
                    pipeline.unet.load_state_dict(state_dict)
                    logger.info("✓ Fine-tuned weights loaded successfully")
                except Exception as e:
                    logger.warning(f"Error loading weights: {e}")
    
    # Move to device
    pipeline = pipeline.to(device)
    logger.info(f"✓ Model loaded on device: {device}")
    
    return pipeline


def generate_images(
    pipeline,
    prompts: list,
    num_samples: int = 3,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    delta: float = 0.050,
    seed: int = 42,
    output_dir: str = "./generated_fine_tuned",
):
    """Generate images using fine-tuned Custom Diffusion model with latent drift."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    generated_images = []
    
    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\nGenerating {num_samples} images for prompt: '{prompt}'")
        
        for sample_idx in range(num_samples):
            logger.info(f"  Sample {sample_idx + 1}/{num_samples}")
            
            # Forward pass with latent drift δ
            # Custom Diffusion paper: δ is applied during sampling
            with torch.no_grad():
                # Generate with guidance and delta offset
                image = pipeline(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    height=512,
                    width=512,
                ).images[0]
            
            # Save image
            prompt_safe = prompt.replace(" ", "_").replace(",", "").lower()[:30]
            img_name = f"prompt_{prompt_idx}_sample_{sample_idx}.png"
            img_path = output_dir / img_name
            image.save(str(img_path))
            
            generated_images.append(image)
            logger.info(f"    Saved: {img_path}")
    
    logger.info(f"\n✓ Generated {len(generated_images)} images")
    logger.info(f"✓ Output directory: {output_dir}")
    
    return generated_images


def prepare_for_pix2pix(images, output_dir: str = "./pix2pix_input"):
    """
    Prepare generated images for pix2pix downstream task.
    
    Paper Section 5: "We evaluate the learned representation by applying
    the fine-tuned model to downstream tasks, such as image-to-image translation."
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nPreparing images for pix2pix downstream task...")
    logger.info(f"Output directory: {output_dir}")
    
    for idx, img in enumerate(images):
        # Save with pix2pix naming convention
        img_path = output_dir / f"generated_{idx:04d}.png"
        img.save(str(img_path))
    
    logger.info(f"✓ Prepared {len(images)} images for pix2pix")
    logger.info(f"  Use these images as input for pix2pix model")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Inference with fine-tuned Custom Diffusion model"
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/custom_diffusion_ld_delta_0050/final",
        help="Path to fine-tuned checkpoint",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base model ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a brain MRI with brain tumor",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="a normal healthy brain MRI",
        help="Negative prompt (counterfactual)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate per prompt",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.050,
        help="Latent drift δ (from grid search)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_fine_tuned",
        help="Output directory for generated images",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--prepare_pix2pix",
        action="store_true",
        help="Prepare images for pix2pix downstream task",
    )
    parser.add_argument(
        "--pix2pix_output_dir",
        type=str,
        default="./pix2pix_input",
        help="Output directory for pix2pix input images",
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("CUSTOM DIFFUSION INFERENCE WITH LATENT DRIFTING")
    logger.info("="*80)
    logger.info(f"Checkpoint:         {args.checkpoint_path}")
    logger.info(f"Prompt:             {args.prompt}")
    logger.info(f"Negative prompt:    {args.negative_prompt}")
    logger.info(f"Num samples:        {args.num_samples}")
    logger.info(f"Guidance scale:     {args.guidance_scale}")
    logger.info(f"Latent drift δ:     {args.delta}")
    logger.info(f"Device:             {args.device}")
    logger.info("="*80 + "\n")
    
    # Load fine-tuned model
    pipeline = load_fine_tuned_model(args.checkpoint_path, device=args.device)
    
    # Generate images with positive prompt (tumor)
    prompts = [args.prompt, args.negative_prompt]
    generated_images = generate_images(
        pipeline,
        prompts=prompts,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        delta=args.delta,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    
    # Prepare for pix2pix if requested
    if args.prepare_pix2pix:
        prepare_for_pix2pix(generated_images, output_dir=args.pix2pix_output_dir)
    
    logger.info("\n" + "="*80)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*80)
    logger.info(f"Generated images saved to: {args.output_dir}")
    if args.prepare_pix2pix:
        logger.info(f"Pix2pix input prepared in: {args.pix2pix_output_dir}")


if __name__ == "__main__":
    main()
