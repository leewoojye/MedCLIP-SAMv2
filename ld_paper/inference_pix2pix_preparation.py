#!/usr/bin/env python3
"""
Image-to-Image Translation with Fine-tuned Custom Diffusion.

Paper Section 5: "We apply the fine-tuned Custom Diffusion model to generate
counterfactual images. For each test image with tumor, we generate the 
corresponding healthy version as a negative example. These paired images
are then evaluated with downstream tasks like pix2pix."

Workflow:
1. Load fine-tuned Custom Diffusion checkpoint (cross-attention tuned)
2. For each test tumor image:
   - Use img2img pipeline with custom diffusion 
   - Generate corresponding negative image (healthy brain)
   - Save paired images for pix2pix downstream task
"""

import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_fine_tuned_img2img_model(checkpoint_path: str, device: str = "cuda"):
    """Load fine-tuned Custom Diffusion as img2img pipeline."""
    logger.info(f"Loading fine-tuned model for img2img: {checkpoint_path}")
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load img2img pipeline (instead of txt2img)
    logger.info("Loading base Stable Diffusion img2img model...")
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
    )
    
    # Load fine-tuned weights
    model_path = checkpoint_path / "model.safetensors"
    if model_path.exists():
        logger.info(f"Loading fine-tuned UNet weights...")
        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(model_path))
            pipeline.unet.load_state_dict(state_dict)
            logger.info("✓ Fine-tuned weights loaded from safetensors")
        except ImportError:
            logger.info("safetensors not available, trying step checkpoint...")
            step_checkpoints = sorted(checkpoint_path.parent.glob("checkpoint_step_*"))
            if step_checkpoints:
                latest_checkpoint = step_checkpoints[-1]
                model_path = latest_checkpoint / "model.safetensors"
                if model_path.exists():
                    from safetensors.torch import load_file
                    state_dict = load_file(str(model_path))
                    pipeline.unet.load_state_dict(state_dict)
                    logger.info(f"✓ Loaded from: {latest_checkpoint.name}")
    
    pipeline = pipeline.to(device)
    logger.info(f"✓ Model on device: {device}\n")
    
    return pipeline


def translate_tumor_to_negative(
    pipeline,
    tumor_image_path: str,
    prompt: str = "a normal healthy brain MRI",
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    seed: int = 42,
) -> Image.Image:
    """
    Translate tumor image to negative (healthy) counterfactual image.
    
    Args:
        pipeline: Fine-tuned diffusion img2img pipeline
        tumor_image_path: Path to test tumor MRI image
        prompt: Target prompt (healthy brain description)
        strength: How much to transform (0.0=no change, 1.0=completely new)
        guidance_scale: Classifier-free guidance strength
        num_inference_steps: Number of diffusion steps
        seed: Random seed for reproducibility
    """
    
    # Load input image
    input_image = Image.open(tumor_image_path).convert("RGB")
    input_image = input_image.resize((512, 512), Image.BICUBIC)
    
    # Set seed
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    # Generate negative image
    with torch.no_grad():
        # img2img with fine-tuned model
        output_image = pipeline(
            prompt=prompt,
            image=input_image,
            strength=strength,  # How much to change (0.0-1.0)
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
    
    return output_image


def process_test_dataset(
    pipeline,
    test_tumor_dir: str,
    output_positive_dir: str,
    output_negative_dir: str,
    prompt: str = "a normal healthy brain MRI",
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    max_images: int = None,
    seed: int = 42,
):
    """
    Process all test tumor images and generate corresponding negative images.
    
    Outputs paired images for pix2pix downstream task:
    - positive_dir: Original tumor images (or can be source for pix2pix)
    - negative_dir: Generated negative/healthy counterfactual images (target for pix2pix)
    """
    
    test_tumor_dir = Path(test_tumor_dir)
    output_positive_dir = Path(output_positive_dir)
    output_negative_dir = Path(output_negative_dir)
    
    output_positive_dir.mkdir(parents=True, exist_ok=True)
    output_negative_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all tumor images
    image_files = sorted([f for f in test_tumor_dir.glob("*.png") if f.is_file()])
    
    if max_images:
        image_files = image_files[:max_images]
    
    logger.info(f"\nProcessing {len(image_files)} test tumor images...")
    logger.info(f"Prompt: '{prompt}'")
    logger.info(f"Strength: {strength} (0.0=no change, 1.0=complete change)")
    logger.info(f"Guidance scale: {guidance_scale}\n")
    
    paired_results = []
    
    for idx, img_path in enumerate(image_files):
        logger.info(f"[{idx+1}/{len(image_files)}] Processing: {img_path.name}")
        
        try:
            # Generate negative image
            negative_image = translate_tumor_to_negative(
                pipeline,
                str(img_path),
                prompt=prompt,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed + idx,  # Different seed for each image
            )
            
            # Save positive (original)
            pos_output_path = output_positive_dir / img_path.name
            img = Image.open(img_path).convert("RGB").resize((512, 512), Image.BICUBIC)
            img.save(str(pos_output_path))
            
            # Save negative (generated)
            neg_output_path = output_negative_dir / img_path.name
            negative_image.save(str(neg_output_path))
            
            paired_results.append({
                "index": idx,
                "filename": img_path.name,
                "positive_path": str(pos_output_path),
                "negative_path": str(neg_output_path),
            })
            
            logger.info(f"  ✓ Positive:  {pos_output_path.name}")
            logger.info(f"  ✓ Negative:  {neg_output_path.name}\n")
        
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            continue
    
    logger.info("="*80)
    logger.info(f"✓ Processed {len(paired_results)} images")
    logger.info(f"  Positive images: {output_positive_dir}")
    logger.info(f"  Negative images: {output_negative_dir}")
    logger.info("="*80)
    logger.info("\nThese paired images can now be used for:")
    logger.info("  1. Pix2pix training (tumor -> healthy translation)")
    logger.info("  2. Downstream task evaluation")
    logger.info("  3. Contrastive learning analysis")
    
    return paired_results


def main():
    parser = argparse.ArgumentParser(
        description="Translate test tumor images to negative/healthy counterfactuals"
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/custom_diffusion_ld_delta_0050/final",
        help="Path to fine-tuned checkpoint",
    )
    parser.add_argument(
        "--test_tumor_dir",
        type=str,
        default="./data/test/tumor",
        help="Directory with test tumor MRI images",
    )
    parser.add_argument(
        "--output_positive_dir",
        type=str,
        default="./pix2pix_dataset/train_A",
        help="Output directory for positive (tumor) images",
    )
    parser.add_argument(
        "--output_negative_dir",
        type=str,
        default="./pix2pix_dataset/train_B",
        help="Output directory for negative (healthy) generated images",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a normal healthy brain MRI",
        help="Target prompt for counterfactual generation",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Strength of transformation (0.0=no change, 1.0=complete change)",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance strength",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion inference steps",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (None=all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("IMAGE-TO-IMAGE TRANSLATION: TUMOR → HEALTHY COUNTERFACTUAL")
    logger.info("="*80)
    logger.info(f"Checkpoint:        {args.checkpoint_path}")
    logger.info(f"Test tumor dir:    {args.test_tumor_dir}")
    logger.info(f"Output positive:   {args.output_positive_dir}")
    logger.info(f"Output negative:   {args.output_negative_dir}")
    logger.info("="*80 + "\n")
    
    # Load fine-tuned model
    pipeline = load_fine_tuned_img2img_model(args.checkpoint_path, device=args.device)
    
    # Process test dataset
    paired_results = process_test_dataset(
        pipeline,
        test_tumor_dir=args.test_tumor_dir,
        output_positive_dir=args.output_positive_dir,
        output_negative_dir=args.output_negative_dir,
        prompt=args.prompt,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        max_images=args.max_images,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
