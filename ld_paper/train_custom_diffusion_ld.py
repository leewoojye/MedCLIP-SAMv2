#!/usr/bin/env python3
"""
Custom Diffusion Fine-Tuning with Latent Drifting (LD).

Paper Section 4.2:
  "Custom Diffusion: fine-tuning only the cross-attention layers.
   This dramatically reduces trainable parameters while maintaining
   the ability to adapt to new domains."

  "For the fine-tuning through Latent Drifting, LD is added to the target
   zT of the forward process, as well as the reverse processes."

Key features:
  - Only cross-attention layers are trainable
  - UNet, VAE, CLIP text encoder are frozen
  - Negative samples (normal brain): counterfactual generation
  - Positive samples (tumor brain): target distribution
  - Latent drift δ applied in reverse process
  - Few-shot fine-tuning with minimal data

Usage (Few-shot: 3-5 images, 250-500 steps):
  python train_custom_diffusion_ld.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --positive_data_dir ./data/train/tumor \
    --negative_data_dir ./normal_sample \
    --output_dir ./checkpoints/custom_diffusion_ld \
    --delta 0.050 \
    --max_training_steps 300 \
    --batch_size 2 \
    --learning_rate 1e-4
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Custom Dataset for Paired Training
# ---------------------------------------------------------------------------

class PairedMRIDataset(Dataset):
    """
    Paired dataset with negative (normal) and positive (tumor) samples.
    
    This enables counterfactual learning where the model learns to
    distinguish between healthy and tumor-bearing brain MRI images.
    """
    
    def __init__(
        self,
        positive_dir: str,
        negative_dir: str,
        resolution: int = 512,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            positive_dir: Directory with tumor MRI images
            negative_dir: Directory with normal MRI images
            resolution: Image resolution (square)
            max_samples: Max number of samples to use (per class)
        """
        self.resolution = resolution
        
        # Load positive (tumor) samples
        self.positive_images = self._load_images_internal(positive_dir, max_samples)
        # Load negative (normal) samples
        self.negative_images = self._load_images_internal(negative_dir, max_samples)
        
        self.num_samples = min(len(self.positive_images), len(self.negative_images))
        
        print(f"Positive samples: {len(self.positive_images)}")
        print(f"Negative samples: {len(self.negative_images)}")
        print(f"Using {self.num_samples} paired samples")
        
        # Prompts
        self.prompt_positive = "a brain MRI with brain tumor"
        self.prompt_negative = "a normal healthy brain MRI"
    
    def _load_images_internal(self, directory: str, max_n: Optional[int] = None) -> List[Image.Image]:
        """Load PNG/JPG images from directory."""
        exts = {".png", ".jpg", ".jpeg"}
        paths = sorted([
            p for p in Path(directory).iterdir()
            if p.suffix.lower() in exts
        ])
        
        if max_n is not None:
            paths = paths[:max_n]
        
        images = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB").resize(
                    (self.resolution, self.resolution), Image.BICUBIC
                )
                images.append(img)
            except Exception as e:
                print(f"  [Warning] {p}: {e}")
        
        return images
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """Return paired positive/negative samples."""
        idx = idx % len(self.positive_images)
        
        pos_img = self.positive_images[idx]
        neg_img = self.negative_images[idx]
        
        # Convert to tensor [0, 1]
        pos_tensor = torch.from_numpy(np.array(pos_img)).permute(2, 0, 1).float() / 255.0
        neg_tensor = torch.from_numpy(np.array(neg_img)).permute(2, 0, 1).float() / 255.0
        
        return {
            "positive_image": pos_tensor,
            "negative_image": neg_tensor,
            "prompt_positive": self.prompt_positive,
            "prompt_negative": self.prompt_negative,
        }


# ---------------------------------------------------------------------------
# Freeze/Unfreeze utilities
# ---------------------------------------------------------------------------

def freeze_params(params):
    """Freeze parameters."""
    for param in params:
        param.requires_grad = False


def unfreeze_params(params):
    """Unfreeze parameters."""
    for param in params:
        param.requires_grad = True


def set_cross_attention_trainable(unet: UNet2DConditionModel):
    """
    Enable gradients only for cross-attention layers in UNet.
    All other parameters remain frozen.
    
    Cross-attention layers: in_layers[1] of each ResnetBlock2D and Transformer2DModel
    """
    # Freeze all parameters initially
    freeze_params(unet.parameters())
    
    # Unfreeze cross-attention layers
    trainable_count = 0
    for name, module in unet.named_modules():
        if "attn2" in name:  # Cross-attention (attn1 is self-attention)
            unfreeze_params(module.parameters())
            trainable_count += sum(p.numel() for p in module.parameters())
    
    logger.info(f"Trainable cross-attention parameters: {trainable_count:,}")


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_custom_diffusion_ld(args):
    """Main training function for Custom Diffusion with Latent Drifting."""
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    
    set_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info("CUSTOM DIFFUSION FINE-TUNING WITH LATENT DRIFTING")
    logger.info(f"{'='*80}\n")
    logger.info(f"Model:              {args.model_id}")
    logger.info(f"Positive data:      {args.positive_data_dir}")
    logger.info(f"Negative data:      {args.negative_data_dir}")
    logger.info(f"Latent drift δ:     {args.delta}")
    logger.info(f"Learning rate:      {args.learning_rate}")
    logger.info(f"Batch size:         {args.batch_size}")
    logger.info(f"Max training steps: {args.max_training_steps}")
    logger.info(f"Mixed precision:    {args.mixed_precision}")
    logger.info(f"{'='*80}\n")
    
    # Load pipeline
    logger.info("Loading Stable Diffusion pipeline...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,  # Always use float32 for VAE to avoid dtype issues
    )
    
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    noise_scheduler = pipeline.scheduler
    
    # Move to device
    device = accelerator.device
    # VAE and text encoder always in float32 (no mixed precision)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    # UNet can be in fp16 if mixed_precision is set
    if args.mixed_precision == "fp16":
        unet = unet.half().to(device)
    else:
        unet = unet.to(device)
    
    # Freeze VAE and text encoder
    logger.info("Freezing VAE and text encoder...")
    freeze_params(vae.parameters())
    freeze_params(text_encoder.parameters())
    
    # Set only cross-attention layers as trainable
    logger.info("Setting cross-attention layers as trainable...")
    set_cross_attention_trainable(unet)
    
    # Create dataset
    logger.info("Loading dataset...")
    dataset = PairedMRIDataset(
        positive_dir=args.positive_data_dir,
        negative_dir=args.negative_data_dir,
        resolution=args.resolution,
        max_samples=args.max_samples,
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
    )
    
    # Setup scheduler
    num_training_steps = args.max_training_steps
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Prepare with accelerator
    logger.info("Preparing with accelerator...")
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Training loop (Step-based, as per Custom Diffusion paper for few-shot learning)
    logger.info(f"Starting training: {args.max_training_steps} steps (few-shot: 3-5 images, 250-500 steps)\n")
    
    progress_bar = tqdm(total=num_training_steps, desc="Training")
    
    global_step = 0
    dataloader_iter = None
    
    while global_step < args.max_training_steps:
        # Reinitialize iterator if exhausted
        if dataloader_iter is None:
            dataloader_iter = iter(train_dataloader)
        
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)  # Reset iterator
            batch = next(dataloader_iter)
        with accelerator.accumulate(unet):
            # Get positive and negative images
            pos_imgs = batch["positive_image"].to(device)
            neg_imgs = batch["negative_image"].to(device)
            
            # Handle prompts - convert to list if needed
            prompts_pos = batch["prompt_positive"]
            prompts_neg = batch["prompt_negative"]
            if isinstance(prompts_pos, str):
                prompts_pos = [prompts_pos] * pos_imgs.shape[0]
            if isinstance(prompts_neg, str):
                prompts_neg = [prompts_neg] * neg_imgs.shape[0]
                
            # Forward pass for both positive and negative examples
            with torch.no_grad():
                # Encode to latent space
                pos_imgs = pos_imgs.float()  # Ensure float32
                neg_imgs = neg_imgs.float()  # Ensure float32
                latents_pos = vae.encode(pos_imgs).latent_dist.sample() * 0.18215
                latents_neg = vae.encode(neg_imgs).latent_dist.sample() * 0.18215
                
                # Encode text
                text_inputs_pos = tokenizer(
                    prompts_pos,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings_pos = text_encoder(
                    text_inputs_pos.input_ids.to(device)
                )[0]
                
                text_inputs_neg = tokenizer(
                    prompts_neg,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings_neg = text_encoder(
                    text_inputs_neg.input_ids.to(device)
                )[0]
            
            # Sample noise and timestep
            noise_pos = torch.randn_like(latents_pos)
            noise_neg = torch.randn_like(latents_neg)
            bsz = latents_pos.shape[0]
            
            timesteps_pos = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz,),
                device=device,
            )
            timesteps_neg = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (bsz,),
                device=device,
            )
            
            # Add noise (forward process) + latent drift
            noisy_latents_pos = noise_scheduler.add_noise(
                latents_pos, noise_pos, timesteps_pos
            )
            noisy_latents_neg = noise_scheduler.add_noise(
                latents_neg, noise_neg, timesteps_neg
            )
            
            # Apply latent drift δ to forward process target
            noisy_latents_pos = noisy_latents_pos + args.delta
            noisy_latents_neg = noisy_latents_neg + args.delta
            
            # Predict noise
            model_pred_pos = unet(
                noisy_latents_pos, timesteps_pos, encoder_hidden_states=text_embeddings_pos
            ).sample
            
            model_pred_neg = unet(
                noisy_latents_neg, timesteps_neg, encoder_hidden_states=text_embeddings_neg
            ).sample
            
            # Compute loss
            loss_pos = F.mse_loss(model_pred_pos, noise_pos)
            loss_neg = F.mse_loss(model_pred_neg, noise_neg)
            
            # Balance: encourage distinguishing between positive and negative
            loss = loss_pos + loss_neg
            
            # Backward
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, unet.parameters()),
                    max_norm=1.0,
                )
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        loss_value = loss.detach().item()
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": f"{loss_value:.4f}", "step": global_step})
        
        # Save checkpoint at regular intervals
        if (global_step + 1) % args.save_steps == 0:
            save_path = Path(args.output_dir) / f"checkpoint_step_{global_step+1}"
            save_path.mkdir(parents=True, exist_ok=True)
            
            accelerator.save_state(str(save_path))
            logger.info(f"Step {global_step+1}/{num_training_steps} - Loss: {loss_value:.6f} - Saved checkpoint")
        
        # Logging
        if (global_step + 1) % args.logging_steps == 0:
            logger.info(f"Step {global_step+1}/{num_training_steps} - Loss: {loss_value:.6f}")
        
        global_step += 1
    
    progress_bar.close()
    
    # Save final model
    logger.info(f"\nSaving final model to {args.output_dir}...")
    accelerator.save_state(str(Path(args.output_dir) / "final"))
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nFinal checkpoint saved to: {args.output_dir}")
    logger.info(f"Latent drift δ: {args.delta}")
    logger.info(f"Fine-tuned method: Custom Diffusion (cross-attention only)")
    logger.info(f"UNet status: Frozen (except cross-attention)")
    logger.info(f"VAE status: Frozen")
    logger.info(f"Text encoder status: Frozen")


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Custom Diffusion Fine-Tuning with Latent Drifting"
    )
    
    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
    )
    
    # Data
    parser.add_argument(
        "--positive_data_dir",
        type=str,
        required=True,
        help="Directory with positive samples (tumor images)",
    )
    parser.add_argument(
        "--negative_data_dir",
        type=str,
        required=True,
        help="Directory with negative samples (normal images)",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    
    # Latent Drifting
    parser.add_argument(
        "--delta",
        type=float,
        default=0.050,
        help="Optimal latent drift δ from grid search",
    )
    
    # Training
    parser.add_argument("--max_training_steps",  type=int,   default=300,
                        help="Total training steps (Few-shot: typically 250-500). Custom Diffusion paper recommends few-shot with 3-5 images")
    parser.add_argument("--batch_size",          type=int,   default=2)
    parser.add_argument("--resolution",          type=int,   default=512)
    parser.add_argument("--learning_rate",       type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps",     type=int,   default=50,
                        help="Warmup steps for few-shot learning")
    parser.add_argument("--mixed_precision",     type=str,   default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed",                type=int,   default=42)
    
    # Checkpointing
    parser.add_argument("--output_dir",      type=str, default="./checkpoints/custom_diffusion_ld")
    parser.add_argument("--save_steps",      type=int, default=10)
    parser.add_argument("--logging_steps",   type=int, default=5)
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_custom_diffusion_ld(args)
