"""
Basic Fine-Tuning of Stable Diffusion with Latent Drifting (LD).

Paper Section 4.1:
  "Stable Diffusion basic fine-tuning: fine-tuning the denoising U-Net
   while freezing the rest of the components."

  "For the fine-tuning through Latent Drifting, LD is added to the target
   zT of the forward process, as well as the reverse processes."

The LD training modification:
  Standard: noisy_z = sqrt(ᾱt)·z0 + sqrt(1-ᾱt)·ε,   loss = ||ε̂θ - ε||²
  With LD  : noisy_z = noisy_z + δ                     (shifts forward target)
             At inference: prev_sample += δ             (shifts reverse mean)

We fine-tune only the U-Net denoising network; the VAE encoder/decoder
and CLIP text encoder are frozen throughout training.

Usage
-----
accelerate launch train_basic_ft.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --train_data_dir ./data/train \
    --output_dir ./checkpoints \
    --delta 0.1 \
    --num_train_epochs 150 \
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-6 \
    --mixed_precision fp16
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

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
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm

from dataset import BrainTumorMRIDataset

logger = get_logger(__name__, log_level="INFO")


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basic Fine-Tuning of Stable Diffusion with Latent Drifting"
    )
    # ---- Model -----------------------------------------------------------
    parser.add_argument(
        "--model_id",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="HuggingFace model ID or local path.",
    )
    # ---- Data ------------------------------------------------------------
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--output_dir",     type=str, default="./checkpoints")
    parser.add_argument("--logging_dir",    type=str, default="./logs")
    # ---- Latent Drifting -------------------------------------------------
    parser.add_argument(
        "--delta",
        type=float,
        default=0.1,
        help="Latent drift δ added to noisy latents during training (Section 3.3).",
    )
    # ---- Training hyperparameters ----------------------------------------
    parser.add_argument("--resolution",                  type=int,   default=512)
    parser.add_argument("--train_batch_size",            type=int,   default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int,   default=4)
    parser.add_argument("--num_train_epochs",            type=int,   default=150)
    parser.add_argument("--max_train_steps",             type=int,   default=None)
    parser.add_argument("--learning_rate",               type=float, default=5e-6)
    parser.add_argument("--lr_scheduler",                type=str,   default="cosine")
    parser.add_argument("--lr_warmup_steps",             type=int,   default=200)
    parser.add_argument("--mixed_precision",             type=str,   default="fp16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing",      action="store_true", default=True)
    parser.add_argument("--seed",                        type=int,   default=42)
    parser.add_argument("--use_diverse_prompts",         action="store_true", default=True)
    # ---- Logging ---------------------------------------------------------
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--report_to",           type=str, default="tensorboard")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    prompts      = [e["prompt"] for e in examples]
    return {"pixel_values": pixel_values.float(), "prompts": prompts}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, args.logging_dir),
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # ---- Load components -------------------------------------------------
    logger.info(f"Loading {args.model_id} …")
    tokenizer    = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_id, subfolder="text_encoder")
    vae          = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    unet         = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_id, subfolder="scheduler")

    # ---- Freeze VAE + text encoder (Basic FT: only U-Net is trained) ----
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # ---- Optimiser -------------------------------------------------------
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # ---- Dataset & DataLoader -------------------------------------------
    train_dataset = BrainTumorMRIDataset(
        data_dir=args.train_data_dir,
        resolution=args.resolution,
        use_diverse_prompts=args.use_diverse_prompts,
        augment=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    # ---- LR scheduler ---------------------------------------------------
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ---- Accelerator prepare --------------------------------------------
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    text_encoder = text_encoder.to(accelerator.device)
    vae          = vae.to(accelerator.device)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    # ---- Logging ---------------------------------------------------------
    logger.info("***** Starting Basic Fine-Tuning with Latent Drifting *****")
    logger.info(f"  Dataset size        : {len(train_dataset)}")
    logger.info(f"  Epochs              : {args.num_train_epochs}")
    logger.info(f"  Batch size          : {args.train_batch_size}")
    logger.info(f"  Grad accum steps    : {args.gradient_accumulation_steps}")
    logger.info(f"  Max training steps  : {args.max_train_steps}")
    logger.info(f"  Learning rate       : {args.learning_rate}")
    logger.info(f"  Latent Drift δ      : {args.delta}")

    if accelerator.is_main_process:
        accelerator.init_trackers("latent_drifting_ft", config=vars(args))

    # ---- Training loop --------------------------------------------------
    global_step    = 0
    first_epoch    = 0
    progress_bar   = tqdm(
        range(global_step, args.max_train_steps),
        desc="Training steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # ---- Encode images to latent space ----------------------
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # ---- Sample noise ---------------------------------------
                # Standard: ε ~ N(0, I)
                noise = torch.randn_like(latents)

                # ---- Latent Drifting: shift noise target (forward process)
                # "LD is added to the target zT of the forward process"
                # We add δ to the noise before creating noisy latents,
                # effectively sampling from N(δ, I) instead of N(0,I).
                if args.delta != 0.0:
                    noise_ld = noise + args.delta
                else:
                    noise_ld = noise

                # ---- Sample random timesteps ----------------------------
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # ---- Forward diffusion (add noise to latents) -----------
                noisy_latents = noise_scheduler.add_noise(latents, noise_ld, timesteps)

                # ---- Encode text prompts --------------------------------
                with torch.no_grad():
                    text_ids = tokenizer(
                        batch["prompts"],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(accelerator.device)
                    encoder_hidden_states = text_encoder(text_ids)[0]

                # ---- Predict noise (UNet forward) -----------------------
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                # ---- MSE loss against the unshifted noise ε -----------
                # The model learns to predict the noise; the LD shift
                # adapts the latent distribution toward the medical domain.
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # ---- Backward pass -------------------------------------
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # ---- Logging & checkpointing --------------------------------
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(ckpt_dir)
                        logger.info(f"Saved checkpoint → {ckpt_dir}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # ---- Save final model -----------------------------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.model_id,
            unet=unet,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipeline.save_pretrained(args.output_dir)
        logger.info(f"Model saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
