#!/bin/bash
# Custom Diffusion Fine-Tuning with Latent Drifting (δ* = +0.050)
# 
# Paper reference:
#   - Section 4.2: Custom Diffusion approach
#   - Section 3.3: Latent Drifting mechanism
#   - Only cross-attention layers are trainable
#   - UNet, VAE, CLIP text encoder are frozen
#   - Paired training with positive (tumor) and negative (normal) samples

cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/ld_paper

source activate medclipsamv2

python train_custom_diffusion_ld.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --positive_data_dir ./data/train/tumor \
    --negative_data_dir ./normal_sample \
    --output_dir ./checkpoints/custom_diffusion_ld_delta_0050 \
    --delta 0.050 \
    --num_epochs 100 \
    --batch_size 2 \
    --resolution 512 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 100 \
    --mixed_precision fp16 \
    --seed 42 \
    --save_steps 10 \
    --logging_steps 5
