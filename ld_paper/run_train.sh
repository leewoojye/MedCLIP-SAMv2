#!/bin/bash

cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/ld_paper

# Clean up old logs
rm -f train_custom_diffusion_ld.log

# Run training with unbuffered output
# Custom Diffusion: Few-shot fine-tuning with 3-5 images, 250-500 steps (Paper section 4.2)
python -u train_custom_diffusion_ld.py \
    --model_id CompVis/stable-diffusion-v1-4 \
    --positive_data_dir ./data/train/tumor \
    --negative_data_dir ./normal_sample \
    --output_dir ./checkpoints/custom_diffusion_ld_delta_0050 \
    --delta 0.050 \
    --max_training_steps 300 \
    --batch_size 2 \
    --resolution 512 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 50 \
    --mixed_precision no \
    --seed 42 \
    --save_steps 30 \
    --logging_steps 5 \
    2>&1 | tee train_custom_diffusion_ld.log

echo "Training completed!"
