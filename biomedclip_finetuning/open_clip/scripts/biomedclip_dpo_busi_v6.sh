#!/bin/bash

# BiomedCLIP DPO Training on BUSI (v6 - Local Noise Sigma 60, Beta 10)
# Goal: Train model to recognize tumor area by destroying it in negative samples.

cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src

export PYTHONPATH=$PYTHONPATH:$(pwd)

nohup python open_clip_train/main.py \
    --train-data /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/data/udiat_dpo.csv \
    --csv-separator "," \
    --csv-img-key filename \
    --csv-img-neg-key filename_neg \
    --csv-caption-key Caption \
    --csv-caption-neg-key Caption_neg \
    --lr=5e-7 \
    --wd=0.01 \
    --warmup 10 \
    --epochs=3 \
    --batch-size=32 \
    --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --dpo-loss \
    --dpo-ref-checkpoint "" \
    --beta-dpo 10.0 \
    --name "biomedclip_dpo_udiat_v11" > train_udiat.log 2>&1 &
