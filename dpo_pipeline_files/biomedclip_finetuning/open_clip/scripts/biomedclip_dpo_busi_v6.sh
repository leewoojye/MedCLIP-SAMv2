#!/bin/bash

# BiomedCLIP DPO Training on BUSI (v6 - Local Noise Sigma 60, Beta 10)
# Goal: Train model to recognize tumor area by destroying it in negative samples.

cd biomedclip_finetuning/open_clip/src

export PYTHONPATH=$PYTHONPATH:$(pwd)

python open_clip_train/main.py \
    --train-data data/breast_dataset/busi_train_dpo_v6.csv \
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
    --pretrained /home/bongdong2/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/snapshots/9f341de24bfb00180f1b847274256e9b65a3a32e/open_clip_pytorch_model.bin \
    --dpo-loss \
    --beta-dpo 10.0 \
    --name "biomedclip_dpo_busi_v6"
