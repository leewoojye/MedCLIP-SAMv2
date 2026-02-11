#!/bin/bash

# Move to the source directory
cd biomedclip_finetuning/open_clip/src

# Full path to your existing model checkpoint
# If this is a HuggingFace style directory, you might need to point solely to the bin file 
# or ensure the loader can handle it. 
# open_clip usually expects a .pt file for resume/pretrained, or a HF Hub ID.
# If loading a local HF model directly fails, you might need to convert it or use the HF Hub path if valid.
# Here we attempt to load it as a pretrained weight file.
PRETRAINED_MODEL="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model/pytorch_model.bin"

echo "Starting Fine-tuning with DHN-Entropy Loss..."
echo "Model: BioMedCLIP (Microsoft)"
echo "Pretrained Weights: $PRETRAINED_MODEL"

# Note: Adjust --train-data to your actual dataset CSV path.
# The current path 'data/medpix_dataset/medpix_dataset.csv' is a placeholder from the original script.

CUDA_VISIBLE_DEVICES=0 python3 -m open_clip_train.main \
    --batch-size 16 \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs="logs/finetune_dhn_entropy" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data "data/medpix_dataset/medpix_dataset_clean.csv" \
    --csv-img-key filename \
    --csv-caption-key Caption \
    --lr=1e-5 \
    --wd=0.1 \
    --warmup 100 \
    --epochs=10 \
    --model "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" \
    --dhn-feature-entropy-loss \
    --entropy-weight 0.05 \
    --temperature-dhnnce 0.5 \
    --alpha-dhnnce 0.0 \
    --beta1-dhnnce 0.1 \
    --beta2-dhnnce 0.1
