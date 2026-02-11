#!/bin/bash

# Move to the source directory
cd biomedclip_finetuning/open_clip/src

# Full path to your existing model checkpoint
PRETRAINED_MODEL="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model/pytorch_model.bin"

echo "Starting Fine-tuning with DHN-Feature-Entropy Loss (Sparse Activation)..."
echo "Model: BioMedCLIP (Microsoft)"
echo "Pretrained Weights: $PRETRAINED_MODEL"

# Using --dhn-feature-entropy-loss instead of --dhn-entropy-loss
# entropy-weight controls how sparse/sharp the FEATURE MAP should be.
# Valid range usually 0.001 to 0.1 depending on feature magnitude.

CUDA_VISIBLE_DEVICES=0 python3 -m open_clip_train.main \
    --batch-size 16 \
    --workers 4 \
    --report-to tensorboard \
    --save-frequency 1 \
    --logs="logs/finetune_dhn_feature_entropy" \
    --dataset-type csv \
    --csv-separator="," \
    --train-data "data/medpix_dataset/medpix_dataset_clean.csv" \
    --csv-img-key filename \
    --csv-caption-key Caption \
    --lr=1e-3 \
    --wd=0.1 \
    --warmup 100 \
    --epochs=32 \
    --model "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224" \
    --dhn-base-entropy-loss \
    --entropy-weight 0.05 \
    --temperature-dhnnce 0.5 \
    --alpha-dhnnce 0.0 \
    --beta1-dhnnce 0.15 \
    --beta2-dhnnce 0.15
