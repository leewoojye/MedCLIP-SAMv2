#!/bin/bash
# Run BioMedDDBM Training

# Pos: Source (Tumor)
# Neg: Target (Generated Healthy)

/home/woojye2020/.conda/envs/medclipsamv2/bin/python -m biomed_ddbm.train \
    --pos_dir data/breast_tumors/test_images \
    --neg_dir generated_neg_output/breast_tumors \
    --mask_dir data/breast_tumors/test_masks \
    --output_dir biomed_ddbm_output \
    --epochs 20 \
    --batch_size 8 \
    --lr 1e-4
