#!/bin/bash

# BiomedCLIP DPO Training on BUSI (v6 - Local Noise Sigma 60, Beta 10)
# Goal: Train model to recognize tumor area by destroying it in negative samples.
# 사용법: ./biomedclip_dpo_busi_v6.sh [버전이름] (예: ./biomedclip_dpo_busi_v6.sh biomedclip_dpo_udiat_v14)
VERSION=${1:-biomedclip_dpo_udiat_v20}
EPOCHS=10

cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src

export PYTHONPATH=$PYTHONPATH:$(pwd)

# 학습 후 자동으로 hf_model로 변환하는 명령어를 nohup으로 묶어서 실행
nohup bash -c "
    python open_clip_train/main.py \
        --train-data /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/BUSI_TEST/busi_test_dpo_v2.csv \
        --csv-separator ',' \
        --csv-img-key filename \
        --csv-img-neg-key filename_neg \
        --csv-caption-key Caption \
        --csv-caption-neg-key Caption_neg \
        --lr=5e-7 \
        --wd=0.01 \
        --warmup 10 \
        --epochs=$EPOCHS \
        --batch-size=32 \
        --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
        --dpo-loss \
        --dpo-ref-checkpoint /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model/openclip_model.pt \
        --beta-dpo 10.0 \
        --name '$VERSION' > train_$VERSION.log 2>&1 && \
    cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2 && \
    python /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/convert_script.py --name '$VERSION' --epoch $EPOCHS
" > nohup_$VERSION.out 2>&1 &

echo "Training and conversion for $VERSION started in background."
echo "Logs: /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/train_$VERSION.log"
