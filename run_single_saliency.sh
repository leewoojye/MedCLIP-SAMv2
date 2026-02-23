#!/bin/bash

# Default values
IMAGE_PATH="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/321 copy.png"
TEXT_PROMPT="a brain MRI with a tumor"
OUTPUT_PATH="saliency_output.png"
MODEL_PATH="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model"

# Check if arguments are provided
if [ "$#" -ge 1 ]; then
    IMAGE_PATH=$1
fi

if [ "$#" -ge 2 ]; then
    TEXT_PROMPT=$2
fi

if [ "$#" -ge 3 ]; then
    OUTPUT_PATH=$3
fi

if [ "$#" -ge 4 ]; then
    MODEL_PATH=$4
fi

echo "======================================"
echo " Running Saliency Map Generation"
echo "======================================"
echo "Input Image : ${IMAGE_PATH}"
echo "Text Prompt : '${TEXT_PROMPT}'"
echo "Output path : ${OUTPUT_PATH}"
echo "Model path  : ${MODEL_PATH}"
echo "--------------------------------------"

# Run the python script
python generate_single_saliency.py \
    --image "${IMAGE_PATH}" \
    --text "${TEXT_PROMPT}" \
    --out "${OUTPUT_PATH}" \
    --model_path "${MODEL_PATH}"

echo "======================================"
echo " Finished"
echo "======================================"
