#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate medclipsamv2

DATASET="data/polyp"

# Function to run experiment
run_experiment() {
    EXP_NAME=$1
    MODEL_NAME=$2
    USE_FINETUNED=$3
    JSON_PATH=$4
    VVAR=$5
    VBETA=$6
    VLAYER=$7

    echo "=================================================="
    echo "Running Experiment: $EXP_NAME"
    echo "Model: $MODEL_NAME"
    echo "Finetuned: $USE_FINETUNED"
    echo "JSON: $JSON_PATH"
    echo "Params: vvar=$VVAR, vbeta=$VBETA, vlayer=$VLAYER"
    echo "=================================================="

    SAL_PATH="saliency_map_outputs/${DATASET}/${EXP_NAME}/masks"
    COARSE_PATH="coarse_outputs/${DATASET}/${EXP_NAME}/masks"
    SAM_PATH="sam_outputs/${DATASET}/${EXP_NAME}/masks"

    rm -rf "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"
    mkdir -p "${SAL_PATH}" "${COARSE_PATH}" "${SAM_PATH}"

    # Construct finetuned flag
    FINETUNED_FLAG=""
    if [ "$USE_FINETUNED" = "true" ]; then
        FINETUNED_FLAG="--finetuned --checkpoint-path saliency_maps/model"
    fi

    conda run -n medclipsamv2 python saliency_maps/generate_saliency_maps.py \
    --input-path ${DATASET}/test_images \
    --output-path ${SAL_PATH} \
    --model-name ${MODEL_NAME} \
    ${FINETUNED_FLAG} \
    --json-path ${JSON_PATH} \
    --reproduce \
    --vvar ${VVAR} \
    --vbeta ${VBETA} \
    --vlayer ${VLAYER} \
    --seed 12

    conda run -n medclipsamv2 python postprocessing/postprocess_saliency_maps.py \
    --input-path ${DATASET}/test_images \
    --output-path ${COARSE_PATH} \
    --sal-path ${SAL_PATH} \
    --postprocess kmeans \
    --filter

    conda run -n medclipsamv2 python segment-anything/prompt_sam.py \
    --input ${DATASET}/test_images \
    --mask-input ${COARSE_PATH} \
    --output ${SAM_PATH} \
    --model-type vit_h \
    --checkpoint segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth \
    --prompts boxes

    conda run -n medclipsamv2 python evaluation/eval.py \
    --gt_path ${DATASET}/test_masks \
    --seg_path ${SAM_PATH}
}

# 1. Baseline (Current polyp script config)
run_experiment "baseline" "BiomedCLIP" "true" "saliency_maps/text_prompts/polyp_testing.json" 0.1 1.0 7

# 2. Stock BiomedCLIP (Like Brain/Breast)
run_experiment "stock_biomedclip" "BiomedCLIP" "false" "saliency_maps/text_prompts/polyp_testing.json" 0.3 1.0 9

# 3. Simple Prompts (Need to create json first)
# Create simple JSON
conda run -n medclipsamv2 python -c "import json, os; 
files = sorted(os.listdir('data/polyp/test_images')); 
data = {f: 'A colonoscopy image of a polyp' for f in files if f.endswith(('.jpg', '.png'))}; 
with open('saliency_maps/text_prompts/polyp_simple.json', 'w') as f: json.dump(data, f, indent=2)"

run_experiment "simple_prompts" "BiomedCLIP" "false" "saliency_maps/text_prompts/polyp_simple.json" 0.3 1.0 9
