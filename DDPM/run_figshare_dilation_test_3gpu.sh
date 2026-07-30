#!/bin/bash
set -u

PROJECT_ROOT="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
PYTHON="${PROJECT_ROOT}/.venv_ddpm_blackwell/bin/python"
IMAGE_DIR="${PROJECT_ROOT}/data/Figshare/test_figshare3064/images"
MASK_DIR="${PROJECT_ROOT}/data/Figshare/test_figshare3064/masks"
CHECKPOINT="${PROJECT_ROOT}/checkpont/step_122000.pt"
OUTPUT_ROOT="${PROJECT_ROOT}/DDPM/dilation_tests/step122000"
LOG_DIR="${OUTPUT_ROOT}/logs"
SAMPLES=(
    1790.png
    992.png
    688.png
    2744.png
    1106.png
    2290.png
    146.png
    71.png
    2763.png
)
GPUS=(0 1 2)
DILATIONS=(0 9 15)

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

pids=()
for slot in 0 1 2; do
    gpu="${GPUS[$slot]}"
    dilation="${DILATIONS[$slot]}"
    worker_log="${LOG_DIR}/dilation_${dilation}_gpu${gpu}.log"
    echo "Starting dilation=${dilation} on physical GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
        -m DDPM.sample_figshare_dilation_test \
        --image-dir "${IMAGE_DIR}" \
        --mask-dir "${MASK_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_ROOT}/dilation_${dilation}" \
        --sample-names "${SAMPLES[@]}" \
        --dilation-size "${dilation}" \
        --physical-gpu "${gpu}" \
        --crop-size 224 \
        --context-margin 16 \
        --diffusion-steps 1000 \
        --noise-schedule linear \
        --seed 42 \
        --weights ema \
        --blur-sigma 1.075 \
        >"${worker_log}" 2>&1 &
    pids+=("$!")
done

status=0
for slot in 0 1 2; do
    if wait "${pids[$slot]}"; then
        echo "Dilation ${DILATIONS[$slot]} completed successfully."
    else
        worker_status=$?
        echo "Dilation ${DILATIONS[$slot]} failed with status ${worker_status}." >&2
        status=1
    fi
done

exit "${status}"
