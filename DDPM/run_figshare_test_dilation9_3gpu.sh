#!/bin/bash
set -u

PROJECT_ROOT="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
PYTHON="${PROJECT_ROOT}/.venv_ddpm_blackwell/bin/python"
IMAGE_DIR="${PROJECT_ROOT}/data/Figshare/test_figshare3064/images"
MASK_DIR="${PROJECT_ROOT}/data/Figshare/test_figshare3064/masks"
CHECKPOINT="${PROJECT_ROOT}/checkpont/step_122000.pt"
OUTPUT_DIR="${PROJECT_ROOT}/generated_neg_output/figshare_test_ddpm_step122000_dilation9"
LOG_DIR="${PROJECT_ROOT}/DDPM/logs/figshare_test_ddpm_step122000_dilation9"
WORLD_SIZE=3
DILATION_SIZE=9
GPUS=(0 1 2)

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

if [ ! -x "${PYTHON}" ]; then
    echo "Missing Python environment: ${PYTHON}" >&2
    exit 1
fi
if [ ! -f "${CHECKPOINT}" ]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

pids=()
cleanup() {
    if [ "${#pids[@]}" -gt 0 ]; then
        kill "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup INT TERM

for rank in 0 1 2; do
    gpu="${GPUS[$rank]}"
    worker_log="${LOG_DIR}/gpu${gpu}_rank${rank}.log"
    echo "Starting rank=${rank} physical_gpu=${gpu} log=${worker_log}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
        -m DDPM.sample_figshare_dilated_sharded \
        --image-dir "${IMAGE_DIR}" \
        --mask-dir "${MASK_DIR}" \
        --checkpoint "${CHECKPOINT}" \
        --output-dir "${OUTPUT_DIR}" \
        --rank "${rank}" \
        --world-size "${WORLD_SIZE}" \
        --physical-gpu "${gpu}" \
        --dilation-size "${DILATION_SIZE}" \
        --crop-size 224 \
        --context-margin 16 \
        --diffusion-steps 1000 \
        --noise-schedule linear \
        --seed 42 \
        --weights ema \
        --blur-sigma 1.075 \
        --skip-existing \
        --log-every 1 \
        >"${worker_log}" 2>&1 &
    pids+=("$!")
done

status=0
for rank in 0 1 2; do
    if wait "${pids[$rank]}"; then
        echo "Rank ${rank} completed successfully."
    else
        worker_status=$?
        echo "Rank ${rank} failed with status ${worker_status}." >&2
        status=1
    fi
done

exit "${status}"
