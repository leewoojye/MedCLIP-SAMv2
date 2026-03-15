#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
PYTHON_BIN="${ROOT}/.venv_ldm/bin/python"
GPU_ID="${GPU_ID:-1}"
IMAGE_ID="${IMAGE_ID:-100}" # Default test ID if not provided. Will grab first otherwise.

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${ROOT}/disease_conditioned_manipulation/vendor"
export KMP_DUPLICATE_LIB_OK=TRUE
export KMP_INIT_AT_FORK=FALSE
export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU

DELTA_CONFIG="${ROOT}/ld_sdinpaint_hybrid/checkpoints/mri_sd_inpaint_ld_hybrid/selected_delta.json"
INFER_OUTPUT_DIR="${ROOT}/ld_paper/outputs/brain_pix2pix_zero_ld_paper_finetuned"
MODEL_ID="/home/woojye2020/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14"

# Set these once the finetuning script finishes or provide new UNets
TRAIN_OUTPUT_UNET="${ROOT}/ld_sdinpaint_hybrid/checkpoints/mri_sd_inpaint_ld_hybrid/final/unet"

# Data sets. Defaults to standard repository structure.
TEST_IMAGE_DIR="${ROOT}/data/brain_tumors/test_images"
TEST_MASK_DIR="${ROOT}/data/brain_tumors/test_masks"

cd "${ROOT}"

echo "Running paper-reproduction inference (Disease-Conditioned Manipulation PIx2Pix Zero)"
"${PYTHON_BIN}" "${ROOT}/ld_paper/run_brain_pix2pix_zero_ld_paper.py" \
  --deltas -0.2 -0.1 -0.05 0.0 0.05 0.1 0.2 \
  --test-image-dir "${TEST_IMAGE_DIR}" \
  --test-mask-dir "${TEST_MASK_DIR}" \
  --image-id "${IMAGE_ID}" \
  --output-dir "${INFER_OUTPUT_DIR}" \
  --model-id "${MODEL_ID}" \
  --source-prompt "A brain MRI of a 65 year old with Alzheimer's Disease" \
  --target-prompt "A brain MRI of a 65 year old Cognitively Normal" \
  --num-inference-steps 50 \
  --guidance-scale 7.5 \
  --inversion-guidance-scale 1.0 \
  --cross-attention-guidance-amount 0.1 \
  --attention-preservation-weight 1.0 \
  --delta-scale 1.0 \
  --delta-application cumulative \
  --delta-schedule constant \
  --mask-edit \
  --preserve-background \
  --match-input-mode \
  --num-opt-steps 1 \
  --seed 42

echo "Done"
