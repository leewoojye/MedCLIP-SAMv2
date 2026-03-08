#!/bin/bash

# Master runner for BUSI_TEST multi-GPU inference
# BUSI 전체를 테스트셋으로 해서 추론진행
# 테스트셋만 동일하다면 모델명/생성법 달라져도 적용가능

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
LOG_DIR="$ROOT_DIR/logs/busi_test_runs"
mkdir -p "$LOG_DIR"

echo "Starting BUSI_TEST inference on 3 GPUs..."

# GPU 0: Baseline
echo "[GPU 0] Starting Baseline..."
CUDA_VISIBLE_DEVICES=3 bash "$ROOT_DIR/zeroshot_scripts/zeroshot_busi_test_baseline.sh" > "$LOG_DIR/baseline.log" 2>&1 &
P1=$!

# GPU 1: DPO
echo "[GPU 1] Starting DPO (v15 candidate)..."
CUDA_VISIBLE_DEVICES=1 bash "$ROOT_DIR/zeroshot_scripts/zeroshot_busi_test_dpo.sh" > "$LOG_DIR/dpo.log" 2>&1 &
P2=$!

# GPU 2: DHN
echo "[GPU 2] Starting DHN..."
CUDA_VISIBLE_DEVICES=2 bash "$ROOT_DIR/zeroshot_scripts/zeroshot_busi_test_dhn.sh" > "$LOG_DIR/dhn.log" 2>&1 &
P3=$!

echo "Processes started (PIDs: $P1, $P2, $P3)"
echo "Logs are being written to $LOG_DIR"
echo "You can monitor them using: tail -f $LOG_DIR/dpo.log"

# Wait for all to finish
wait $P1 $P2 $P3

echo "All BUSI_TEST inference runs completed!"
