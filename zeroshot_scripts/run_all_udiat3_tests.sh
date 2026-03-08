#!/bin/bash

# Master runner for UDIAT3 multi-GPU inference

ROOT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
LOG_DIR="$ROOT_DIR/logs/udiat3_test_runs"
mkdir -p "$LOG_DIR"

echo "Starting UDIAT3 inference on 3 GPUs..."

# GPU 3: Baseline
echo "[GPU 3] Starting Baseline (High-Fidelity)..."
CUDA_VISIBLE_DEVICES=3 bash "$ROOT_DIR/zeroshot_scripts/zeroshot_udiat3_baseline_test.sh" > "$LOG_DIR/baseline_hf.log" 2>&1 &
P1=$!

# GPU 1: DPO
echo "[GPU 1] Starting DPO..."
CUDA_VISIBLE_DEVICES=1 bash "$ROOT_DIR/zeroshot_scripts/zeroshot_udiat3_dpo_test.sh" > "$LOG_DIR/dpo.log" 2>&1 &
P2=$!

# GPU 2: DHN
echo "[GPU 2] Starting DHN..."
CUDA_VISIBLE_DEVICES=2 bash "$ROOT_DIR/zeroshot_scripts/zeroshot_udiat3_dhn_test.sh" > "$LOG_DIR/dhn.log" 2>&1 &
P3=$!

echo "Processes started (PIDs: $P1, $P2, $P3)"
echo "Logs are being written to $LOG_DIR"
echo "You can monitor them using: tail -f $LOG_DIR/baseline_hf.log"

# Wait for all to finish
wait $P1 $P2 $P3

echo "All UDIAT3 inference runs completed!"
