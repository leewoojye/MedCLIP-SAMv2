#!/bin/bash

# Run all UDIAT4 tests in parallel (Relocated to zeroshot_scripts)
# Robustly find script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit

# Path to root relative to this script
ROOT_DIR=".."
LOG_DIR="$ROOT_DIR/logs/udiat4_test_runs"

mkdir -p "$LOG_DIR"

echo "Starting UDIAT4 Baseline Zeroshot Test..."
bash zeroshot_udiat3_baseline_test.sh > "$LOG_DIR/baseline.log" 2>&1 &

echo "Starting UDIAT4 DHN Zeroshot Test..."
bash zeroshot_udiat3_dhn_test.sh > "$LOG_DIR/dhn.log" 2>&1 &

echo "Starting UDIAT4 DPO Zeroshot Test..."
bash zeroshot_udiat3_dpo_test.sh > "$LOG_DIR/dpo.log" 2>&1 &

echo "All tests started. Monitor logs in logs/udiat4_test_runs/"
wait
echo "All UDIAT4 tests completed."
