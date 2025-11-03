#!/bin/bash
set -e

echo "Launching BigBoss v7 – Qwen3 8B Benchmark (CPU-offload)"
MODEL_PATH="/home/munna/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
OFFLOAD_DIR="/home/munna/model_offload_qwen3_8b_ext4"
RESULTS_ROOT="/home/munna/results_bigboss_v10"
LOG_DIR="/home/munna/logs"
LOG_FILE="$LOG_DIR/qwen3_v7_$(date +%s).log"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

# Check env
if [[ -z "$MAX_NEW_TOKENS" ]]; then
  MAX_NEW_TOKENS=8
fi

echo "Tokens: $MAX_NEW_TOKENS"
echo "────────────────────────────────────────────"

# Activate environment
source ~/bigboss-env-py312/bin/activate

# Verify Hugging Face login
echo "Checking Hugging Face login..."
if hf auth whoami &>/dev/null; then
  echo "Auth confirmed."
else
  echo "Not logged in. Run 'hf auth login' first."
  exit 1
fi

# Launch benchmark
echo "Starting benchmark..."
python3 ~/bigboss_scripts/benchmark_qwen3_8b_cpu_v7.py | tee "$LOG_FILE"

echo "Benchmark finished. Log saved at: $LOG_FILE"
