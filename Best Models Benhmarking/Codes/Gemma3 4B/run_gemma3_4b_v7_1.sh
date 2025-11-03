#!/bin/bash
set -e
echo "Launching BigBoss v7.1 – Gemma3 4B (CPU-offload)"
echo "Tokens: ${MAX_NEW_TOKENS:-12}"
echo "────────────────────────────────────────────"

source ~/bigboss-env-py312/bin/activate

if hf auth whoami >/dev/null 2>&1; then
    echo "Auth confirmed."
else
    echo "Hugging Face not logged in! Run: hf auth login"
    exit 1
fi

LOG_FILE=~/logs/live_gemma3_v7_1_$(date +%s).log
SCRIPT=~/bigboss_scripts/benchmark_gemma3_4b_v7_1.py
mkdir -p ~/logs

echo "Starting benchmark..."
python3 "$SCRIPT" 2>&1 | tee "$LOG_FILE"
