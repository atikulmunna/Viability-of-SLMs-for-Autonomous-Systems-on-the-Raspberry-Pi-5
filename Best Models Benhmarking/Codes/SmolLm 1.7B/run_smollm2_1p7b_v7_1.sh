#!/bin/bash
echo "Launching BigBoss v7.1 – SmolLM2 1.7B Benchmark"
echo "Tokens: ${MAX_NEW_TOKENS:-24}"
echo "────────────────────────────────────────────"

# Activate environment
source ~/bigboss-env-py312/bin/activate

# Verify login
if ! hf auth whoami >/dev/null 2>&1; then
    echo "Not logged in to Hugging Face! Please run: hf auth login"
    exit 1
fi
echo "Auth confirmed."

# Ensure swapfile
if ! swapon --show | grep -q swapfile; then
    echo "Creating 4GB swapfile..."
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
else
    echo "Existing swapfile detected — skipping creation."
fi

# Run the benchmark
python3 ~/bigboss_scripts/benchmark_smollm2_1p7b_v7_1.py
