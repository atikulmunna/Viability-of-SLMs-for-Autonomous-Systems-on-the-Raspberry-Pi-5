#!/bin/bash
echo "Launching BigBoss v7.1 – DeepSeek-R1 (Qwen 7B) Benchmark"
echo "Tokens: ${MAX_NEW_TOKENS:-12}"
echo "────────────────────────────────────────────"
source ~/bigboss-env-py312/bin/activate
echo "Auth confirmed."

SWAPFILE="/swapfile_bigboss"
SWAPSIZE_GB=6

# --- SMART SWAP SETUP ---
if ! swapon --show | grep -q "$SWAPFILE"; then
    echo "Creating temporary ${SWAPSIZE_GB} GB swapfile for DeepSeek benchmark..."
    sudo fallocate -l ${SWAPSIZE_GB}G $SWAPFILE
    sudo chmod 600 $SWAPFILE
    sudo mkswap $SWAPFILE >/dev/null
    sudo swapon $SWAPFILE
    echo "Swap enabled: $(swapon --show | grep $SWAPFILE | awk '{print $3}')"
else
    echo "Existing swapfile detected — skipping creation."
fi

# --- RUN BENCHMARK ---
python3 ~/bigboss_scripts/benchmark_deepseek_r1_qwen7b_v7_1.py
STATUS=$?

# --- CLEANUP ---
if [ $STATUS -eq 0 ]; then
    echo "Benchmark finished cleanly."
else
    echo "Benchmark exited with code $STATUS."
fi

if swapon --show | grep -q "$SWAPFILE"; then
    echo "Removing temporary swapfile..."
    sudo swapoff $SWAPFILE
    sudo rm -f $SWAPFILE
    echo "Swapfile cleaned up."
fi
