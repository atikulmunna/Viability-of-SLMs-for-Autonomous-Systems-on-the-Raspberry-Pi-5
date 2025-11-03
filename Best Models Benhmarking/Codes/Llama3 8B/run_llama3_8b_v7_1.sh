#!/usr/bin/env bash
# run_llama3_8b_v7_1.sh
# Launcher for Llama3.1 8B Benchmark (BigBoss v7.1)

export VENV="${HOME}/bigboss-env-py312"
export SCRIPT="${HOME}/bigboss_scripts/benchmark_llama3_8b_v7_1.py"
export LOGDIR="${HOME}/logs"
export LOG="${LOGDIR}/run_llama3_8b_v7_1.log"
export MODEL_ID="meta-llama/Llama-3.1-8B"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-12}"

mkdir -p "${LOGDIR}"
echo "Activating Python environment..."
source "${VENV}/bin/activate"

echo "Checking Hugging Face login..."
python3 - <<'PY' 2>/dev/null || true
from huggingface_hub import whoami
try:
    print("Authenticated as:", whoami(token=True).get("name", "(unknown)"))
except Exception as e:
    print("Hugging Face login check failed:", e)
PY

echo "Launching BigBoss v7.1 – Llama3.1 8B Benchmark"
echo "Tokens: ${MAX_NEW_TOKENS}"
echo "────────────────────────────────────────────"

python3 "${SCRIPT}" 2>&1 | tee -a "${LOG}"
