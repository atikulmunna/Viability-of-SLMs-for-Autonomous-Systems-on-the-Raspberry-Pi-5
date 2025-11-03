#!/usr/bin/env python3
import os, time, json, csv, psutil, torch, datetime, subprocess, glob
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIGURATION
snapshots = glob.glob("/home/munna/.cache/huggingface/hub/models--google--gemma-3-4b-it/snapshots/*")
MODEL_PATH = max(snapshots, key=os.path.getmtime) if snapshots else "google/gemma-3-4b-it"
OFFLOAD_DIR = "/home/munna/model_offload_gemma3_4b_ext4"
PROMPTS_FILE = "/home/munna/bigboss_scripts/prompts_v4_six.json"
CORRECTNESS_CHECKER = "/home/munna/bigboss_scripts/correctness_checkers.py"
RESULTS_ROOT = "/home/munna/results_bigboss_v10"
BACKUP_ROOT = "/home/munna/backups_bigboss"
LOG_PATH = "/home/munna/logs/live_gemma3_v7_1.log"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 12))
os.makedirs(OFFLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_ROOT, exist_ok=True)
os.makedirs(BACKUP_ROOT, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# HELPERS
def log_line(text):
    t = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{t} {text}", flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{t} {text}\n")

def get_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        return 0.0

def save_progress():
    with open(os.path.join(RUN_DIR, "outputs.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "output", "elapsed_s"])
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(RUN_DIR, "telemetry.json"), "w") as f:
        json.dump(telemetry, f, indent=2)
    json.dump({"completed_prompts": len(results), "timestamp": time.time()},
              open(os.path.join(BACKUP_ROOT, f"{RUN_ID}_resume_state.json"), "w"))
    log_line(f"Backup saved ({len(results)}/{total}).")

# RUN CONTEXT
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ID = f"gemma3_cpu_run_{ts}"
RUN_DIR = os.path.join(RESULTS_ROOT, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

# LOG HEADER
log_line("BigBoss v7.1 – Gemma3 4B Benchmark (FP32 Offload + Resume + Telemetry)")
log_line(f"Model path: {MODEL_PATH}")
log_line(f"Run directory: {RUN_DIR}\n")

# AUTH CHECK
try:
    from huggingface_hub import whoami
    whoami()
    log_line("Hugging Face auth OK.")
except Exception as e:
    log_line(f"Auth check failed: {e}")

# LOAD TOKENIZER & MODEL
log_line("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
log_line("Tokenizer ready.\n")

log_line("Loading model (FP32 offload-safe)… please wait.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,
)
log_line("Model loaded successfully.\n")

# LOAD PROMPTS (robust)
if os.path.exists(PROMPTS_FILE):
    with open(PROMPTS_FILE, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "prompts" in data:
            prompts = data["prompts"]
        else:
            prompts = list(data.values())
    elif isinstance(data, list):
        if all(isinstance(p, dict) and "prompt" in p for p in data):
            prompts = [p["prompt"] for p in data]
        else:
            prompts = data
    else:
        prompts = [str(data)]
else:
    prompts = ["Say hello to Raspberry Pi 5!"]

prompts = [p for p in prompts if isinstance(p, str) and p.strip()]
total = len(prompts)
if total == 0:
    raise ValueError(f"No valid prompts found in {PROMPTS_FILE}")

log_line(f"Running {total} prompts × {MAX_NEW_TOKENS} tokens each…\n")

# BENCHMARK LOOP
results, telemetry = [], []
start_time = time.time()

for i, prompt in enumerate(prompts, 1):
    t0 = time.time()
    try:
        inputs = tok(prompt, return_tensors="pt").input_ids
    except Exception as e:
        log_line(f"Encoding error for prompt {i}: {e}")
        continue

    with torch.no_grad():
        output = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    gen_text = tok.decode(output[0], skip_special_tokens=True)

    elapsed = round(time.time() - t0, 2)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    temp = get_temp()

    results.append({"id": i, "prompt": prompt, "output": gen_text, "elapsed_s": elapsed})
    telemetry.append({
        "prompt_id": i,
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_percent": mem.percent,
        "swap_used_gb": round(swap.used / (1024**3), 2),
        "swap_percent": swap.percent,
        "temp_c": temp,
        "elapsed_s": elapsed
    })

    log_line(f"[{i}/{total}] ({elapsed:.1f}s) Temp={temp:.1f}°C | RAM={mem.percent:.1f}% | SWAP={swap.percent:.1f}%")
    log_line(f"↳ {gen_text[:100]}…\n")
    save_progress()

    if i % 3 == 0 and os.path.exists(CORRECTNESS_CHECKER):
        log_line("Running partial correctness check…")
        os.system(f"python3 {CORRECTNESS_CHECKER} {RUN_DIR}/outputs.csv > {RUN_DIR}/correctness_partial_{i}.txt")
        log_line(f"correctness_partial_{i}.txt written.\n")

# SUMMARY
if len(results) > 0:
    avg_latency = round(sum(r["elapsed_s"] for r in results) / len(results), 2)
else:
    avg_latency = 0.0

summary = {
    "total_prompts": total,
    "tokens_per_prompt": MAX_NEW_TOKENS,
    "avg_latency_s": avg_latency,
    "runtime_min": round((time.time() - start_time) / 60, 2),
    "timestamp": ts,
}
with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

log_line("Benchmark complete.")
log_line(f"Results saved to: {RUN_DIR}")
log_line(f"Average latency: {avg_latency}s per prompt")

if os.path.exists(CORRECTNESS_CHECKER):
    log_line("Running final correctness check…")
    os.system(f"python3 {CORRECTNESS_CHECKER} {RUN_DIR}/outputs.csv > {RUN_DIR}/correctness.txt")
    log_line("Correctness report saved.\n")
else:
    log_line("Skipping correctness checker (not found).\n")
