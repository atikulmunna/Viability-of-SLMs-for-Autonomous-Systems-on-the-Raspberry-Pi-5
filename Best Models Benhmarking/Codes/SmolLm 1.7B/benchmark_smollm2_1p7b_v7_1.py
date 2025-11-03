import os, time, json, csv, psutil, torch, datetime, subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

#  CONFIG 
MODEL_PATH = "/home/munna/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct/snapshots/31b70e2e869a7173562077fd711b654946d38674"
OFFLOAD_DIR = "/home/munna/model_offload_smollm2_1p7b_ext4"
PROMPTS_FILE = "/home/munna/bigboss_scripts/prompts_v4_six.json"
CORRECTNESS_CHECKER = "/home/munna/bigboss_scripts/correctness_checkers.py"
RESULTS_ROOT = "/home/munna/results_bigboss_v10"
BACKUP_ROOT = "/home/munna/backups_bigboss"
LOG_PATH = "/home/munna/logs/benchmark_smollm2_1p7b_v7_1.log"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 24))
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_ROOT, f"smollm2_cpu_run_{ts}")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(BACKUP_ROOT, exist_ok=True)

def log_line(msg):
    """Print + append to live log"""
    print(msg, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] {msg}\n")

def get_temp():
    """Read CPU temperature"""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        return 0.0

log_line(f"BigBoss v7.1 – SmolLM2-1.7B Benchmark (FP32 Offload + Resume + Telemetry)")
log_line(f"Model path: {MODEL_PATH}")
log_line(f"Run directory: {RUN_DIR}\n")

#  TOKENIZER 
log_line("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
log_line("Tokenizer ready.\n")

#  MODEL LOAD 
log_line("Loading model (FP32 offload-safe)… please wait.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,
)
log_line("Model loaded successfully.\n")

#  PROMPTS 
if not os.path.exists(PROMPTS_FILE):
    raise FileNotFoundError(f"Prompt file missing: {PROMPTS_FILE}")
with open(PROMPTS_FILE, "r") as f:
    data = json.load(f)
prompts = data["prompts"] if isinstance(data, dict) and "prompts" in data else data
if not isinstance(prompts, list) or not prompts:
    raise ValueError(f"No valid prompts found in {PROMPTS_FILE}")

total = len(prompts)
log_line(f"Running {total} prompts × {MAX_NEW_TOKENS} tokens each…\n")

results, telemetry = [], []
start_time = time.time()

#  Resume detection 
existing_output = os.path.join(RUN_DIR, "outputs.csv")
if os.path.exists(existing_output):
    with open(existing_output, "r") as f:
        prev = list(csv.DictReader(f))
    done_ids = {int(r["id"]) for r in prev}
    results.extend(prev)
    log_line(f"Resuming from checkpoint: {len(done_ids)}/{total} prompts already completed.")
else:
    done_ids = set()
    log_line("No previous checkpoint found, starting fresh.")

avg_eta = 0.0

def save_progress():
    """Write outputs and telemetry"""
    with open(os.path.join(RUN_DIR, "outputs.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "output", "elapsed_s"])
        w.writeheader(); w.writerows(results)
    with open(os.path.join(RUN_DIR, "telemetry.json"), "w") as f:
        json.dump(telemetry, f, indent=2)

#  BENCHMARK LOOP 
for i, prompt in enumerate(prompts, 1):
    if i in done_ids:
        continue

    t0 = time.time()
    try:
        inputs = tok(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        log_line(f"Generation error for prompt {i}: {e}")
        text = ""

    elapsed = round(time.time() - t0, 2)
    avg_eta = (avg_eta * (i - 1) + elapsed) / i
    mem = psutil.virtual_memory(); swap = psutil.swap_memory(); temp = get_temp()

    eta_remaining = avg_eta * (total - i)
    eta_h, eta_m = divmod(eta_remaining / 60, 60)

    results.append({"id": i, "prompt": prompt, "output": text, "elapsed_s": elapsed})
    telemetry.append({
        "prompt_id": i,
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "swap_used_gb": round(swap.used / (1024**3), 2),
        "temp_c": temp,
        "elapsed_s": elapsed
    })

    log_line(f"[{i}/{total}] ({elapsed:.1f}s) Temp={temp:.1f}°C | RAM={mem.percent:.1f}% | SWAP={swap.percent:.1f}% | ETA≈{int(eta_h)}h {int(eta_m)}m")
    log_line(f"↳ {text[:120]}...\n")

    # Save progress every 2 prompts
    if i % 2 == 0 or i == total:
        save_progress()
        backup_path = os.path.join(BACKUP_ROOT, os.path.basename(RUN_DIR))
        os.makedirs(backup_path, exist_ok=True)
        for fn in ("outputs.csv", "telemetry.json"):
            src = os.path.join(RUN_DIR, fn)
            if os.path.exists(src):
                subprocess.run(["cp", src, backup_path])
        log_line(f"Backup saved ({i}/{total}).")

    # Partial correctness every 3 prompts
    if i % 3 == 0 and os.path.exists(CORRECTNESS_CHECKER):
        log_line("Running partial correctness check…")
        os.system(f"python3 {CORRECTNESS_CHECKER} {RUN_DIR}/outputs.csv > {RUN_DIR}/correctness_partial_{i}.txt")
        log_line(f"correctness_partial_{i}.txt written.\n")

# SUMMARY 
avg_latency = round(sum(float(r["elapsed_s"]) for r in results) / len(results), 2)
summary = {
    "total_prompts": total,
    "tokens_per_prompt": MAX_NEW_TOKENS,
    "avg_latency_s": avg_latency,
    "runtime_min": round((time.time() - start_time) / 60, 2),
    "timestamp": ts
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
