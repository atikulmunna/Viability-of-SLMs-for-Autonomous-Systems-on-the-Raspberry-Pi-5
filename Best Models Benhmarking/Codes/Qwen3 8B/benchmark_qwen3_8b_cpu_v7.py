import os, time, json, csv, psutil, torch, datetime, subprocess, shutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

#  CONFIG 
MODEL_PATH = "/home/munna/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
OFFLOAD_DIR = "/home/munna/model_offload_qwen3_8b_ext4"
PROMPTS_FILE = "/home/munna/bigboss_scripts/prompts_v5_lite.json"
CORRECTNESS_CHECKER = "/home/munna/bigboss_scripts/correctness_checkers.py"
RESULTS_ROOT = "/home/munna/results_bigboss_v10"
BACKUP_DIR = "/home/munna/backups_bigboss"
LOG_PATH = "/home/munna/logs/live_qwen3_v7.log"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 16))

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_ROOT, f"qwen3_cpu_run_{ts}")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# UTILITIES 
def log_line(text):
    """Print and append to the live log."""
    print(text, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")

def get_temp():
    """Read CPU temperature (Raspberry Pi)."""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        return 0.0

def backup_run():
    """Backup current progress folder."""
    dest = os.path.join(BACKUP_DIR, os.path.basename(RUN_DIR))
    try:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(RUN_DIR, dest)
        log_line(f"Backup saved to {dest}")
    except Exception as e:
        log_line(f"Backup failed: {e}")

#  INIT 
log_line("BigBoss v7.0 – Qwen3 8B Benchmark (FP32 Offload + Resume + Backup + Telemetry)")
log_line(f"Run directory: {RUN_DIR}\n")

#  LOAD TOKENIZER 
log_line("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
log_line("Tokenizer ready.\n")

#  LOAD MODEL 
log_line("Loading model (FP32 offload-safe)... please wait.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,
)
log_line("Model loaded successfully.\n")

#  LOAD PROMPTS 
if os.path.exists(PROMPTS_FILE):
    with open(PROMPTS_FILE, "r") as f:
        data = json.load(f)
    prompts = [p["text"] for p in data.get("prompts", [])]
else:
    prompts = ["Say hello to Raspberry Pi 5!"]

total = len(prompts)

#  RESUME DETECTION 
outputs_path = os.path.join(RUN_DIR, "outputs.csv")
telemetry_path = os.path.join(RUN_DIR, "telemetry.json")

completed_ids = set()
if os.path.exists(outputs_path):
    try:
        with open(outputs_path, "r") as f:
            reader = csv.DictReader(f)
            completed_ids = {int(r["id"]) for r in reader}
    except Exception:
        pass

done = len(completed_ids)
remaining = total - done
log_line(f"Checkpoint scan: {done}/{total} completed, {remaining} remaining.\n")

#  ETA ESTIMATION 
avg_prev = 3600  # default 1 hour per prompt
if done > 0:
    try:
        with open(outputs_path, "r") as f:
            reader = csv.DictReader(f)
            times = [float(r["elapsed_s"]) for r in reader if float(r["elapsed_s"]) > 0]
            if times:
                avg_prev = sum(times) / len(times)
    except Exception:
        pass

eta_sec = avg_prev * remaining
eta_hr, eta_min = int(eta_sec // 3600), int((eta_sec % 3600) // 60)
log_line(f"ETA: {eta_hr}h {eta_min}m (avg {avg_prev:.0f}s/prompt)\n")

results, telemetry = [], []
if os.path.exists(outputs_path):
    with open(outputs_path, "r") as f:
        reader = csv.DictReader(f)
        results = list(reader)
if os.path.exists(telemetry_path):
    with open(telemetry_path, "r") as f:
        telemetry = json.load(f)

def save_progress():
    """Write progress to disk."""
    with open(outputs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "output", "elapsed_s"])
        w.writeheader()
        w.writerows(results)
    with open(telemetry_path, "w") as f:
        json.dump(telemetry, f, indent=2)

#  MAIN LOOP 
log_line(f"Starting run: {remaining} prompts × {MAX_NEW_TOKENS} tokens each...\n")
start_time = time.time()

for i, prompt in enumerate(prompts, 1):
    if i in completed_ids:
        continue

    t0 = time.time()
    try:
        inputs = tok.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen_text = tok.decode(output[0], skip_special_tokens=True)

        elapsed = round(time.time() - t0, 2)
        mem, swap = psutil.virtual_memory(), psutil.swap_memory()
        temp = get_temp()

        results.append({"id": i, "prompt": prompt, "output": gen_text, "elapsed_s": elapsed})
        telemetry.append({
            "prompt_id": i,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "temp_c": temp,
            "elapsed_s": elapsed
        })

        log_line(f"[{i}/{total}] ({elapsed:.1f}s) Temp={temp:.1f}°C | RAM={mem.percent:.1f}% | SWAP={swap.percent:.1f}%")
        log_line(f"↳ {gen_text[:100]}...\n")

        save_progress()
        if i % 2 == 0:
            backup_run()

        if i % 3 == 0 and os.path.exists(CORRECTNESS_CHECKER):
            log_line("Running partial correctness check...")
            os.system(f"python3 {CORRECTNESS_CHECKER} {RUN_DIR}/outputs.csv > {RUN_DIR}/correctness_partial_{i}.txt")
            log_line(f"correctness_partial_{i}.txt written.\n")

    except Exception as e:
        Path(RUN_DIR, f"error_prompt_{i}.txt").write_text(f"Prompt: {prompt}\nError: {e}")
        log_line(f"Error on prompt {i}: {e}")
        continue

#  SUMMARY 
avg_latency = round(sum(float(r["elapsed_s"]) for r in results) / len(results), 2) if results else 0
summary = {
    "total_prompts": total,
    "completed_prompts": len(results),
    "tokens_per_prompt": MAX_NEW_TOKENS,
    "avg_latency_s": avg_latency,
    "runtime_min": round((time.time() - start_time) / 60, 2),
    "timestamp": ts,
}
with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

log_line("\nBenchmark complete.")
log_line(f"Results saved to: {RUN_DIR}")
log_line(f"Completed {len(results)}/{total} prompts.")
log_line(f"Average latency: {avg_latency}s per prompt")

if os.path.exists(CORRECTNESS_CHECKER):
    log_line("Running final correctness check...")
    os.system(f"python3 {CORRECTNESS_CHECKER} {RUN_DIR}/outputs.csv > {RUN_DIR}/correctness_final.txt")
    log_line("Correctness report saved.\n")
else:
    log_line("Skipping correctness checker (not found).")
