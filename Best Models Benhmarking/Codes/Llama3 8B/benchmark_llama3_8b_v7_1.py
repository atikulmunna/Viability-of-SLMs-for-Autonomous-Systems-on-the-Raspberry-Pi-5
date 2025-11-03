import os, time, json, csv, psutil, torch, datetime, subprocess, shutil, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

MODEL_ID = "meta-llama/Llama-3.1-8B"
OFFLOAD_DIR = "/home/munna/model_offload_llama3_8b_ext4"
PROMPTS_FILE = "/home/munna/bigboss_scripts/prompts_v4_six.json"
CORRECTNESS_CHECKER = "/home/munna/bigboss_scripts/correctness_checkers.py"
RESULTS_ROOT = "/home/munna/results_bigboss_v10"
LOG_PATH = "/home/munna/logs/benchmark_llama3_8b_v7_1.log"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 12))

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_ROOT, f"llama3_cpu_run_{ts}")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

def log_line(text):
    print(text, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.datetime.now().isoformat()} {text}\n")

def get_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        return None

log_line("BigBoss v7.1 – Llama3.1 8B Benchmark (FP32 Offload + Resume + Telemetry)")
log_line(f"Model ID: {MODEL_ID}")
log_line(f"Run directory: {RUN_DIR}\n")

#  Download snapshot if not cached 
try:
    if not os.path.isdir(MODEL_ID) and "/" in MODEL_ID:
        local_path = snapshot_download(repo_id=MODEL_ID, token=True)
        log_line(f"Model cached locally at: {local_path}")
        MODEL_PATH = local_path
    else:
        MODEL_PATH = MODEL_ID
except Exception as e:
    log_line(f"Snapshot download failed: {e}")
    MODEL_PATH = MODEL_ID

#  Load Tokenizer 
log_line("Loading tokenizer...")
try:
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, trust_remote_code=False)
    log_line("Tokenizer ready.\n")
except Exception as e:
    log_line(f"Tokenizer load failed: {e}")
    sys.exit(1)

#  Load Model 
log_line("Loading model (FP32 offload-safe)... please wait.")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        trust_remote_code=False,
    )
    log_line("Model loaded successfully.\n")
except Exception as e:
    log_line(f"Model load failed: {e}")
    sys.exit(1)

#  Load Prompts 
if not os.path.exists(PROMPTS_FILE):
    log_line(f"Prompts file not found: {PROMPTS_FILE}")
    sys.exit(1)

try:
    with open(PROMPTS_FILE, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        prompts = data.get("prompts", list(data.values()))
    elif isinstance(data, list):
        prompts = data
    else:
        prompts = [str(data)]
except Exception as e:
    log_line(f"Could not read prompts file: {e}")
    prompts = []

if not prompts:
    log_line("No valid prompts found. Exiting.")
    sys.exit(1)

total = len(prompts)
log_line(f"Running {total} prompts × {MAX_NEW_TOKENS} tokens each...\n")

results, telemetry = [], []
start_time = time.time()

def save_progress():
    with open(os.path.join(RUN_DIR, "outputs.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "output", "elapsed_s"])
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(RUN_DIR, "telemetry.json"), "w") as f:
        json.dump(telemetry, f, indent=2)

#  MAIN LOOP 
for i, prompt in enumerate(prompts, 1):
    t0 = time.time()
    try:
        enc = tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
    except Exception as e:
        log_line(f"[{i}/{total}] Encoding error: {e}")
        continue

    try:
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        text = tok.decode(out[0], skip_special_tokens=True)
    except Exception as e:
        log_line(f"[{i}/{total}] Generation failed: {e}")
        save_progress()
        continue

    elapsed = round(time.time() - t0, 2)
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    temp = get_temp()

    results.append({
        "id": i,
        "prompt": prompt,
        "output": text,
        "elapsed_s": elapsed
    })
    telemetry.append({
        "prompt_id": i,
        "temp_c": temp,
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "swap_used_gb": round(swap.used / (1024**3), 2),
        "elapsed_s": elapsed
    })

    log_line(f"[{i}/{total}] ({elapsed:.1f}s) Temp={temp}°C | RAM={mem.percent:.1f}% | SWAP={swap.percent:.1f}%")
    log_line(f"↳ {text[:200]}...\n")

    if i % 3 == 0 or i == total:
        save_progress()
        log_line(f"Progress saved ({i}/{total}).")

#  SUMMARY 
if results:
    avg_latency = round(sum(r["elapsed_s"] for r in results) / len(results), 2)
    summary = {
        "total_prompts": total,
        "tokens_per_prompt": MAX_NEW_TOKENS,
        "avg_latency_s": avg_latency,
        "runtime_min": round((time.time() - start_time) / 60, 2),
        "timestamp": ts,
    }
    with open(os.path.join(RUN_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log_line("\nBenchmark complete.")
    log_line(f"Results saved to: {RUN_DIR}")
    log_line(f"Average latency: {avg_latency}s per prompt")

if os.path.exists(CORRECTNESS_CHECKER):
    log_line("Running final correctness checker...")
    os.system(f"python3 {CORRECTNESS_CHECKER} {RUN_DIR}/outputs.csv > {RUN_DIR}/correctness.txt")
    log_line("Correctness report saved.")
else:
    log_line("Skipping correctness checker (not found).")
