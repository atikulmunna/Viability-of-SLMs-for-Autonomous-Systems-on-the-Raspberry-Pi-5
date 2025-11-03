import os, time, json, csv, psutil, torch, datetime, subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
OFFLOAD_DIR = "/home/munna/model_offload_deepseek_qwen7b_ext4"
PROMPTS_FILE = "/home/munna/bigboss_scripts/prompts_v4_six.json"
CORRECTNESS_CHECKER = "/home/munna/bigboss_scripts/correctness_checkers.py"
RESULTS_ROOT = "/home/munna/results_bigboss_v10"
LOG_PATH = "/home/munna/logs/live_deepseek_qwen7b_v7_1.log"

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 12))
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(RESULTS_ROOT, f"deepseek_qwen7b_run_{ts}")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_line(text):
    print(text, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(text + "\n")

def get_temp():
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.strip().replace("temp=", "").replace("'C", ""))
    except Exception:
        return 0.0

log_line(f"[{ts}] BigBoss v7.1 – DeepSeek R1 (Qwen 7B) Benchmark (FP32 Offload + Resume + Telemetry)")
log_line(f"[{ts}] Model path: {MODEL_PATH}")
log_line(f"[{ts}] Run directory: {RUN_DIR}\n")

#  Tokenizer 
log_line(f"[{ts}] Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
log_line(f"[{ts}] Tokenizer ready.\n")

#  Model Load 
log_line(f"[{ts}] Loading model (FP32 offload-safe)… please wait.")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,
)
log_line(f"[{ts}] Model loaded successfully.\n")

#  Prompts 
with open(PROMPTS_FILE, "r") as f:
    data = json.load(f)
prompts = data["prompts"] if "prompts" in data else data
if not prompts or not isinstance(prompts, list):
    raise ValueError(f"No valid prompts found in {PROMPTS_FILE}")

total = len(prompts)
log_line(f"[{ts}] Running {total} prompts × {MAX_NEW_TOKENS} tokens each…\n")

results, telemetry = [], []
start_time = time.time()

def save_progress():
    with open(os.path.join(RUN_DIR, "outputs.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prompt", "output", "elapsed_s"])
        w.writeheader()
        w.writerows(results)
    with open(os.path.join(RUN_DIR, "telemetry.json"), "w") as f:
        json.dump(telemetry, f, indent=2)

for i, prompt in enumerate(prompts, 1):
    try:
        t0 = time.time()
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen_text = tok.decode(output[0], skip_special_tokens=True)

        elapsed = round(time.time() - t0, 2)
        mem, swap = psutil.virtual_memory(), psutil.swap_memory()
        temp = get_temp()

        results.append({
            "id": i, "prompt": prompt,
            "output": gen_text, "elapsed_s": elapsed
        })
        telemetry.append({
            "prompt_id": i,
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "swap_used_gb": round(swap.used / (1024**3), 2),
            "temp_c": temp, "elapsed_s": elapsed
        })

        log_line(f"[{ts}] [{i}/{total}] ({elapsed:.1f}s) Temp={temp:.1f}°C | RAM={mem.percent:.1f}% | SWAP={swap.percent:.1f}%")
        log_line(f"[{ts}] ↳ {gen_text[:100]}...\n")

        if i % 3 == 0 or i == total:
            save_progress()
            log_line(f"[{ts}] Backup saved ({i}/{total}).")

        if i % 3 == 0 and os.path.exists(CORRECTNESS_CHECKER):
            log_line(f"[{ts}] Running partial correctness check…")
            os.system(f"python3 {CORRECTNESS_CHECKER} {RUN_DIR}/outputs.csv > {RUN_DIR}/correctness_partial_{i}.txt")
            log_line(f"[{ts}] correctness_partial_{i}.txt written.\n")

    except Exception as e:
        log_line(f"[{ts}] Error on prompt {i}: {e}")
        save_progress()

#  Summary 
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

log_line(f"[{ts}] Benchmark complete.")
log_line(f"[{ts}] Results saved to: {RUN_DIR}")
log_line(f"[{ts}] Average latency: {avg_latency}s per prompt")

if os.path.exists(CORRECTNESS_CHECKER):
    log_line(f"[{ts}] Running final correctness check…")
    os.system(f"python3 {CORRECTNESS_CHECKER} {RUN_DIR}/outputs.csv > {RUN_DIR}/correctness.txt")
    log_line(f"[{ts}] Correctness report saved.\n")
else:
    log_line(f"[{ts}] Skipping correctness checker (not found).\n")
