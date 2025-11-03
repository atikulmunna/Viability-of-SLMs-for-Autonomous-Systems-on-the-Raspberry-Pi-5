import os, glob, json, statistics, pandas as pd, matplotlib.pyplot as plt

DATA_DIR = "/home/munna/results_bigboss_v10"
OUT_DIR = os.path.join(DATA_DIR, "aggregated_report")
os.makedirs(OUT_DIR, exist_ok=True)

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

summary_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/summary.json"), recursive=True))
telemetry_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/telemetry.json"), recursive=True))
correctness_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/correctness_summary.json"), recursive=True))
outputs_files = sorted(glob.glob(os.path.join(DATA_DIR, "**/outputs.csv"), recursive=True))

rows = []
for i in range(max(len(summary_files), len(telemetry_files), len(correctness_files))):
    s = load_json(summary_files[i]) if i < len(summary_files) else None
    t = load_json(telemetry_files[i]) if i < len(telemetry_files) else None
    c = load_json(correctness_files[i]) if i < len(correctness_files) else None

    run_label = s.get("timestamp") if s and "timestamp" in s else f"run_{i+1}"
    avg_latency = s.get("avg_latency_s") if s else None
    total_prompts = s.get("total_prompts") if s else None
    tokens_per_prompt = s.get("tokens_per_prompt") if s else None

    mean_temp = mean_ram = mean_swap = None
    if t and isinstance(t, list) and len(t) > 0:
        temps = [x.get("temp_c") or x.get("temp") for x in t if isinstance(x.get("temp_c") or x.get("temp"), (int, float))]
        if temps: mean_temp = round(statistics.mean(temps), 2)
        rams = [x.get("ram_used_gb") or x.get("ram_used") for x in t if isinstance(x.get("ram_used_gb") or x.get("ram_used"), (int, float))]
        if rams: mean_ram = round(statistics.mean(rams), 2)
        swaps = [x.get("swap_used_gb") or x.get("swap_used") for x in t if isinstance(x.get("swap_used_gb") or x.get("swap_used"), (int, float))]
        if swaps: mean_swap = round(statistics.mean(swaps), 2)

    accuracy = None
    if c and isinstance(c, dict):
        accuracy = c.get("accuracy") or c.get("score") or c.get("exact_match")

    rows.append({
        "run_label": run_label,
        "summary_file": os.path.basename(summary_files[i]) if i < len(summary_files) else None,
        "telemetry_file": os.path.basename(telemetry_files[i]) if i < len(telemetry_files) else None,
        "correctness_file": os.path.basename(correctness_files[i]) if i < len(correctness_files) else None,
        "total_prompts": total_prompts,
        "tokens_per_prompt": tokens_per_prompt,
        "avg_latency_s": avg_latency,
        "mean_temp_c": mean_temp,
        "mean_ram_gb": mean_ram,
        "mean_swap_gb": mean_swap,
        "accuracy": accuracy,
    })

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "bigboss_v7_summary.csv")
df.to_csv(csv_path, index=False)

# Bar plots
if df["avg_latency_s"].notnull().any():
    plt.figure(figsize=(8,3))
    df_plot = df.dropna(subset=["avg_latency_s"])
    plt.bar(df_plot["run_label"], df_plot["avg_latency_s"])
    plt.ylabel("Avg latency (s)")
    plt.title("Average Latency per Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "avg_latency.png"))
    plt.close()

if df["mean_temp_c"].notnull().any():
    plt.figure(figsize=(8,3))
    df_plot = df.dropna(subset=["mean_temp_c"])
    plt.bar(df_plot["run_label"], df_plot["mean_temp_c"])
    plt.ylabel("Temperature (°C)")
    plt.title("Mean Temperature per Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mean_temp.png"))
    plt.close()

# Markdown summary
md_report = os.path.join(OUT_DIR, "bigboss_v7_report.md")
with open(md_report, "w", encoding="utf-8") as f:
    f.write("#BigBoss v7 Benchmark Report\n\n")
    f.write(df.to_markdown(index=False))
    f.write("\n\n---\n\n## Notes\n- Data collected from results_bigboss_v10 folder\n")
    f.write("- Each run’s correctness, telemetry, and summary merged here.\n")

print(f"\nReport generated:\n- {csv_path}\n- {md_report}\n- avg_latency.png / mean_temp.png")
