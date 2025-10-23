import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIG
FAMILY_FILES = [
    "benchmark_results_qwen.csv",
    "benchmark_results_gemma.csv",
    "benchmark_results_deepseek.csv",
    "benchmark_results_llama.csv",
    "benchmark_results_smollm.csv"
]
OUTPUT_CSV = "benchmark_results_all_families.csv"
os.makedirs("figures", exist_ok=True)

# LOAD and COMBINE
dfs = []
for file in FAMILY_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        df["Family"] = os.path.basename(file).replace("benchmark_results_", "").replace(".csv", "")
        dfs.append(df)
    else:
        print(f"Missing: {file}")

if not dfs:
    raise FileNotFoundError("No CSVs found for analysis!")

combined_df = pd.concat(dfs, ignore_index=True)
combined_df.to_csv(OUTPUT_CSV, index=False)
print(f"Combined results saved to {OUTPUT_CSV}")

# BASIC STATS
summary = (
    combined_df.groupby("Family")
    .agg({
        "Avg_Latency_s": "mean",
        "Avg_Tokens_per_s": "mean",
        "Avg_CPU_%": "mean",
        "Avg_RAM_%": "mean",
        "Avg_Temp_C": "mean",
        "Accuracy_%": "mean",
        "Performance_Score": "mean"
    })
    .reset_index()
)
print("\nFamily-level Summary:")
print(summary)

# VISUALIZATIONS
sns.set_theme(style="whitegrid")

# Latency per family
plt.figure(figsize=(10,6))
sns.barplot(data=summary, x="Family", y="Avg_Latency_s", palette="coolwarm")
plt.title("Average Latency per Model Family (s)")
plt.tight_layout()
plt.savefig("figures/family_latency.png")

# Tokens per second
plt.figure(figsize=(10,6))
sns.barplot(data=summary, x="Family", y="Avg_Tokens_per_s", palette="viridis")
plt.title("Average Token Generation Speed (tokens/sec)")
plt.tight_layout()
plt.savefig("figures/family_tps.png")

# RAM usage
plt.figure(figsize=(10,6))
sns.barplot(data=summary, x="Family", y="Avg_RAM_%", palette="magma")
plt.title("Average RAM Usage per Family (%)")
plt.tight_layout()
plt.savefig("figures/family_ram.png")

# Temperature
plt.figure(figsize=(10,6))
sns.barplot(data=summary, x="Family", y="Avg_Temp_C", palette="cubehelix")
plt.title("Average Temperature per Family (°C)")
plt.tight_layout()
plt.savefig("figures/family_temp.png")

# PERFORMANCE RADAR
from math import pi

categories = ["Avg_Latency_s", "Avg_Tokens_per_s", "Accuracy_%", "Performance_Score"]
num_vars = len(categories)
angles = [i / float(num_vars) * 2 * pi for i in range(num_vars)]
angles += angles[:1]

plt.figure(figsize=(8,8))

for _, row in summary.iterrows():
    values = row[categories].values.flatten().tolist()
    # Normalize (inverting latency for better interpretation)
    normalized_values = []
    for c, v in zip(categories, values):
        if c == "Avg_Latency_s":
            v = 1 / v if v != 0 else 0
        normalized_values.append(v / max(summary[c]))
    normalized_values += normalized_values[:1]
    
    plt.polar(angles, normalized_values, linewidth=2, label=row["Family"])
    plt.fill(angles, normalized_values, alpha=0.25)

plt.title("Model Family Comparison Radar Chart", size=14, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.savefig("figures/family_radar.png")
print("\n Graphs (including radar) saved in 'figures/' folder!")
