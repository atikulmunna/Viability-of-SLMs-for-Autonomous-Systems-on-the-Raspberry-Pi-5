import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os

#  CONFIG 
RESULT_FILES = [
    "benchmark_results_qwen.csv",
    "benchmark_results_gemma.csv",
    "benchmark_results_deepseek.csv",
    "benchmark_results_llama.csv",
    "benchmark_results_smollm.csv"
]
OUTPUT_COMBINED = "benchmark_results_all_families.csv"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
 

# Load and combine all results
dfs = []
for file in RESULT_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        family_name = file.replace("benchmark_results_", "").replace(".csv", "")
        df["Family"] = family_name
        dfs.append(df)
    else:
        print(f"Warning: {file} not found, skipping.")

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(OUTPUT_COMBINED, index=False)
print(f"Combined results saved to {OUTPUT_COMBINED}")

# Calculate family-level averages
summary = combined.groupby("Family").agg({
    "Avg_Latency_s": "mean",
    "Avg_Tokens_per_s": "mean",
    "Avg_RAM_%": "mean",
    "Accuracy_%": "mean"
}).reset_index()

# Add Composite Performance Score (raw)
summary["Composite_Performance_Score"] = (
    (summary["Avg_Tokens_per_s"] * summary["Accuracy_%"]) /
    (summary["Avg_Latency_s"] * (summary["Avg_RAM_%"] + 1))
)

# Normalized 0–1 scaling for each metric
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

summary["TokensNorm"] = normalize(summary["Avg_Tokens_per_s"])
summary["AccNorm"] = normalize(summary["Accuracy_%"])
summary["LatNorm"] = normalize(summary["Avg_Latency_s"])
summary["RAMNorm"] = normalize(summary["Avg_RAM_%"])

# Weighted Normalized Score
summary["Normalized_Weighted_Score"] = (
    0.4 * summary["TokensNorm"] +
    0.3 * summary["AccNorm"] +
    0.2 * (1 - summary["LatNorm"]) +
    0.1 * (1 - summary["RAMNorm"])
)

# Save final summary
summary.to_csv("family_summary_scores.csv", index=False)
print("\nFamily-level Summary:")
print(summary)

#  Visuals 

sns.set_theme(style="whitegrid")

# Latency
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="Family", y="Avg_Latency_s", palette="coolwarm")
plt.title("Average Latency by Family", fontsize=14)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "latency_by_family.png"))
plt.close()

# Tokens per Second
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="Family", y="Avg_Tokens_per_s", palette="viridis")
plt.title("Average Tokens per Second by Family", fontsize=14)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "tokens_by_family.png"))
plt.close()

# RAM Usage
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="Family", y="Avg_RAM_%", palette="magma")
plt.title("Average RAM Usage (%) by Family", fontsize=14)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "ram_by_family.png"))
plt.close()

# Accuracy
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="Family", y="Accuracy_%", palette="cubehelix")
plt.title("Average Accuracy by Family", fontsize=14)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "accuracy_by_family.png"))
plt.close()

# Dual Score Comparison
plt.figure(figsize=(9, 5))
bar_width = 0.4
r1 = range(len(summary))
r2 = [x + bar_width for x in r1]

plt.bar(r1, summary["Composite_Performance_Score"], color='skyblue', width=bar_width, label='Raw Composite')
plt.bar(r2, summary["Normalized_Weighted_Score"], color='limegreen', width=bar_width, label='Normalized Weighted')

plt.xticks([r + bar_width/2 for r in range(len(summary))], summary["Family"], rotation=15)
plt.ylabel("Score")
plt.title("Composite vs Normalized Weighted Performance Scores", fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "composite_vs_normalized.png"))
plt.close()

print(f"\nAll visualizations saved to {FIGURES_DIR}/")
