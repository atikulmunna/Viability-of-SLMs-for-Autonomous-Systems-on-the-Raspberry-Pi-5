import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from math import pi

# CONFIG 
CSV_FILE = "benchmark_results_all_families.csv"
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#  LOAD and CLEAN DATA 
df = pd.read_csv(CSV_FILE)

# Select relevant columns (adjust if needed)
metrics = [
    "Avg_Tokens_per_s",
    "Avg_Latency_s",
    "Avg_RAM_%",
    "Accuracy_%"
]

# Drop incomplete rows
df = df.dropna(subset=metrics)

#  NORMALIZATION 
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df["TokensNorm"] = normalize(df["Avg_Tokens_per_s"])
df["AccNorm"] = normalize(df["Accuracy_%"])
df["LatNorm"] = normalize(df["Avg_Latency_s"])
df["RAMNorm"] = normalize(df["Avg_RAM_%"])

#  WEIGHTED SCORE 
df["Normalized_Weighted_Score"] = (
    0.4 * df["TokensNorm"] +
    0.3 * df["AccNorm"] +
    0.2 * (1 - df["LatNorm"]) +
    0.1 * (1 - df["RAMNorm"])
)

#  BAR CHARTS 
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
colors = sns.color_palette("husl", len(df))

sns.barplot(data=df, x="Family", y="Avg_Latency_s", ax=axes[0], palette="coolwarm")
axes[0].set_title("Average Latency (s)")

sns.barplot(data=df, x="Family", y="Avg_Tokens_per_s", ax=axes[1], palette="viridis")
axes[1].set_title("Average Tokens per Second")

sns.barplot(data=df, x="Family", y="Avg_RAM_%", ax=axes[2], palette="magma")
axes[2].set_title("Average RAM Usage (%)")

sns.barplot(data=df, x="Family", y="Accuracy_%", ax=axes[3], palette="cubehelix")
axes[3].set_title("Average Accuracy (%)")

for ax in axes:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "family_metric_bars_v2.png"), dpi=300)
plt.close()

print("✅ Saved: family_metric_bars_v2.png")

#  RADAR CHART 
categories = ["TokensNorm", "AccNorm", "LatNorm", "RAMNorm"]
num_vars = len(categories)

angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for i, row in df.iterrows():
    values = [row[c] for c in categories]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=row["Family"])
    ax.fill(angles, values, alpha=0.15)

plt.xticks(angles[:-1], ["Tokens", "Accuracy", "Latency", "RAM"], color='grey', size=10)
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.title("Normalized Performance Radar Chart", size=14, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

plt.savefig(os.path.join(OUTPUT_DIR, "family_radar_chart_v2.png"), dpi=300)
plt.close()

print("Saved: family_radar_chart_v2.png")

#  SAVE SUMMARY 
summary_path = os.path.join(OUTPUT_DIR, "family_summary_v2.csv")
df.to_csv(summary_path, index=False)
print(f"Saved detailed summary to {summary_path}")

#  DISPLAY TOP FAMILY 
top_family = df.loc[df["Normalized_Weighted_Score"].idxmax()]
print("\nTop Performing Family (Normalized Composite):")
print(top_family[["Family", "Normalized_Weighted_Score", "Avg_Latency_s", "Avg_Tokens_per_s", "Avg_RAM_%", "Accuracy_%"]])
