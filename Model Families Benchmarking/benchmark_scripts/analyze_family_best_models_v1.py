import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths
results_path = "benchmark_results_all_families.csv"
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Load combined results
df = pd.read_csv(results_path)

# Handle missing or invalid data
df = df.dropna(subset=["Accuracy_%", "Avg_Latency_s", "Avg_Tokens_per_s", "Avg_RAM_%", "Performance_Score"])

# Normalize metrics for fair comparison
metrics = ["Accuracy_%", "Performance_Score", "Avg_Tokens_per_s"]
inverse_metrics = ["Avg_Latency_s", "Avg_RAM_%"]

for col in metrics:
    df[col + "_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

for col in inverse_metrics:
    df[col + "_norm"] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Weighted composite score
df["Composite_Score"] = (
    0.4 * df["Accuracy_%_norm"]
    + 0.3 * df["Performance_Score_norm"]
    + 0.15 * df["Avg_Tokens_per_s_norm"]
    + 0.1 * df["Avg_Latency_s_norm"]
    + 0.05 * df["Avg_RAM_%_norm"]
)

# Rank models within families
df["Rank_in_Family"] = df.groupby("Family")["Composite_Score"].rank(ascending=False)

# Pick best model per family
best_per_family = df.loc[df.groupby("Family")["Composite_Score"].idxmax()].reset_index(drop=True)

# Save results
best_per_family.to_csv("best_models_per_family.csv", index=False)
print("Best models per family saved to best_models_per_family.csv")

print("\nTop model from each family:\n")
print(best_per_family[["Family", "Model", "Composite_Score", "Accuracy_%", "Avg_Latency_s", "Avg_RAM_%", "Performance_Score"]])

#  Visualization  
sns.set_theme(style="whitegrid")

#Composite Score per Family
plt.figure(figsize=(8, 5))
sns.barplot(data=best_per_family, x="Family", y="Composite_Score", palette="crest")
plt.title("Composite Score of Top Model per Family")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "best_models_composite_scores.png"))
plt.close()

#Accuracy vs Latency trade-off
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=best_per_family,
    x="Avg_Latency_s",
    y="Accuracy_%",
    hue="Family",
    s=120,
    palette="tab10",
    edgecolor="black"
)
plt.title("Accuracy vs Latency (Best Model per Family)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "accuracy_vs_latency_best_models.png"))
plt.close()

# Radar Chart Alternative – Performance Comparison
from math import pi

categories = ["Accuracy_%", "Performance_Score", "Avg_Tokens_per_s", "Avg_Latency_s", "Avg_RAM_%"]
N = len(categories)

values = []
labels = best_per_family["Family"].tolist()

for _, row in best_per_family.iterrows():
    val = [
        row["Accuracy_%_norm"],
        row["Performance_Score_norm"],
        row["Avg_Tokens_per_s_norm"],
        row["Avg_Latency_s_norm"],
        row["Avg_RAM_%_norm"]
    ]
    values.append(val)

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

for i, val in enumerate(values):
    val += val[:1]
    ax.plot(angles, val, linewidth=2, linestyle='solid', label=labels[i])
    ax.fill(angles, val, alpha=0.15)

plt.xticks(angles[:-1], categories, color='grey', size=8)
plt.title("Radar Comparison of Top Models per Family", size=11, pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "radar_best_models.png"))
plt.close()
