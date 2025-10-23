import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from math import pi

# CONFIGURATION
CSV_FILE = "benchmark_results_all_families.csv"
OUTPUT_DIR = "figures/showdown"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD AND CLEAN DATA
df = pd.read_csv(CSV_FILE)
required = ["Family", "Model", "Avg_Tokens_per_s", "Avg_Latency_s", "Avg_RAM_%", "Accuracy_%"]
df = df.dropna(subset=required)

# NORMALIZATION
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df["TokensNorm"] = normalize(df["Avg_Tokens_per_s"])
df["AccNorm"] = normalize(df["Accuracy_%"])
df["LatNorm"] = normalize(df["Avg_Latency_s"])
df["RAMNorm"] = normalize(df["Avg_RAM_%"])

df["CompositeNormScore"] = (
    0.4 * df["TokensNorm"] +
    0.3 * df["AccNorm"] +
    0.2 * (1 - df["LatNorm"]) +
    0.1 * (1 - df["RAMNorm"])
)

# PER-FAMILY TOP 5
families = df["Family"].unique()
summary_rows = []

for fam in families:
    fam_df = df[df["Family"] == fam].copy()
    fam_df = fam_df.sort_values(by="CompositeNormScore", ascending=False)
    top5 = fam_df.head(5)

    # Save family CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{fam}_top5.csv")
    top5.to_csv(csv_path, index=False)
    print(f"Saved {fam} top 5 summary → {csv_path}")

    # Collect for radar chart
    categories = ["TokensNorm", "AccNorm", "LatNorm", "RAMNorm"]
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    for i, row in top5.iterrows():
        values = [row[c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["Model"])
        ax.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], ["Tokens", "Accuracy", "Latency", "RAM"], color='grey', size=10)
    plt.title(f"Top 5 Models in {fam} Family", size=14, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()

    img_path = os.path.join(OUTPUT_DIR, f"{fam}_top5_radar.png")
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"Saved radar chart for {fam} → {img_path}")

    # Save top model for mega showdown
    summary_rows.append(top5.iloc[0])

#Top Model of Each Family
top_families_df = pd.DataFrame(summary_rows)

categories = ["TokensNorm", "AccNorm", "LatNorm", "RAMNorm"]
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for i, row in top_families_df.iterrows():
    values = [row[c] for c in categories]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{row['Family']} ({row['Model']})")
    ax.fill(angles, values, alpha=0.1)

plt.xticks(angles[:-1], ["Tokens", "Accuracy", "Latency", "RAM"], color='grey', size=10)
ax.set_rlabel_position(30)
plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
plt.title("Top-5 Showdown: Best Model from Each Family", size=15, y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

showdown_img = os.path.join(OUTPUT_DIR, "mega_showdown_radar.png")
plt.savefig(showdown_img, dpi=300)
plt.close()

print(f"\nSaved Mega Showdown Radar → {showdown_img}")

# FINAL SUMMARY
leaderboard_path = os.path.join(OUTPUT_DIR, "top5_showdown_leaderboard.csv")
top_families_df.to_csv(leaderboard_path, index=False)
print(f" Leaderboard saved → {leaderboard_path}")

best = top_families_df.loc[top_families_df["CompositeNormScore"].idxmax()]
print("\n Absolute Top Model Across Families:")
print(best[["Family", "Model", "CompositeNormScore", "Avg_Latency_s", "Avg_Tokens_per_s", "Avg_RAM_%", "Accuracy_%"]])


# PERFORMANCE HEATMAP (Top Models Across Families)
import seaborn as sns

# Prepare data for heatmap
metrics = ["TokensNorm", "AccNorm", "LatNorm", "RAMNorm", "CompositeNormScore"]
heatmap_df = top_families_df.set_index(["Family", "Model"])[metrics]

# Order columns logically
heatmap_df = heatmap_df[["TokensNorm", "AccNorm", "LatNorm", "RAMNorm", "CompositeNormScore"]]

plt.figure(figsize=(8, 4))
sns.heatmap(
    heatmap_df,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    linewidths=0.5,
    cbar_kws={'label': 'Normalized Performance'},
    square=False
)
plt.title("Performance Heatmap — Top Model of Each Family", fontsize=13, pad=15)
plt.ylabel("Model Family")
plt.xlabel("Metric")
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, "top_models_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()

print(f"Heatmap saved → {heatmap_path}")
