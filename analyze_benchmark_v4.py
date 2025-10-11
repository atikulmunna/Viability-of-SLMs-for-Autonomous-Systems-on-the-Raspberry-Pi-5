import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

CSV_FILE = "benchmark_summary.csv"
sns.set(style="whitegrid", context="notebook")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file '{CSV_FILE}' not found!")

df = pd.read_csv(CSV_FILE)

def find_col(df, options):
    for col in options:
        if col in df.columns:
            return col
    return None

lat_col = find_col(df, ["Avg_Latency_s", "Latency_s", "end_to_end_latency_s"])
tok_col = find_col(df, ["Avg_Tokens_per_s", "tokens_per_s"])
cpu_col = find_col(df, ["Avg_CPU_%"])
ram_col = find_col(df, ["Avg_RAM_%"])
temp_col = find_col(df, ["Avg_Temp_C"])
acc_col = find_col(df, ["Accuracy_%"])
perf_col = find_col(df, ["Performance_Score"])

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# === SORT MODELS ===
sort_col = perf_col or acc_col or lat_col
ascending = False if sort_col in [perf_col, acc_col] else True
df = df.sort_values(by=sort_col, ascending=ascending)

# === VISUALIZATION ===
num_models = len(df)
fig_height = max(16, num_models * 1.0)
fig, axes = plt.subplots(3, 2, figsize=(18, fig_height))
fig.suptitle("📊 Benchmark Analysis (v7)", fontsize=20, weight="bold", y=0.995)

def plot_bar(ax, x, y, title, palette):
    sns.barplot(x=x, y=y, data=df, ax=ax, palette=palette)
    ax.set_title(title, fontsize=14, weight="bold", pad=10)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    for label in ax.get_yticklabels():
        label.set_wrap(True)

plot_bar(axes[0, 0], lat_col, "Model", "Average Latency (s)", "viridis")
plot_bar(axes[0, 1], tok_col, "Model", "Average Tokens per Second", "mako")
plot_bar(axes[1, 0], cpu_col, "Model", "Average CPU Usage (%)", "crest")
plot_bar(axes[1, 1], ram_col, "Model", "Average RAM Usage (%)", "rocket")

if temp_col:
    plot_bar(axes[2, 0], temp_col, "Model", "Average Temperature (°C)", "coolwarm")
else:
    axes[2, 0].axis("off")

if perf_col:
    plot_bar(axes[2, 1], perf_col, "Model", "Performance Score", "flare")
elif acc_col:
    plot_bar(axes[2, 1], acc_col, "Model", "Accuracy (%)", "flare")
else:
    axes[2, 1].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# === RADAR CHART ===
metrics = [c for c in [lat_col, tok_col, cpu_col, ram_col, temp_col, acc_col, perf_col] if c]

if len(metrics) >= 3:
    print("\n Generating radar chart...")
    df_norm = df.copy()
    for col in metrics:
        if "Latency" in col:
            df_norm[col] = 1 / (df[col] + 1e-9)
        else:
            df_norm[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    labels = metrics
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for _, row in df_norm.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, label=row["Model"], linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    plt.xticks(angles[:-1], labels, fontsize=10)
    plt.yticks([], [])
    plt.title("Model Comparison Radar Chart", size=16, weight="bold", pad=20)
    plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.show()

# === TOP 3 MODELS SUMMARY ===
top_col = perf_col or acc_col or lat_col
top_df = df.head(3)[["Model", top_col, lat_col, tok_col, cpu_col, ram_col, temp_col, acc_col, perf_col]].fillna("N/A")

print("\n Top 3 Models Summary:\n")
for idx, row in top_df.iterrows():
    print(f" {row['Model']}")
    print(f"   Performance Score: {row.get(perf_col, 'N/A')}")
    print(f"   Accuracy: {row.get(acc_col, 'N/A')}")
    print(f"   Latency: {row.get(lat_col, 'N/A')} s")
    print(f"   Tokens/s: {row.get(tok_col, 'N/A')}")
    print(f"   CPU: {row.get(cpu_col, 'N/A')} %, RAM: {row.get(ram_col, 'N/A')} %, Temp: {row.get(temp_col, 'N/A')} °C\n")

# === SAVE TO MARKDOWN ===
md = ["#  Top 3 Model Summary\n"]
md.append("| Rank | Model | Performance Score | Accuracy (%) | Latency (s) | Tokens/s | CPU (%) | RAM (%) | Temp (°C) |")
md.append("|------|--------|------------------|---------------|--------------|-----------|----------|----------|-----------|")

for i, (_, row) in enumerate(top_df.iterrows(), 1):
    md.append(f"| {i} | {row['Model']} | {row.get(perf_col, 'N/A')} | {row.get(acc_col, 'N/A')} | "
              f"{row.get(lat_col, 'N/A')} | {row.get(tok_col, 'N/A')} | {row.get(cpu_col, 'N/A')} | "
              f"{row.get(ram_col, 'N/A')} | {row.get(temp_col, 'N/A')} |")

with open("top_models_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md))

