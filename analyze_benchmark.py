import pandas as pd
import os

CSV_FILE = "benchmark_results.csv"
SUMMARY_FILE = "benchmark_summary.csv"

def analyze_benchmark():
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found. Please run benchmark.py first.")
        return

    print(f"📂 Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    numeric_fields = [
        'end_to_end_latency_s',
        'eval_tokens',
        'eval_duration_s',
        'tokens_per_second',
        'cpu_percent',
        'ram_percent'
    ]
    df[numeric_fields] = df[numeric_fields].apply(pd.to_numeric, errors='coerce')

    df['accuracy_numeric'] = df['accuracy'].apply(lambda x: 1 if 'Pass' in str(x) else 0)

    summary = df.groupby('model').agg({
        'end_to_end_latency_s': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std'],
        'cpu_percent': ['mean', 'std'],
        'ram_percent': ['mean', 'std'],
        'accuracy_numeric': 'mean'
    }).reset_index()

    summary.columns = [
        'Model',
        'Avg_Latency_s', 'Std_Latency_s',
        'Avg_Tokens_per_s', 'Std_Tokens_per_s',
        'Avg_CPU_%', 'Std_CPU_%',
        'Avg_RAM_%', 'Std_RAM_%',
        'Accuracy_%'
    ]

    summary['Accuracy_%'] = (summary['Accuracy_%'] * 100).round(2)

    summary = summary.round(3)

    print("\n📊 Benchmark Summary (averaged by model):")
    print(summary.to_string(index=False))

    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"\n✅ Summary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    analyze_benchmark()
