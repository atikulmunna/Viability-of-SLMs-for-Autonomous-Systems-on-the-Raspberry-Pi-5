import os
import json
import time
import argparse
import pandas as pd
from utils.benchmark_utils import run_model_test

def run_family_benchmark(family_name, prompts_file):
    # Load prompts
    with open(prompts_file, "r") as f:
        prompts = json.load(f)

    families = {
        "qwen": ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "qwen3:8b"],
        "deepseek": ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:8b"],
        "gemma": ["gemma3:270m", "gemma3:1b", "gemma3:4b"],
        "llama": ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"],
        "smollm": ["smollm2:135m", "smollm2:360m", "smollm2:1.7b"]
    }

    if family_name not in families:
        print(f" Unknown family '{family_name}'. Available families: {', '.join(families.keys())}")
        return

    models = families[family_name]
    print(f"\n--- Running {family_name.upper()} Family Benchmark ---\n")

    results = []

    for model in models:
        print(f" Testing model: {model}")
        for tier_name, tier_prompts in prompts.items():
            for prompt in tier_prompts:
                print(f"Running {tier_name} prompt: {prompt[:60]}...")
                result = run_model_test(model, prompt, tier_name)
                result["family"] = family_name
                results.append(result)
                time.sleep(1)  # small delay between runs

    # Save results
    df = pd.DataFrame(results)
    out_path = f"benchmark_results_{family_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n Results saved to {out_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark models by family")
    parser.add_argument("--family", type=str, required=True, help="Model family name (e.g., qwen, gemma, llama)")
    parser.add_argument("--prompts", type=str, default="prompts_tiered.json", help="Path to prompts JSON file")
    args = parser.parse_args()

    run_family_benchmark(args.family, args.prompts)
