import requests
import json
import time
import csv
import os
import psutil

MODELS_TO_TEST = [
    "qwen2:0.5b",
    "gemma2:2b",
    "phi3:mini"
]

PROMPTS_TO_TEST = [
    {"name": "math_basic_addition", "command": "2 + 2"},
    {"name": "math_complex_equation", "command": "(12 * 7) - (5 ** 2)"},
    {"name": "math_word_problem", "command": "If a train travels 60 km in 1.5 hours, what is its average speed in km/h?"},

    {"name": "instruction_rephrase", "command": "Rephrase 'The quick brown fox jumps over the lazy dog' in simpler English."},
    {"name": "instruction_summary", "command": "Summarize the sentence 'Artificial intelligence enables machines to learn from data' in five words."},

    {"name": "knowledge_capital", "command": "What is the capital of Japan?"},
    {"name": "knowledge_invention", "command": "Who invented the telephone?"},

    {"name": "translation_simple", "command": "Translate 'Good morning' into French."},
    {"name": "translation_reverse", "command": "Translate 'Buenos días' into English."},

    {"name": "logic_condition_true", "command": "If it is raining, take an umbrella. It is raining."},
    {"name": "logic_condition_false", "command": "If the light is red, stop. The light is green."}
]

SYSTEM_PROMPT_TEMPLATE = (
    'User command: "{command}". Interpret this command and respond ONLY with a JSON '
    'object with "action" and "target" keys. For math, use action "calculate" and target as the expression.'
)

CSV_FILE = "benchmark_results.csv"

def get_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_percent = psutil.virtual_memory().percent
    return cpu_percent, ram_percent

def run_benchmark():
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'model', 'prompt_name', 'end_to_end_latency_s',
            'eval_tokens', 'eval_duration_s', 'tokens_per_second',
            'cpu_percent', 'ram_percent', 'accuracy'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for model in MODELS_TO_TEST:
            print(f"\n--- Testing Model: {model} ---")
            for prompt_info in PROMPTS_TO_TEST:
                prompt_name = prompt_info["name"]
                command = prompt_info["command"]
                full_prompt = SYSTEM_PROMPT_TEMPLATE.format(command=command)

                print(f"  Running prompt: '{prompt_name}'...")

                cpu, ram = get_system_usage()

                data = {
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {"seed": 42}
                }

                start_time = time.time()
                try:
                    response = requests.post(
                        "http://localhost:11434/api/generate", json=data, timeout=300
                    )
                    response.raise_for_status()
                    api_response = response.json()
                except requests.exceptions.RequestException as e:
                    print(f"    ERROR: API request failed for model {model}. {e}")
                    continue
                end_time = time.time()

                latency = end_time - start_time
                eval_count = api_response.get('eval_count', 0)
                eval_duration_ns = api_response.get('eval_duration', 1)
                eval_duration_s = eval_duration_ns / 1_000_000_000
                tokens_per_sec = eval_count / eval_duration_s if eval_duration_s > 0 else 0

                accuracy = "Fail"
                try:
                    json.loads(api_response.get('response', ''))
                    accuracy = "Pass"
                except json.JSONDecodeError:
                    accuracy = "Fail (Invalid JSON)"

                writer.writerow({
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'model': model,
                    'prompt_name': prompt_name,
                    'end_to_end_latency_s': round(latency, 4),
                    'eval_tokens': eval_count,
                    'eval_duration_s': round(eval_duration_s, 4),
                    'tokens_per_second': round(tokens_per_sec, 2),
                    'cpu_percent': cpu,
                    'ram_percent': ram,
                    'accuracy': accuracy
                })
                print(f"    Done. Latency: {latency:.2f}s, Tokens/sec: {tokens_per_sec:.2f}, Accuracy: {accuracy}")

    print(f"\nBenchmark complete. Results saved to {CSV_FILE}")

if __name__ == "__main__":
    run_benchmark()
