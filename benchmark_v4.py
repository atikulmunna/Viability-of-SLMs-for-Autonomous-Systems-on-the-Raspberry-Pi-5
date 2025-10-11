import requests
import json
import time
import csv
import os
import psutil
import subprocess
import re
from datetime import timedelta, datetime

MODELS_TO_TEST = [
    "qwen3:8b",
    "smollm2:1.7b",
    "mathstral:latest",
    "deepseek-coder:1.3b",
    "mistral:7b",
    "orca-mini:3b",
    "stablelm2:1.6b",
    "tinyllama:latest",
    "phi3:mini",
    "gemma2:2b",
    "qwen2:0.5b"
]

PROMPTS_TO_TEST = [
    # --- Agent-style Prompts (20) ---
    {"name": "turn_on_light", "command": "Turn on the living room light."},
    {"name": "turn_off_fan", "command": "Switch off the bedroom fan."},
    {"name": "set_alarm", "command": "Set an alarm for 6:30 AM tomorrow."},
    {"name": "open_browser", "command": "Open the web browser and go to YouTube."},
    {"name": "weather_check", "command": "Check today's weather forecast for London."},
    {"name": "play_music", "command": "Play some relaxing jazz music."},
    {"name": "system_status", "command": "Report current system CPU and memory usage."},
    {"name": "start_timer", "command": "Start a 5-minute timer."},
    {"name": "send_email", "command": "Send an email to Alex saying 'Meeting rescheduled to 4 PM'."},
    {"name": "translate_phrase", "command": "Translate 'Good morning, how are you?' into French."},
    {"name": "get_time", "command": "What time is it right now?"},
    {"name": "control_volume", "command": "Increase the system volume by 20 percent."},
    {"name": "find_file", "command": "Find a file named 'budget_report.pdf'."},
    {"name": "check_battery", "command": "Check the current battery percentage."},
    {"name": "launch_app", "command": "Open the calculator app."},
    {"name": "bluetooth_on", "command": "Turn on Bluetooth."},
    {"name": "wifi_status", "command": "Check if Wi-Fi is connected."},
    {"name": "record_voice", "command": "Start recording a voice note for 30 seconds."},
    {"name": "calendar_event", "command": "Add 'Project Meeting' to the calendar for Monday at 10 AM."},
    {"name": "smart_home_temp", "command": "Set the thermostat in the living room to 22 degrees Celsius."},

    # --- Analytical & Creative Prompts (20) ---
    {"name": "math_simple", "command": "What is 23 multiplied by 47?"},
    {"name": "math_reasoning", "command": "If a train travels 120 km in 2 hours, what is its average speed?"},
    {"name": "logical_reasoning", "command": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?"},
    {"name": "story_creative", "command": "Write a short, two-sentence story about a robot discovering music."},
    {"name": "summarization", "command": "Summarize this: 'Artificial Intelligence enables machines to learn from data and make decisions with minimal human intervention.'"},
    {"name": "coding_task", "command": "Write a Python function that returns the factorial of a number."},
    {"name": "analytical_thinking", "command": "Why might a smaller AI model respond faster but less accurately than a larger one?"},
    {"name": "translation_task", "command": "Translate 'Knowledge is power' into Spanish."},
    {"name": "definition_task", "command": "Define the term 'edge computing' in one sentence."},
    {"name": "creative_analogy", "command": "Create an analogy comparing a neural network to the human brain."},
    {"name": "compare_models", "command": "Compare large and small language models in terms of speed and accuracy."},
    {"name": "data_interpretation", "command": "If CPU usage increases but latency stays the same, what does that imply about model efficiency?"},
    {"name": "critical_thinking", "command": "What are the ethical implications of AI systems used in hiring decisions?"},
    {"name": "cause_effect", "command": "If temperature rises, what happens to the density of air?"},
    {"name": "math_advanced", "command": "Solve for x: 3x + 7 = 19."},
    {"name": "language_reasoning", "command": "What is the opposite of 'complicated' and use it in a sentence."},
    {"name": "creative_task", "command": "Invent a new use for a paperclip that doesn't involve holding papers."},
    {"name": "science_question", "command": "Explain how photosynthesis works in one paragraph."},
    {"name": "moral_reasoning", "command": "Is it always wrong to tell a lie? Explain briefly."},
    {"name": "summary_task", "command": "Summarize why energy efficiency is important for embedded AI systems."}
]

SYSTEM_PROMPT_TEMPLATE = (
    'User command: "{command}". Interpret this command and respond ONLY with a JSON '
    'object with "action" and "target" keys. For math, use action "calculate" and target as the expression.'
)

CSV_FILE = "benchmark_results.csv"
LIVE_LOG_FILE = "benchmark_live.log"


def timestamp_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_system_usage():
    """Return (cpu_percent, ram_percent)."""
    cpu_percent = psutil.cpu_percent(interval=1)
    ram_percent = psutil.virtual_memory().percent
    return cpu_percent, ram_percent

def get_temp_c():
    """Return Raspberry Pi CPU temp in Celsius (float), or None."""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.replace("temp=", "").replace("'C", "").strip())
    except Exception:
        return None

def safe_json_parse(text):
    """
    Try to extract JSON-like substring and parse it.
    Attempts mild fixes:
      - extract {...} substring
      - convert single quotes to double quotes
      - remove trailing commas inside objects/lists (simple)
      - remove newlines
    Returns parsed object on success, otherwise None.
    """
    if not text or not isinstance(text, str):
        return None

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return None
    json_str = match.group(0)

    json_str = json_str.replace("\n", " ")
    json_str = json_str.replace("\r", " ")
    json_str = json_str.strip()

    json_str = re.sub(r"(?<!\")'(?P<m>[^']*?)'(?!\")", r'"\g<m>"', json_str)

    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        try:
            repaired = json_str.encode('utf-8', 'replace').decode('unicode_escape')
            return json.loads(repaired)
        except Exception:
            return None

def truncate_text(s, n=250):
    """Return the first n characters of s, safe for logs."""
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= n:
        return s
    return s[:n-1] + "…"

def write_detailed_log(logfile_handle, model, prompt_name, command, raw_response,
                       latency, tokens_per_sec, cpu, ram, temp_c, accuracy, error_msg=None):
    """Write a multi-line detailed entry to the live log file with timestamp."""
    now = timestamp_now()
    header = f"\n=== {now} | MODEL: {model} | PROMPT: {prompt_name} ===\n"
    logfile_handle.write(header)
    logfile_handle.write(f"Command: {command}\n")
    logfile_handle.write(f"Latency_s: {latency:.4f}\n")
    logfile_handle.write(f"Tokens_per_s: {tokens_per_sec:.4f}\n")
    logfile_handle.write(f"CPU_%: {cpu}\n")
    logfile_handle.write(f"RAM_%: {ram}\n")
    logfile_handle.write(f"Temp_C: {temp_c}\n")
    logfile_handle.write(f"Accuracy: {accuracy}\n")
    if error_msg:
        logfile_handle.write(f"Error: {error_msg}\n")
    logfile_handle.write("Raw response (truncated):\n")
    logfile_handle.write(truncate_text(raw_response, 1000) + "\n")
    logfile_handle.write("=" * 60 + "\n\n")
    logfile_handle.flush()

# --- Main Benchmark Function ---

def run_benchmark():
    file_exists = os.path.isfile(CSV_FILE)
    live_log = open(LIVE_LOG_FILE, "a", encoding="utf-8")

    with open(CSV_FILE, 'a', newline='', encoding="utf-8") as csvfile:
        fieldnames = [
            'timestamp', 'model', 'prompt_name', 'end_to_end_latency_s',
            'eval_tokens', 'eval_duration_s', 'tokens_per_second',
            'cpu_percent', 'ram_percent', 'temperature_C', 'accuracy'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
            csvfile.flush()

        completed = set()
        if file_exists:
            with open(CSV_FILE, 'r', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    completed.add((row['model'], row['prompt_name']))

        total_tests = len(MODELS_TO_TEST) * len(PROMPTS_TO_TEST)
        done_tests = len(completed)
        start_time_all = time.time()

        try:
            for model in MODELS_TO_TEST:
                print(f"\n=== Testing Model: {model} ===")
                for prompt_info in PROMPTS_TO_TEST:
                    prompt_name = prompt_info["name"]

                    if (model, prompt_name) in completed:
                        done_tests += 0  
                        continue

                    command = prompt_info["command"]
                    full_prompt = SYSTEM_PROMPT_TEMPLATE.format(command=command)

                    cpu, ram = get_system_usage()
                    temp_c = get_temp_c()

                    data = {
                        "model": model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {"seed": 42}
                    }

                    start_time = time.time()
                    api_response = {}
                    raw_response_text = ""
                    error_msg = None
                    try:
                        response = requests.post(
                            "http://localhost:11434/api/generate",
                            json=data,
                            timeout=300
                        )
                        response.raise_for_status()
                        api_response = response.json()

                        if isinstance(api_response, dict):
                            raw_response_text = api_response.get('response') or api_response.get('output') or api_response.get('text') or ""
                            # Some servers wrap content deeper; handle lists
                            if not raw_response_text and 'choices' in api_response:
                                try:

                                    choices = api_response.get('choices')
                                    if isinstance(choices, list) and len(choices) > 0:
                                        choice = choices[0]
                                        raw_response_text = (choice.get('message') or {}).get('content') if isinstance(choice.get('message'), dict) else choice.get('content', '')
                                except Exception:
                                    pass
                        else:
                            raw_response_text = str(api_response)
                    except requests.exceptions.RequestException as e:
                        error_msg = str(e)
                    except ValueError as e:
                        error_msg = f"Invalid JSON from server: {e}"

                    end_time = time.time()
                    latency = end_time - start_time

                    eval_count = api_response.get('eval_count', 0) if isinstance(api_response, dict) else 0
                    eval_duration_ns = api_response.get('eval_duration', 1) if isinstance(api_response, dict) else 1
                    eval_duration_s = eval_duration_ns / 1_000_000_000 if eval_duration_ns else 0
                    tokens_per_sec = (eval_count / eval_duration_s) if eval_duration_s > 0 else 0.0

                    parsed = safe_json_parse(raw_response_text)
                    accuracy = "Pass" if parsed else "Fail (Invalid JSON)"

                    write_detailed_log(
                        live_log, model, prompt_name, full_prompt, raw_response_text,
                        latency, tokens_per_sec, cpu, ram, temp_c, accuracy, error_msg
                    )

                    writer.writerow({
                        'timestamp': timestamp_now(),
                        'model': model,
                        'prompt_name': prompt_name,
                        'end_to_end_latency_s': round(latency, 4),
                        'eval_tokens': eval_count,
                        'eval_duration_s': round(eval_duration_s, 4),
                        'tokens_per_second': round(tokens_per_sec, 2),
                        'cpu_percent': cpu,
                        'ram_percent': ram,
                        'temperature_C': round(temp_c, 2) if temp_c is not None else "N/A",
                        'accuracy': accuracy
                    })
                    csvfile.flush()

                    done_tests += 1
                    elapsed = time.time() - start_time_all
                    avg_time = elapsed / done_tests if done_tests > 0 else 0
                    remaining = total_tests - done_tests
                    eta = timedelta(seconds=int(avg_time * remaining))
                    progress_pct = (done_tests / total_tests) * 100

                    print(f"[{progress_pct:.1f}%] {model} | {prompt_name} | Latency: {latency:.2f}s | "
                          f"Temp: {temp_c}°C | Tokens/s: {tokens_per_sec:.2f} | Acc: {accuracy} | ETA: {eta}")

        finally:
            live_log.close()

    print(f"\n✅ Benchmark complete! Results appended to: {CSV_FILE}")
    print(f"✅ Live detailed log: {LIVE_LOG_FILE}")

if __name__ == "__main__":
    run_benchmark()


