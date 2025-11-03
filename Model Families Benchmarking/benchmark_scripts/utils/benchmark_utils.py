import time
import psutil
import random

def simulate_temperature():
    base_temp = 55 + random.uniform(-2, 2)
    fluctuation = random.uniform(0, 8)
    return base_temp + fluctuation

def run_model_test(model_name, prompt, tier_name):

    # Start metrics
    start_time = time.time()
    cpu_before = psutil.cpu_percent(interval=None)
    ram_before = psutil.virtual_memory().percent
    temp_before = simulate_temperature()

    # Simulate processing
    simulated_latency = random.uniform(1.0, 60.0)
    time.sleep(simulated_latency / 50.0)  # speed up testing
    simulated_tps = random.uniform(1.0, 30.0)

    # End metrics
    cpu_after = psutil.cpu_percent(interval=None)
    ram_after = psutil.virtual_memory().percent
    temp_after = simulate_temperature()
    end_time = time.time()

    # Compute averages
    avg_cpu = (cpu_before + cpu_after) / 2
    avg_ram = (ram_before + ram_after) / 2
    avg_temp = (temp_before + temp_after) / 2
    latency = end_time - start_time

    # Simulate accuracy and performance score
    accuracy = random.uniform(60.0, 95.0)
    perf_score = (accuracy / (latency + 0.01)) * (simulated_tps / 10)

    return {
        "Model": model_name,
        "Prompt": prompt[:80],
        "Tier": tier_name,
        "Avg_Latency_s": latency,
        "Avg_Tokens_per_s": simulated_tps,
        "Avg_CPU_%": avg_cpu,
        "Avg_RAM_%": avg_ram,
        "Avg_Temp_C": avg_temp,
        "Accuracy_%": accuracy,
        "Performance_Score": perf_score
    }
