import json
import subprocess
import time

with open("model_families.json", "r") as f:
    model_families = json.load(f)

print("\n===  Starting Model Pull Process ===\n")

for family, models in model_families.items():
    print(f"\n Pulling models for {family} family...\n")
    for model in models:
        print(f" Pulling {model} ...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f" Successfully pulled {model}\n")
            else:
                print(f" Failed to pull {model}. Error:\n{result.stderr}\n")
        except Exception as e:
            print(f" Exception while pulling {model}: {e}")
        time.sleep(3)  

print("\n All model families processed!\n")
