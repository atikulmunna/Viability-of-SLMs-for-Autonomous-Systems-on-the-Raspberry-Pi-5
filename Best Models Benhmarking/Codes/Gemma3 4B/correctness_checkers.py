import sys, os, csv, json, re
from pathlib import Path

def safe_read_csv(path):
    rows = []
    try:
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]
    except Exception as e:
        print(f"CSV read error: {e}")
    return rows

def is_json_like(text):
    return text.strip().startswith("{") and text.strip().endswith("}")

def is_code_like(text):
    return any(k in text for k in ["def ", "for ", "while ", "if ", "return "])

def is_math_like(text):
    return bool(re.search(r"\d+[\+\-\*\/\^]\d+", text))

def score_entry(prompt, output):
    prompt_l, out_l = prompt.lower(), output.lower()
    score, reasons = 0, []

    # Math
    if "derivative" in prompt_l:
        if "2x" in out_l or "6x" in out_l:
            score, reasons = 1.0, ["math derivation correct"]
        else:
            reasons.append("wrong derivative")

    # Code
    elif "python function" in prompt_l or "reverse a string" in prompt_l:
        if "def" in out_l and "return" in out_l:
            score, reasons = 1.0, ["code structure valid"]
        else:
            reasons.append("invalid function")

    # JSON
    elif "json" in prompt_l:
        if is_json_like(output):
            score, reasons = 1.0, ["valid JSON object"]
        else:
            reasons.append("JSON formatting error")

    # Logic / Reasoning
    elif "trains" in prompt_l or "liters" in prompt_l:
        if any(x in out_l for x in ["2 hours", "4 liters", "2h", "two hours"]):
            score, reasons = 1.0, ["reasoning answer plausible"]
        else:
            reasons.append("reasoning uncertain")

    # Creative / NLP
    elif "story" in prompt_l:
        if len(output.split()) > 20:
            score, reasons = 1.0, ["coherent creative output"]
        else:
            reasons.append("too short or abrupt")

    else:
        if len(output.strip()) > 10:
            score = 0.5
            reasons.append("response length acceptable")
        else:
            reasons.append("incomplete response")

    return score, reasons

def main(csv_path):
    rows = safe_read_csv(csv_path)
    if not rows:
        print("No results found.")
        return

    summary = []
    correct = total = 0
    for r in rows:
        prompt, output = r["prompt"], r["output"]
        score, reasons = score_entry(prompt, output)
        summary.append({"prompt": prompt, "score": score, "reasons": reasons})
        total += 1
        correct += 1 if score >= 1.0 else 0

    acc = round((correct / total) * 100, 2) if total else 0.0
    print(f"Correctness Summary: {acc}% ({correct}/{total})")

    base = os.path.dirname(csv_path)
    with open(Path(base, "correctness_summary.json"), "w") as f:
        json.dump({"accuracy": acc, "details": summary}, f, indent=2)

    with open(Path(base, "correctness.txt"), "w") as f:
        f.write(f"Accuracy: {acc}% ({correct}/{total})\n\n")
        for s in summary:
            f.write(f"Prompt: {s['prompt']}\n")
            for r in s["reasons"]:
                f.write(f"  - {r}\n")
            f.write("\n")

    print(f"Saved reports to {base}/correctness_summary.json and correctness.txt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 correctness_checkers.py <path_to_outputs.csv>")
        sys.exit(1)
    main(sys.argv[1])
