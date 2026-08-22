import pandas as pd
import json
import os
import yaml

LOG_PATH = "logs/ai/control_log.jsonl"
POLICY_PATH = "config/building_policy.yaml"


def _current_yaml_threshold():
    """2.2 fix: this used to print a hardcoded '0.70' regardless of what
    config/building_policy.yaml actually said (it says 0.65) -- always
    read the live value instead of a stale literal."""
    try:
        with open(POLICY_PATH, "r") as f:
            return yaml.safe_load(f)["gate"]["ccs_threshold"]
    except Exception:
        return None


def justify_threshold():
    if not os.path.exists(LOG_PATH):
        print("Run a simulation for at least 1000 steps first to generate data.")
        return

    data = []
    with open(LOG_PATH, "r") as f:
        for line in f:
            data.append(json.loads(line))
    
    df = pd.DataFrame(data)
    # Filter for AI decisions only (ignore Failsafe/Holding)
    ai_only = df[df['source'].isin(['AI', 'AI (Corrected)'])]
    
    if ai_only.empty:
        print("No AI decisions found in logs to calibrate against.")
        return

    mean_ccs = ai_only['ccs'].mean()
    std_ccs = ai_only['ccs'].std()
    
    # We want a threshold that accepts "High Confidence" moves 
    # but filters out the bottom 15% (outliers/hallucinations)
    suggested_threshold = mean_ccs - (1.0 * std_ccs)

    current_threshold = _current_yaml_threshold()

    print("--- GATE CALIBRATION REPORT ---")
    print(f"Sample Size: {len(ai_only)} decisions")
    print(f"Mean CCS: {mean_ccs:.3f}")
    print(f"Std Dev: {std_ccs:.3f}")
    print(f"Recommended Threshold (Mean - 1 Sigma): {suggested_threshold:.2f}")
    print(f"Current YAML Threshold: {current_threshold if current_threshold is not None else 'could not read ' + POLICY_PATH}")
    print("\nNOTE: this only describes the distribution of CCS scores actually")
    print("produced in ONE run at the CURRENT threshold -- it does not tell you")
    print("how approval rate, savings, or violations would change at a")
    print("DIFFERENT threshold. For that (Blueprint 1.2's actual calibration")
    print("sweep), use scripts/calibrate_ccs_sweep.py instead, and prefer its")
    print("output for ARCHITECTURE.md's threshold justification.")

if __name__ == "__main__":
    justify_threshold()