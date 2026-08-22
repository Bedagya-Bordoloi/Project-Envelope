"""
scripts/compare_models.py

Pulls a representative sample of real (t_in, t_out, forecast, carbon)
inputs from an existing control_log.jsonl, replays each one through
agents.strategist.Strategist twice -- once per model -- and records
latency and whether the proposal clears SentinelGate.check(). Needs
GROQ_API_KEY set (loaded via .env, same as main.py) and a control log
that already has enough rows to sample from (see calibrate_ccs_sweep.py
or a normal `python main.py` run).

NOTE: the source log only stores t_in/t_out/setpoint/reason, not the
forecast window or humidity that were live at that step -- those aren't
persisted per-row. This script reconstructs a same-shape forecast via
EnergyPlusBridge._parse_epw_drybulb + the row's step (so the replay
sees the real weather trajectory the run actually had), and uses a
fixed humidity (50%) since it isn't logged. That's a real limitation,
not hidden: if you want exact replay fidelity, log `humidity` and
`forecast` per-row in main.py's log_entry first, then point this
script at a re-run.

Usage:
    python scripts/compare_models.py --log logs/ai/control_log.jsonl --n 15

Produces:
    logs/model_comparison.csv
    Printed summary table: model | avg latency (s) | CCS pass rate | n
"""

import argparse
import json
import os
import random
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.strategist import Strategist
from core.sentinel_gate import SentinelGate
from core.energyplus_bridge import EnergyPlusBridge
from core.seasonality import get_baseline_setpoint

# FIX (post-Round-3): Groq deprecated llama-3.1-8b-instant and
# llama-3.3-70b-versatile (shut down 2026-08-16); calls now 404 with
# "model_not_found". Swapped to Groq's recommended small/large pair --
# same 8B-vs-70B-style comparison, current model IDs. Re-verify against
# console.groq.com/docs/models if this is stale by the time you read it.
DEFAULT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def load_policy(path="config/building_policy.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_sample_rows(log_path, n, seed=7):
    rows = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("source") in ("AI", "AI (Corrected)", "AI (Stabilized)", "FAILSAFE"):
                rows.append(row)
    if not rows:
        raise SystemExit(f"No usable AI/FAILSAFE rows found in {log_path}. "
                          f"Run a simulation first.")
    random.Random(seed).shuffle(rows)
    # Mix of normal, near-boundary (small ccs margin), and rejected cases.
    rejected = [r for r in rows if r.get("source") == "FAILSAFE"]
    near_boundary = [r for r in rows if r.get("ccs") is not None and abs(r["ccs"] - 0.65) < 0.05]
    normal = [r for r in rows if r not in rejected and r not in near_boundary]

    sample = []
    for bucket in (rejected, near_boundary, normal):
        take = min(len(bucket), max(1, n // 3))
        sample.extend(bucket[:take])
    sample = sample[:n] if len(sample) >= n else (sample + normal)[:n]
    return sample


def replay(rows, models, policy, epw_path):
    forecast_temps = EnergyPlusBridge._parse_epw_drybulb(epw_path)
    results = []

    for model in models:
        print(f"\n=== {model} ===")
        # FIX: policy now threaded through so this replay reads the same
        # real season thresholds / baseline value the live loop uses
        # (see agents/strategist.py, core/seasonality.py) instead of the
        # old hardcoded 14/22/"22.0C".
        strategist = Strategist(model=model, policy=policy)
        gate = SentinelGate(policy)  # fresh gate per model so hysteresis state doesn't leak across models
        # FIX: `last` is the previous *setpoint* -- SentinelGate.check()
        # uses it for both the hysteresis dwell/delta check and the
        # rate_penalty term in the CCS score. This used to be passed
        # `t_in` (current indoor temp) by mistake, silently corrupting
        # rate_penalty/HOLD logic for every replayed row. Track a running
        # last_setpoint across the loop instead, mirroring
        # ProjectEnvelope.decide()'s self.last_setpoint in main.py.
        last_setpoint = get_baseline_setpoint(policy)

        for row in rows:
            t_in, t_out = row["t_in"], row["t_out"]
            step = row.get("step", 0)
            hour_index = min(step // 6, len(forecast_temps) - 1)
            forecast = forecast_temps[hour_index: hour_index + 3]
            humidity = 50.0  # not persisted per-row in control_log.jsonl -- see module docstring

            start = time.perf_counter()
            try:
                proposal = strategist.decide(t_in, t_out, forecast, carbon="Medium")
                elapsed = time.perf_counter() - start
                outcome, ccs, reason, _ = gate.check(
                    proposal["setpoint"], last_setpoint, proposal["confidence"], t_in, humidity, t_out
                )
                passed = outcome == "APPROVED"
                if passed:
                    last_setpoint = proposal["setpoint"]
            except Exception as e:
                elapsed = time.perf_counter() - start
                outcome, ccs, passed = "ERROR", None, False
                print(f"  step {step}: ERROR ({e})")

            results.append({
                "model": model, "step": step, "t_in": t_in, "t_out": t_out,
                "latency_s": round(elapsed, 3), "gate_outcome": outcome,
                "ccs": round(ccs, 3) if ccs is not None else None, "passed": passed,
            })

    return pd.DataFrame(results)


def summarize(df):
    summary = df.groupby("model").agg(
        avg_latency_s=("latency_s", "mean"),
        ccs_pass_rate=("passed", "mean"),
        n=("passed", "count"),
    ).reset_index()
    summary["avg_latency_s"] = summary["avg_latency_s"].round(3)
    summary["ccs_pass_rate"] = summary["ccs_pass_rate"].round(3)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/ai/control_log.jsonl")
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--out", default="logs/model_comparison.csv")
    args = parser.parse_args()

    policy = load_policy()
    epw_path = policy["paths"]["epw"]

    rows = load_sample_rows(args.log, args.n)
    print(f"Replaying {len(rows)} decision points from {args.log} through {args.models}")

    df = replay(rows, args.models, policy, epw_path)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote per-decision detail to {args.out}")

    summary = summarize(df)
    print("\n--- MODEL JUSTIFICATION TABLE ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
