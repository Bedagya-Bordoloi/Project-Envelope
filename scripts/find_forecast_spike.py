"""
scripts/find_forecast_spike.py

Step 1: scan the loaded .epw for a real outdoor-temp spike (a jump of
several degrees C within a few hours), matching the threshold that
agents/strategist.py's Strategist._check_lookahead() now actually uses
(LOOKAHEAD_SPIKE_DELTA_C). Reports the candidate hour(s) and the
corresponding sim step range (step ~= hour_index * 6, per
EnergyPlusBridge's stepping), so you know which window of
logs/ai/control_log.jsonl to pull for the "before/after" slide.

Step 2 (--plot): given a step range (from step 1, or your own), plots
t_in / t_out / setpoint for just that window from a control log, and
marks whether lookahead_triggered=True fired ahead of the spike --
this is the chart the blueprint asks for as the standalone "before/after"
proof that the Strategist moved setpoint ahead of the outdoor jump,
not after it.

Usage:
    python scripts/find_forecast_spike.py
    python scripts/find_forecast_spike.py --plot --step-start 1200 --step-end 1400
"""

import argparse
import json
import os

import yaml

from core.energyplus_bridge import EnergyPlusBridge
from agents.strategist import Strategist

SPIKE_DELTA_C = Strategist.LOOKAHEAD_SPIKE_DELTA_C


def load_policy(path="config/building_policy.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def find_spikes(epw_path, hours_window=6, delta_c=SPIKE_DELTA_C):
    temps = EnergyPlusBridge._parse_epw_drybulb(epw_path)
    candidates = []
    for i in range(len(temps) - hours_window):
        window = temps[i:i + hours_window]
        rise = max(window) - temps[i]
        fall = temps[i] - min(window)
        if rise >= delta_c or fall >= delta_c:
            direction = "rising" if rise >= fall else "falling"
            magnitude = rise if direction == "rising" else fall
            step = i * 6  # EnergyPlusBridge.get_forward_weather(): hour_index = step_counter // 6
            candidates.append({
                "hour": i, "step": step, "direction": direction,
                "magnitude_c": round(magnitude, 1), "from_c": round(temps[i], 1),
                "window": [round(t, 1) for t in window],
            })
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", default="logs/ai/control_log.jsonl")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--step-start", type=int)
    parser.add_argument("--step-end", type=int)
    parser.add_argument("--out", default="logs/lookahead_demo.png")
    args = parser.parse_args()

    policy = load_policy()
    epw_path = policy["paths"]["epw"]

    if not args.plot:
        candidates = find_spikes(epw_path)
        if not candidates:
            print(f"No spikes >= {SPIKE_DELTA_C}C found in {epw_path}. Try a smaller "
                  f"--hours-window or lower delta by editing SPIKE_DELTA_C.")
            return
        print(f"Found {len(candidates)} spike candidate(s) >= {SPIKE_DELTA_C}C in {epw_path}:\n")
        for c in candidates[:20]:
            print(f"  hour {c['hour']:>5} (step ~{c['step']:>6}): {c['direction']:>7} "
                  f"{c['magnitude_c']}C from {c['from_c']}C  window={c['window']}")
        print(f"\nPick one, then run a full simulation, then:\n"
              f"  python scripts/find_forecast_spike.py --plot "
              f"--step-start <step-300> --step-end <step+300>")
        return

    if args.step_start is None or args.step_end is None:
        raise SystemExit("--plot requires --step-start and --step-end (see the non-plot run above).")

    if not os.path.exists(args.log):
        raise SystemExit(f"{args.log} not found -- run `python main.py` first.")

    rows = []
    with open(args.log, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.step_start <= row.get("step", -1) <= args.step_end:
                rows.append(row)

    if not rows:
        raise SystemExit(f"No log rows in step range [{args.step_start}, {args.step_end}] in {args.log}.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [r["step"] for r in rows]
    t_in = [r["t_in"] for r in rows]
    t_out = [r["t_out"] for r in rows]
    setpoint = [r.get("setpoint") for r in rows]
    lookahead_steps = [r["step"] for r in rows if r.get("lookahead_triggered")]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, t_out, label="Outdoor temp", color="deepskyblue", linestyle=":")
    ax.plot(steps, t_in, label="Indoor temp", color="orange")
    ax.plot(steps, setpoint, label="AI setpoint", color="limegreen", linestyle="--")
    for s in lookahead_steps:
        ax.axvline(s, color="red", alpha=0.3, linewidth=1)
    if lookahead_steps:
        ax.axvline(lookahead_steps[0], color="red", alpha=0.3, linewidth=1, label="Lookahead triggered")

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Forward look-ahead: setpoint moving ahead of the outdoor spike")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")
    if not lookahead_steps:
        print("NOTE: no rows in this window have lookahead_triggered=True -- either the "
              "spike wasn't in this step range, or cadence_steps skipped over the trigger "
              "tick. Widen --step-start/--step-end or re-check the candidate from the "
              "non-plot run.")


if __name__ == "__main__":
    main()
