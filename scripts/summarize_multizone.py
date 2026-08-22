"""
scripts/summarize_multizone.py

Blueprint 1.1 -- Verify step.

"Verify: Dashboard/log shows two independent zone temperature traces and
two independent gate decisions per cadence tick."

`python main.py --multizone` writes one JSONL log per zone under
logs/multizone/ (zone_one.jsonl, zone_two.jsonl, ... one file per name in
config/building_policy.yaml's multizone.zone_names). This script reads
every *.jsonl file in that directory and produces:

  - logs/multizone/multizone_summary.csv   -- one row per zone: rows
    logged, approval rate (AI-sourced / total), HOLD rate, comfort
    violations, correction rate, and final cumulative_kwh (note: this is
    the bridge-wide COMBINED total, identical across zones -- see
    core/energyplus_bridge.py's fix note #5 on why per-zone kWh may be
    all-zero).
  - logs/multizone/multizone_temps.png     -- indoor temp trace per zone
    (+ shared outdoor temp) on one chart, so you can visually confirm the
    two zones are NOT moving in lockstep.

Usage:
    python scripts/summarize_multizone.py
    python scripts/summarize_multizone.py --log-dir logs/multizone
"""

import argparse
import glob
import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_zone_log(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def summarize(log_dir="logs/multizone", out_csv=None, out_png=None):
    out_csv = out_csv or os.path.join(log_dir, "multizone_summary.csv")
    out_png = out_png or os.path.join(log_dir, "multizone_temps.png")

    paths = sorted(glob.glob(os.path.join(log_dir, "*.jsonl")))
    if not paths:
        print(f"No .jsonl logs found in {log_dir}. Run "
              f"`python main.py --multizone` first (needs "
              f"models/two_zone_controlled.idf -- generate it with "
              f"`python scripts/patch_idf.py` -- plus a working "
              f"EnergyPlus + Groq environment).")
        return

    zone_frames = {}
    for path in paths:
        zone = os.path.splitext(os.path.basename(path))[0]
        df = _load_zone_log(path)
        if df.empty:
            print(f"WARNING: {path} produced no rows, skipping.")
            continue
        zone_frames[zone] = df

    if not zone_frames:
        print("No usable per-zone data found.")
        return

    # --- Summary table ---
    summary_rows = []
    for zone, df in zone_frames.items():
        total = len(df)
        approved = df["source"].isin(["AI", "AI (Corrected)"]).sum() if "source" in df else 0
        held = df["source"].astype(str).str.contains("Stabilized", na=False).sum() if "source" in df else 0
        failsafe = df["source"].astype(str).str.contains("FAILSAFE", na=False).sum() if "source" in df else 0
        corrected = df["source"].astype(str).str.contains("Corrected", na=False).sum() if "source" in df else 0
        violations = df["reason"].astype(str).str.contains("REJECTED: Violation", na=False).sum() if "reason" in df else 0
        final_kwh = df["cumulative_kwh"].iloc[-1] if "cumulative_kwh" in df and total else None

        summary_rows.append({
            "zone": zone,
            "rows": total,
            "approval_rate_pct": round(100 * approved / total, 1) if total else None,
            "hold_rate_pct": round(100 * held / total, 1) if total else None,
            "correction_rate_pct": round(100 * corrected / total, 1) if total else None,
            "failsafe_rate_pct": round(100 * failsafe / total, 1) if total else None,
            "comfort_violations_seen": int(violations),
            "final_cumulative_kwh_combined": final_kwh,
        })

    summary_df = pd.DataFrame(summary_rows)
    os.makedirs(log_dir, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    print("--- MULTI-ZONE SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {out_csv}")

    # --- Temperature trace chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    outdoor_plotted = False
    for zone, df in zone_frames.items():
        if "step" not in df or "t_in" not in df:
            continue
        ax.plot(df["step"], df["t_in"], label=f"{zone} indoor")
        if not outdoor_plotted and "t_out" in df:
            ax.plot(df["step"], df["t_out"], label="outdoor", linestyle=":", color="gray")
            outdoor_plotted = True

    ax.set_xlabel("Simulation step")
    ax.set_ylabel("Temperature (C)")
    ax.set_title("Blueprint 1.1: independent per-zone temperature traces")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")

    if len(zone_frames) > 1:
        zones = list(zone_frames.keys())
        # Quick, honest sanity signal: are the two zones' final indoor
        # temps identical to several decimal places? If so, something is
        # probably wired to the same actuator twice rather than two
        # independent ones -- worth a manual look before demoing.
        last_temps = {z: df["t_in"].iloc[-1] for z, df in zone_frames.items() if "t_in" in df}
        if len(set(round(v, 3) for v in last_temps.values())) == 1 and len(last_temps) > 1:
            print("\nNOTE: all zones' final indoor temps are identical to 3 "
                  "decimal places. That CAN happen legitimately (both zones "
                  "converging to the same setpoint), but if it holds across "
                  "the whole run it's worth double-checking that each zone "
                  "really has its own actuator handle rather than both "
                  "writing to the same one.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs/multizone")
    args = parser.parse_args()
    summarize(log_dir=args.log_dir)
