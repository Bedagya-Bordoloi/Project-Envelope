"""
scripts/calibrate_ccs_sweep.py

Blueprint 1.2 -- CCS calibration sweep.

The existing scripts/calibrate_ccs.py (kept, unchanged) only computes
mean/std of the CCS scores actually seen in ONE run at ONE threshold --
that tells you about the distribution of scores, not about how the
system *behaves* at a different threshold. This script does the real
sweep the blueprint asks for: it re-runs the full AI pipeline once per
candidate ccs_threshold, and for each run computes:

  - approval rate      (AI-sourced rows / total rows)
  - energy savings %   (vs. a baseline run you provide once, since the
                        baseline schedule doesn't depend on ccs_threshold
                        and re-running it per threshold would be wasted
                        wall-clock time)
  - comfort violations (rows where SentinelGate's PMV check failed --
                        i.e. every row whose reason starts with
                        "REJECTED: Violation", counted as a violation
                        the AI PROPOSED, not one that reached the
                        building -- the gate's whole job is to stop
                        those before they're applied. If you want
                        "violations that reached the zone" instead,
                        post-filter calibration_report.csv's per-sweep
                        logs/sweep/sweep_<t>.jsonl on actual t_in vs.
                        your comfort band.)

Usage:
    # 1. Run the baseline once (not swept, doesn't depend on threshold):
    python main.py --baseline

    # 2. Run the sweep (each point re-runs the full AI pipeline -- this
    #    is wall-clock-expensive; reduce --thresholds for a quick pass):
    python scripts/calibrate_ccs_sweep.py --baseline-log logs/baseline/control_log.jsonl

Produces:
    logs/sweep/sweep_<threshold>.jsonl   (one full control log per point)
    logs/sweep/calibration_report.csv    (the table)
    logs/sweep/calibration_sweep.png     (the plot -- approval rate /
                                           savings % / violations vs.
                                           threshold, per blueprint 1.2)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

import pandas as pd
import yaml

POLICY_PATH = "config/building_policy.yaml"
AI_LOG_PATH = "logs/ai/control_log.jsonl"
SWEEP_DIR = "logs/sweep"

DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
# FIX (Round-2 audit finding: "the gate's calibration story doesn't
# account for the hard violation clause"): SentinelGate used to require
# violation_severity == 0 no matter what ccs_threshold was set to, so a
# threshold-only sweep could never show/explain long FAILSAFE streaks in
# warm weather. core/sentinel_gate.py now reads a configurable
# gate.max_violation_severity (default 0.0, i.e. identical to the old
# hardcoded behavior). This script can now sweep that dimension too --
# pass --violation-tolerances to opt in; default keeps the old
# threshold-only behavior unchanged.
DEFAULT_VIOLATION_TOLERANCES = [0.0]


# BUGFIX (audit finding): the old version of this function did
# `yaml.safe_load()` then `yaml.dump()` to change two numbers. PyYAML's
# dumper does not preserve comments -- confirmed by actually running it
# against config/building_policy.yaml, every one of the file's ~103
# "FIX (Round-N audit)..." explanation comments (documenting *why*
# cadence is 12, why enforce_baseline_direction matters, why the model
# string changed, etc.) was silently deleted on the very first sweep
# point, even at default settings, because the restore-on-exit logic
# only restored the *values*, not the comments. That's permanent data
# loss on a file the user was handed, on the very first run of this
# script (or generate_evidence.py, which calls it).
#
# Fix: never round-trip the whole file through yaml.safe_load/yaml.dump
# for a two-value change. Instead, do a targeted regex replace of just
# the `ccs_threshold:` and `max_violation_severity:` value lines under
# `gate:`, leaving every other byte of the file -- comments, key order,
# blank lines, formatting -- untouched. Each key is matched by a
# line-anchored regex (not a generic string search), so this can't
# accidentally match an occurrence of the word inside a comment (e.g.
# the sentence above that mentions "ccs_threshold" in prose).
_CCS_THRESHOLD_FIELD_RE = re.compile(
    r"^(?P<prefix>[ \t]*ccs_threshold:[ \t]*)\S+(?P<suffix>.*)$",
    re.MULTILINE,
)
_MAX_VIOLATION_SEVERITY_FIELD_RE = re.compile(
    r"^(?P<prefix>[ \t]*max_violation_severity:[ \t]*)\S+(?P<suffix>.*)$",
    re.MULTILINE,
)


def _set_threshold(threshold: float, violation_tolerance: float = 0.0):
    with open(POLICY_PATH, "r") as f:
        text = f.read()

    new_text, n = _CCS_THRESHOLD_FIELD_RE.subn(
        lambda m: f"{m.group('prefix')}{float(threshold)}{m.group('suffix')}",
        text, count=1,
    )
    if n != 1:
        raise RuntimeError(
            f"Expected exactly 1 'ccs_threshold:' line in {POLICY_PATH}, found {n}. "
            f"Refusing to write the file to avoid corrupting it -- check that the "
            f"key still exists under 'gate:' in the expected 'key: value' format."
        )

    new_text, n = _MAX_VIOLATION_SEVERITY_FIELD_RE.subn(
        lambda m: f"{m.group('prefix')}{float(violation_tolerance)}{m.group('suffix')}",
        new_text, count=1,
    )
    if n != 1:
        raise RuntimeError(
            f"Expected exactly 1 'max_violation_severity:' line in {POLICY_PATH}, "
            f"found {n}. Refusing to write the file to avoid corrupting it."
        )

    with open(POLICY_PATH, "w") as f:
        f.write(new_text)


def _load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _final_kwh(rows):
    for row in reversed(rows):
        if "cumulative_kwh" in row and row["cumulative_kwh"] is not None:
            return float(row["cumulative_kwh"])
    return None


def _estimate_wall_clock_minutes(n_points, policy):
    """Rough, deliberately conservative estimate of how long `n_points`
    sequential `python main.py` runs will take, dominated by rate-limiter
    waiting (EnergyPlus compute time is comparatively small). Printed
    up front so a multi-hour sweep doesn't look identical to a hang --
    see the audit finding this is fixing."""
    cadence = policy.get("strategist", {}).get("cadence_steps", 12)
    max_rpm = policy.get("strategist", {}).get("max_requests_per_minute", 25)
    steps_per_year = 365 * 24 * 4  # 15-min zone timesteps, full year
    ticks_per_run = steps_per_year / max(1, cadence)
    # Worst case: every tick also needs a second (correction) call.
    calls_per_run = ticks_per_run * 2
    minutes_per_run = calls_per_run / max(1, max_rpm)
    return minutes_per_run * n_points


def run_sweep(thresholds, violation_tolerances, baseline_log_path, restore_config,
              run_period_days=None):
    os.makedirs(SWEEP_DIR, exist_ok=True)

    baseline_rows = _load_jsonl(baseline_log_path)
    if not baseline_rows:
        print(f"WARNING: no baseline log found at {baseline_log_path}. "
              f"Run `python main.py --baseline` first if you want savings %% "
              f"in the report. Continuing with approval-rate/violations only.")
    baseline_kwh = _final_kwh(baseline_rows) if baseline_rows else None

    records = []
    for tol in violation_tolerances:
        for t in thresholds:
            label = f"ccs_threshold={t:.2f}, max_violation_severity={tol:.2f}"
            print(f"\n=== Sweeping {label} ===")
            _set_threshold(t, tol)

            # main.py appends to logs/ai/control_log.jsonl -- clear any prior
            # content for THIS sweep point so rows from an earlier point
            # don't leak into this one's stats.
            os.makedirs(os.path.dirname(AI_LOG_PATH), exist_ok=True)
            if os.path.exists(AI_LOG_PATH):
                os.remove(AI_LOG_PATH)

            main_cmd = [sys.executable, "main.py"]
            if run_period_days:
                main_cmd += ["--run-period-days", str(run_period_days)]

            point_start = time.perf_counter()
            result = subprocess.run(main_cmd)
            print(f"  ({(time.perf_counter() - point_start) / 60.0:.1f} min for this point)")
            if result.returncode != 0:
                print(f"WARNING: `python main.py` exited {result.returncode} at "
                      f"{label} -- skipping this point.")
                continue

            dest = os.path.join(SWEEP_DIR, f"sweep_t{t:.2f}_v{tol:.2f}.jsonl")
            shutil.copy(AI_LOG_PATH, dest)

            rows = _load_jsonl(dest)
            if not rows:
                print(f"WARNING: {dest} produced no rows -- skipping.")
                continue

            df = pd.DataFrame(rows)
            total = len(df)
            approved = df["source"].str.startswith(("AI", "AI (")).sum() if total else 0
            approval_rate = approved / total if total else 0.0

            # Rejections that the correction pass ALSO failed end up as
            # "FAILSAFE" with a "Gate Override:" reason -- those, plus any
            # row whose reason string shows a violation, count as comfort
            # violations the AI proposed (and the gate caught).
            violations = df["reason"].fillna("").str.contains(
                "Violation", case=False
            ).sum()

            ai_kwh = _final_kwh(rows)
            savings_pct = None
            if baseline_kwh and ai_kwh is not None and baseline_kwh > 0:
                savings_pct = (baseline_kwh - ai_kwh) / baseline_kwh * 100

            records.append({
                "threshold": t,
                "max_violation_severity": tol,
                "total_rows": total,
                "approval_rate": round(approval_rate, 4),
                "comfort_violations": int(violations),
                "ai_kwh": ai_kwh,
                "savings_pct": round(savings_pct, 2) if savings_pct is not None else None,
            })
            print(f"  approval_rate={approval_rate:.3f}  violations={violations}  "
                  f"ai_kwh={ai_kwh}  savings_pct={records[-1]['savings_pct']}")

    if restore_config is not None:
        _set_threshold(*restore_config)
        print(f"\nRestored {POLICY_PATH} gate.ccs_threshold/max_violation_severity to {restore_config}.")

    report_df = pd.DataFrame(records)
    report_path = os.path.join(SWEEP_DIR, "calibration_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\nWrote {report_path}")
    return report_df


def plot_report(report_df):
    if report_df.empty:
        print("Nothing to plot (empty sweep results).")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tolerances = sorted(report_df["max_violation_severity"].unique()) if "max_violation_severity" in report_df else [0.0]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("CCS threshold")
    ax1.set_ylabel("Approval rate", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Comfort violations (count)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # One line per violation-tolerance value (usually just one, [0.0], if
    # --violation-tolerances wasn't passed -- identical to the old plot).
    markers = ["o", "D", "^", "v", "P", "X"]
    for i, tol in enumerate(tolerances):
        sub = report_df[report_df["max_violation_severity"] == tol].sort_values("threshold")
        m = markers[i % len(markers)]
        ax1.plot(sub["threshold"], sub["approval_rate"], marker=m, color="tab:blue",
                  linestyle="-" if i == 0 else "--", alpha=1.0 if i == 0 else 0.6,
                  label=f"Approval rate (tol={tol:.2f})")
        ax2.plot(sub["threshold"], sub["comfort_violations"], marker=m, color="tab:red",
                  linestyle="-" if i == 0 else "--", alpha=1.0 if i == 0 else 0.6,
                  label=f"Violations (tol={tol:.2f})")

    if report_df["savings_pct"].notna().any():
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("outward", 60))
        ax3.set_ylabel("Energy savings %", color="tab:green")
        ax3.tick_params(axis="y", labelcolor="tab:green")
        for i, tol in enumerate(tolerances):
            sub = report_df[report_df["max_violation_severity"] == tol].sort_values("threshold")
            m = markers[i % len(markers)]
            ax3.plot(sub["threshold"], sub["savings_pct"], marker=m, color="tab:green",
                      linestyle="-" if i == 0 else "--", alpha=1.0 if i == 0 else 0.6)

    if len(tolerances) > 1:
        ax1.legend(loc="lower left", fontsize=8)
        ax2.legend(loc="lower right", fontsize=8)

    fig.suptitle("CCS threshold (and violation-tolerance) calibration sweep")
    fig.tight_layout()
    out_path = os.path.join(SWEEP_DIR, "calibration_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--violation-tolerances", type=float, nargs="+",
                         default=DEFAULT_VIOLATION_TOLERANCES,
                         help="Sweep gate.max_violation_severity too (Round-2 audit fix). "
                              "Default [0.0] reproduces the old threshold-only sweep exactly.")
    parser.add_argument("--baseline-log", default="logs/baseline/control_log.jsonl")
    parser.add_argument("--no-restore", action="store_true",
                         help="Don't restore config/building_policy.yaml's original threshold/tolerance when done.")
    args = parser.parse_args()

    with open(POLICY_PATH, "r") as f:
        original_policy = yaml.safe_load(f)
    original_threshold = original_policy["gate"]["ccs_threshold"]
    original_tolerance = original_policy["gate"].get("max_violation_severity", 0.0)

    report_df = run_sweep(
        args.thresholds,
        args.violation_tolerances,
        args.baseline_log,
        restore_config=None if args.no_restore else (original_threshold, original_tolerance),
    )
    plot_report(report_df)


if __name__ == "__main__":
    main()
