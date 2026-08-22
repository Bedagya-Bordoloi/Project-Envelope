"""
scripts/generate_evidence.py

Round-2 audit finding #3: DEMO_REHEARSAL.md's Part 0 evidence table names
calibration_report.csv, multizone_summary.csv, compare_models's CSV,
bacnet_adapter_test.json, etc. -- but the scripts that produce them had
never actually been run in the submitted zip. "Code exists" != "we have
the number." This script runs the whole table in one shot, in a sane
order, and tells you plainly which artifacts it produced vs. which steps
failed (so you know exactly what still needs attention before a judge
asks to see one).

This does NOT replace the manual live-failsafe rehearsal in
DEMO_REHEARSAL.md Part 2 -- that one has to be done by hand, live.

Usage:
    python scripts/generate_evidence.py
    python scripts/generate_evidence.py --skip multizone bacnet   # skip slow/optional steps

Each step's stdout/stderr is captured to logs/evidence/<step>.log so a
failure doesn't scroll off your terminal.
"""

import argparse
import os
import subprocess
import sys
import time

LOG_DIR = "logs/evidence"

# FIX (audit finding: "generate_evidence.py can take half a day to over
# a day and gives zero terminal feedback while running"): the old code
# used subprocess.run(cmd, stdout=logf, ...), which blocks silently
# until the whole step finishes -- for the AI/baseline/sweep steps that
# can be hours, so a long-running step looks identical to a hang. This
# prints a short "still running" heartbeat to the console (while still
# capturing full output to logs/evidence/<step>.log exactly as before)
# so the person watching the terminal can tell the difference between
# "working" and "stuck".
_HEARTBEAT_INTERVAL_S = 30.0

def build_steps(run_period_days=None, sweep_thresholds=None):
    """Build the STEPS list. run_period_days (audit fix): the full-year
    default makes baseline_run/ai_run/calibration_sweep take somewhere
    between 30 min and multiple hours EACH (see calibrate_ccs_sweep.py's
    module docstring for the math), so calibration_sweep alone re-runs
    the AI pipeline 9x sequentially on top of that. Pass --quick-days N
    to shorten every AI-pipeline step's RunPeriod to N days for a fast
    iteration pass; omit it (the default) to run the real full-year
    numbers you'd actually put on a slide."""
    run_period_args = ["--run-period-days", str(run_period_days)] if run_period_days else []
    sweep_args = ["--baseline-log", "logs/baseline/control_log.jsonl"] + run_period_args
    if sweep_thresholds:
        sweep_args += ["--thresholds"] + [str(t) for t in sweep_thresholds]

    return [
        ("llm_connectivity", [sys.executable, "scripts/check_llm_connectivity.py"],
         "Confirms the Strategist is actually reaching Groq before anything else "
         "(no point generating a savings number that's secretly all-fallback)."),
        ("baseline_run", [sys.executable, "main.py", "--baseline"] + run_period_args,
         "logs/baseline/control_log.jsonl"),
        ("ai_run", [sys.executable, "main.py"] + run_period_args,
         "logs/ai/control_log.jsonl"),
        ("calibration_sweep", [sys.executable, "scripts/calibrate_ccs_sweep.py"] + sweep_args,
         "logs/sweep/calibration_report.csv, logs/sweep/calibration_sweep.png"),
        ("forecast_spike", [sys.executable, "scripts/find_forecast_spike.py"],
         "pre-cool/pre-heat demo evidence"),
        ("compare_models", [sys.executable, "scripts/compare_models.py"],
         "8B vs 70B latency/CCS-pass-rate table"),
        ("multizone", [sys.executable, "main.py", "--multizone"] + run_period_args,
         "logs/multizone/*.jsonl"),
        ("multizone_summary", [sys.executable, "scripts/summarize_multizone.py"],
         "logs/multizone/multizone_summary.csv, multizone_temps.png"),
        ("bacnet_test", [sys.executable, "scripts/test_bacnet_adapter.py"],
         "logs/bacnet_adapter_test.json"),
        ("ev_demo", [sys.executable, "demos/ev_charging_demo.py"],
         "logs/ev_demo/ev_charging_log.jsonl"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", nargs="*", default=[],
                         help="Step names to skip, e.g. --skip multizone bacnet_test")
    parser.add_argument("--quick-days", type=int, default=None,
                         help="Audit fix for wall-clock infeasibility: shorten every "
                              "AI-pipeline step's (baseline/ai/sweep/multizone) RunPeriod "
                              "to N days instead of the full year, via main.py's/"
                              "calibrate_ccs_sweep.py's --run-period-days. A full year at "
                              "the default cadence_steps=12 / max_requests_per_minute=25 "
                              "can take hours PER step; --quick-days 14 finishes in "
                              "minutes. Use the full year (omit this flag) only for the "
                              "numbers you actually put on a slide.")
    parser.add_argument("--sweep-thresholds", type=float, nargs="+", default=None,
                         help="Forwarded to calibrate_ccs_sweep.py --thresholds. Use with "
                              "--quick-days to also shrink the sweep's point count, e.g. "
                              "--sweep-thresholds 0.55 0.65 0.75 instead of the default 9.")
    args = parser.parse_args()

    if args.quick_days:
        print(f"[generate_evidence] --quick-days {args.quick_days}: baseline/ai/sweep/"
              f"multizone steps will run a {args.quick_days}-day RunPeriod, NOT the full "
              f"year. Numbers from this pass are for iteration only -- rerun without "
              f"--quick-days before trusting a savings/violations number on a slide.")

    steps = build_steps(run_period_days=args.quick_days, sweep_thresholds=args.sweep_thresholds)

    os.makedirs(LOG_DIR, exist_ok=True)

    results = []
    for name, cmd, produces in steps:
        if name in args.skip:
            print(f"[skip] {name}")
            results.append((name, "SKIPPED", produces))
            continue

        print(f"\n=== {name}: {' '.join(cmd)} ===")
        log_path = os.path.join(LOG_DIR, f"{name}.log")
        step_start = time.perf_counter()
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            last_heartbeat = step_start
            while True:
                try:
                    returncode = proc.wait(timeout=_HEARTBEAT_INTERVAL_S)
                    break
                except subprocess.TimeoutExpired:
                    now = time.perf_counter()
                    if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                        last_heartbeat = now
                        elapsed_min = (now - step_start) / 60.0
                        print(f"  ...{name} still running "
                              f"({elapsed_min:.1f} min elapsed, see {log_path} "
                              f"for live detail)", flush=True)
        result = subprocess.CompletedProcess(cmd, returncode)

        if result.returncode == 0:
            print(f"[ok]   {name} -> {produces}")
            results.append((name, "OK", produces))
        else:
            print(f"[FAIL] {name} exited {result.returncode} -- see {log_path}")
            results.append((name, f"FAILED (exit {result.returncode})", produces))

    print("\n" + "=" * 70)
    print("EVIDENCE GENERATION SUMMARY")
    print("=" * 70)
    for name, status, produces in results:
        print(f"  {status:<20} {name:<20} -> {produces}")
    print("=" * 70)

    failed = [r for r in results if r[1].startswith("FAILED")]
    if failed:
        print(f"\n{len(failed)} step(s) failed. Check logs/evidence/<step>.log for each "
              f"before trusting any slide number that depends on it.")
        sys.exit(1)
    else:
        print("\nAll steps completed. Cross-check the artifact list above against "
              "DEMO_REHEARSAL.md's Part 0 table before building slides.")


if __name__ == "__main__":
    main()
