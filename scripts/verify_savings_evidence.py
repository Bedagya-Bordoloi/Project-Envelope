"""
scripts/verify_savings_evidence.py

Priority 4 (Audit Blueprint Section 4 -- "Pre-submission verification
pass"). Everything up through Priority 3 changes CODE; this script
changes nothing -- it reads whatever control_log.jsonl files a clean
`python main.py` / `python main.py --baseline` pair already produced
and checks the three things the blueprint says must hold before you
treat the resulting Live Savings % as real:

  1. The AI log's winter setpoints actually vary step to step (not the
     old frozen-21.2C symptom) -- visual proof the LLM loop is live,
     not silently all-fallback.
  2. The AI instance's cumulative_cool_kwh barely moves through deep
     winter -- i.e. the Priority 1a cooling-ceiling clamp is actually
     holding in the real run, not just in the code.
  3. A meaningful share of scored decisions came from a real LLM call
     (llm_ok=True), not the hardcoded physical-target fallback.

This does NOT run the simulation itself -- see scripts/generate_evidence.py
for that. Run this AFTER a clean pair of runs
(rm logs/*/control_log.jsonl; python main.py --baseline; python main.py),
pointed at the resulting logs.

Usage:
    python scripts/verify_savings_evidence.py
    python scripts/verify_savings_evidence.py \
        --ai-log logs/ai/control_log.jsonl \
        --baseline-log logs/baseline/control_log.jsonl \
        --winter-limit 14.0 --cool-tolerance-kwh 0.05

Exit code is 0 if every check passes (savings number is safe to present
as-is), 1 if any check fails or a log is missing/empty (fix or re-run
before presenting), 2 if a check is inconclusive (e.g. no winter rows
in this run's window -- read the printed reason and judge for yourself).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_log(path):
    """Reads a control_log.jsonl into a list of dicts. Returns None if the
    file doesn't exist or is empty -- callers treat that as "re-run
    first", not a crash."""
    if not os.path.exists(path):
        return None
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
    return rows or None


def check_setpoint_variance(ai_rows, winter_limit):
    """Check 1: winter setpoints must not be a single frozen value --
    that was the exact symptom (21.2C on every row, step 12 to 912)
    that started this whole audit."""
    winter = [r for r in ai_rows if r.get("t_out") is not None and r["t_out"] < winter_limit]
    if not winter:
        return None, "no winter rows (t_out < %.1fC) in this run's window -- can't check" % winter_limit
    distinct = len({round(r["setpoint"], 2) for r in winter if r.get("setpoint") is not None})
    if distinct <= 1:
        return False, f"FROZEN -- only {distinct} distinct winter setpoint value across {len(winter)} rows"
    return True, f"{distinct} distinct winter setpoint values across {len(winter)} rows"


def check_winter_cooling_ceiling(ai_rows, winter_limit, tolerance_kwh):
    """Check 2: the Priority 1a clamp's actual effect -- AI's cumulative
    cooling energy should barely move through deep winter. Needs the
    Priority 1b split (cumulative_cool_kwh) to be present in the log;
    older logs from before that fix won't have it."""
    winter = [r for r in ai_rows if r.get("t_out") is not None and r["t_out"] < winter_limit]
    if not winter:
        return None, "no winter rows (t_out < %.1fC) in this run's window -- can't check" % winter_limit
    if "cumulative_cool_kwh" not in winter[0]:
        return None, "cumulative_cool_kwh not in this log -- re-run to pick up the Priority 1b split-energy logging"
    cool_start = winter[0]["cumulative_cool_kwh"]
    cool_end = winter[-1]["cumulative_cool_kwh"]
    delta = cool_end - cool_start
    ok = delta <= tolerance_kwh
    verdict = "holding" if ok else "LEAKING -- clamp is not preventing winter cooling draw"
    return ok, f"AI cooling-kWh moved {delta:.3f} kWh across {len(winter)} winter rows ({verdict}; tolerance {tolerance_kwh:.2f} kWh)"


def check_llm_success_rate(ai_rows, min_rate):
    """Check 3: a meaningful share of scored (non-Holding) decisions
    actually reached Groq, rather than the hardcoded physical-target
    fallback -- llm_ok is None only for the no-op Holding path, which
    this excludes rather than treating as a failure."""
    scored = [r for r in ai_rows if r.get("llm_ok") is not None]
    if not scored:
        return None, "no scored decisions (llm_ok always None) -- can't check"
    ok_count = sum(1 for r in scored if r["llm_ok"])
    rate = ok_count / len(scored)
    passed = rate >= min_rate
    return passed, f"{ok_count}/{len(scored)} scored decisions reached a real LLM call ({rate*100:.1f}%, minimum {min_rate*100:.0f}%)"


def compute_savings_pct(ai_rows, baseline_rows):
    ai_total = ai_rows[-1].get("cumulative_kwh")
    baseline_total = baseline_rows[-1].get("cumulative_kwh")
    if ai_total is None or baseline_total is None or baseline_total == 0:
        return None
    return (baseline_total - ai_total) / baseline_total * 100.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ai-log", default="logs/ai/control_log.jsonl")
    parser.add_argument("--baseline-log", default="logs/baseline/control_log.jsonl")
    parser.add_argument("--winter-limit", type=float, default=14.0,
                         help="Matches config/building_policy.yaml's seasonality.winter_limit.")
    parser.add_argument("--cool-tolerance-kwh", type=float, default=0.05,
                         help="How much winter cooling-kWh drift to tolerate before flagging a leak.")
    parser.add_argument("--min-llm-success-rate", type=float, default=0.5,
                         help="Fraction of scored decisions that must have reached a real LLM call.")
    args = parser.parse_args()

    ai_rows = load_log(args.ai_log)
    baseline_rows = load_log(args.baseline_log)

    if ai_rows is None:
        print(f"[verify_savings_evidence] {args.ai_log} is missing or empty -- run `python main.py` first.")
        sys.exit(1)
    if baseline_rows is None:
        print(f"[verify_savings_evidence] {args.baseline_log} is missing or empty -- run `python main.py --baseline` first.")
        sys.exit(1)

    checks = [
        ("Winter setpoints vary (not frozen)", check_setpoint_variance(ai_rows, args.winter_limit)),
        ("Winter cooling-ceiling clamp holding", check_winter_cooling_ceiling(ai_rows, args.winter_limit, args.cool_tolerance_kwh)),
        ("LLM success rate", check_llm_success_rate(ai_rows, args.min_llm_success_rate)),
    ]

    print("=== Priority 4 pre-submission verification ===")
    any_fail = False
    any_inconclusive = False
    for name, (result, detail) in checks:
        if result is None:
            tag = "SKIP"
            any_inconclusive = True
        elif result:
            tag = "PASS"
        else:
            tag = "FAIL"
            any_fail = True
        print(f"[{tag}] {name}: {detail}")

    savings_pct = compute_savings_pct(ai_rows, baseline_rows)
    print()
    if savings_pct is not None:
        print(f"Live Savings %: {savings_pct:.2f}% (AI cumulative_kwh vs baseline cumulative_kwh, last logged row)")
    else:
        print("Live Savings %: could not compute (cumulative_kwh missing from one of the logs)")

    print()
    if any_fail:
        print("VERDICT: Do NOT present the current Live Savings % as-is -- at least one check FAILED above.")
        sys.exit(1)
    elif any_inconclusive:
        print("VERDICT: Inconclusive -- at least one check was SKIPped (see reasons above). Use judgement.")
        sys.exit(2)
    else:
        print("VERDICT: All checks passed -- the Live Savings % above is safe to present.")
        sys.exit(0)


if __name__ == "__main__":
    main()
