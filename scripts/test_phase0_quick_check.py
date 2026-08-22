"""
scripts/test_phase0_quick_check.py

Standalone verification for Rework Blueprint Phase 0 (5.1): the always-on
fast-loop safety net. Runs with zero EnergyPlus/network dependency -- pure
control-logic tests against core/sentinel_gate.py's SentinelGate.quick_check()
and its wiring into main.py's ProjectEnvelope.decide().

Run this after any future change to sentinel_gate.py or main.py's decide()
to confirm Phase 0 hasn't silently regressed.

Usage:
    python scripts/test_phase0_quick_check.py

Exit code 0 if all checks pass, 1 otherwise.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strategist.__init__ requires a GROQ_API_KEY to exist (even a fake one is
# fine -- this test never actually calls out to Groq, since the quick_check
# trip path returns before any Strategist.decide() call happens).
os.environ.setdefault("GROQ_API_KEY", "test_dummy_key_for_phase0_verification")

import yaml

from core.sentinel_gate import SentinelGate
from main import ProjectEnvelope


def test_quick_check_unit():
    print("=" * 70)
    print("Unit tests: SentinelGate.quick_check()")
    print("=" * 70)

    policy = yaml.safe_load(open("config/building_policy.yaml"))
    gate = SentinelGate(policy)

    cases = [
        (policy["safety"]["temp_min_c"] - 0.1, True, "just below floor"),
        (policy["safety"]["temp_min_c"], False, "exactly at floor (boundary, no trip)"),
        (22.0, False, "normal mid-range"),
        (policy["safety"]["temp_max_c"], False, "exactly at ceiling (boundary, no trip)"),
        (policy["safety"]["temp_max_c"] + 0.1, True, "just above ceiling"),
    ]

    ok = True
    for t_in, expect_trip, label in cases:
        tripped, reason = gate.quick_check(t_in)
        passed = tripped == expect_trip
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  t_in={t_in:>6.2f}  "
              f"tripped={tripped!s:<5} expected={expect_trip!s:<5}  ({label})")

    # quick_check must be side-effect-free with respect to the real
    # scored-decision state (hysteresis dwell counter, override-rate
    # history) -- it's a smoke detector, not a scored decision.
    state_isolated = (gate._steps_since_change == 0) and (len(gate._decision_history) == 0)
    print(f"  {'PASS' if state_isolated else 'FAIL'}  quick_check does not mutate "
          f"hysteresis/override-rate state")
    ok &= state_isolated

    return ok


class _FakeBridge:
    step_counter = 5  # deliberately NOT a multiple of any realistic cadence_steps
    cumulative_kwh = 12.34
    cumulative_heat_kwh = 12.34
    cumulative_cool_kwh = 0.0
    zone_names = ["ZONE ONE"]


def test_main_integration():
    print()
    print("=" * 70)
    print("Integration tests: ProjectEnvelope.decide() off-cadence wiring")
    print("=" * 70)

    policy = yaml.safe_load(open("config/building_policy.yaml"))
    ok = True

    # Case A: off-cadence step, in-bounds temp -> must take the OLD
    # "Holding" path unchanged, with NO log line written (this is the
    # pre-Phase-0 behavior for the common case; Phase 0 must not add log
    # volume/noise on every physical step, only on genuine trips).
    tmp_log = tempfile.mktemp(suffix=".jsonl")
    env = ProjectEnvelope(policy, tmp_log)
    env.attach_bridge(_FakeBridge())

    setpoint, source = env.decide(t_in=22.0, t_out=5.0, humidity=45.0)
    case_a_ok = (source == "Holding")
    with open(tmp_log) as f:
        lines = f.readlines()
    case_a_ok &= (len(lines) == 0)
    print(f"  {'PASS' if case_a_ok else 'FAIL'}  in-bounds off-cadence step -> "
          f"source={source!r}, log lines written={len(lines)} (expect 'Holding', 0)")
    ok &= case_a_ok
    os.remove(tmp_log)

    # Case B: off-cadence step, temp below the hard floor -> quick_check
    # MUST fire even though this isn't a cadence tick. This is the actual
    # point of Phase 0: the old code would have silently held here with
    # zero verification until the next scheduled LLM call.
    tmp_log = tempfile.mktemp(suffix=".jsonl")
    env = ProjectEnvelope(policy, tmp_log)
    env.attach_bridge(_FakeBridge())

    below_floor = policy["safety"]["temp_min_c"] - 1.5
    setpoint, source = env.decide(t_in=below_floor, t_out=-15.0, humidity=45.0)
    case_b_ok = (source == "FAILSAFE (QuickCheck)")
    case_b_ok &= (setpoint != below_floor)  # failsafe must have actually intervened
    with open(tmp_log) as f:
        lines = f.readlines()
    case_b_ok &= (len(lines) == 1)
    if lines:
        entry = json.loads(lines[0])
        case_b_ok &= ("QUICK-CHECK TRIP" in entry.get("reason", ""))
        case_b_ok &= (entry.get("llm_ok") is None)  # no LLM was involved this tick
    print(f"  {'PASS' if case_b_ok else 'FAIL'}  below-floor off-cadence step -> "
          f"source={source!r}, setpoint={setpoint}, log lines written={len(lines)}")
    ok &= case_b_ok
    os.remove(tmp_log)

    return ok


def main():
    unit_ok = test_quick_check_unit()
    integration_ok = test_main_integration()

    print()
    print("=" * 70)
    if unit_ok and integration_ok:
        print("[PASS] Phase 0 (fast-loop safety net) verified.")
        sys.exit(0)
    else:
        print("[FAIL] Phase 0 verification found a regression -- see failures above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
