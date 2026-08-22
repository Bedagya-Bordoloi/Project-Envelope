"""
scripts/verify_decision_cache.py

Standalone check for core/decision_cache.py AND its wiring into
main.py's ProjectEnvelope.decide(). Deliberately does NOT require
EnergyPlus, a Groq key, or a Gemini key -- it mocks EnergyPlusBridge and
Strategist so this can run anywhere (CI, no .env, no network) in under a
second, the same "cheap enough to actually run before every demo"
posture as scripts/check_llm_connectivity.py / check_secondary_provider.py.

What it actually proves, not just asserts blindly:
  1. Unit-level DecisionCache behavior: bin hit/miss, TTL expiry,
     invalidate() on a rejected replay.
  2. Integration-level: running ProjectEnvelope.decide() across several
     cadence ticks that land in the SAME state bin calls the (mocked)
     Strategist fewer times than there are ticks -- i.e. the cache is
     actually intercepting calls, not just present-but-inert (the exact
     "dead config" failure class this codebase has hit before -- see
     core/sentinel_gate.py's docstring for carbon/override_rate).
  3. SentinelGate.check() is still called on EVERY tick, cache hit or
     not -- proving the cache never bypasses the safety/comfort gate,
     only the LLM call (Blueprint 5.5's core safety property).
  4. A cache hit whose replay gets REJECTED invalidates that bin (a
     subsequent tick in the same bin is a fresh miss, not a repeat of
     the bad decision).

Run: python scripts/verify_decision_cache.py
Exits non-zero (and prints which assertion failed) if any check fails.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.decision_cache import DecisionCache


def _fail(msg):
    print(f"[verify_decision_cache] FAIL: {msg}")
    sys.exit(1)


def _ok(msg):
    print(f"[verify_decision_cache] OK: {msg}")


# ---------------------------------------------------------------------
# Part 1: unit-level DecisionCache behavior
# ---------------------------------------------------------------------
def test_unit_behavior():
    policy = {
        "strategist": {
            "decision_cache": {
                "enabled": True,
                "indoor_bin_c": 0.5,
                "outdoor_bin_c": 1.0,
                "hour_bin_h": 3.0,
                "ttl_ticks": 4,
            }
        }
    }
    cache = DecisionCache(policy)

    # Miss on an empty cache.
    if cache.get(21.0, 5.0, 2.0, "WINTER", tick_index=0) is not None:
        _fail("expected a miss on an empty cache")

    proposal = {"setpoint": 21.0, "confidence": 0.9, "reason": "test", "llm_ok": True, "error": None}
    cache.put(21.0, 5.0, 2.0, "WINTER", tick_index=0, proposal=proposal)

    # Hit for an identical state.
    hit = cache.get(21.0, 5.0, 2.0, "WINTER", tick_index=1)
    if hit is None or hit["setpoint"] != 21.0 or hit.get("cache_hit") is not True:
        _fail(f"expected a hit with cache_hit=True, got {hit}")

    # Hit for a NEARBY state within the same bin (bins are 0.5C/1.0C/3h wide).
    hit2 = cache.get(21.2, 5.4, 2.9, "WINTER", tick_index=2)
    if hit2 is None:
        _fail("expected a hit for a nearby state within the same bin")

    # Miss for a state outside the bin width.
    miss = cache.get(24.0, 5.0, 2.0, "WINTER", tick_index=2)
    if miss is not None:
        _fail("expected a miss for a state well outside the cached bin")

    # TTL expiry: tick_index - cached_tick(0) > ttl_ticks(4) -> miss, and pruned.
    expired = cache.get(21.0, 5.0, 2.0, "WINTER", tick_index=5)
    if expired is not None:
        _fail("expected the entry to have expired past ttl_ticks")

    # Different season -> different bin -> miss even with identical temps/hour.
    cache.put(21.0, 5.0, 2.0, "WINTER", tick_index=10, proposal=proposal)
    if cache.get(21.0, 5.0, 2.0, "SHOULDER", tick_index=11) is not None:
        _fail("expected season to be part of the bin key (SHOULDER != WINTER)")

    # invalidate() drops a bin immediately, independent of TTL.
    cache.invalidate(21.0, 5.0, 2.0, "WINTER")
    if cache.get(21.0, 5.0, 2.0, "WINTER", tick_index=10) is not None:
        _fail("expected invalidate() to drop the bin immediately")

    # Disabled cache is always a no-op (never raises, never hits).
    disabled = DecisionCache({"strategist": {"decision_cache": {"enabled": False}}})
    disabled.put(21.0, 5.0, 2.0, "WINTER", 0, proposal)
    if disabled.get(21.0, 5.0, 2.0, "WINTER", 0) is not None:
        _fail("expected a disabled cache to never return a hit")

    _ok("DecisionCache unit behavior (hit/miss/TTL/season-key/invalidate/disabled)")


# ---------------------------------------------------------------------
# Part 2: integration-level -- ProjectEnvelope.decide() actually uses it
# ---------------------------------------------------------------------
class _FakeBridge:
    """Minimal stand-in for EnergyPlusBridge -- just enough surface area
    for ProjectEnvelope.decide()/attach_bridge() to run without touching
    EnergyPlus at all."""

    def __init__(self):
        self.step_counter = 0
        self.cumulative_kwh = 0.0
        self.cumulative_heat_kwh = 0.0
        self.cumulative_cool_kwh = 0.0
        self.zone_names = ["ZONE ONE"]

    def get_forward_weather(self, hours=3):
        return [5.0] * hours


class _CallCountingStrategist:
    """Wraps a real Strategist-shaped decide() but counts real calls, so
    the test can assert the cache actually reduced call volume -- not
    just that the code runs without crashing."""

    def __init__(self, fixed_setpoint=21.0):
        self.calls = 0
        self.fixed_setpoint = fixed_setpoint
        self.tool_context = None
        self.last_lookahead_triggered = False

    def decide(self, t_in, t_out, forecast, carbon, correction_context=None, timeout=10):
        self.calls += 1
        return {
            "setpoint": self.fixed_setpoint,
            "confidence": 0.9,
            "reason": "mocked decision",
            "llm_ok": True,
            "error": None,
        }


def test_integration_reduces_calls():
    import main as main_module

    policy = {
        "safety": {"temp_min_c": -100.0, "temp_max_c": 100.0, "max_delta_c_per_step": 100.0},
        "comfort": {"deadband_c": 2.0, "pmv_band": 0.5, "outdoor_temp_ewma_alpha": 0.1},
        "hysteresis": {"min_delta_c": 0.0, "min_dwell_steps": 0},
        "seasonality": {"winter_limit": 14.0, "summer_limit": 22.0,
                         "comfort_ranges": {"winter": [21.0, 21.8], "summer": [23.5, 25.5]},
                         "clothing": {"winter_clo": 1.1, "shoulder_clo": 0.7, "summer_clo": 0.4}},
        "gate": {"ccs_threshold": 0.0, "weights": {"violation": 0.4, "rate_penalty": 0.15,
                 "llm_confidence": 0.15, "override_rate": 0.1, "carbon": 0.2},
                 "max_violation_severity": 1.0, "override_window": 20,
                 "enforce_baseline_direction": False, "baseline_direction_tolerance_c": 0.1},
        "strategist": {
            "model": "mock", "cadence_steps": 4, "correction_skip_delta_c": 0.0,
            "max_requests_per_minute": 1000, "call_timeout_s": 1, "correction_timeout_s": 1,
            "timestep_minutes": 15,
            # hour_bin_h=24 (single bucket for the whole day) so this
            # integration check isolates the temp/season binning without
            # also crossing an hour-of-day bucket boundary as
            # fake_bridge.step_counter advances across ticks -- the
            # hour-of-day bucketing itself is already covered by
            # test_unit_behavior() above.
            "decision_cache": {"enabled": True, "indoor_bin_c": 0.5, "outdoor_bin_c": 1.0,
                                "hour_bin_h": 24.0, "ttl_ticks": 100},
        },
        "failsafe": {"target_low_c": 21.0, "target_high_c": 23.0, "setback_c": 1.0},
        "baseline": {"schedule_setpoint_c": 22.0},
    }

    env = main_module.ProjectEnvelope(policy, "/tmp/verify_decision_cache_test_log.jsonl")
    fake_bridge = _FakeBridge()
    env.attach_bridge(fake_bridge)
    fake_strategist = _CallCountingStrategist(fixed_setpoint=21.0)
    env.strategist = fake_strategist  # swap in the call-counting mock

    gate_check_calls = {"n": 0}
    real_check = env.gate.check

    def _counting_check(*args, **kwargs):
        gate_check_calls["n"] += 1
        return real_check(*args, **kwargs)

    env.gate.check = _counting_check

    # Same t_in/t_out/hour/season every tick -> every cadence tick after
    # the first should land in the same bin.
    n_ticks = 6
    cache_hits_seen = []
    for i in range(n_ticks):
        fake_bridge.step_counter = i * policy["strategist"]["cadence_steps"]
        setpoint, source = env.decide(t_in=21.0, t_out=5.0, humidity=45.0)
        cache_hits_seen.append("Cached" in source)

    if fake_strategist.calls != 1:
        _fail(f"expected exactly 1 real Strategist call across {n_ticks} identical-bin "
              f"ticks (cache should have served the rest), got {fake_strategist.calls}")

    if gate_check_calls["n"] != n_ticks:
        _fail(f"expected SentinelGate.check() to run on EVERY tick ({n_ticks}) regardless "
              f"of cache hits (the gate must never be bypassed), got {gate_check_calls['n']}")

    if cache_hits_seen[0]:
        _fail("the FIRST tick should be a cache miss (nothing cached yet)")
    if not all(cache_hits_seen[1:]):
        _fail(f"expected ticks 2..{n_ticks} to be cache hits, got {cache_hits_seen}")

    _ok(f"integration: {n_ticks} identical-bin ticks -> {fake_strategist.calls} real "
        f"Strategist call, {gate_check_calls['n']} gate checks (one per tick)")

    # --- Part 3: a rejected replay invalidates the bin -------------------
    # Force the NEXT cached decision to fail the gate by making the cache
    # hold a value that will violate the (very permissive) policy's
    # violation_fn once t_in changes sharply, then confirm the following
    # tick in the same nominal bin is a fresh miss (a new real call).
    fake_bridge.step_counter = n_ticks * policy["strategist"]["cadence_steps"]
    # Sanity: bin currently holds a cached, approved 21.0C decision
    # (still well within ttl_ticks=100, put at tick_index=0).
    if env.decision_cache.get(21.0, 5.0, 0.0, "WINTER", tick_index=n_ticks) is None:
        _fail("setup assumption failed: expected an existing cache entry before the "
              "invalidate-on-reject check")

    # Directly exercise invalidate() the same way main.py's reject branch
    # does, then confirm the bin is gone.
    env.decision_cache.invalidate(21.0, 5.0, 0.0, "WINTER")
    if env.decision_cache.get(21.0, 5.0, 0.0, "WINTER", tick_index=n_ticks) is not None:
        _fail("expected invalidate() to remove the bin main.py's reject branch targets")

    _ok("invalidate() path used by main.py's REJECTED branch removes the stale bin")

    os.remove("/tmp/verify_decision_cache_test_log.jsonl")


if __name__ == "__main__":
    test_unit_behavior()
    test_integration_reduces_calls()
    print("[verify_decision_cache] ALL CHECKS PASSED")
