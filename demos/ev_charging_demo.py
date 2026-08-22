"""
demos/ev_charging_demo.py

"""

import json
import os
import random

import yaml

from core.sentinel_gate import SentinelGate

LOG_PATH = "logs/ev_demo/ev_charging_log.jsonl"

# Domain policy: same shape as building_policy.yaml's gate section, so
# SentinelGate.__init__ doesn't need to change to accept it.
EV_POLICY = {
    "gate": {
        "ccs_threshold": 0.65,
        "weights": {"violation": 0.40, "rate_penalty": 0.15, "llm_confidence": 0.15, "carbon": 0.20},
    },
    "hysteresis": {"min_delta_c": 0.3, "min_dwell_steps": 2},
    # Reused as the EWMA-smoothing knob for grid-demand load, same role
    # as outdoor-temp smoothing plays for HVAC.
    "comfort": {"outdoor_temp_ewma_alpha": 0.2},
}

PANEL_CAPACITY_KW = 7.2  # hard ceiling -- exceeding this is the "violation"
MIN_CHARGE_KW = 0.0


def capacity_violation_fn(proposed_kw, current_load_kw, _unused_humidity, load_ewma_kw):
    total_kw = load_ewma_kw + proposed_kw
    headroom_kw = PANEL_CAPACITY_KW - total_kw
    if headroom_kw >= 0:
        severity = 0.0
    else:
        # Same shape as the PMV severity calc: 0 at the boundary, scales
        # up smoothly past it instead of a hard cliff.
        severity = min(abs(headroom_kw) / PANEL_CAPACITY_KW, 1.0)
    label = f"load {total_kw:.2f}kW vs {PANEL_CAPACITY_KW}kW panel (headroom {headroom_kw:+.2f}kW)"
    return severity, {"total_kw": total_kw, "headroom_kw": headroom_kw}, label


def simulate_load_curve(n_steps=200, seed=3):
    """Fake site demand: a diurnal sine wave + noise, same role .epw plays for HVAC."""
    rng = random.Random(seed)
    import math
    curve = []
    for i in range(n_steps):
        base = 3.5 + 2.5 * math.sin(2 * math.pi * (i % 96) / 96.0)  # ~daily cycle
        noise = rng.uniform(-0.4, 0.4)
        curve.append(max(0.0, base + noise))
    return curve


def toy_strategist_propose(current_load_kw, headroom_hint_kw):
    """Stand-in for a real EV-charging strategist: greedy but capped by
    a rough headroom hint, with a bit of noise so proposals vary."""
    target = max(MIN_CHARGE_KW, min(7.2, headroom_hint_kw + random.uniform(-0.5, 0.5)))
    confidence = 0.85 if 0 <= target <= 7.2 else 0.5
    return {"setpoint": round(target, 2), "confidence": confidence}


def run_demo(n_steps=200):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    gate = SentinelGate(EV_POLICY, violation_fn=capacity_violation_fn, initial_setpoint=3.0)
    load_curve = simulate_load_curve(n_steps)
    last_rate = 3.0

    approved_count = 0
    rejected_count = 0
    hold_count = 0

    for step, site_load in enumerate(load_curve):
        headroom_hint = max(0.0, PANEL_CAPACITY_KW - site_load)
        proposal = toy_strategist_propose(site_load, headroom_hint)

        outcome, ccs, reason, detail = gate.check(
            proposal["setpoint"], last_rate, proposal["confidence"],
            proposal["setpoint"],  # "indoor_temp" slot -> proposed_kw, per capacity_violation_fn contract
            0.0,                    # unused humidity slot
            site_load,              # "t_out" slot -> current site load (gets EWMA-smoothed by the gate)
            unit="kW",
        )

        if outcome == "APPROVED":
            last_rate = proposal["setpoint"]
            approved_count += 1
        elif outcome == "HOLD":
            hold_count += 1
        else:
            rejected_count += 1

        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "step": step, "site_load_kw": round(site_load, 3),
                "proposed_kw": proposal["setpoint"], "applied_kw": last_rate,
                "outcome": outcome, "ccs": round(ccs, 3), "reason": reason,
            }) + "\n")

    total = approved_count + rejected_count + hold_count
    print("--- EV CHARGING DEMO (Tier 3 generalization) ---")
    print(f"Steps: {total}")
    print(f"Approved: {approved_count} ({approved_count/total*100:.1f}%)")
    print(f"Held:     {hold_count} ({hold_count/total*100:.1f}%)")
    print(f"Rejected: {rejected_count} ({rejected_count/total*100:.1f}%)  <- capacity violations the gate caught")
    print(f"\nLog written to {LOG_PATH}")
    print("This ran the SAME SentinelGate class and the SAME compute_ccs() "
          "scoring used for HVAC (core/sentinel_gate.py), with only the "
          "violation_fn swapped for a real panel-capacity check -- an "
          "independent log, from a different domain, produced by the same gate.")


if __name__ == "__main__":
    run_demo()
