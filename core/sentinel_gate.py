"""
core/sentinel_gate.py

Tier 3 note (Improvement Blueprint): the blueprint's generic-domain demo
assumes SentinelGate.check() can be reused verbatim by swapping the
comfort band. That's no longer literally true here -- this gate scores
PMV thermal comfort, not a min/max band, so bolting an EV charge rate
into calculate_pmv() would produce a number but not a meaningful one.

Instead, the domain-agnostic part (HOLD/hysteresis + the CCS weighted
score: violation, rate_penalty, llm_confidence, override_rate, carbon)
is factored out of the PMV-specific part (how "violation_severity" gets
computed). SentinelGate takes an optional `violation_fn` -- default is
the existing PMV calculation, unchanged in behavior. A different domain
(see demos/ev_charging_demo.py) supplies its own violation_fn and gets
the exact same HOLD logic and CCS math, honestly, instead of a
relabeled thermal-comfort score.

--- Fixes applied after the Round-2 log audit -----------------------------

1. `carbon` weight was dead: compute_ccs() always used its hardcoded
   default (carbon_score=0.8) because check() never received an actual
   carbon-intensity value from the caller. It's now a real parameter of
   check(), so main.py can pass a numeric score derived from
   bms_mcp.tools.carbon_intensity_level() (see carbon_score_from_level()
   there) and the weight actually varies the score.

2. `override_rate` weight was declared in building_policy.yaml but never
   referenced anywhere in compute_ccs(), so the 5 declared weights only
   summed to 0.90 in practice. It's now backed by a real signal: a
   rolling window of this gate's own recent APPROVED/REJECTED outcomes.
   A gate that's been getting overridden (rejected) a lot recently scores
   lower on this term than one that's been approving cleanly -- i.e. it's
   a live "how trustworthy have my own recent decisions been" signal, not
   a relabeling of anything else already in the formula. Weight is looked
   up with .get(..., 0.0) so callers (like the EV demo) that don't
   declare an `override_rate` weight are unaffected.

3. The hard `violation_severity == 0` approval clause meant
   calibrate_ccs_sweep.py's threshold-only sweep could never explain (or
   fix) long FAILSAFE streaks during e.g. a hot spell -- a proposal 0.01
   over the PMV band was rejected no matter what ccs_threshold was set
   to. That clause is now `violation_severity <= self.max_violation_severity`,
   config-driven via policy["gate"]["max_violation_severity"] (defaults to
   0.0, i.e. IDENTICAL behavior to before unless you explicitly change
   it). This makes "how strict is comfort enforcement" an actual tunable
   dimension scripts/calibrate_ccs_sweep.py can sweep instead of a silent
   hardcoded cliff -- see that script's --violation-tolerances flag.
"""

from collections import deque
import math

from core.comfort import calculate_pmv, PMVOutOfRangeError
from core.seasonality import violates_baseline_direction


def compute_ccs(weights, violation_severity, rate_penalty, llm_conf,
                 carbon_score=0.8, override_score=1.0):
    """The domain-agnostic Confidence-to-Commit Score. Shared by every
    SentinelGate instance regardless of what violation_fn is plugged in.

    carbon_score / override_score default to values that reproduce the
    OLD (buggy) behavior if a caller doesn't pass them -- but check()
    below always passes real ones now. Both weight lookups use .get(...)
    so policies that don't declare "override_rate" (e.g. the EV demo's
    EV_POLICY) simply score 0 on that term instead of raising a KeyError.
    """
    return (weights["violation"] * (1 - min(violation_severity, 1.0)) +
            weights["rate_penalty"] * (1 - rate_penalty) +
            weights["llm_confidence"] * float(llm_conf) +
            weights.get("override_rate", 0.0) * float(override_score) +
            weights.get("carbon", 0.0) * float(carbon_score))


class SentinelGate:
    def __init__(self, policy: dict, violation_fn=None, initial_setpoint: float = 22.0):
        """
        violation_fn: optional callable (proposed, indoor_temp, humidity, t_out_ewma)
            -> (violation_severity: float in [0, 1+], detail: dict, label: str)
            Defaults to the PMV thermal-comfort calculation below.
            `detail`/`label` are only used for the human-readable reason string.
        initial_setpoint: starting "last active" value for the hysteresis
            dwell logic -- 22.0C for HVAC; a domain like EV charging should
            pass its own sane starting value instead of reusing a temperature.
        """
        self.policy = policy
        self.ccs_threshold = float(policy["gate"]["ccs_threshold"])
        self.w = policy["gate"]["weights"]
        self.violation_fn = violation_fn or self._pmv_violation

        # Fix #3: was a hardcoded `violation_severity == 0` check. Now
        # config-driven; default 0.0 keeps old behavior unchanged.
        self.max_violation_severity = float(policy["gate"].get("max_violation_severity", 0.0))

        # Structural guardrail (opt-in, default off -- see
        # config/building_policy.yaml's gate.enforce_baseline_direction
        # comment and core/seasonality.violates_baseline_direction()).
        self.enforce_baseline_direction = bool(policy["gate"].get("enforce_baseline_direction", False))
        self.baseline_direction_tolerance_c = float(policy["gate"].get("baseline_direction_tolerance_c", 0.1))

        # Fix #2: rolling approve/reject history backing the
        # "override_rate" CCS term. Window length is config-driven
        # (policy["gate"]["override_window"], default 20 decisions);
        # HOLD outcomes don't count -- they never reach a scored decision.
        override_window = int(policy["gate"].get("override_window", 20))
        self._decision_history = deque(maxlen=max(1, override_window))

        # BUGFIX (audit finding: reported crashes ~15min into an AI run,
        # runs looking "frozen" with the step count barely moving, and
        # overall energy cost coming in ABOVE baseline instead of below):
        # config/building_policy.yaml declares a `safety:` block
        # (temp_min_c/temp_max_c/max_delta_c_per_step) under a "SAFETY &
        # COMFORT" heading, but a repo-wide search turns up ZERO reads of
        # it anywhere -- the same class of dead-config bug this file's
        # docstring already documents for `carbon` and `override_rate`
        # (both of which WERE wired up), except this one is the actual
        # hard safety clamp and it shipped completely inert. The only
        # thing standing between a raw LLM proposal and the actuator was
        # the soft PMV/CCS score -- nothing hard-stopped an extreme
        # setpoint or a large single-tick jump. A big jump forces
        # EnergyPlus to subdivide the zone timestep into many extra HVAC
        # system timesteps to converge (see energyplus_bridge.py's
        # docstring on this) -- that both burns real energy recovering
        # from the swing (a plausible source of AI-costs-more-than-
        # baseline results) and can make a run look stalled (each tick
        # takes far longer in real time), and enough severe-error/
        # convergence trouble can trip EnergyPlus's own abort threshold
        # (a plausible source of a crash partway through, not at step 0).
        # Defaults (inf/inf/inf) keep old behavior unchanged for any
        # policy that omits `safety` entirely.
        safety = policy.get("safety", {}) or {}
        self.temp_min_c = float(safety.get("temp_min_c", -math.inf))
        self.temp_max_c = float(safety.get("temp_max_c", math.inf))
        self.max_delta_c_per_step = float(safety.get("max_delta_c_per_step", math.inf))

        # State memory for stability
        self._t_out_ewma = None
        self._last_active_setpoint = float(initial_setpoint)
        self._steps_since_change = 0

    def _update_ewma(self, t_out):
        alpha = self.policy["comfort"]["outdoor_temp_ewma_alpha"]
        if self._t_out_ewma is None: self._t_out_ewma = t_out
        else: self._t_out_ewma = (alpha * t_out) + ((1 - alpha) * self._t_out_ewma)
        return self._t_out_ewma

    def _pmv_violation(self, proposed, indoor_temp, humidity, t_out_ewma):
        try:
            pmv_val, clo = calculate_pmv(indoor_temp, humidity, t_out_ewma)
            # BUGFIX (dead config -- see chat): this used to hardcode 0.5
            # twice instead of reading comfort.pmv_band from
            # config/building_policy.yaml -- the two happened to match by
            # coincidence, but editing pmv_band in the YAML silently did
            # nothing. Now it actually controls gate strictness.
            pmv_band = self.policy.get("comfort", {}).get("pmv_band", 0.5)
            severity = max(0.0, (abs(pmv_val) - pmv_band) / pmv_band)
            return severity, {"pmv": pmv_val, "clo": clo}, f"PMV {pmv_val:.2f} (Clo {clo:.2f})"
        except Exception:
            return 1.0, {"pmv": 0, "clo": 1.0}, "PMV calc error"  # safety first on math error

    def quick_check(self, t_in):
        """Rework Blueprint 5.1 -- Phase 0: the always-on fast-loop safety
        net. Deliberately NOT the same thing as the hard safety clamp
        inside check() above (which validates a PROPOSED setpoint before
        it's committed). This validates the CURRENTLY MEASURED indoor
        temperature, and is meant to be called on every physical
        EnergyPlus step regardless of strategist cadence -- see
        main.py's ProjectEnvelope.decide(), which now calls this before
        its `step % cadence_steps` gate, not after.

        Why this is needed even though check()'s clamp exists: that
        clamp only runs when the slow loop actually fires (a cadence
        tick). Between ticks, the old code path returned
        `self.last_setpoint, "Holding"` unconditionally -- if reality
        drifted out of bounds while waiting for the next cadence tick
        (a fast weather swing, an equipment fault, a stale trajectory),
        NOTHING was watching until the next scheduled LLM call, which
        could be up to `cadence_steps` away. This closes that window:
        the building is now supervised every physical step, not just at
        reasoning time, without spending a single LLM call to do it --
        it's a pure hard-bounds comparison, deliberately cheap enough to
        run unconditionally.

        Intentionally simple and cheap on purpose: no PMV, no CCS
        scoring, no gate weights, no hysteresis/dwell state touched, and
        does NOT feed self._decision_history (this isn't a scored
        approve/reject decision about a proposal -- it's closer to a
        smoke detector than a thermostat, and shouldn't affect the
        "override_rate" trust signal check() uses for real decisions).

        Returns (tripped: bool, reason: str | None). tripped=False on
        the common case (temp_min_c/temp_max_c default to +/-inf when
        `safety` is omitted from policy -- see __init__ -- so this is a
        guaranteed no-op unless a real safety band is configured, same
        opt-in-by-config posture as the rest of this file's safety
        features).
        """
        if t_in < self.temp_min_c:
            return True, (f"QUICK-CHECK TRIP: measured indoor temp {t_in:.2f}C is below "
                          f"the hard safety floor {self.temp_min_c:.1f}C.")
        if t_in > self.temp_max_c:
            return True, (f"QUICK-CHECK TRIP: measured indoor temp {t_in:.2f}C is above "
                          f"the hard safety ceiling {self.temp_max_c:.1f}C.")
        return False, None

    @property
    def recent_override_rate(self):
        """Fraction of the last N scored decisions (approve/reject, not
        HOLD) that were REJECTED. 0.0 if no history yet (optimistic
        default -- a fresh gate hasn't overridden anything)."""
        if not self._decision_history:
            return 0.0
        rejects = sum(1 for approved in self._decision_history if not approved)
        return rejects / len(self._decision_history)

    def check(self, proposed, last, llm_conf, indoor_temp, humidity, t_out,
              carbon_score=0.8, unit="C", season=None, baseline_setpoint=None):
        """
        carbon_score: numeric grid-carbon-intensity score in [0, 1], higher
            = more favorable grid conditions. Pass the real value (e.g. via
            bms_mcp.tools.carbon_score_from_level()) -- see fix #1 above.
            Defaults to 0.8 only so any old caller that doesn't pass it
            keeps working, not as an endorsement of a fake constant.
        season / baseline_setpoint: only used by the opt-in baseline-
            direction guardrail (self.enforce_baseline_direction). Callers
            that don't classify a season (e.g. the EV demo, or
            scripts/compare_models.py's replay) can simply omit these --
            the guardrail is a no-op unless both are provided AND the
            policy has enforce_baseline_direction: true.
        """
        t_out_ewma = self._update_ewma(t_out)

        # 0. HARD SAFETY CLAMP (see __init__ comment) -- checked before
        # anything else, unconditionally, whether or not enforce_
        # baseline_direction is on and regardless of hysteresis state.
        # This is a REJECT, not a silent clamp: the correction pass gets
        # a clear reason string to work from, and if that also fails the
        # tick falls through to FailsafeController, which is itself
        # already bounded by policy["failsafe"]'s own low/high/setback.
        if proposed < self.temp_min_c or proposed > self.temp_max_c:
            reason = (f"REJECTED: Safety -- proposed {proposed:.2f}{unit} is outside "
                      f"the hard safety band [{self.temp_min_c:.1f}, {self.temp_max_c:.1f}]{unit}.")
            self._decision_history.append(False)
            return "REJECTED", 0.0, reason, {}

        step_delta = abs(proposed - last)
        if step_delta > self.max_delta_c_per_step:
            reason = (f"REJECTED: Safety -- requested change of {step_delta:.2f}{unit} in one "
                      f"step exceeds max_delta_c_per_step ({self.max_delta_c_per_step:.1f}{unit}).")
            self._decision_history.append(False)
            return "REJECTED", 0.0, reason, {}

        # 1. THE HOLD LOGIC (Stability = Profit)
        delta = abs(proposed - self._last_active_setpoint)
        min_delta = self.policy["hysteresis"]["min_delta_c"]
        min_dwell = self.policy["hysteresis"]["min_dwell_steps"]

        if delta < min_delta and self._steps_since_change < min_dwell:
            self._steps_since_change += 1
            return "HOLD", 1.0, f"HOLD: Change of {delta:.2f}{unit} is too small. Staying steady.", {}

        # 2. VIOLATION CALCULATION (pluggable; PMV by default)
        violation_severity, detail, label = self.violation_fn(proposed, indoor_temp, humidity, t_out_ewma)

        # 3. SCORING (CCS) -- domain-agnostic
        rate_penalty = min(abs(proposed - last) / 2.0, 1.0)
        override_score = 1.0 - self.recent_override_rate  # fix #2: real signal, not unused
        ccs = compute_ccs(self.w, violation_severity, rate_penalty, llm_conf,
                           carbon_score=carbon_score, override_score=override_score)

        # fix #3: tunable tolerance instead of a hardcoded "== 0" cliff.
        approved = ccs >= self.ccs_threshold and violation_severity <= self.max_violation_severity

        # Structural guardrail: even a proposal that clears CCS + comfort
        # can still be the wrong direction relative to baseline (e.g. a
        # winter setpoint warmer than baseline's fixed 22.0C, which can
        # only draw MORE heating energy than baseline, never less). Only
        # active when the caller supplies season/baseline_setpoint AND
        # the policy has opted in -- see config/building_policy.yaml.
        guardrail_tripped = False
        if approved and self.enforce_baseline_direction and season is not None and baseline_setpoint is not None:
            guardrail_tripped = violates_baseline_direction(
                proposed, baseline_setpoint, season, self.baseline_direction_tolerance_c
            )
            if guardrail_tripped:
                approved = False

        if approved:
            self._last_active_setpoint = proposed
            self._steps_since_change = 0
            reason = f"APPROVED: CCS {ccs:.2f} | {label}"
        elif guardrail_tripped:
            reason = (f"REJECTED: Guardrail -- {proposed:.2f}{unit} is the wrong direction "
                      f"vs. {season} baseline {baseline_setpoint:.2f}{unit} (would cost more "
                      f"energy than baseline, not less).")
        else:
            reason = f"REJECTED: Violation ({label}). Propose a value closer to the last approved setpoint."

        # Record this scored decision (not HOLD) for the override_rate window.
        self._decision_history.append(approved)

        return "APPROVED" if approved else "REJECTED", ccs, reason, detail
