"""
core/trigger_engine.py

"""

DAY_START_HOUR = 6.0
NIGHT_START_HOUR = 22.0


def _schedule_bucket(hour_of_day):
    """Coarse day/night bucket standing in for an occupancy signal -- see
    module docstring. Returns "DAY" or "NIGHT"."""
    if DAY_START_HOUR <= hour_of_day < NIGHT_START_HOUR:
        return "DAY"
    return "NIGHT"


class TriggerEngine:
    """Stateful per-orchestrator (one instance per ProjectEnvelope, i.e.
    per zone in --multizone mode, same lifetime as its SentinelGate/
    DecisionCache) -- deliberately NOT a module-level singleton, unlike
    agents/strategist.py's _RateLimiter: trigger state (t_in/t_out/season
    at last decision) is meaningfully different per zone, not a shared
    account-level budget."""

    def __init__(self, policy):
        trigger_cfg = (policy or {}).get("strategist", {}).get("trigger", {}) or {}
        cadence_steps = int((policy or {}).get("strategist", {}).get("cadence_steps", 12))
        self.cadence_steps = max(1, cadence_steps)  # fallback ceiling, always present

        self.enabled = bool(trigger_cfg.get("enabled", False))
        self.deviation_deadband_c = float(trigger_cfg.get("deviation_deadband_c", 0.4))
        self.outdoor_delta_threshold_c = float(trigger_cfg.get("outdoor_delta_threshold_c", 1.5))
        # Never let a misconfigured/omitted max_staleness_steps make this
        # LESS responsive than the pre-existing fixed cadence was --
        # cap it at a generous multiple of cadence_steps rather than
        # trusting an arbitrary YAML value blindly.
        configured_staleness = int(trigger_cfg.get("max_staleness_steps", self.cadence_steps * 8))
        self.max_staleness_steps = (
            min(configured_staleness, self.cadence_steps * 8)
            if configured_staleness > 0 else self.cadence_steps
        )
        self.debounce_steps = int(trigger_cfg.get("debounce_steps", 2))

        # State as of the last time the slow loop actually ran -- None
        # until the very first decision (see evaluate() below).
        self._last_step = None
        self._last_t_in = None
        self._last_t_out = None
        self._last_schedule_bucket = None
        self._last_season = None
        self._last_fire_step = None  # for debounce; distinct from note_decision

    def note_decision(self, step, t_in, t_out, hour_of_day, season):
        """Called by main.py immediately after the slow loop actually
        executes (real LLM call or decision-cache hit), so the NEXT
        evaluate() call measures drift from THIS point forward. Must
        NOT be called on a step where evaluate() returned False (that
        would erase the very drift the next evaluate() needs to see)."""
        self._last_step = step
        self._last_t_in = t_in
        self._last_t_out = t_out
        self._last_schedule_bucket = _schedule_bucket(hour_of_day)
        self._last_season = season

    def evaluate(self, step, t_in, t_out, hour_of_day, season):
        """Returns (should_fire: bool, reason: str | None).

        reason is one of "initial_decision" / "deviation" /
        "forecast_shift" / "schedule_boundary" / "max_staleness" /
        "cadence_ceiling" -- this is also the canonical trigger-reason
        vocabulary the Blueprint 5.6 explainability column (Phase 5)
        surfaces, so no renaming will be needed once that phase lands.

        Never raises. The very first call (no prior decision yet)
        always fires, matching the old `step % cadence_steps` gate's
        behavior at step 0 -- the system must get one real decision
        before there's anything to measure drift against. A disabled
        engine (strategist.trigger.enabled: false) reproduces the OLD
        fixed-cadence behavior exactly, so this can be toggled without
        touching main.py.
        """
        if self._last_step is None:
            return True, "initial_decision"

        if not self.enabled:
            if step % self.cadence_steps == 0:
                return True, "cadence_ceiling"
            return False, None

        steps_since = step - self._last_step

        # Debounce: suppress a re-firing within debounce_steps of the
        # last FIRING (not the last note_decision call -- a "Holding"
        # step never calls note_decision), UNLESS waiting would itself
        # violate max_staleness -- that ceiling is never subject to
        # debounce suppression.
        if (self._last_fire_step is not None
                and step - self._last_fire_step < self.debounce_steps
                and steps_since < self.max_staleness_steps):
            return False, None

        if steps_since >= self.max_staleness_steps:
            reason = "max_staleness"
        elif abs(t_in - self._last_t_in) > self.deviation_deadband_c:
            reason = "deviation"
        elif abs(t_out - self._last_t_out) > self.outdoor_delta_threshold_c:
            reason = "forecast_shift"
        elif (_schedule_bucket(hour_of_day) != self._last_schedule_bucket
              or season != self._last_season):
            reason = "schedule_boundary"
        elif step % self.cadence_steps == 0:
            # Backstop, not the primary mechanism anymore -- see module
            # docstring. Keeps a bound on reasoning latency even with a
            # loosely-tuned trigger config.
            reason = "cadence_ceiling"
        else:
            return False, None

        self._last_fire_step = step
        return True, reason
