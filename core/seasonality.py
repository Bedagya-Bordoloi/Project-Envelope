"""
core/seasonality.py

Single source of truth for two things that were previously hardcoded in
more than one place and had drifted out of sync with config/building_policy.yaml:

1. Season classification (winter/summer/shoulder). This used to be a
   literal `t_out < 14` / `t_out > 22` inline in
   agents/strategist.py's _build_prompt(), while
   config/building_policy.yaml's seasonality.winter_limit/summer_limit
   sat there completely unread ("dead config" -- winter_limit had
   drifted to a stale 12.0 that was never actually exercised, the same
   class of bug as the earlier dead carbon-weight fix).

2. The baseline setpoint the AI is supposed to beat. This was hardcoded
   as the literal string "22.0C" in the Strategist's prompt instead of
   reading policy["baseline"]["schedule_setpoint_c"] -- the one value
   BaselineController itself actually uses.

Both agents/strategist.py's prompt-building AND core/sentinel_gate.py's
optional baseline-direction guardrail now import from here, so the two
can no longer independently drift apart the way winter_limit did.
"""

WINTER = "WINTER"
SUMMER = "SUMMER"
SHOULDER = "SHOULDER"


def classify_season(t_out, policy):
    """Returns WINTER / SUMMER / SHOULDER using the SAME thresholds the
    policy YAML declares (seasonality.winter_limit / summer_limit)
    instead of each caller hardcoding its own copy that can silently
    drift out of sync.

    Defaults (14.0 / 22.0) match the Strategist's previous hardcoded
    behavior, so a policy dict that omits `seasonality` entirely still
    classifies exactly as before.
    """
    limits = policy.get("seasonality", {}) if policy else {}
    winter_limit = float(limits.get("winter_limit", 14.0))
    summer_limit = float(limits.get("summer_limit", 22.0))
    if t_out < winter_limit:
        return WINTER
    if t_out > summer_limit:
        return SUMMER
    return SHOULDER


def get_baseline_setpoint(policy):
    """The fixed schedule setpoint BaselineController runs -- i.e. the
    real number the AI is trying to beat, instead of a hardcoded "22.0C"
    string duplicated in a prompt.
    """
    baseline = policy.get("baseline", {}) if policy else {}
    return float(baseline.get("schedule_setpoint_c", 22.0))


def violates_baseline_direction(proposed, baseline_setpoint, season, tolerance_c=0.1):
    """Guardrail check -- opt-in via policy["gate"]["enforce_baseline_direction"]
    (see core/sentinel_gate.py). Returns True if `proposed` moves the
    setpoint the WRONG direction relative to baseline for the given
    season, i.e. a direction that can only cost MORE energy than
    baseline, never less, regardless of what the LLM's reasoning
    claimed:

      WINTER: a lower setpoint means less heating draw (cheaper). The
        wrong direction is proposing something WARMER than baseline --
        that can only pull more heating energy than baseline would.
      SUMMER: a higher setpoint means less cooling draw (cheaper). The
        wrong direction is proposing something COOLER than baseline.
      SHOULDER: no hard direction constraint. The Strategist is already
        instructed to match baseline here, and there's no single
        "cheaper direction" in the shoulder season the way there is in
        winter/summer -- so this always returns False for SHOULDER.

    `tolerance_c` avoids tripping on trivial float noise right at the
    baseline value (default 0.1C).
    """
    if season == WINTER:
        return proposed > baseline_setpoint + tolerance_c
    if season == SUMMER:
        return proposed < baseline_setpoint - tolerance_c
    return False
