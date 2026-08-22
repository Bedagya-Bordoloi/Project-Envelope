"""
mcp/tools.py

The four tools named in the problem statement: get_state, set_hvac,
get_weather, get_carbon_intensity.

carbon_intensity_level() is now the single source of truth for the
simulated carbon signal -- both this tool and main.py's control loop
(which needs the same value to pass into SentinelGate.check()) call it,
so the two can never drift out of sync. Entirely YAML-driven: which
profile is active, how long each level lasts, and what labels exist are
all read from policy["carbon"] -- add a new profile in the YAML and it
works here with no code change.
"""

from dataclasses import dataclass


@dataclass
class ToolContext:
    """Shared handle to whatever the tools need to read/write."""
    bridge: object            # core.energyplus_bridge.EnergyPlusBridge instance
    policy: dict
    carbon_profile: str = "flat_medium"   # kept for backward compat; policy["carbon"] is authoritative


# ---------------------------------------------------------------------------
# Carbon signal -- single source of truth, YAML-driven
# ---------------------------------------------------------------------------
def carbon_intensity_level(step: int, policy: dict) -> str:
    """
    Cycles through policy["carbon"]["profiles"][<active profile>] every
    policy["carbon"]["cycle_steps"] control steps. Add a new profile, or
    change how long each level lasts, purely in the YAML.
    """
    carbon_cfg = (policy or {}).get("carbon", {})
    profile_name = carbon_cfg.get("profile", "flat_medium")
    profiles = carbon_cfg.get("profiles", {"flat_medium": ["Low", "Medium", "High"]})
    levels = profiles.get(profile_name, ["Low", "Medium", "High"])
    cycle_steps = int(carbon_cfg.get("cycle_steps", 30))
    if not levels or cycle_steps <= 0:
        return "Medium"
    idx = (step // cycle_steps) % len(levels)
    return levels[idx]


def carbon_score_from_level(level: str, policy: dict) -> float:
    """
    Fix (Round-2 audit finding #1): SentinelGate.compute_ccs() used to be
    called with NO carbon value at all, so it silently fell back to its
    hardcoded default (0.8) on every single tick -- the 20%-weighted
    "carbon intensity" term in the CCS score never actually varied.

    This is the single source of truth for turning the simulated
    Low/Medium/High carbon label into the numeric [0, 1] score
    SentinelGate.check()'s carbon_score parameter expects. Higher score =
    more favorable grid conditions (lower emissions) = more favorable to
    the CCS term. Both main.py (which needs the numeric score for
    gate.check()) and anything else that wants it call this, so there's
    exactly one place that defines what "Low/Medium/High" numerically
    means -- no drift between them.

    Mapping is policy-driven via policy["carbon"]["score_map"]; falls back
    to a sane default if not set.
    """
    carbon_cfg = (policy or {}).get("carbon", {})
    score_map = carbon_cfg.get("score_map", {"Low": 1.0, "Medium": 0.6, "High": 0.2})
    return float(score_map.get(level, score_map.get("Medium", 0.6)))


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def get_state(ctx: ToolContext) -> dict:
    """Return the latest sensed state from the live EnergyPlus instance."""
    b = ctx.bridge
    return {
        "indoor_temp_c": b.last_indoor_temp,
        "outdoor_temp_c": b.last_outdoor_temp,
        "humidity_pct": getattr(b, "last_humidity", None),
        "last_setpoint_c": b.last_setpoint,
        "cumulative_kwh": round(getattr(b, "cumulative_kwh", 0.0), 4),
        "step": b.step_counter,
    }


def set_hvac(ctx: ToolContext, setpoint_c: float, confidence: float = 0.8,
             reason: str = "", trajectory: list = None) -> dict:
    """
    Stage a new HVAC cooling setpoint proposal for Sentinel Gate review.
    This is the Strategist's ACTION tool -- calling it is how the model
    commits to a final decision for this cadence tick. It does NOT bypass
    the Sentinel Gate: main.py's control loop is what actually calls
    set_actuator_value after the gate approves it.

    Rework Blueprint 5.3: `trajectory` (optional) is a short list of
    {"offset_min": .., "setpoint_c": ..} anchors -- see
    agents/strategist.py's _build_prompt()/decide() for the real parsing
    path main.py's control loop actually uses (this MCP-exposed function
    is a separate, external-tool-facing entry point via bms_mcp/server.py
    and stages only the immediate setpoint on ctx.bridge; it does not
    itself drive the fast loop's interpolation).
    """
    ctx.bridge.pending_setpoint = float(setpoint_c)
    return {
        "staged_setpoint_c": float(setpoint_c),
        "confidence": float(confidence),
        "reason": reason,
        "trajectory": trajectory,
        "status": "pending_gate_review",
    }


def get_weather(ctx: ToolContext, hours: int = 3) -> dict:
    """
    Forward look-ahead using the simulation's OWN known future weather from
    the loaded .epw file -- a deterministic proxy for a live forecast, not
    live data.
    """
    window = ctx.bridge.get_forward_weather(hours=hours)
    return {
        "hours": hours,
        "outdoor_temp_forecast_c": window,
        "source": "epw_lookahead (simulated future, not a live forecast)",
    }


def get_carbon_intensity(ctx: ToolContext) -> dict:
    """
    Explicitly simulated grid carbon-intensity signal, not a live API.
    Uses carbon_intensity_level() so this always matches whatever the
    gate itself is scoring against -- no drift between the Strategist's
    view and the Governor's view.
    """
    step = getattr(ctx.bridge, "step_counter", 0)
    level = carbon_intensity_level(step, ctx.policy)
    profile = (ctx.policy or {}).get("carbon", {}).get("profile", ctx.carbon_profile)
    return {"carbon_intensity": level, "source": "simulated", "profile": profile}


# ---------------------------------------------------------------------------
# JSON-schema tool definitions (Groq function-calling compatible)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": "Get the current sensed indoor/outdoor temperature, humidity, last setpoint, and cumulative energy use.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_hvac",
            "description": (
                "Submit your FINAL decision: a short trajectory of HVAC "
                "cooling setpoint anchors spanning the next few hours "
                "(Rework Blueprint 5.3 -- MPC-style receding horizon, not "
                "a single instantaneous value), your confidence in it, "
                "and a short reason. Call this exactly once, after you've "
                "gathered whatever state/weather/carbon context you need. "
                "This is how you commit to an action. 'setpoint_c' is your "
                "IMMEDIATE target (offset_min=0) -- this is the only value "
                "Sentinel Gate actually scores/approves this tick. "
                "'trajectory' anchors beyond it are advisory guidance for "
                "how you'd continue if nothing changes before your next "
                "real decision."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "setpoint_c": {"type": "number", "description": "Proposed IMMEDIATE cooling setpoint in Celsius (offset_min=0 of the trajectory)"},
                    "trajectory": {
                        "type": "array",
                        "description": (
                            "Optional (but preferred) list of 3-4 anchors spanning the "
                            "next few hours, e.g. [{\"offset_min\":0,\"setpoint_c\":21.0}, "
                            "{\"offset_min\":60,\"setpoint_c\":20.5}, "
                            "{\"offset_min\":120,\"setpoint_c\":20.0}, "
                            "{\"offset_min\":180,\"setpoint_c\":20.5}]. The first anchor "
                            "MUST be offset_min:0 and should match setpoint_c above."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "offset_min": {"type": "number", "description": "Minutes from now"},
                                "setpoint_c": {"type": "number", "description": "Setpoint in Celsius at this offset"},
                            },
                        },
                    },
                    "confidence": {"type": "number", "description": "Your confidence in this proposal, 0.0-1.0", "default": 0.8},
                    "reason": {"type": "string", "description": "One short sentence explaining the proposal"},
                },
                "required": ["setpoint_c"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the next N hours of outdoor temperature from the simulation's known future weather (a forecast proxy, not live data).",
            "parameters": {
                "type": "object",
                "properties": {
                    "hours": {"type": "integer", "description": "Look-ahead window in hours", "default": 3}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_carbon_intensity",
            "description": "Get the current simulated grid carbon-intensity level (Low/Medium/High).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

DISPATCH = {
    "get_state": get_state,
    "set_hvac": set_hvac,
    "get_weather": get_weather,
    "get_carbon_intensity": get_carbon_intensity,
}


def call_tool(name: str, ctx: ToolContext, **kwargs) -> dict:
    """Dispatch a tool call by name. Raises KeyError if the tool is unknown."""
    if name not in DISPATCH:
        raise KeyError(f"Unknown tool: {name}")
    return DISPATCH[name](ctx, **kwargs)
