import os
import json
import time
import re
import threading
from groq import Groq, RateLimitError
from openai import OpenAI as _OpenAIClient, RateLimitError as _OpenAIRateLimitError
from bms_mcp.tools import TOOL_SCHEMAS, call_tool
from core.seasonality import classify_season, get_baseline_setpoint
from core.provider_pool import ProviderPool

_RATE_LIMIT_EXCEPTIONS = (RateLimitError, _OpenAIRateLimitError)

MAX_INTERNAL_RETRIES = 3

TRAJECTORY_ANCHOR_COUNT = 4
TRAJECTORY_SPAN_MIN = 180.0  

_RATE_LIMITER_LOCK = threading.Lock()
_shared_rate_limiter = None


class _RateLimiter:
    """Thread-safe minimum-interval pacer. wait() blocks (if needed) so
    that no two calls through this limiter are closer together than
    60/max_per_minute seconds, then returns."""

    def __init__(self, max_per_minute):
        self._min_interval = 60.0 / max(1, max_per_minute)
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self):
        with self._lock:
            now = time.perf_counter()
            delay = self._next_allowed - now
            if delay > 0:
                time.sleep(delay)
                now = time.perf_counter()
            self._next_allowed = now + self._min_interval


def _get_shared_rate_limiter(max_per_minute):
    global _shared_rate_limiter
    with _RATE_LIMITER_LOCK:
        if _shared_rate_limiter is None:
            _shared_rate_limiter = _RateLimiter(max_per_minute)
        return _shared_rate_limiter


RETRY_BACKOFF_S = [0.5, 1.5, 3.0]      
RATE_LIMIT_BACKOFF_S = [2.0, 5.0, 10.0] 

MAX_HONORED_RETRY_AFTER_S = 8.0


_DAILY_QUOTA_MARKERS = (
    "per day", "requests per day", "tokens per day",
    "rpd", "tpd", "daily limit", "daily quota",
)


def _looks_like_daily_quota_error(exc):
    """Best-effort text match -- see module comment above. False
    (default to the old short-cooldown behavior) if nothing matches, so
    a provider whose error text doesn't fit this pattern is never held
    back longer than the genuine per-minute case warrants."""
    text = str(exc).lower()
    try:
        body = getattr(exc, "response", None)
        if body is not None:
            text += " " + str(getattr(body, "text", "")).lower()
    except Exception:
        pass
    return any(marker in text for marker in _DAILY_QUOTA_MARKERS)

DECIDE_WALL_CLOCK_BUDGET_S = 30.0


def _retry_after_seconds(exc):
    """Best-effort extraction of the API's own Retry-After hint from a
    RateLimitError's underlying HTTP response, capped at
    MAX_HONORED_RETRY_AFTER_S (see comment above -- a real 429's
    Retry-After can be minutes long, and this loop runs synchronously
    inside the EnergyPlus callback, so it must never be allowed to block
    the sim indefinitely). Returns None if not present/parseable --
    callers fall back to RATE_LIMIT_BACKOFF_S.
    """
    try:
        header = exc.response.headers.get("retry-after")
        if header is not None:
            return min(max(0.0, float(header)), MAX_HONORED_RETRY_AFTER_S)
    except Exception:
        pass
    return None

class Strategist:
    def __init__(self, model="openai/gpt-oss-20b", tool_context=None, policy=None):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set - check your .env file.")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.tool_context = tool_context

        self.policy = policy
 
        max_rpm = (policy or {}).get("strategist", {}).get("max_requests_per_minute", 25)
        self._rate_limiter = _get_shared_rate_limiter(max_rpm)
        self.last_latency_s = 0
        self.last_tool_calls = []


        secondary_cfg = (policy or {}).get("strategist", {}).get("secondary_provider", {}) or {}
        self.secondary_client = None
        self.secondary_model = None
        if secondary_cfg.get("enabled", False):
            key_env = secondary_cfg.get("api_key_env", "SECONDARY_LLM_API_KEY")
            secondary_key = os.environ.get(key_env)
            base_url = secondary_cfg.get("base_url")
            if secondary_key and base_url:
                self.secondary_client = _OpenAIClient(api_key=secondary_key, base_url=base_url)
                self.secondary_model = secondary_cfg.get("model", self.model)
            else:
                print(f"[Strategist] secondary_provider.enabled is true but {key_env} "
                      "or base_url is missing -- running with primary provider only.")


        self.provider_pool = self._build_provider_pool(policy, max_rpm)


        self.last_used_fallback = False
        self.last_error = None

    def _build_provider_pool(self, policy, primary_rpm):
        """Rework Blueprint 5.4: assembles the ProviderPool from whatever
        is configured in `policy`. The primary (Groq, self.client) is
        ALWAYS included -- it's a hard requirement (GROQ_API_KEY is
        checked at the top of __init__), so the pool never ends up
        empty. secondary_provider (Gemini, if enabled and its key/
        base_url are present -- see above) and any entries under
        policy["strategist"]["provider_pool"]["additional_providers"]
        (config/building_policy.yaml) are appended if THEIR
        `enabled: true` and their own api_key_env is set. A misconfigured
        or disabled additional provider is skipped with a warning, same
        fail-safe convention as secondary_provider -- it never crashes
        the run, it just isn't added to the pool.
        """
        strategist_cfg = (policy or {}).get("strategist", {}) or {}
        providers = [{
            "name": "groq",
            "client": self.client,
            "model": self.model,
            "rpm": primary_rpm,

            "rpd": strategist_cfg.get("rpd", 1000),
        }]

        secondary_cfg = (policy or {}).get("strategist", {}).get("secondary_provider", {}) or {}
        if self.secondary_client is not None:
            providers.append({
                "name": secondary_cfg.get("name", "gemini"),
                "client": self.secondary_client,
                "model": self.secondary_model,
                "rpm": secondary_cfg.get("rpm", 15),

                "rpd": secondary_cfg.get("rpd", 0),
            })

        pool_cfg = (policy or {}).get("strategist", {}).get("provider_pool", {}) or {}
        for entry in pool_cfg.get("additional_providers", []) or []:
            name = entry.get("name")
            if not name:
                print("[Strategist] provider_pool.additional_providers entry missing "
                      "'name' -- skipping.")
                continue
            if not entry.get("enabled", False):
                continue
            key_env = entry.get("api_key_env")
            base_url = entry.get("base_url")
            model = entry.get("model")
            api_key = os.environ.get(key_env) if key_env else None
            if not (api_key and base_url and model):
                print(f"[Strategist] provider_pool provider '{name}' is enabled but "
                      f"{key_env or 'api_key_env'}/base_url/model is missing -- "
                      "skipping this provider (pool continues without it).")
                continue
            providers.append({
                "name": name,
                "client": _OpenAIClient(api_key=api_key, base_url=base_url),
                "model": model,
                "rpm": entry.get("rpm", 25),

                "rpd": entry.get("rpd", 0),
            })

        return ProviderPool(providers)

    LOOKAHEAD_SPIKE_DELTA_C = 4.0

    def _check_lookahead(self, outdoor_temp, forecast_window):
        """
        Real implementation (blueprint 1.3): scans the forecast window
        (the next few hours of outdoor drybulb from get_forward_weather())
        for a swing large enough to justify pre-empting it now instead of
        reacting after it lands. Returns (triggered: bool, direction: str
        or None, magnitude_c: float) and also sets self.last_lookahead_*
        for main.py's logging.
        """
        self.last_lookahead_triggered = False
        self.last_lookahead_direction = None
        self.last_lookahead_magnitude_c = 0.0

        if not forecast_window:
            return False, None, 0.0

        deltas = [f - outdoor_temp for f in forecast_window]
        max_rise = max(deltas) if deltas else 0.0
        max_fall = min(deltas) if deltas else 0.0

        if max_rise >= self.LOOKAHEAD_SPIKE_DELTA_C:
            self.last_lookahead_triggered = True
            self.last_lookahead_direction = "rising"
            self.last_lookahead_magnitude_c = round(max_rise, 2)
        elif abs(max_fall) >= self.LOOKAHEAD_SPIKE_DELTA_C:
            self.last_lookahead_triggered = True
            self.last_lookahead_direction = "falling"
            self.last_lookahead_magnitude_c = round(abs(max_fall), 2)

        return self.last_lookahead_triggered, self.last_lookahead_direction, self.last_lookahead_magnitude_c

    def _build_prompt(self, t_in, t_out, forecast, carbon, internal_error=None, correction_context=None):
        """Cognitive Strategy Engine: Determines the financial path based on seasonal physics."""

        baseline_setpoint = get_baseline_setpoint(self.policy) if self.policy is not None else 22.0
        season_label = classify_season(t_out, self.policy) if self.policy is not None else (
            "WINTER" if t_out < 14 else ("SUMMER" if t_out > 22 else "SHOULDER")
        )


        comfort_ranges = (self.policy.get("seasonality", {}).get("comfort_ranges", {})
                           if self.policy is not None else {})
        winter_lo, winter_hi = comfort_ranges.get("winter", [21.0, 21.8])
        summer_lo, summer_hi = comfort_ranges.get("summer", [23.5, 25.5])

        if season_label == "WINTER":
            season, advice = "WINTER", (
                f"Lowering temp saves heating. TARGET: {winter_lo:.1f}C to {winter_hi:.1f}C.")
        elif season_label == "SUMMER":
            season, advice = "SUMMER", (
                f"PROFIT RULE: Raising temp saves AC energy. Aim for {summer_lo:.1f}C to {summer_hi:.1f}C.")
        else:
            season, advice = "SHOULDER (Mild)", f"MATCH BASELINE ({baseline_setpoint:.1f}C) to avoid wasting energy on setpoint jitter."

        triggered, direction, magnitude = self._check_lookahead(t_out, forecast)
        if triggered and direction == "rising":
            lookahead_note = (
                f"[FORECAST ALERT] Outdoor temp is forecast to climb ~{magnitude:.1f}C over the "
                f"next {len(forecast)}h ({forecast}). PRE-COOL NOW: propose a lower setpoint ahead "
                f"of the spike rather than waiting to react to it."
            )
        elif triggered and direction == "falling":
            lookahead_note = (
                f"[FORECAST ALERT] Outdoor temp is forecast to drop ~{magnitude:.1f}C over the "
                f"next {len(forecast)}h ({forecast}). PRE-HEAT NOW: propose a setpoint that gets "
                f"ahead of the drop rather than waiting to react to it."
            )
        elif forecast:
            lookahead_note = f"[FORECAST] Next {len(forecast)}h outdoor: {forecast} (no significant swing -- hold current strategy)."
        else:
            lookahead_note = "[FORECAST] Not available this tick."

        prompt = f"""You are the 'Eco-Loop' BMS Strategist. GOAL: Beat {baseline_setpoint:.1f}C Baseline profit.

[SITUATION] 
- Outdoor: {t_out:.1f}C | Indoor: {t_in:.1f}C | Season: {season} | Grid carbon: {carbon}
- Strategy: {advice}
- {lookahead_note}
- Stability: Do not change setpoint unless weather or carbon trends shift significantly.

[TASK]
Propose a short TRAJECTORY: {TRAJECTORY_ANCHOR_COUNT} setpoint anchors spanning the next ~{TRAJECTORY_SPAN_MIN / 60.0:.0f}h of the forecast above (Rework Blueprint 5.3 -- a receding-horizon plan, not one instantaneous value), e.g. offsets 0/60/120/180 minutes. The offset_min=0 anchor is your IMMEDIATE proposal -- it is the ONLY anchor Sentinel Gate actually scores/approves this tick. Later anchors are advisory: they smooth the fast control loop's interpolation between now and your next real decision, and will be superseded early if conditions shift before then.
Maximize profit while ensuring 100% Gate Approval on the immediate (offset_min=0) anchor.
You MUST call the 'set_hvac' tool with a 'trajectory' array of {{"offset_min":.., "setpoint_c":..}} objects (offset_min=0 first, matching setpoint_c).

IMPORTANT: Output valid JSON tool calls only. Do not provide setpoints as plain text."""

        if correction_context:
            prompt += (
                f"\n\n[SELF-CORRECTION -- READ CAREFULLY]: Your previous IMMEDIATE "
                f"(offset_min=0) proposal for this tick was REJECTED by the Sentinel "
                f"Gate for this reason: \"{correction_context}\". "
                f"You must propose a DIFFERENT offset_min=0 setpoint that directly "
                f"addresses this rejection (e.g. a smaller change from the last "
                f"approved setpoint, or a value that keeps comfort/PMV within range). "
                f"Do not repeat the same value."
            )

        if internal_error:
            prompt += f"\n\n[RETRY ALERT]: Your previous attempt had an error: {internal_error}. Use correct keys: 'setpoint_c' (required, immediate target), 'trajectory' (optional array of anchors), 'confidence', 'reason'."

        return prompt

    def _parse_trajectory(self, args):
        """Rework Blueprint 5.3: normalize whatever the LLM returned into
        a sorted list of {"offset_min": float, "setpoint": float} anchors
        with a guaranteed offset_min=0 first anchor -- the IMMEDIATE
        target, which is also the only value SentinelGate ever scores/
        approves (see core/sentinel_gate.py's check() and main.py's
        decide()). Accepts either the new 'trajectory' array or a flat
        'setpoint_c'/'setpoint' single value (so a model that ignores the
        new instructions, or a caller still on the pre-Phase-3 tool
        schema, degrades to a valid 1-anchor "trajectory" instead of
        crashing -- exactly the old single-setpoint behavior).

        Returns None if there's no usable setpoint anywhere in `args`.
        """
        anchors = []
        raw_trajectory = args.get("trajectory")
        if isinstance(raw_trajectory, list) and raw_trajectory:
            for item in raw_trajectory:
                try:
                    offset = float(item.get("offset_min", 0.0))
                    setpoint = item.get("setpoint_c")
                    if setpoint is None:
                        setpoint = item.get("setpoint")
                    if setpoint is None:
                        continue
                    anchors.append({"offset_min": offset, "setpoint": float(setpoint)})
                except (AttributeError, TypeError, ValueError):
                    continue 
            anchors.sort(key=lambda a: a["offset_min"])

        if not anchors:
            flat = args.get("setpoint_c")
            if flat is None:
                flat = args.get("setpoint")
            if flat is None:
                return None
            return [{"offset_min": 0.0, "setpoint": float(flat)}]

        if anchors[0]["offset_min"] != 0.0:

            anchors.insert(0, {"offset_min": 0.0, "setpoint": anchors[0]["setpoint"]})

        return anchors

    def decide(self, t_in, t_out, forecast=None, carbon="Medium", correction_context=None, timeout=10):
        """Requests a decision with forced tool use and defensive parsing.

        Rework Blueprint 5.4 -- Phase 4: this loop now round-robins
        through self.provider_pool (core/provider_pool.py) instead of
        hardcoding "try Groq, and only after a 429 try Gemini once."
        A 1-provider pool (just Groq, the default with no secondary/
        additional providers configured) degenerates to exactly the old
        single-provider retry/backoff behavior -- see the branch below
        that waits out a cooldown instead of giving up immediately when
        there's only one provider to come back to.

        Returns a dict with the proposal PLUS two diagnostic fields:
          - "llm_ok": True if this proposal came from a real, successfully
            parsed tool-call response from SOME pooled provider. False if
            every attempt across the whole pool failed and this is the
            hardcoded physical-target fallback.
          - "error": the last exception/parse-failure message if llm_ok is
            False, else None. Surface this in logs/dashboards -- don't
            swallow it silently (that's exactly the bug that made a
            fully-fallback-driven year of decisions look like "AI" in the
            control log).
        """
        start = time.perf_counter()
        self.last_tool_calls = []
        current_error = None

        attempts_budget = max(MAX_INTERNAL_RETRIES, len(self.provider_pool))

        for attempt in range(attempts_budget):
            provider = self.provider_pool.next_available()

            if provider is None:

                status = self.provider_pool.status()
                soonest_free_s = min(
                    (s["cooldown_s"] for s in status.values()), default=0.0
                )
                elapsed = time.perf_counter() - start
                if soonest_free_s <= 0 or elapsed + soonest_free_s > DECIDE_WALL_CLOCK_BUDGET_S:
                    current_error = (current_error or "") + " | ProviderPool: every provider is currently in cooldown."
                    break
                current_error = ((current_error or "") +
                                  f" | ProviderPool: every provider cooling down, "
                                  f"waiting {soonest_free_s:.1f}s for the soonest to clear.")
                time.sleep(soonest_free_s)
                continue

            is_rate_limit = False
            rate_limit_exc = None  
                                    
            try:
                prompt = self._build_prompt(
                    t_in, t_out, forecast, carbon,
                    internal_error=current_error,
                    correction_context=correction_context,
                )

                self.provider_pool.rate_limiter_for(provider["name"]).wait()

                self.provider_pool.record_request(provider["name"])

                resp = provider["client"].chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=provider["model"],
                    tools=TOOL_SCHEMAS,
                    tool_choice={"type": "function", "function": {"name": "set_hvac"}},
                    timeout=timeout
                )

                message = resp.choices[0].message
                if not message.tool_calls:
                    current_error = f"No tool call detected. [{provider['name']}]"
                else:
                    self.last_tool_calls = [tc.function.name for tc in message.tool_calls]
                    args = json.loads(message.tool_calls[0].function.arguments)

                    trajectory = self._parse_trajectory(args)

                    if trajectory is not None:
                        self.last_latency_s = time.perf_counter() - start
                        self.last_used_fallback = False
                        self.last_error = None
                        reason = str(args.get("reason", "Adaptive optimization."))
                        if provider["name"] != "groq":
                            reason += f" [{provider['name']}]"
                        return {
                            "setpoint": trajectory[0]["setpoint"],
                            "trajectory": trajectory,
                            "confidence": float(args.get("confidence", 0.9)),
                            "reason": reason,
                            "llm_ok": True,
                            "error": None,
                        }
                    else:
                        current_error = f"Missing 'trajectory' and 'setpoint_c' keys. [{provider['name']}]"

            except _RATE_LIMIT_EXCEPTIONS as e:
                current_error = f"RateLimitError ({provider['name']}): {e}"
                is_rate_limit = True
                rate_limit_exc = e
            except Exception as e:
                current_error = f"{type(e).__name__} ({provider['name']}): {e}"

            if is_rate_limit:
                if _looks_like_daily_quota_error(rate_limit_exc):
                    retry_hint = _retry_after_seconds(rate_limit_exc)
                    self.provider_pool.mark_daily_exhausted(provider["name"])
                    current_error = (current_error or "") + " [daily quota -- provider held back until next window]"
                else:
                    backoff = _retry_after_seconds(rate_limit_exc) or RATE_LIMIT_BACKOFF_S[min(attempt, len(RATE_LIMIT_BACKOFF_S) - 1)]
                    self.provider_pool.mark_rate_limited(provider["name"], backoff)
                if attempt == attempts_budget - 1:
                    break
                continue

            if attempt == attempts_budget - 1:
                break

            backoff = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]

            elapsed = time.perf_counter() - start
            if elapsed + backoff > DECIDE_WALL_CLOCK_BUDGET_S:
                current_error = (current_error or "") + " (decide() wall-clock budget exceeded; falling back)"
                break

            time.sleep(backoff)

        self.last_latency_s = time.perf_counter() - start
        self.last_used_fallback = True
        self.last_error = current_error
        fallback = 21.2 if t_out < 15 else (24.5 if t_out > 24 else 22.0)
        return {
            "setpoint": fallback,

            "trajectory": [{"offset_min": 0.0, "setpoint": fallback}],
            "confidence": 0.5,
            "reason": "System-enforced physical target.",
            "llm_ok": False,
            "error": current_error,
        }
