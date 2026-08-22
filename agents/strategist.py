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

# Rework Blueprint 5.4 -- Phase 4: decide() below now treats a rate limit
# from ANY pooled provider uniformly, regardless of which SDK raised it.
# Groq's own SDK raises its own groq.RateLimitError; every other provider
# in the pool is reached through the openai SDK's OpenAI-compatible
# client (see _build_provider_pool() below), which raises
# openai.RateLimitError instead -- a different class, same meaning
# ("this account is over its per-minute/per-day budget right now").
# Catching only one of the two would silently treat the other
# provider's rate limits as generic errors (still handled, just with the
# wrong backoff schedule and without marking that provider's pool
# cooldown) -- see decide()'s except clause.
_RATE_LIMIT_EXCEPTIONS = (RateLimitError, _OpenAIRateLimitError)

MAX_INTERNAL_RETRIES = 3

# Rework Blueprint 5.3 -- Phase 3: trajectory planning instead of
# single-setpoint calls. The Strategist is asked for a short receding
# horizon (a few anchors spanning a few hours of forecast) instead of one
# instantaneous setpoint; main.py's fast loop linearly interpolates
# between anchors every physical step (see main.py's
# ProjectEnvelope._interpolate_trajectory()) instead of holding flat
# until the next slow-loop firing and then jumping. Only the offset_min=0
# anchor (the IMMEDIATE target) is ever scored/approved by SentinelGate
# -- see this module's _parse_trajectory() and main.py's decide().
TRAJECTORY_ANCHOR_COUNT = 4
TRAJECTORY_SPAN_MIN = 180.0  # 3 hours, matching the blueprint's own example

# BUGFIX (this round -- the "only step 12 was ever real, everything after
# is (Fallback)" bug): every fix so far in this file has been REACTIVE --
# catch a 429 after it happens, then back off. Nothing ever stopped calls
# from being fired faster than the account's real budget in the first
# place. EnergyPlus can simulate a 15-minute zone timestep in a tiny
# fraction of a real second, so cadence_steps: 12 (~3 sim-hours apart in
# SIMULATED time) can still mean a burst of real Groq calls landing within
# a handful of real WALL-CLOCK seconds -- comfortably blowing through
# Groq's free-tier ~30 requests/minute budget almost immediately. Once
# that happens, every following call keeps arriving faster than the
# per-minute budget refills, so the account never recovers for the rest
# of the run: exactly one genuine "AI" row (before the budget was blown),
# then "(Fallback)" forever after -- which also explains why savings
# erode over time: the fallback is a fixed 21.2C, and it drifts further
# from optimal as outdoor temp climbs out of deep winter.
#
# Fix: a small process-wide pacer (_RateLimiter below) that makes every
# REAL Groq call wait, if needed, so consecutive calls are never closer
# together than `strategist.max_requests_per_minute` in building_policy.
# yaml allows -- BEFORE hitting the API, not after. This is shared across
# every Strategist instance in the process (module-level singleton),
# because multizone mode creates one Strategist per zone but they all
# share the SAME Groq API key / account-level budget.
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

# FIX (post-Round-3 fallback-streak audit): the dashboard showed the AI
# instance pinned to the hardcoded fallback (21.2C) for hundreds of
# consecutive cadence ticks, correlating with cumulative energy flipping
# to a small loss. That is NOT SentinelGate rejecting anything -- a
# replay of the exact fallback proposal through the real gate shows it
# APPROVED/HOLD every time, never REJECTED (see the diagnostic run
# referenced in ROUND3_FIXES.md). The actual cause: MAX_INTERNAL_RETRIES
# was 2 with a flat time.sleep(0.5) between attempts -- so the ENTIRE
# retry budget for a single decide() call was ~1 second. A real Groq
# rate limit (expected under sustained load: thousands of cadence ticks
# across a year-long sim, all hitting the same API key) typically needs
# several seconds to clear. The old logic burned through both retries
# and gave up in ~1s, so once rate-limited once, EVERY subsequent
# cadence tick repeated the exact same losing race and fell back again
# -- a sustained streak, not an occasional blip. A mocked repro
# (agents/strategist.py's git history / ROUND3_FIXES.md has the
# transcript) confirms: old logic -> 0/1 recovery within ~1s of a 3-call
# rate-limit burst; new logic below -> recovers within the same
# decide() call by actually waiting long enough.
#
# Fix: (1) explicitly catch groq.RateLimitError separately from other
# exceptions and back off using the API's own Retry-After header when
# present, falling back to a real (multi-second) schedule otherwise --
# not the same flat 0.5s used for a JSON-parse hiccup. (2) One extra
# retry attempt (3 instead of 2) since a rate limit clearing is a
# time-based event, not a random one -- more attempts with real backoff
# meaningfully raises the odds of recovering within a single decide()
# call, whereas more attempts with a flat 0.5s changed nothing.
RETRY_BACKOFF_S = [0.5, 1.5, 3.0]        # non-rate-limit errors (parse issues, transient 5xx, etc.)
RATE_LIMIT_BACKOFF_S = [2.0, 5.0, 10.0]  # rate limits specifically -- needs real wall-clock time

# BUGFIX (this round): a real Groq 429 can carry a Retry-After header of
# several MINUTES (a sustained token/request-limit exhaustion, e.g. Groq's
# own documented example "Please try again in 6m 11.52s" -- not a
# theoretical case, this is the normal shape of a real rate-limit response
# under sustained load). _retry_after_seconds() previously honored that
# value with NO ceiling and handed it straight to time.sleep() -- and this
# whole retry loop runs SYNCHRONOUSLY inside EnergyPlus's per-timestep
# callback (core/energyplus_bridge.py's _callback), so a single real 429
# could freeze the ENTIRE simulation for minutes. Worse, main.py can call
# decide() a SECOND time for the correction pass on the same tick, so one
# unlucky tick could block 20+ minutes of real wall-clock time -- which
# from the outside (a baseline instance making zero network calls
# finishing the full year almost instantly in the same window) looks
# exactly like the AI instance crashed/hung, not like it's "still going."
# Fix: cap whatever the server asks for to a small ceiling. This still
# respects a real rate limit (we still back off, and still don't hammer
# the API) without being able to stall the sim by unbounded amounts --
# if the real reset time is longer than the cap, decide() will simply
# fall through to the safe physical-target fallback for this tick and
# get a fresh chance on the next cadence tick, rather than blocking.
MAX_HONORED_RETRY_AFTER_S = 8.0

# ROUND 7 FIX -- see core/provider_pool.py's module docstring for the full
# story. A 429 whose error text names a PER-DAY limit (Groq's own body
# reads "...on tokens per day (TPD): Limit 200,000 · Used 199,336...";
# Gemini's free-tier equivalent reads similarly for RPD) was previously
# treated exactly like an ordinary per-minute 429 -- capped at an 8-10s
# cooldown via mark_rate_limited(). Once a provider's DAILY budget is
# actually gone, an 8s cooldown does nothing: the very next attempt (and
# every attempt after it for the rest of the real day) gets the same 429
# again. Dashboard runs showed exactly this: Groq and Gemini alternating
# 429s every single tick from step ~348 onward, each one burning up to
# DECIDE_WALL_CLOCK_BUDGET_S before falling to FAILSAFE, for the rest of
# the run.
#
# _looks_like_daily_quota_error() does simple substring matching on the
# exception's own text (both groq.RateLimitError and openai.
# RateLimitError expose a human-readable message that includes the
# provider's own words for which limit was hit) to tell "clears in a few
# seconds" apart from "clears at the next daily reset" -- see decide()'s
# except clause, which routes to provider_pool.mark_daily_exhausted()
# instead of mark_rate_limited() when this returns True.
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

# Hard ceiling on total wall-clock time a single decide() call (across all
# its internal retries/backoffs) may take. decide() runs synchronously
# inside EnergyPlus's per-timestep callback -- there is nothing else
# advancing the simulation while it blocks -- so this must stay small
# relative to how often cadence ticks happen, not just "generous."
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
        # FIX: season thresholds (14/22) and the "Beat 22.0C Baseline"
        # prompt text used to be hardcoded here instead of reading
        # config/building_policy.yaml's seasonality.*/baseline.* keys.
        # `policy` is optional so any caller that hasn't been updated
        # yet (e.g. an older script) still gets identical behavior via
        # core/seasonality.py's defaults -- see _build_prompt() below.
        self.policy = policy
        # BUGFIX: proactive pacer (see module docstring above) -- shared
        # across every Strategist instance in the process since they all
        # burn the same Groq account's per-minute budget. Default 25
        # stays safely under Groq's free-tier ~30 req/min with headroom
        # for the correction pass; override via
        # policy["strategist"]["max_requests_per_minute"] if your account
        # tier allows more (or needs less).
        max_rpm = (policy or {}).get("strategist", {}).get("max_requests_per_minute", 25)
        self._rate_limiter = _get_shared_rate_limiter(max_rpm)
        self.last_latency_s = 0
        self.last_tool_calls = []

        # Priority 3b (blueprint Section 1.6/3): OPTIONAL second reasoning
        # tier -- a genuinely DIFFERENT provider (here: Gemini's OpenAI
        # compatibility layer) reached via base_url override, so this
        # doesn't add a new *account* dependency on the primary provider
        # -- see config/building_policy.yaml's strategist.secondary_provider.
        # This is deliberately NOT a second Groq account/key: the
        # blueprint's "5 pooled keys" section flags that as a ToS
        # anti-circumvention risk, while a different provider's free tier
        # is governed by its own separate ToS. Off by default; only ever
        # tried when the PRIMARY provider's failure was specifically a
        # rate limit (see decide() below), never for a schema/parse/network
        # error that a different provider wouldn't fix either.
        #
        # BUGFIX (this round): this used to construct `Groq(api_key=...,
        # base_url=...)` on the theory that "Groq's SDK is just an
        # OpenAI-compatible client, so pointing it at any OpenAI-compatible
        # base_url works." That's false in a way that only shows up once
        # you actually call it: groq-python hardcodes the request path as
        # `{base_url}/openai/v1/chat/completions` (correct ONLY because
        # Groq's own base_url, https://api.groq.com, has no path segment
        # of its own). Gemini's documented OpenAI-compatible base_url,
        # https://generativelanguage.googleapis.com/v1beta/openai/, already
        # ends in .../openai/ -- so the Groq client was posting to
        # .../v1beta/openai/openai/v1/chat/completions, which doesn't
        # exist -> 404 on every call, not just under rate-limit load.
        # Google's own docs (https://ai.google.dev/gemini-api/docs/openai)
        # show this exact base_url used with the real `openai` package,
        # whose client posts to `{base_url}/chat/completions` with no
        # extra prefix -- i.e. .../v1beta/openai/chat/completions, which
        # is the real endpoint. Fix: use the openai SDK for the secondary
        # client instead of reusing the Groq class. (If you ever point
        # secondary_provider at a provider whose OpenAI-compat base_url
        # has NO path segment, like Together.ai's api.together.xyz, the
        # Groq-class trick would have worked there too -- Gemini's base_url
        # specifically is the one shape that breaks it.)
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

        # Rework Blueprint 5.4 -- Phase 4: fold whatever got configured
        # above (primary Groq + optional secondary/Gemini) PLUS any
        # additional providers from policy["strategist"]["provider_pool"]
        # ["additional_providers"] (config/building_policy.yaml) into one
        # ProviderPool (core/provider_pool.py). decide() below now
        # round-robins through this pool instead of the old hardcoded
        # "try Groq, on 429 try Gemini once" block -- see decide()'s
        # docstring and the module docstring above for why this is a
        # generalization, not a behavior change, when only Groq is
        # configured (a 1-provider pool degenerates to the exact old
        # single-provider retry/backoff behavior).
        self.provider_pool = self._build_provider_pool(policy, max_rpm)

        # --- Diagnostics (fix: silent-fallback visibility) -----------------
        # Every prior run showed exactly two confidence values across the
        # whole year (0.5 and 0.9) -- both hardcoded constants from the
        # fallback/defensive-parsing paths below, never a genuine LLM
        # confidence score. That means the real Groq call was failing on
        # (almost) every single attempt, but nothing surfaced that fact --
        # main.py logged every approved fallback proposal as plain
        # "source": "AI", indistinguishable from a real LLM decision.
        #
        # These two attributes are set on every decide() call so main.py
        # can log accurately (source: "AI" vs "AI (Fallback)") and so a
        # human/dashboard can see at a glance whether the LLM is actually
        # being used. See scripts/check_llm_connectivity.py for a 2-minute
        # standalone check of this same thing.
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
            # ROUND 7: proactive daily-request budget (see core/
            # provider_pool.py). Default 1000 matches Groq's published
            # free-tier RPD for openai/gpt-oss-20b as of this writing --
            # override via strategist.rpd if your account tier differs
            # (0 disables proactive tracking; the reactive
            # mark_daily_exhausted() path still protects you either way).
            "rpd": strategist_cfg.get("rpd", 1000),
        }]

        secondary_cfg = (policy or {}).get("strategist", {}).get("secondary_provider", {}) or {}
        if self.secondary_client is not None:
            providers.append({
                "name": secondary_cfg.get("name", "gemini"),
                "client": self.secondary_client,
                "model": self.secondary_model,
                "rpm": secondary_cfg.get("rpm", 15),
                # ROUND 7: 0 (unset) by default -- Gemini's free-tier RPD
                # varies by model/region and isn't reliably published the
                # way Groq's is, so this ships un-tracked proactively
                # rather than risk a wrong number silently under- or
                # over-throttling it. Set strategist.secondary_provider.rpd
                # once you've confirmed your account's real limit (see
                # scripts/check_secondary_provider.py); the reactive
                # mark_daily_exhausted() path already protects this
                # provider regardless.
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
                # ROUND 7: same convention as secondary_provider above --
                # 0/unset by default, set entry.rpd once you've confirmed
                # the real per-provider daily limit.
                "rpd": entry.get("rpd", 0),
            })

        return ProviderPool(providers)

    # A jump of this many degrees C within the forecast window counts as
    # a "spike" worth pre-empting -- see _check_lookahead().
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

        # PHYSICAL LOGIC ENGINE
        # FIX: season thresholds and the baseline value below now come
        # from config/building_policy.yaml via core/seasonality.py
        # (classify_season/get_baseline_setpoint) instead of being
        # hardcoded (14/22/"22.0C") independently of the values
        # BaselineController and SentinelGate's guardrail actually use.
        baseline_setpoint = get_baseline_setpoint(self.policy) if self.policy is not None else 22.0
        season_label = classify_season(t_out, self.policy) if self.policy is not None else (
            "WINTER" if t_out < 14 else ("SUMMER" if t_out > 22 else "SHOULDER")
        )

        # BUGFIX (dead config -- see chat): "TARGET: 21.0C to 21.3C" / "Aim
        # for 24.5C to 25.5C" were hardcoded here and had quietly drifted
        # out of sync with seasonality.comfort_ranges in
        # config/building_policy.yaml (which declares winter [21.0, 21.8]
        # and summer [23.5, 25.5]) -- editing comfort_ranges in the YAML
        # silently did nothing, same failure class as the winter_limit/
        # baseline-setpoint drift already fixed elsewhere in this file.
        # Now the advice text is generated FROM comfort_ranges directly.
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
            # The 'Shoulder Season' fix: Match baseline to stop energy-wasting transitions
            season, advice = "SHOULDER (Mild)", f"MATCH BASELINE ({baseline_setpoint:.1f}C) to avoid wasting energy on setpoint jitter."

        # FORWARD LOOK-AHEAD (blueprint 1.3): previously `forecast` was
        # accepted as a parameter but never made it into the prompt, so
        # the Strategist could only ever react to the current instant.
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

        # FIX: this used to be accepted as a `decide()` kwarg and then
        # silently dropped -- `_build_prompt()` never had a
        # `correction_context` parameter, so main.py's Reflective
        # Self-Correction pass ("call decide() again with the rejection
        # reason") re-sent the EXACT SAME prompt as the first attempt.
        # That's why the year-long log showed zero "AI (Corrected)" rows:
        # the "corrected" proposal was never actually informed by why it
        # was rejected, so it just failed the gate the same way again (or
        # fell through to the same hardcoded fallback). Now the gate's
        # actual rejection reason is injected back into the prompt.
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
                    continue  # skip one malformed anchor rather than discarding the whole trajectory
            anchors.sort(key=lambda a: a["offset_min"])

        if not anchors:
            flat = args.get("setpoint_c")
            if flat is None:
                flat = args.get("setpoint")
            if flat is None:
                return None
            return [{"offset_min": 0.0, "setpoint": float(flat)}]

        if anchors[0]["offset_min"] != 0.0:
            # Guarantee an offset_min=0 anchor exists: SentinelGate always
            # scores the IMMEDIATE target, not whichever anchor the model
            # happened to list first.
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

        # At least one attempt per provider in the pool, but never fewer
        # than the original MAX_INTERNAL_RETRIES (preserves the old
        # single-provider retry count exactly when the pool has only
        # Groq in it).
        attempts_budget = max(MAX_INTERNAL_RETRIES, len(self.provider_pool))

        for attempt in range(attempts_budget):
            provider = self.provider_pool.next_available()

            if provider is None:
                # Every provider in the pool is currently in cooldown.
                # With a 1-provider pool this is exactly the old
                # behavior (wait out the same provider's backoff, then
                # retry it) -- with an N-provider pool this branch is
                # only reached in the rare case where ALL of them
                # happen to be cooling down simultaneously, since
                # next_available() would otherwise have handed us a
                # different, still-healthy provider instead of landing
                # here at all.
                # ROUND 7: status() now returns a dict-of-dicts (cooldown_s/
                # daily_used/daily_limit per provider, see core/
                # provider_pool.py) instead of a flat {name: seconds} map --
                # pull out just the cooldown seconds for the "how long until
                # the soonest provider frees up" check below.
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
            rate_limit_exc = None  # kept outside the except block -- Python 3 deletes
                                    # the `except ... as e` binding once the block exits
            try:
                prompt = self._build_prompt(
                    t_in, t_out, forecast, carbon,
                    internal_error=current_error,
                    correction_context=correction_context,
                )

                # BUGFIX: pace the REAL network call, not the retry loop
                # around it -- one pacer PER PROVIDER (core/
                # provider_pool.py's _RateLimiter) so this keeps EACH
                # provider under ITS OWN per-minute budget, whichever one
                # the pool hands us this attempt, instead of just
                # reacting to a 429 after the fact (see module docstring
                # above).
                self.provider_pool.rate_limiter_for(provider["name"]).wait()

                # ROUND 7: record this as a real attempt against the
                # provider's own rolling 24h budget (see core/
                # provider_pool.py's record_request() docstring) BEFORE
                # making the call -- counting attempted, not just
                # successful, calls is the conservative choice: a call
                # that fails for a non-quota reason still consumed one of
                # the provider's real request slots.
                self.provider_pool.record_request(provider["name"])

                resp = provider["client"].chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=provider["model"],
                    tools=TOOL_SCHEMAS,
                    # FORCED TOOL CHOICE: Prevents Error 400 by making 'set_hvac' mandatory
                    tool_choice={"type": "function", "function": {"name": "set_hvac"}},
                    timeout=timeout
                )

                message = resp.choices[0].message
                if not message.tool_calls:
                    current_error = f"No tool call detected. [{provider['name']}]"
                else:
                    self.last_tool_calls = [tc.function.name for tc in message.tool_calls]
                    args = json.loads(message.tool_calls[0].function.arguments)

                    # Rework Blueprint 5.3: DEFENSIVE PARSING -- accept a
                    # 'trajectory' array (preferred) or fall back to a
                    # flat 'setpoint_c'/'setpoint' as a degenerate
                    # 1-anchor trajectory. See _parse_trajectory()'s
                    # docstring.
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

            # FIX: RateLimitError used to fall into the same generic
            # `except Exception` branch as a JSON parse hiccup, both
            # backed off with the same flat 0.5s -- nowhere near enough
            # for a real 429 to clear (see the module docstring's mocked
            # repro). Handle it separately with real backoff. Catches
            # BOTH Groq's own RateLimitError and openai.RateLimitError
            # (every other pooled provider goes through the openai SDK
            # -- see _build_provider_pool()) -- see _RATE_LIMIT_EXCEPTIONS
            # at the top of this module for why both are needed.
            except _RATE_LIMIT_EXCEPTIONS as e:
                current_error = f"RateLimitError ({provider['name']}): {e}"
                is_rate_limit = True
                rate_limit_exc = e
            except Exception as e:
                current_error = f"{type(e).__name__} ({provider['name']}): {e}"

            if is_rate_limit:
                # ROUND 7: a 429 is no longer treated as one uniform
                # thing. If the error text itself says this is a PER-DAY
                # limit (see _looks_like_daily_quota_error() above), a
                # short cooldown is pointless -- the very next attempt
                # would just get the same 429 again, for the rest of the
                # real day. Route that case to mark_daily_exhausted()
                # instead, which holds the provider back until its own
                # rolling 24h window is expected to clear. An ordinary
                # per-minute/burst 429 still gets the short, real-backoff
                # treatment exactly as before.
                if _looks_like_daily_quota_error(rate_limit_exc):
                    retry_hint = _retry_after_seconds(rate_limit_exc)
                    # _retry_after_seconds() caps at MAX_HONORED_RETRY_AFTER_S
                    # (8s) for the sim-freeze reason explained there -- that
                    # cap is correct for a per-minute 429 but far too short
                    # to be a meaningful hint for a genuine daily-quota
                    # reset, so it's deliberately NOT passed through here.
                    # mark_daily_exhausted() falls back to its own rolling
                    # 24h estimate when no hint is given.
                    self.provider_pool.mark_daily_exhausted(provider["name"])
                    current_error = (current_error or "") + " [daily quota -- provider held back until next window]"
                else:
                    backoff = _retry_after_seconds(rate_limit_exc) or RATE_LIMIT_BACKOFF_S[min(attempt, len(RATE_LIMIT_BACKOFF_S) - 1)]
                    # Put THIS provider into cooldown for the backoff window
                    # -- this is the entire point of pooling: instead of
                    # sleeping here and retrying the SAME rate-limited
                    # provider (the old 2-tier behavior), loop straight back
                    # around and let next_available() hand us a DIFFERENT,
                    # presumably still-healthy provider on the very next
                    # attempt with no local time.sleep() at all. Only when
                    # every provider is simultaneously cooling down does the
                    # `provider is None` branch above actually wait.
                    self.provider_pool.mark_rate_limited(provider["name"], backoff)
                if attempt == attempts_budget - 1:
                    break
                continue

            if attempt == attempts_budget - 1:
                break  # last attempt failed -- no point sleeping before falling to the fallback below

            # Non-rate-limit failure (parse/schema/network/5xx on
            # whichever provider we just tried) -- same real backoff as
            # before rather than immediately hammering the next provider,
            # since a systemic issue (e.g. a malformed prompt) is just as
            # likely to repeat on a different provider.
            backoff = RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)]

            # BUGFIX (defense in depth, alongside the MAX_HONORED_RETRY_AFTER_S
            # cap above): this whole decide() call runs synchronously inside
            # EnergyPlus's per-timestep callback, so it must have a hard
            # ceiling on total wall-clock time no matter what combination of
            # per-call `timeout`, retry attempts, and backoffs is in play.
            # If we're already past budget, stop retrying now and fall
            # through to the physical-target fallback below instead of
            # sleeping further -- a fresh cadence tick a few hours of sim
            # time later gets another chance at a real LLM call.
            elapsed = time.perf_counter() - start
            if elapsed + backoff > DECIDE_WALL_CLOCK_BUDGET_S:
                current_error = (current_error or "") + " (decide() wall-clock budget exceeded; falling back)"
                break

            time.sleep(backoff)

        # Smart Policy-Driven Fallback (Reduces 'Failsafe' losses)
        self.last_latency_s = time.perf_counter() - start
        self.last_used_fallback = True
        self.last_error = current_error
        fallback = 21.2 if t_out < 15 else (24.5 if t_out > 24 else 22.0)
        return {
            "setpoint": fallback,
            # Rework Blueprint 5.3: a degenerate 1-anchor "trajectory" so
            # main.py's fast-loop interpolation (_interpolate_trajectory)
            # has a consistent shape to work with regardless of whether
            # this tick's proposal came from a real trajectory-bearing
            # LLM response or this hardcoded physical-target fallback.
            "trajectory": [{"offset_min": 0.0, "setpoint": fallback}],
            "confidence": 0.5,
            "reason": "System-enforced physical target.",
            "llm_ok": False,
            "error": current_error,
        }
