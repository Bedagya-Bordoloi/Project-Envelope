# Round 7 — fixes applied after "both Groq AND Gemini 429ing by step ~360"

## Changed files
- `core/provider_pool.py`
- `agents/strategist.py`
- `config/building_policy.yaml`

## The bug

Dashboard runs showed the `Explainable AI Decisions` table filling with
`fallback_error` entries alternating between `RateLimitError (groq)` and
`RateLimitError (gemini)` from step ~348 onward — only ~1% into a
35,040-step year-long run — with Live Savings eroding (5.4% → 2.3%) as
every tick fell through to the hardcoded fallback / FAILSAFE instead of a
real LLM decision.

Two separate contributing causes:

1. **`mark_rate_limited()` treated every 429 the same way**, capping the
   cooldown at single-digit seconds (correct for an ordinary per-minute
   burst). But Groq's `openai/gpt-oss-20b` free tier separately enforces
   **1,000 requests/day and 200,000 tokens/day** — a real limit hit
   during the run (matching the exact `199,950/200,000`-style TPD
   exhaustion already seen in `ROUND3_FIXES.md`). Once that's actually
   gone, the real reset is hours away, not seconds — but the pool kept
   retrying both providers every ~10s for the rest of the run, each
   attempt burning wall-clock time before falling through to FAILSAFE.
2. **The event-triggered scheduler's thresholds (`deviation_deadband_c:
   0.4`, `outdoor_delta_threshold_c: 1.5`) were tight enough that during
   ordinary winter drift they fired almost every cadence tick** —
   nowhere near the blueprint's own framing ("only large temperature
   shifts get a real call"). That kept real call volume high enough,
   early enough in the run, to reach Groq's daily cap well before the
   simulated year finished.

## The fix

**1. `core/provider_pool.py` — daily quota is now tracked, not just
per-minute:**
- `mark_daily_exhausted(name, reset_hint_s=None)` — a new method,
  distinct from `mark_rate_limited()`. Puts a provider in cooldown until
  its own rolling 24h window is expected to clear (not a short fixed
  backoff).
- `record_request(name)` — call this immediately before a real network
  call is sent. Tracks a rolling 24h request count per provider.
- `next_available()` now also skips a provider that's already at its own
  configured `rpd` (requests-per-day) budget, so the pool stops routing
  to a provider it already knows is exhausted instead of finding out via
  a wasted 429 round-trip.
- `status()` now returns `{name: {"cooldown_s", "daily_used",
  "daily_limit"}}` instead of a flat `{name: seconds}` map (updated the
  one call site in `agents/strategist.py` accordingly).

**2. `agents/strategist.py` — 429s are classified, not treated
uniformly:**
- `_looks_like_daily_quota_error(exc)` — substring-matches an
  exception's own error text for daily-quota language ("per day", "RPD",
  "TPD", "daily limit"/"daily quota"). Verified against the real Groq TPD
  error shape (`"...on tokens per day (TPD): Limit 200000, Used
  199336..."`) and a Gemini-style RPD message — both correctly classified
  as daily; an ordinary "per minute" message correctly classified as not.
- `decide()`'s rate-limit handling now branches on this: a daily-quota
  429 routes to `mark_daily_exhausted()`; an ordinary one still routes to
  `mark_rate_limited()` with the existing short real-backoff schedule.
- `decide()` now calls `provider_pool.record_request()` immediately
  before every real network attempt, so the proactive daily budget stays
  accurate even on attempts that later fail for an unrelated reason.
- `_build_provider_pool()` reads an optional `rpd` from each provider's
  config block and passes it through.

**3. `config/building_policy.yaml`:**
- `strategist.rpd: 1000` — matches Groq's published free-tier RPD for
  `openai/gpt-oss-20b`. `secondary_provider.rpd` and each
  `additional_providers[].rpd` default to `0` (untracked proactively —
  Gemini/Cerebras/OpenRouter daily caps vary too much to guess safely;
  the reactive `mark_daily_exhausted()` path still protects them
  regardless of this number). Verify the Groq number against
  `console.groq.com/settings/limits` for your account tier before a long
  run.
- `strategist.trigger.deviation_deadband_c`: `0.4` → `0.8`, and
  `outdoor_delta_threshold_c`: `1.5` → `3.0` — secondary mitigation:
  fewer real calls get triggered by routine drift in the first place,
  reducing how fast a run approaches the daily cap at all. Still catches
  genuine comfort-relevant drift and real storm fronts; just stops
  firing on every minor wobble.

## What this does NOT change

The reactive per-minute pacing from Round 6 (`ProviderPool._RateLimiter`,
one instance per provider) is untouched — this round is specifically
about the *daily* ceiling, which is a different limit enforced
independently of the per-minute one (Groq's own docs: "You can hit any
limit type depending on which threshold you reach first").

## Verification

`core/provider_pool.py`'s new methods were smoke-tested standalone
(no live API keys needed, matching this module's existing
transport-agnostic design):
- Exhausting a provider's proactive `rpd` budget correctly removes it
  from `next_available()`'s rotation while leaving other pooled
  providers unaffected.
- `mark_daily_exhausted()` on a single-provider pool correctly produces
  a ~24h cooldown (`86399.99s` in the test run), not the old single-digit
  seconds — confirmed via `status()`.
- `mark_rate_limited()`'s ordinary short-cooldown path is unchanged
  (`5.0s` in, `~5.0s` reported).
- The rolling 24h window correctly reopens (provider becomes available
  again) once `now - window_start >= 86400s`.
- `_looks_like_daily_quota_error()` correctly classifies the real Groq
  TPD error shape and a Gemini-style RPD message as daily-quota, and an
  ordinary "per minute" message as not.
