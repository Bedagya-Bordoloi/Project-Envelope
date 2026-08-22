# Round 6 — fixes applied after the "only step 12 was ever real" bug

## Changed files
- `agents/strategist.py`
- `main.py`
- `config/building_policy.yaml`

## 1. Crash / freeze: unbounded Retry-After could stall the whole simulation

`_retry_after_seconds()` honored Groq's real `Retry-After` header with no
ceiling. A genuine 429 under sustained load can carry a header of several
minutes (Groq's own docs show examples like "try again in 6m 11.52s"), and
this retry loop runs synchronously inside EnergyPlus's per-timestep
callback -- so one real rate-limit hit could freeze the entire simulation
for 20+ minutes with `main.py` able to call `decide()` twice per tick
(initial + correction). This is what made the AI instance look "stuck at
step 12" while the baseline instance (zero network calls) finished the
full year almost instantly.

Fix: capped the honored `Retry-After` at `MAX_HONORED_RETRY_AFTER_S` (8s)
and added a hard `DECIDE_WALL_CLOCK_BUDGET_S` (30s) ceiling on total time
a single `decide()` call may spend across all its retries. Verified with a
mocked 371s Retry-After header: 24.8 min -> 16s.

## 2. Permanent fallback: every call after the first real success failed

Capping the reactive backoff (fix #1) surfaced a second, previously-masked
problem: with the retry loop now failing fast instead of hanging, control
logs showed exactly ONE genuine "AI" row (the very first cadence tick)
followed by nothing but "(Fallback)" rows for the rest of the run, with
Live Savings shrinking over time (8.3% -> 6.2% -> 5.0%) as the fixed
21.2C fallback drifted further from optimal outside deep winter.

Root cause: nothing in this codebase ever paced outbound Groq calls --
every fix to date was REACTIVE (catch a 429, then back off). EnergyPlus
can simulate a 15-minute zone timestep in a tiny fraction of a real
second, so `cadence_steps: 12` (3 SIMULATED hours) can still mean a burst
of real Groq calls landing within a handful of real WALL-CLOCK seconds --
blowing through Groq's free-tier ~30 req/min budget almost immediately.
Once blown, every following call keeps arriving faster than the budget
refills, so the account never recovers for the rest of the run.

Fix: a process-wide `_RateLimiter` (shared across every `Strategist`
instance, since multizone mode creates one per zone but they all share
one Groq API key) paces real calls to stay under
`strategist.max_requests_per_minute` (default 25, headroom under Groq's
free-tier 30/min) BEFORE hitting the API, not after. Verified with a
40-call burst across two Strategist instances: real call rate held at
25.6 req/min with a strict >=2.4s floor between calls, instead of firing
all 40 instantly.
