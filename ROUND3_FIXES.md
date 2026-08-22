# Round 3 — fixes applied after the log audit

This round starts from your `project_envelope_updated_v3` zip (all Tier 0–4
blueprint work kept intact: multizone, BACnet path, lookahead, EV
generalization demo, etc.) and fixes the logic bugs the full 116-row AI log
audit surfaced. Nothing in Tier 0–4's *structure* was removed — every fix
below is additive or corrective inside the same files.

## Changed files
- `agents/strategist.py`
- `core/sentinel_gate.py`
- `bms_mcp/tools.py`
- `main.py`
- `config/building_policy.yaml`
- `scripts/calibrate_ccs_sweep.py`

## New files
- `scripts/check_llm_connectivity.py` — the "confirm in 2 minutes" check
- `scripts/generate_evidence.py` — runs the whole DEMO_REHEARSAL.md evidence
  table in one shot

---

## 1. Reflective self-correction was a no-op (zero `AI (Corrected)` rows all year)

**Bug:** `main.py` called `self.strategist.decide(..., correction_context=reason)`
on the correction pass, but `Strategist.decide()` accepted that kwarg and
then dropped it — `_build_prompt()` had no `correction_context` parameter at
all. The "corrected" attempt sent the model the *exact same prompt* as the
first try, so it just failed the gate the same way again.

**Fix:** `_build_prompt()` now takes `correction_context` and, when present,
appends a `[SELF-CORRECTION]` block quoting the gate's actual rejection
reason and instructing the model to propose a different value. `decide()`
threads it through on every attempt of the correction pass.

## 2. Silent fallback made a possibly-100%-fallback year look like "AI"

**Bug:** Every confidence value in the whole log was one of two hardcoded
constants (`0.5` from the emergency fallback, `0.9` from the missing-key
default) — never a value a real LLM actually returned. `strategist.py`
swallowed every exception into a single `current_error` string and never
surfaced it; `main.py` logged the fallback's approved proposals as plain
`"source": "AI"`, indistinguishable from a real LLM decision.

**Fix:**
- `Strategist.decide()` now returns `llm_ok: bool` and `error: str | None`
  on every call, and sets `self.last_used_fallback` / `self.last_error`.
- `main.py` tags the log's `source` field honestly: `"AI"` only when the
  LLM genuinely answered, `"AI (Fallback)"` / `"AI (Corrected) (Fallback)"`
  / `"AI (Stabilized) (Fallback)"` otherwise.
- Every log row now also carries `"llm_ok"` and `"fallback_error"` fields.
- New `scripts/check_llm_connectivity.py` calls `decide()` a few times
  standalone and tells you plainly whether Groq is actually being reached —
  run this *before* trusting any other number.

## 3. `carbon` weight (20% of the CCS score) was dead

**Bug:** `SentinelGate.check()` never received a carbon value from the
caller, so `compute_ccs()` always used its hardcoded default
(`carbon_score=0.8`) — the term never varied.

**Fix:** New `bms_mcp/tools.py:carbon_score_from_level()` maps the
simulated `Low/Medium/High` label to a real `[0,1]` score
(`config/building_policy.yaml`'s new `carbon.score_map`). `main.py` computes
it every tick and passes it into `gate.check(..., carbon_score=...)`, which
forwards it into `compute_ccs()`. Verified: two otherwise-identical gate
calls with `carbon_score=1.0` vs `0.0` now produce different CCS values.

## 4. `override_rate` weight (10%) was declared but never used

**Bug:** `building_policy.yaml` declared an `override_rate: 0.10` gate
weight; `compute_ccs()` never referenced it. The 5 declared weights only
summed to 0.90 in practice.

**Fix:** `SentinelGate` now keeps a rolling window (`gate.override_window`,
default 20) of its own recent approve/reject outcomes and exposes
`recent_override_rate`. `check()` computes `override_score = 1 -
recent_override_rate` and passes it into `compute_ccs()`, which uses
`weights.get("override_rate", 0.0)` (so callers like the EV demo that don't
declare the key are unaffected — verified, the EV demo still runs clean).

## 5. Gate's hard `violation_severity == 0` clause wasn't tunable

**Bug:** Approval required `violation_severity == 0` no matter what
`ccs_threshold` was set to, so `calibrate_ccs_sweep.py`'s threshold-only
sweep could never explain (or fix) long `FAILSAFE` streaks in warm weather —
a proposal 0.01 PMV over the line was rejected regardless of threshold.

**Fix:** New `gate.max_violation_severity` config (default `0.0`, i.e.
*identical* behavior unless you change it). `check()` now tests
`violation_severity <= self.max_violation_severity`.
`scripts/calibrate_ccs_sweep.py` gained a `--violation-tolerances` flag to
sweep this dimension alongside threshold; default keeps the old
threshold-only sweep unchanged.

## 6. Cadence (300 steps = 3.1 days) + dwell (compounds to 9.4 days) were too sparse

**Bug:** `cadence_steps: 300` meant the Strategist only ran once every 75
hours. `hysteresis.min_dwell_steps` is counted in *cadence ticks*, not raw
steps, so `min_dwell_steps: 3` at that cadence meant up to **9.4 days**
where the gate wouldn't even reconsider a near-identical proposal.

**Fix:** `cadence_steps` is now `12` (3 hours — in the blueprint's
recommended 1–4h range). At the same `min_dwell_steps: 3`, the real dwell
time is now `3 × 12 × 15min = 9 hours` — a defensible "don't jitter for
under half a day," not a two-week freeze. Comments in the YAML spell out
the tick-vs-step distinction explicitly so it doesn't get silently
re-broken later. (Trade-off noted in-file: lower cadence = more Groq calls;
watch your rate-limit tier if you push below ~12.)

## 7. No evidence artifacts had actually been generated

**Not a code bug**, but addressed with `scripts/generate_evidence.py`,
which runs the entire `DEMO_REHEARSAL.md` Part-0 table
(baseline → AI run → calibration sweep → forecast-spike → model comparison
→ multizone → BACnet test → EV demo) in one shot, logs each step to
`logs/evidence/<step>.log`, and prints a pass/fail summary so you know
exactly which slide numbers are backed by a real file before a judge asks.

---

---

## What to run, in order, before trusting any number again

```bash
python scripts/check_llm_connectivity.py      # confirm the LLM is real first
python scripts/generate_evidence.py           # regenerate every evidence artifact
```

If `check_llm_connectivity.py` reports fallbacks, fix that (rotate/verify
`GROQ_API_KEY`, check the model string, check network egress to
`api.groq.com`) **before** re-running the sweep or trusting any savings
number — a savings/CCS number computed while every decision is secretly the
hardcoded fallback isn't measuring what the deck says it's measuring.

---

## Post-Round-3 addendum — Groq model deprecation

`llama-3.1-8b-instant` and `llama-3.3-70b-versatile` were deprecated by
Groq on 2026-06-17 and fully shut down 2026-08-16. Calls now return a
`404 model_not_found`, which will *also* look like "the LLM is failing"
via the `llm_ok`/`fallback_error` diagnostics added above — worth knowing
these are two separate issues that can look identical from the log:
silent-fallback-due-to-a-real-bug (fixed above) vs.
silent-fallback-due-to-an-upstream-model-retirement (this one).

Swapped every hardcoded reference to Groq's recommended replacements:
- `agents/strategist.py`'s default `model=` kwarg
- `config/building_policy.yaml`'s `strategist.model`
- `scripts/compare_models.py`'s `DEFAULT_MODELS`

`openai/gpt-oss-20b` replaces the 8B tier, `openai/gpt-oss-120b` replaces
the 70B tier. If you're reading this well after August 2026, re-verify
these are still active with `curl
https://api.groq.com/openai/v1/models -H "Authorization: Bearer
$GROQ_API_KEY"` or console.groq.com/docs/models before assuming — model
lineups on hosted-inference providers change fast, and `scripts/check_llm_connectivity.py`'s raw error message is the fastest way to catch it if they change again.

