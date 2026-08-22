import argparse
import os
import sys
import json
import yaml
import time
from dotenv import load_dotenv

from core.energyplus_bridge import EnergyPlusBridge
from core.sentinel_gate import SentinelGate
from core.failsafe_controller import FailsafeController
from core.baseline_controller import BaselineController
from core.decision_cache import DecisionCache
from core.trigger_engine import TriggerEngine
from agents.strategist import Strategist
from bms_mcp.tools import ToolContext, carbon_intensity_level, carbon_score_from_level
from core.seasonality import classify_season, get_baseline_setpoint
from integrations.bacnet_adapter import build_adapter_from_policy

load_dotenv()

def load_policy(path="config/building_policy.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _quick_idf_path(idf_path, days):
    """Audit fix for wall-clock infeasibility: return a path to a sibling
    copy of `idf_path` with its RunPeriod shortened to `days` days
    (starting Jan 1) instead of the full year, for fast iteration. The
    original idf is never modified -- see scripts/patch_idf.py's
    patch_run_period(). Regenerated on every call (cheap, deterministic)
    so it always reflects whatever idf_path currently points at."""
    import datetime
    from scripts.patch_idf import patch_run_period

    days = int(days)
    if days < 1 or days > 365:
        raise ValueError("--run-period-days must be between 1 and 365.")
    # 2023: an arbitrary non-leap reference year, used only to turn "N
    # days from Jan 1" into a calendar month/day -- the actual simulated
    # year comes from the .epw weather file, not this date arithmetic.
    end_date = datetime.date(2023, 1, 1) + datetime.timedelta(days=days - 1)

    root, ext = os.path.splitext(idf_path)
    out_path = f"{root}.quick{days}d{ext}"
    return patch_run_period(idf_path, out_path, end_date.month, end_date.day)

def _reset_control_log(path):
    """BUGFIX (savings spike/decay after a restart -- see chat): every
    log write elsewhere in this file uses open(path, "a") -- correct for
    appending WITHIN a run, but nothing ever cleared the file BETWEEN
    runs. EnergyPlusBridge.step_counter and cumulative_kwh always start
    fresh at 0 for a new process, but the log file didn't -- so
    restarting `python main.py` kept appending a new step-0 sequence
    onto whatever an earlier run (possibly a completed full year) had
    already written. ui/app.py's dashboard then aligns AI/Baseline by
    "the last row at or before the shared step," which picks up the
    freshly-restarted run's tiny early cumulative_kwh against a stale
    high-step-count row from the OTHER log, producing exactly the
    "98% savings, then rapidly decaying" pattern -- a log-mixing
    artifact, not a real result. Truncating here means every fresh
    `python main.py` (or --baseline) invocation starts this mode's log
    at a clean, single, self-consistent run.
    """
    open(path, "w").close()


class ProjectEnvelope:
    """AI-controlled instance with adaptive safety gating."""
    def __init__(self, policy, control_log_path):
        self.policy = policy
        self.control_log_path = control_log_path
        _reset_control_log(control_log_path)
        self.gate = SentinelGate(policy)
        self.failsafe = FailsafeController(policy)
        # FIX: policy now threaded through so the Strategist reads real
        # season thresholds / baseline value instead of hardcoding them
        # (see agents/strategist.py, core/seasonality.py).
        self.strategist = Strategist(model=policy["strategist"]["model"], policy=policy)
        self.cadence_steps = policy["strategist"]["cadence_steps"]
        # FIX (dead config): policy["strategist"]["call_timeout_s"]/
        # "correction_timeout_s" were declared in building_policy.yaml
        # but never actually reached Strategist.decide() -- both calls
        # below used decide()'s hardcoded default (timeout=10) instead.
        # Read once here so the values in the YAML are the values
        # actually used.
        self.call_timeout_s = float(policy["strategist"].get("call_timeout_s", 10))
        self.correction_timeout_s = float(policy["strategist"].get("correction_timeout_s", 5))
        # Priority 3a: see config/building_policy.yaml's strategist.correction_skip_delta_c.
        self.correction_skip_delta_c = float(policy["strategist"].get("correction_skip_delta_c", 0.0))
        # Rework Blueprint 5.5 -- Phase 1: state-bin decision cache. See
        # core/decision_cache.py's module docstring for the full
        # reasoning; wired in below inside decide().
        self.decision_cache = DecisionCache(policy)
        # Rework Blueprint 5.2 -- Phase 2: event-triggered scheduler. See
        # core/trigger_engine.py's module docstring for the full
        # reasoning; wired in below inside decide(), replacing the old
        # `step % self.cadence_steps != 0: return ..., "Holding"` gate.
        self.trigger_engine = TriggerEngine(policy)
        # Rework Blueprint 5.2: with firings no longer tied to fixed
        # cadence, "cadence ticks" (the unit core/decision_cache.py's
        # ttl_ticks is expressed in) now means "slow-loop firings" --
        # incremented once per decide() call that actually reaches the
        # cache/Strategist below, regardless of what triggered it.
        self._decision_counter = 0
        self.timestep_minutes = float(policy["strategist"].get("timestep_minutes", 15))
        self.last_setpoint = 22.0
        # Rework Blueprint 5.3 -- Phase 3: the currently-committed
        # trajectory (list of {"offset_min","setpoint"} anchors, sorted
        # ascending) the fast loop interpolates between on every physical
        # step where the slow loop does NOT fire, plus the physical step
        # at which it was committed (interpolation offsets are measured
        # from here). None until the first real decision -- see
        # _interpolate_trajectory()/_set_flat_trajectory() below.
        self._trajectory = None
        self._trajectory_start_step = 0
        self._bridge = None

    def attach_bridge(self, bridge):
        self._bridge = bridge
        self.strategist.tool_context = ToolContext(bridge=bridge, policy=self.policy)

    def _get_carbon_intensity(self, step):
        return carbon_intensity_level(step, self.policy)

    def _set_flat_trajectory(self, step, setpoint):
        """Rework Blueprint 5.3: reset the committed trajectory to a
        single flat anchor at `setpoint`, starting now. Used whenever
        this tick's outcome is NOT a freshly-approved trajectory (a
        quick_check trip, a FAILSAFE override, a skipped/failed
        correction) -- the fast loop must not keep interpolating toward a
        stale plan that was superseded by a safety override, so this
        gives it a safe, unambiguous "just hold here" target instead."""
        self._trajectory = [{"offset_min": 0.0, "setpoint": float(setpoint)}]
        self._trajectory_start_step = step

    def _interpolate_trajectory(self, step):
        """Rework Blueprint 5.3 -- the fast loop's half of the two-clock
        architecture. Called on every physical step where the slow loop
        does NOT fire this tick (the "Holding" path in decide() below) so
        the actuator glides between the last committed trajectory's
        anchors instead of holding perfectly flat until the next
        trigger/cadence fire and then jumping.

        Deliberately does NOT re-invoke SentinelGate: the trajectory was
        already approved as a whole when its offset_min=0 anchor cleared
        the gate (see decide() below) -- interpolating between two
        already-approved-adjacent anchors is a pure fast-loop control
        action, not a new AI decision. The result IS still clamped to the
        gate's hard safety band (temp_min_c/temp_max_c), the same
        Blueprint 5.1 "local deterministic safety check" spirit as
        quick_check(), since only the offset_min=0 anchor was individually
        gate-checked -- later anchors came from the same tool call and
        were never independently verified against the hard bounds.
        """
        if not self._trajectory:
            return self.last_setpoint

        anchors = self._trajectory
        elapsed_min = (step - self._trajectory_start_step) * self.timestep_minutes

        if elapsed_min <= anchors[0]["offset_min"]:
            value = anchors[0]["setpoint"]
        elif elapsed_min >= anchors[-1]["offset_min"]:
            value = anchors[-1]["setpoint"]
        else:
            value = anchors[-1]["setpoint"]  # overwritten below unless something's malformed
            for a, b in zip(anchors, anchors[1:]):
                if a["offset_min"] <= elapsed_min <= b["offset_min"]:
                    span = b["offset_min"] - a["offset_min"]
                    if span <= 0:
                        value = b["setpoint"]
                    else:
                        frac = (elapsed_min - a["offset_min"]) / span
                        value = a["setpoint"] + frac * (b["setpoint"] - a["setpoint"])
                    break

        return min(max(value, self.gate.temp_min_c), self.gate.temp_max_c)

    @staticmethod
    def _tag_source(label, llm_ok):
        # Fix (silent-fallback visibility): previously every approved
        # proposal was logged as plain "AI"/"AI (Corrected)" regardless of
        # whether it came from a real Groq tool-call or the hardcoded
        # physical-target fallback. The whole year-long log turned out to
        # be indistinguishable either way -- this is what makes them
        # distinguishable again.
        return label if llm_ok else f"{label} (Fallback)"

    def decide(self, t_in, t_out, humidity, zone_name=None):
        # zone_name: accepted so EnergyPlusBridge can call this uniformly
        # whether it's driving one zone or several (Blueprint 1.1) --
        # unused here because a single ProjectEnvelope instance already
        # belongs to exactly one zone in --multizone mode (see
        # MultiZoneOrchestrator below), and to the implicit single zone
        # otherwise. Recorded in the log line purely for traceability.
        step = self._bridge.step_counter

        # Rework Blueprint 5.1 -- Phase 0: fast-loop safety net. Runs on
        # EVERY physical step, unconditionally, BEFORE the cadence gate
        # below -- this is what makes the system genuinely "24/7
        # surveillance" rather than surveillance-only-at-reasoning-time.
        # Deliberately placed ahead of the cadence check: the old code
        # returned early on non-cadence steps with zero verification at
        # all, so a real drift between cadence ticks would go completely
        # unwatched until the next scheduled LLM call. See
        # SentinelGate.quick_check()'s docstring for the full reasoning.
        # Cheap by construction (no LLM call, no PMV/CCS scoring) so
        # running it every step is not a performance concern.
        quick_tripped, quick_reason = self.gate.quick_check(t_in)
        if quick_tripped:
            self.last_setpoint = self.failsafe.decide(t_in)
            # Rework Blueprint 5.3: a quick_check trip is a hard safety
            # override -- the fast loop must not keep interpolating
            # toward whatever trajectory was in flight before the trip
            # (it may be exactly what caused the drift). Reset to a flat
            # hold at the safe failsafe value so post-trip Holding steps
            # resume from here, not from a stale plan.
            self._set_flat_trajectory(step, self.last_setpoint)
            # NOT run through _tag_source(): that helper's "(Fallback)"
            # suffix specifically means "the LLM was asked and failed."
            # A quick_check trip never asks the LLM anything at this tick
            # at all -- it's a pre-emptive hard-bounds intervention, not
            # an LLM fallback. Keeping the label distinct matters for the
            # explainability log (Blueprint Section 2 / 5.6): a judge
            # reading the log should be able to tell "the LLM failed" and
            # "the fast safety loop caught a real drift" apart at a glance.
            source = "FAILSAFE (QuickCheck)"
            log_entry = {
                "step": int(step), "t_in": round(t_in, 2), "t_out": round(t_out, 2),
                "setpoint": float(self.last_setpoint), "source": source, "reason": quick_reason,
                "ccs": None, "lookahead_triggered": False,
                "cumulative_kwh": round(self._bridge.cumulative_kwh, 4),
                "cumulative_heat_kwh": round(self._bridge.cumulative_heat_kwh, 4),
                "cumulative_cool_kwh": round(self._bridge.cumulative_cool_kwh, 4),
                "zone": zone_name or self._bridge.zone_names[0],
                "carbon": None, "carbon_score": None,
                "llm_ok": None, "fallback_error": None,
                "cache_hit": False,  # schema consistency with the main log entry below
                "trigger_reason": None,  # quick_check trips are outside the slow loop -- see main log entry below
                "trajectory": self._trajectory,  # Rework Blueprint 5.3: flat reset -- see _set_flat_trajectory() above
            }
            # Same defense-in-depth reasoning as the main log write further
            # down: a logging hiccup must never be why a physically-safe
            # decision fails to reach the actuator.
            try:
                with open(self.control_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as log_err:
                print(f"[ProjectEnvelope] WARNING: failed to write control log "
                      f"entry for step {step} (quick-check trip): {log_err}")
            return self.last_setpoint, source

        # hour_of_day/season computed up front now -- the old cadence gate
        # didn't need them until after it passed, but the Rework Blueprint
        # 5.2 trigger evaluator needs season for its schedule_boundary
        # check BEFORE deciding whether to fire at all.
        hour_of_day = (step * self.timestep_minutes / 60.0) % 24.0
        season = classify_season(t_out, self.policy)

        # Rework Blueprint 5.2 -- Phase 2: event-triggered scheduler,
        # replacing the old `if step % self.cadence_steps != 0: return
        # ..., "Holding"` gate. Fires the slow loop on deviation /
        # forecast-shift / schedule-boundary / max-staleness instead of
        # blind fixed cadence -- see core/trigger_engine.py's module
        # docstring. A disabled trigger engine (strategist.trigger.
        # enabled: false) reproduces the exact old fixed-cadence
        # behavior via its own cadence_ceiling fallback.
        should_fire, trigger_reason = self.trigger_engine.evaluate(step, t_in, t_out, hour_of_day, season)
        if not should_fire:
            # Rework Blueprint 5.3: the fast loop interpolates along the
            # currently-committed trajectory instead of holding perfectly
            # flat -- see _interpolate_trajectory()'s docstring.
            self.last_setpoint = self._interpolate_trajectory(step)
            return self.last_setpoint, "Holding"

        forecast = self._bridge.get_forward_weather(3)
        carbon = self._get_carbon_intensity(step)
        # Fix: carbon_score is now a real numeric value derived from the
        # same "Low/Medium/High" label the Strategist sees, and gets
        # threaded into gate.check() below -- previously nothing was
        # passed and the gate's carbon term was a hardcoded constant.
        carbon_score = carbon_score_from_level(carbon, self.policy)
        # Fed to the gate's opt-in baseline-direction guardrail below --
        # see core/seasonality.py and config/building_policy.yaml's
        # gate.enforce_baseline_direction (off by default).
        baseline_setpoint = get_baseline_setpoint(self.policy)

        # Rework Blueprint 5.5 -- Phase 1: state-bin decision cache.
        # tick_index counts SLOW-LOOP FIRINGS (see self._decision_counter's
        # comment in __init__), matching the same convention
        # hysteresis.min_dwell_steps already uses (a monotonically
        # increasing "how many decisions have happened" counter, not raw
        # physical steps) -- see core/decision_cache.py's module docstring
        # for why that unit matters.
        self._decision_counter += 1
        tick_index = self._decision_counter
        cache_key_args = (t_in, t_out, hour_of_day, season)
        cached_proposal = self.decision_cache.get(*cache_key_args, tick_index)

        source = "FAILSAFE"
        reason = "System Init"
        final_ccs = None
        lookahead_triggered = False
        llm_ok = None          # None until we know; logged for diagnostics
        fallback_error = None
        cache_hit = cached_proposal is not None

        try:
            # 1. AI Reasoning -- or, on a cache hit, reuse a recent
            # APPROVED decision for a close-enough state bin instead of
            # spending a real LLM call (see core/decision_cache.py).
            # lookahead_triggered stays False on a cache hit: no forecast
            # scan actually ran this tick (that's the whole point of the
            # cache), so there's nothing real to report there -- the fast
            # loop's quick_check() (Phase 0) is still watching every
            # physical step regardless.
            if cache_hit:
                proposal = cached_proposal
                llm_ok = proposal.get("llm_ok", True)
                fallback_error = proposal.get("error")
            else:
                proposal = self.strategist.decide(t_in, t_out, forecast, carbon, timeout=self.call_timeout_s)
                lookahead_triggered = getattr(self.strategist, "last_lookahead_triggered", False)
                llm_ok = proposal.get("llm_ok", True)
                fallback_error = proposal.get("error")

            # 2. Gating -- a cached proposal is re-verified against THIS
            # tick's actual measurements exactly like a fresh one; the
            # cache never bypasses the gate, only the LLM call.
            outcome, ccs, reason, _ = self.gate.check(
                proposal["setpoint"], self.last_setpoint, proposal["confidence"],
                t_in, humidity, t_out, carbon_score=carbon_score,
                season=season, baseline_setpoint=baseline_setpoint,
            )
            final_ccs = ccs

            if outcome == "APPROVED":
                self.last_setpoint = proposal["setpoint"]
                # Rework Blueprint 5.3: commit the approved trajectory
                # (or a degenerate 1-anchor one -- see
                # agents/strategist.py's _parse_trajectory()) for the
                # fast loop to interpolate along until the next firing.
                self._trajectory = proposal.get("trajectory") or [
                    {"offset_min": 0.0, "setpoint": proposal["setpoint"]}]
                self._trajectory_start_step = step
                source = self._tag_source("AI (Cached)" if cache_hit else "AI", llm_ok)
                # Store (or refresh the TTL of) this bin only with an
                # already-APPROVED, post-gate proposal -- see
                # DecisionCache.put()'s docstring.
                self.decision_cache.put(*cache_key_args, tick_index, proposal)
            elif outcome == "HOLD":
                source = self._tag_source("AI (Stabilized)", llm_ok)
                final_ccs = 1.0
            else:
                # Rework Blueprint 5.5: a cached proposal that gets
                # REJECTED on replay means this state bin's cached
                # decision is no longer trustworthy -- drop it immediately
                # rather than letting other ticks landing in the same bin
                # keep reusing a decision that just failed re-verification.
                if cache_hit:
                    self.decision_cache.invalidate(*cache_key_args)
                # Priority 3a: a REJECTED proposal that's already within
                # correction_skip_delta_c of the current setpoint is very
                # unlikely to be meaningfully rescued by a second real LLM
                # call -- skip it and go straight to FAILSAFE, saving the
                # tokens for ticks with an actual chance of a corrected
                # APPROVED outcome. Delta is measured against the FIRST
                # proposal (the one that was just rejected), not a guess
                # at what the correction pass would have returned.
                delta = abs(proposal["setpoint"] - self.last_setpoint)
                if self.correction_skip_delta_c > 0 and delta <= self.correction_skip_delta_c:
                    self.last_setpoint, source = self.failsafe.decide(t_in), "FAILSAFE"
                    reason = (f"Correction skipped (delta {delta:.2f}C <= "
                              f"{self.correction_skip_delta_c:.2f}C tolerance): {reason}")
                    self._set_flat_trajectory(step, self.last_setpoint)
                else:
                    # 3. Correction -- correction_context now actually reaches
                    # the prompt (see agents/strategist.py), so this is a real
                    # second attempt informed by the rejection reason, not a
                    # verbatim repeat of the first one.
                    corrected = self.strategist.decide(t_in, t_out, forecast, carbon, correction_context=reason, timeout=self.correction_timeout_s)
                    llm_ok = corrected.get("llm_ok", True)
                    fallback_error = corrected.get("error")
                    outcome2, ccs2, reason2, _ = self.gate.check(
                        corrected["setpoint"], self.last_setpoint, corrected["confidence"],
                        t_in, humidity, t_out, carbon_score=carbon_score,
                        season=season, baseline_setpoint=baseline_setpoint,
                    )
                    final_ccs = ccs2
                    if outcome2 == "APPROVED":
                        self.last_setpoint, source, reason = corrected["setpoint"], self._tag_source("AI (Corrected)", llm_ok), reason2
                        # Same Blueprint 5.5 rule as the first-attempt path:
                        # only ever cache an already-APPROVED, post-gate
                        # proposal -- a corrected proposal that clears the
                        # gate is just as legitimate a cache entry as a
                        # first-attempt one.
                        self._trajectory = corrected.get("trajectory") or [
                            {"offset_min": 0.0, "setpoint": corrected["setpoint"]}]
                        self._trajectory_start_step = step
                        self.decision_cache.put(*cache_key_args, tick_index, corrected)
                    else:
                        self.last_setpoint, source, reason = self.failsafe.decide(t_in), "FAILSAFE", f"Gate Override: {reason2}"
                        self._set_flat_trajectory(step, self.last_setpoint)
        except Exception as e:
            self.last_setpoint, source, reason = self.failsafe.decide(t_in), "FAILSAFE", f"Error: {e}"
            self._set_flat_trajectory(step, self.last_setpoint)

        # Rework Blueprint 5.2: record this as the new "last decision"
        # reference point for the trigger engine's NEXT evaluate() call,
        # regardless of how this tick's decision resolved (cache hit,
        # real LLM call, correction, or an exception falling through to
        # FAILSAFE) -- the slow loop DID run this step, so the drift
        # clock genuinely resets here. Deliberately outside the try/
        # except above: a FAILSAFE-via-exception outcome still means the
        # slow loop consumed this event, and the next evaluate() should
        # measure drift from here, not from whenever it last succeeded.
        self.trigger_engine.note_decision(step, t_in, t_out, hour_of_day, season)

        # 4. Logging
        log_entry = {
            "step": int(step), "t_in": round(t_in, 2), "t_out": round(t_out, 2),
            "setpoint": float(self.last_setpoint), "source": source, "reason": reason,
            "ccs": round(final_ccs, 3) if final_ccs is not None else None,
            "lookahead_triggered": lookahead_triggered,
            "cumulative_kwh": round(self._bridge.cumulative_kwh, 4),
            # Part 1b: split so a cooling-ceiling clamp event (Part 1a)
            # is directly verifiable in kWh, not just inferred from the
            # combined total.
            "cumulative_heat_kwh": round(self._bridge.cumulative_heat_kwh, 4),
            "cumulative_cool_kwh": round(self._bridge.cumulative_cool_kwh, 4),
            "zone": zone_name or self._bridge.zone_names[0],
            "carbon": carbon,
            "carbon_score": round(carbon_score, 3),
            # Diagnostics -- see agents/strategist.py's llm_ok/error fields.
            # llm_ok is None only for the "Holding" no-op path above, which
            # returns before this point.
            "llm_ok": llm_ok,
            "fallback_error": fallback_error,
            # Rework Blueprint 5.5: whether this tick's proposal came from
            # the state-bin cache instead of a real Strategist/LLM call --
            # makes the cache's effect directly verifiable in the log
            # instead of only inferrable from call counts.
            "cache_hit": cache_hit,
            # Rework Blueprint 5.2: why the slow loop fired this step
            # (deviation / forecast_shift / schedule_boundary /
            # max_staleness / cadence_ceiling / initial_decision) -- see
            # core/trigger_engine.py. This is the raw data Phase 5 (5.6)
            # will surface as a dedicated dashboard column; captured here
            # already since it's a direct byproduct of this phase's wiring.
            "trigger_reason": trigger_reason,
            # Rework Blueprint 5.3: the trajectory committed THIS tick
            # (list of {"offset_min","setpoint"} anchors) -- what
            # main.py's fast loop will interpolate along on subsequent
            # "Holding" steps until the next slow-loop firing. None only
            # if something failed before any trajectory (even a flat
            # fallback one) was set, which should not be reachable.
            "trajectory": self._trajectory,
        }
        # BUGFIX (defense in depth): this write happens after the try/except
        # above, so it wasn't itself protected -- any serialization hiccup
        # here (e.g. a stray NaN/Inf slipping into the dict) would propagate
        # straight up through EnergyPlusBridge._callback and kill the whole
        # simulation, which is indistinguishable from "the AI instance
        # crashed" to anyone watching the dashboard. Never let logging
        # itself be why a physically-safe decision (self.last_setpoint is
        # already resolved by this point) fails to reach the actuator.
        try:
            with open(self.control_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as log_err:
            print(f"[ProjectEnvelope] WARNING: failed to write control log "
                  f"entry for step {step}: {log_err}")

        return self.last_setpoint, source

class BaselineOrchestrator:
    """Non-AI comparison instance."""
    def __init__(self, policy, control_log_path):
        self.controller = BaselineController(policy)
        self.control_log_path = control_log_path
        _reset_control_log(control_log_path)
        self._bridge = None
    def attach_bridge(self, bridge):
        self._bridge = bridge
    def decide(self, t_in, t_out, humidity, zone_name=None):
        step = self._bridge.step_counter
        setpoint = self.controller.decide(t_in)
        # BUGFIX (audit finding: baseline runs reported "stopping"/
        # freezing mid-run): unlike ProjectEnvelope.decide(), which wraps
        # its own control_log_path write in try/except so a transient I/O
        # hiccup can never propagate out of the callback, this write was
        # unprotected -- any failure here (disk full, permission hiccup,
        # a stray non-JSON-serializable value) would raise straight up
        # through EnergyPlusBridge._callback and into the EnergyPlus C
        # callback, which typically aborts the WHOLE simulation rather
        # than showing a normal Python traceback -- indistinguishable
        # from "it just stopped." A physically-safe setpoint has already
        # been decided by this point; logging failing is never a reason
        # to take down the run.
        try:
            with open(self.control_log_path, "a") as f:
                f.write(json.dumps({"step": int(step), "t_in": t_in, "t_out": t_out, "setpoint": setpoint,
                                   "source": "Baseline", "reason": "Standard Schedule",
                                   "cumulative_kwh": self._bridge.cumulative_kwh,
                                   "cumulative_heat_kwh": self._bridge.cumulative_heat_kwh,
                                   "cumulative_cool_kwh": self._bridge.cumulative_cool_kwh,
                                   "zone": zone_name or self._bridge.zone_names[0]}) + "\n")
        except Exception as log_err:
            print(f"[BaselineOrchestrator] WARNING: failed to write control log "
                  f"entry for step {step}: {log_err}")
        return setpoint, "Baseline"

def _zone_slug(zone_name):
    """'ZONE ONE' -> 'zone_one', for log filenames."""
    return zone_name.strip().lower().replace(" ", "_")


class MultiZoneOrchestrator:
    """Blueprint 1.1. Owns one COMPLETE ProjectEnvelope stack (its own
    Strategist, SentinelGate, FailsafeController, and log file) PER ZONE,
    so each zone gets an independent proposal and an independent
    accept/reject decision on the same simulated tick -- this is what
    proves the pattern generalizes across zones with independent,
    sometimes-conflicting proposals, rather than one decision fanned out
    to every zone."""

    def __init__(self, policy, zone_names, log_dir):
        self.zones = {
            zone: ProjectEnvelope(policy, os.path.join(log_dir, f"{_zone_slug(zone)}.jsonl"))
            for zone in zone_names
        }
        self._bridge = None

    def attach_bridge(self, bridge):
        self._bridge = bridge
        for env in self.zones.values():
            env.attach_bridge(bridge)

    def decide(self, t_in, t_out, humidity, zone_name=None):
        if zone_name is None or zone_name not in self.zones:
            # Defensive fallback -- EnergyPlusBridge always passes a real
            # zone_name for every configured zone, so this should not be
            # reachable in practice.
            zone_name = next(iter(self.zones))
        return self.zones[zone_name].decide(t_in, t_out, humidity, zone_name=zone_name)


def _run_and_report(bridge, label, output_dir):
    """BUGFIX (audit finding: reports of a run "crashing after ~15 min" or
    "just stopping" with no clear signal either way): previously
    bridge.run() was called with nothing before or after it, so a normal
    full-year completion and a mid-run crash looked identical from the
    terminal -- both just silently returned (or didn't). This wraps the
    call so the three real outcomes are unambiguous: a clean finish
    prints a summary with final step/kWh/wall-clock time; an exception
    prints what failed and re-raises (preserving a non-zero exit code
    for scripts like calibrate_ccs_sweep.py / generate_evidence.py that
    check it); and if EnergyPlus itself aborts (a fatal error inside its
    own C++ engine, e.g. from exceeding its internal severe-error budget)
    this can't catch that -- but points you at the exact file that will
    say so.
    """
    start = time.perf_counter()
    print(f"[main] Starting {label} run...")
    try:
        bridge.run()
    except Exception:
        elapsed = time.perf_counter() - start
        print(f"[main] {label} run RAISED an exception after {elapsed / 60.0:.1f} min "
              f"(step {bridge.step_counter}). Re-raising -- see the traceback below. "
              f"If nothing below looks like a Python error, check "
              f"{os.path.join(output_dir, 'eplusout.err')} for a '**Fatal**' line -- "
              f"that means EnergyPlus itself aborted the run (e.g. exceeded its own "
              f"internal severe-error budget), which a Python try/except can't catch.")
        raise
    elapsed = time.perf_counter() - start
    print(f"[main] {label} run COMPLETE: {bridge.step_counter} steps, "
          f"{bridge.cumulative_kwh:.2f} cumulative kWh "
          f"(heat: {bridge.cumulative_heat_kwh:.2f} kWh, cool: {bridge.cumulative_cool_kwh:.2f} kWh), "
          f"{elapsed / 60.0:.1f} min wall-clock.")


def main():
    policy = load_policy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                         help="Run the fixed-schedule comparison instance instead of the AI-gated one.")
    parser.add_argument("--multizone", action="store_true",
                         help="Blueprint 1.1: run BOTH zones of models/two_zone_controlled.idf "
                              "(generate it first with `python scripts/patch_idf.py`), each with its "
                              "own independent Strategist+SentinelGate+Failsafe stack. Mutually "
                              "exclusive with --baseline -- the multi-zone demo is AI-gated only.")
    parser.add_argument("--run-period-days", type=int, default=None,
                         help="Audit fix for wall-clock infeasibility: shorten the "
                              "simulation's RunPeriod to this many days (starting Jan 1) "
                              "instead of the full year. At the default cadence_steps=12 / "
                              "max_requests_per_minute=25, a full year is ~2,920 Strategist "
                              "calls and can take hours; e.g. --run-period-days 14 finishes "
                              "in minutes. Writes a sibling *.quickNd.idf next to the real "
                              "idf -- the original is never modified. Iteration only: do "
                              "NOT use this for the numbers you put on a slide.")
    parser.add_argument("--cadence-override", type=int, default=None,
                         help="Override config/building_policy.yaml's strategist.cadence_steps "
                              "for this run only (the YAML file itself is never modified). "
                              "Useful for a fast smoke test (a LOW value, e.g. 4, fires many "
                              "Strategist/Gate decisions quickly) vs. a real run (leave unset "
                              "to use the tuned default of 12). NOTE: hysteresis.min_dwell_steps "
                              "is counted in CADENCE TICKS, not raw steps, so a high override "
                              "also stretches out the minimum dwell time between setpoint "
                              "changes by the same factor -- see building_policy.yaml's comment.")
    args = parser.parse_args()

    if args.cadence_override is not None:
        if args.cadence_override < 1:
            parser.error("--cadence-override must be >= 1.")
        old = policy["strategist"]["cadence_steps"]
        policy["strategist"]["cadence_steps"] = args.cadence_override
        print(f"[main] --cadence-override {args.cadence_override}: strategist cadence_steps "
              f"{old} -> {args.cadence_override} for this run only "
              f"(building_policy.yaml on disk is untouched).")
        min_steps_for_one_tick = args.cadence_override
        min_days_for_one_tick = min_steps_for_one_tick * 15 / 60 / 24
        if args.run_period_days is not None and args.run_period_days < min_days_for_one_tick:
            print(f"[main] WARNING: --run-period-days {args.run_period_days} is shorter than "
                  f"one cadence tick ({min_days_for_one_tick:.1f} days at cadence "
                  f"{args.cadence_override}) -- the Strategist/Gate will NEVER be called this "
                  f"run. Every decision will just hold the initial setpoint. Raise "
                  f"--run-period-days above {min_days_for_one_tick:.1f} if you actually want "
                  f"to exercise the AI logic at this cadence.")

    if args.run_period_days:
        print(f"[main] --run-period-days {args.run_period_days}: using a shortened "
              f"{args.run_period_days}-day RunPeriod for faster iteration. This is NOT "
              f"the full year -- do not treat savings/violations numbers from this run "
              f"as final evidence.")

    if args.multizone and args.baseline:
        parser.error("--multizone and --baseline are mutually exclusive; "
                      "the multi-zone demo runs the AI-gated stack on every zone.")

    if args.multizone:
        zone_names = policy.get("multizone", {}).get("zone_names", ["ZONE ONE", "ZONE TWO"])
        log_dir = "logs/multizone"
        os.makedirs(log_dir, exist_ok=True)

        orchestrator = MultiZoneOrchestrator(policy, zone_names, log_dir)

        two_zone_idf = policy.get("paths", {}).get("two_zone_idf", "models/two_zone_controlled.idf")
        if args.run_period_days:
            two_zone_idf = _quick_idf_path(two_zone_idf, args.run_period_days)

        # 2.1's BACnet mirroring is a single-point integration; with two+
        # zones proposing independent setpoints there's no single "the"
        # setpoint to mirror, so multi-zone runs stay BACnet-free until a
        # per-zone point mapping is built (see energyplus_bridge.py's
        # constructor warning for what happens if this is set anyway).
        bridge = EnergyPlusBridge(
            idf=two_zone_idf,
            epw=policy["paths"]["epw"],
            output=log_dir,
            decision_callback=orchestrator.decide,
            zone_names=zone_names,
            bacnet_adapter=None,
            policy=policy,
        )
        orchestrator.attach_bridge(bridge)
        _run_and_report(bridge, "multizone", log_dir)
        return

    mode = "baseline" if args.baseline else "ai"
    log_path = f"logs/{mode}/control_log.jsonl"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    orchestrator = ProjectEnvelope(policy, log_path) if mode == "ai" else BaselineOrchestrator(policy, log_path)

    # 2.1 -- only the AI-gated instance mirrors to a real/simulated BACnet
    # point (config/building_policy.yaml: bacnet.enabled). The baseline
    # counterfactual is a comparison instance, not something a real BMS
    # would ever drive a physical point with, so it stays BACnet-free.
    bacnet_adapter = build_adapter_from_policy(policy) if mode == "ai" else None
    if bacnet_adapter is not None:
        bacnet_adapter.connect()

    controlled_idf = "models/controlled.idf"
    if args.run_period_days:
        controlled_idf = _quick_idf_path(controlled_idf, args.run_period_days)

    bridge = EnergyPlusBridge(
        idf=controlled_idf, 
        epw=policy["paths"]["epw"], 
        output=f"logs/{mode}", 
        decision_callback=orchestrator.decide,
        bacnet_adapter=bacnet_adapter,
        policy=policy,
    )
    orchestrator.attach_bridge(bridge)
    _run_and_report(bridge, mode, f"logs/{mode}")

if __name__ == "__main__":
    main()