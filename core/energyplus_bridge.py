"""
core/energyplus_bridge.py
"""

import sys
import os
import time

from core.seasonality import classify_season, get_baseline_setpoint, WINTER, SUMMER


_HEATING_METER_CANDIDATES = ["DistrictHeatingWater:Facility", "DistrictHeating:Facility"]
_COOLING_METER_CANDIDATES = ["DistrictCooling:Facility"]

# Simulated-time values are floats; two firings for "the same instant" can
# differ by float noise in the 1e-9 range. Round before comparing.
_SIM_TIME_ROUND_DP = 6


def _load_energyplus_api():
    eplus_dir = os.environ.get("EPLUS_DIR", r"C:\EnergyPlusV24-1-0")
    if eplus_dir not in sys.path:
        sys.path.insert(0, eplus_dir)
    try:
        from pyenergyplus.api import EnergyPlusAPI
        return EnergyPlusAPI
    except ImportError as e:
        raise ImportError(
            f"Could not import pyenergyplus from {eplus_dir}. "
            f"Set the EPLUS_DIR environment variable to your EnergyPlus "
            f"install directory."
        ) from e


class EnergyPlusBridge:
    def __init__(self, idf, epw, output, decision_callback, label="AI",
                 zone_name="ZONE ONE", zone_names=None, deadband_c=2.0,
                 track_energy=True, bacnet_adapter=None, policy=None):
        """
        decision_callback: function(indoor_temp_c, outdoor_temp_c,
                            humidity_pct, zone_name=<str>) -> (setpoint_c,
                            source_str). Called once per DISTINCT
                            simulated timestep PER ZONE (see the dedupe
                            fix above, and the multi-zone fix note #5) —
                            owns all Strategist/Gate/Failsafe or
                            baseline-schedule logic. Injected so this file
                            has no dependency on main.py. Callables that
                            don't accept the `zone_name` kwarg still work
                            for the single-zone case (see _call_decision).
        zone_name / zone_names: pass ONE of these. `zone_name` (singular,
                    default "ZONE ONE") is kept for backward compatibility
                    and is equivalent to `zone_names=[zone_name]`. Pass
                    `zone_names=["ZONE ONE", "ZONE TWO"]` (matching
                    models/two_zone_controlled.idf from
                    scripts/patch_idf.py's patch_two_zone()) to drive
                    multiple independent zones in one bridge/process.
        deadband_c: the AI/baseline setpoint is treated as the COOLING
                    target; heating target = setpoint - deadband. Comes
                    from config/building_policy.yaml's comfort.deadband_c.
                    Applied identically to every zone.
        policy: BUGFIX (see chat -- "cooling-ceiling guardrail"). Deriving
                    BOTH the cooling target (= setpoint, unmodified) and
                    the heating target (= setpoint - deadband/2) from one
                    scalar means a winter setpoint chosen purely for its
                    heating-side savings ALSO lowers the cooling ceiling
                    below baseline's -- so on any tick with enough solar/
                    internal gain to approach that ceiling, the AI draws
                    cooling energy baseline never would, quietly eating
                    the heating-side savings. core/seasonality.py's
                    violates_baseline_direction() guardrail only checked
                    the heating framing and never caught this -- it's a
                    real bug independent of Groq/LLM reliability. When
                    `policy` is provided, WINTER ticks clamp the cooling
                    target to never go below the baseline's own setpoint,
                    and SUMMER ticks symmetrically clamp the heating
                    target to never go above baseline's -- so a proposal
                    can only ever be as-good-or-better than baseline on
                    BOTH sides of the deadband, never accidentally worse
                    on the side it wasn't reasoning about. Left None
                    (no clamp, previous behavior) for callers that don't
                    pass a policy dict, e.g. ad-hoc/test bridges.
        bacnet_adapter: optional integrations.bacnet_adapter.BACnetAdapter.
                    When set, every approved setpoint applied to the
                    EnergyPlus actuator (2.1) is ALSO mirrored to this
                    adapter's real/simulated BACnet point, in the same
                    callback, right after the simulation write -- so the
                    "hardware path" and the "digital twin path" are driven
                    by the exact same setpoint value on every tick. Left
                    None by default: the baseline counterfactual instance
                    never gets one, since a real BMS wouldn't run two
                    control loops against one physical point. NOTE: this
                    mirrors a single point, so it's only meaningful with
                    exactly one zone -- multi-zone runs should leave this
                    None (main.py does) until a per-zone BACnet mapping
                    is built.
        """
        EnergyPlusAPI = _load_energyplus_api()
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        self.idf = idf
        self.epw = epw
        self.output = output
        self.decision_callback = decision_callback
        self.label = label
        self.zone_names = list(zone_names) if zone_names else [zone_name]
        self.zone_name = self.zone_names[0]  # back-compat single-zone accessor
        self.deadband_c = float(deadband_c)
        self.policy = policy
        self.track_energy = track_energy
        self.bacnet_adapter = bacnet_adapter
        if bacnet_adapter is not None and len(self.zone_names) > 1:
            print(f"[{label}] WARNING: bacnet_adapter is set with "
                  f"{len(self.zone_names)} zones configured. It will mirror "
                  f"every zone's setpoint onto the SAME BACnet point in "
                  f"sequence, which is almost certainly not what you want. "
                  f"Pass bacnet_adapter=None for multi-zone runs.")

        # Exposed for mcp/tools.py's get_state()/get_weather() and for
        # ui/app.py's dashboard. In multi-zone mode these track the LAST
        # zone processed on a given tick (see per_zone_state for the full
        # per-zone breakdown) -- kept for single-zone backward compat.
        self.last_indoor_temp = None
        self.last_outdoor_temp = None
        self.last_humidity = None
        self.last_setpoint = 22.0
        self.step_counter = 0
        self.pending_setpoint = None
        self.cumulative_kwh = 0.0
        self.cumulative_heat_kwh = 0.0
        self.cumulative_cool_kwh = 0.0

        # Per-zone state, always populated (single-zone runs just have one
        # key). per_zone_state[zone] = {"t_in", "t_out", "humidity",
        # "setpoint", "source"}; zone_kwh[zone] = best-effort per-zone
        # cumulative kWh (see fix note #5 -- may stay 0.0). Part 1b: split
        # into zone_heat_kwh/zone_cool_kwh so the cooling-ceiling clamp's
        # effect is directly measurable per zone; zone_kwh kept as their
        # sum for anything still reading the combined figure.
        self.per_zone_state = {z: {} for z in self.zone_names}
        self.zone_kwh = {z: 0.0 for z in self.zone_names}
        self.zone_heat_kwh = {z: 0.0 for z in self.zone_names}
        self.zone_cool_kwh = {z: 0.0 for z in self.zone_names}

        self._epw_forecast = None
        self._handles_resolved = False
        # Per-zone handle storage: {zone_name: {"t_in":.., "humidity":..,
        # "cooling_actuator":.., "heating_actuator":.., "ideal_heat_energy":..,
        # "ideal_cool_energy":..}}
        self._zone_handles = {}
        self._t_out_handle = None
        self._heating_meter_handle = -1
        self._cooling_meter_handle = -1
        self._energy_meter_warned = False
        self._per_zone_energy_warned = False

        self._last_sim_time_key = None

        self._heartbeat_interval_s = 30.0
        self._last_heartbeat_t = time.perf_counter()
        self._run_start_t = time.perf_counter()

    def get_forward_weather(self, hours=3):
        """
        Returns the next `hours` outdoor dry-bulb temps from the .epw file
        already loaded by EnergyPlus - a deterministic proxy for a live
        forecast, honestly labeled as such wherever it's shown/logged.
        """
        if self._epw_forecast is None:
            self._epw_forecast = self._parse_epw_drybulb(self.epw)
        hour_index = min(self.step_counter // 6, len(self._epw_forecast) - 1)
        return self._epw_forecast[hour_index: hour_index + hours]

    @staticmethod
    def _parse_epw_drybulb(epw_path):
        temps = []
        with open(epw_path, "r", encoding="latin-1") as f:
            lines = f.readlines()
        for line in lines[8:]:
            parts = line.strip().split(",")
            if len(parts) > 6:
                try:
                    temps.append(float(parts[6]))
                except ValueError:
                    continue
        return temps or [20.0]

    def _resolve_handles(self, state_ptr):
        """Resolve all variable/actuator/meter handles once, after warmup.
        Loops over self.zone_names so single- and multi-zone runs share
        one code path."""
        self._t_out_handle = self.api.exchange.get_variable_handle(
            state_ptr, "Site Outdoor Air Drybulb Temperature", "Environment"
        )
        if self._t_out_handle == -1:
            raise RuntimeError(f"[{self.label}] Could not resolve the site "
                                f"outdoor drybulb temperature variable.")

        for zone in self.zone_names:
            t_in_h = self.api.exchange.get_variable_handle(
                state_ptr, "Zone Mean Air Temperature", zone
            )
            humidity_h = self.api.exchange.get_variable_handle(
                state_ptr, "Zone Air Relative Humidity", zone
            )
            cooling_h = self.api.exchange.get_actuator_handle(
                state_ptr, "Zone Temperature Control", "Cooling Setpoint", zone
            )
            heating_h = self.api.exchange.get_actuator_handle(
                state_ptr, "Zone Temperature Control", "Heating Setpoint", zone
            )

            if t_in_h == -1:
                raise RuntimeError(
                    f"[{self.label}] Variable handle not found for zone "
                    f"'{zone}'. Check the zone name matches the IDF."
                )
            if cooling_h == -1 or heating_h == -1:
                raise RuntimeError(
                    f"[{self.label}] Thermostat actuator(s) not found for "
                    f"zone '{zone}'. This means the loaded IDF ({self.idf}) "
                    f"has no ZoneControl:Thermostat exposing them for that "
                    f"zone. Run scripts/patch_idf.py's patch() (single "
                    f"zone) or patch_two_zone() (two zones) to generate an "
                    f"IDF with the right zones patched, and point this "
                    f"instance at that file."
                )
            if humidity_h == -1:
                print(f"[{self.label}] Note: 'Zone Air Relative Humidity' "
                      f"variable not found for zone '{zone}'; PMV comfort "
                      f"scoring will fall back to a fixed 50% RH assumption.")

            ideal_heat_h = self.api.exchange.get_variable_handle(
                state_ptr, "Zone Ideal Loads Supply Air Total Heating Energy", zone
            )
            ideal_cool_h = self.api.exchange.get_variable_handle(
                state_ptr, "Zone Ideal Loads Supply Air Total Cooling Energy", zone
            )
            if ideal_heat_h == -1 and ideal_cool_h == -1 and not self._per_zone_energy_warned:
                self._per_zone_energy_warned = True
                print(f"[{self.label}] Note: per-zone Ideal Loads energy "
                      f"variables not resolved (zone '{zone}' and likely "
                      f"others) -- zone_kwh will stay 0.0 for all zones. "
                      f"The combined cumulative_kwh total is still tracked "
                      f"normally via the facility meter.")

            self._zone_handles[zone] = {
                "t_in": t_in_h,
                "humidity": humidity_h,
                "cooling_actuator": cooling_h,
                "heating_actuator": heating_h,
                "ideal_heat_energy": ideal_heat_h,
                "ideal_cool_energy": ideal_cool_h,
            }

        if self.track_energy:
            for name in _HEATING_METER_CANDIDATES:
                h = self.api.exchange.get_meter_handle(state_ptr, name)
                if h != -1:
                    self._heating_meter_handle = h
                    break
            for name in _COOLING_METER_CANDIDATES:
                h = self.api.exchange.get_meter_handle(state_ptr, name)
                if h != -1:
                    self._cooling_meter_handle = h
                    break
            if self._heating_meter_handle == -1 and self._cooling_meter_handle == -1:
                print(f"[{self.label}] WARNING: no heating/cooling facility meter "
                      f"resolved from candidates {_HEATING_METER_CANDIDATES + _COOLING_METER_CANDIDATES}. "
                      f"Energy tracking will report 0.0 kWh. Check "
                      f"{os.path.join(self.output, 'eplusout.rdd')} after a run "
                      f"for the exact meter names available in your EnergyPlus "
                      f"version and update _HEATING_METER_CANDIDATES / "
                      f"_COOLING_METER_CANDIDATES in this file.")

        self._handles_resolved = True

    def _call_decision(self, t_in, t_out, humidity, zone):
        """Calls decision_callback with the zone_name kwarg when the
        callback supports it, falling back to the 3-arg call otherwise --
        keeps single-zone callbacks written before Blueprint 1.1 working
        unmodified.

        BUGFIX (audit finding: reports of a run crashing/stopping partway
        through with no clear error): decision_callback is normally
        expected to never raise (main.py's ProjectEnvelope/
        BaselineOrchestrator both catch their own internal errors), but
        "normally" isn't a guarantee -- and an exception escaping this
        function propagates straight into EnergyPlus's C callback, which
        typically aborts the entire simulation rather than showing a
        normal Python traceback, which is indistinguishable from "it
        just crashed." Last line of defense: catch anything unexpected
        here too and hold at the last known setpoint rather than ever
        letting a Python-side bug take down the whole run.
        """
        try:
            return self.decision_callback(t_in, t_out, humidity, zone_name=zone)
        except TypeError:
            try:
                return self.decision_callback(t_in, t_out, humidity)
            except Exception as e:
                print(f"[{self.label}] WARNING: decision_callback raised {type(e).__name__}: {e} "
                      f"-- holding at last setpoint for zone '{zone}' this tick.")
                return self.per_zone_state.get(zone, {}).get("setpoint", self.last_setpoint), "FAILSAFE (Callback Error)"
        except Exception as e:
            print(f"[{self.label}] WARNING: decision_callback raised {type(e).__name__}: {e} "
                  f"-- holding at last setpoint for zone '{zone}' this tick.")
            return self.per_zone_state.get(zone, {}).get("setpoint", self.last_setpoint), "FAILSAFE (Callback Error)"

    def _callback(self, state_ptr):
        if self.api.exchange.warmup_flag(state_ptr):
            return

        sim_time_key = round(self.api.exchange.current_sim_time(state_ptr), _SIM_TIME_ROUND_DP)
        if sim_time_key == self._last_sim_time_key:
            return
        self._last_sim_time_key = sim_time_key

        self.step_counter += 1

        now = time.perf_counter()
        if now - self._last_heartbeat_t >= self._heartbeat_interval_s:
            self._last_heartbeat_t = now
            sim_day = sim_time_key / 24.0
            elapsed = now - self._run_start_t
            print(f"[{self.label}] ...still running: simulated day "
                  f"{sim_day:.1f}, step {self.step_counter}, "
                  f"{elapsed / 60.0:.1f} min elapsed", flush=True)

        if not self._handles_resolved:
            self._resolve_handles(state_ptr)

        t_out = self.api.exchange.get_variable_value(state_ptr, self._t_out_handle)
        self.last_outdoor_temp = t_out

        if self.track_energy:
            heat_j = 0.0
            cool_j = 0.0
            if self._heating_meter_handle != -1:
                heat_j = self.api.exchange.get_meter_value(state_ptr, self._heating_meter_handle)
            if self._cooling_meter_handle != -1:
                cool_j = self.api.exchange.get_meter_value(state_ptr, self._cooling_meter_handle)
            self.cumulative_heat_kwh += heat_j / 3_600_000.0
            self.cumulative_cool_kwh += cool_j / 3_600_000.0
            self.cumulative_kwh = self.cumulative_heat_kwh + self.cumulative_cool_kwh

        for zone in self.zone_names:
            h = self._zone_handles[zone]
            t_in = self.api.exchange.get_variable_value(state_ptr, h["t_in"])
            humidity = (
                self.api.exchange.get_variable_value(state_ptr, h["humidity"])
                if h["humidity"] != -1 else 50.0
            )

            if h["ideal_heat_energy"] != -1:
                zone_heat = self.api.exchange.get_variable_value(
                    state_ptr, h["ideal_heat_energy"]) / 3_600_000.0
                self.zone_heat_kwh[zone] += zone_heat
                self.zone_kwh[zone] += zone_heat
            if h["ideal_cool_energy"] != -1:
                zone_cool = self.api.exchange.get_variable_value(
                    state_ptr, h["ideal_cool_energy"]) / 3_600_000.0
                self.zone_cool_kwh[zone] += zone_cool
                self.zone_kwh[zone] += zone_cool

            # SENSE -> REASON -> VERIFY lives in main.py; this file only
            # calls into it (once per zone) and applies whatever setpoint
            # comes back to THAT zone's actuators.
            setpoint, source = self._call_decision(t_in, t_out, humidity, zone)
            cooling_setpoint = setpoint
            heating_setpoint = setpoint - (self.deadband_c / 2.0)
            clamped = False
            if self.policy is not None:
                season = classify_season(t_out, self.policy)
                baseline_setpoint = get_baseline_setpoint(self.policy)
                if season == WINTER and cooling_setpoint < baseline_setpoint:
                    cooling_setpoint = baseline_setpoint
                    clamped = True
                elif season == SUMMER:
                    baseline_heating_setpoint = baseline_setpoint - (self.deadband_c / 2.0)
                    if heating_setpoint > baseline_heating_setpoint:
                        heating_setpoint = baseline_heating_setpoint
                        clamped = True
            self.api.exchange.set_actuator_value(state_ptr, h["cooling_actuator"], cooling_setpoint)
            self.api.exchange.set_actuator_value(state_ptr, h["heating_actuator"], heating_setpoint)

            self.per_zone_state[zone] = {
                "t_in": t_in, "t_out": t_out, "humidity": humidity,
                "setpoint": setpoint, "source": source,
                "cooling_ceiling_clamped": clamped,
            }

            self.last_indoor_temp = t_in
            self.last_humidity = humidity
            self.last_setpoint = setpoint

            # 2.1 -- mirror to BACnet. Only sane with exactly one zone;
            # see the constructor warning above.
            if self.bacnet_adapter is not None:
                try:
                    self.bacnet_adapter.write_setpoint(setpoint)
                except Exception:
                    pass
    def run(self):
        self.api.runtime.callback_begin_system_timestep_before_predictor(
            self.state, self._callback
        )

        if self.track_energy and hasattr(self.api.exchange, "request_variable"):
            for zone in self.zone_names:
                try:
                    self.api.exchange.request_variable(
                        self.state, "Zone Ideal Loads Supply Air Total Heating Energy", zone)
                    self.api.exchange.request_variable(
                        self.state, "Zone Ideal Loads Supply Air Total Cooling Energy", zone)
                except Exception:
                    pass
        self.api.runtime.run_energyplus(self.state, [
            "-d", self.output,
            "-w", self.epw,
            self.idf,
        ])
