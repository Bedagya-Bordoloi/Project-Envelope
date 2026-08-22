"""
scripts/patch_idf.py

CRITICAL FINDING beyond the code bugs: models/baseline.idf is the stock
EnergyPlus "1ZoneUncontrolled" example file. It contains a Zone object and
envelope surfaces, but literally zero HVAC objects - no
ZoneControl:Thermostat, no ZoneHVAC:EquipmentConnections, no conditioning
equipment of any kind. That's *why* indoor temp tracked outdoor temp 1:1
in the demo log (down to -18C): there was never anything in the building
model capable of holding a setpoint, regardless of what the Python side
does. Even with every code fix applied, `get_actuator_handle(..., "Zone
Temperature Control", "Cooling Setpoint", ...)` will return -1 forever
against this file.

This script appends a minimal-but-real HVAC system to a COPY of
baseline.idf (never edits the original, so it stays valid as the
untouched baseline for Feature 2's comparison):
  - ScheduleTypeLimits: Temperature, Control Type
  - Schedule:Compact: constant heating/cooling setpoint schedules + a
    dual-setpoint control-type schedule
  - ThermostatSetpoint:DualSetpoint
  - ZoneControl:Thermostat  (this is what makes the
    "Zone Temperature Control"/"Cooling Setpoint" actuator exist)
  - Zone air/return nodes, NodeList, ZoneHVAC:EquipmentConnections
  - ZoneHVAC:IdealLoadsAirSystem + ZoneHVAC:EquipmentList
    (a textbook "perfect" HVAC unit - satisfies the thermostat setpoint
    exactly, which is standard practice for control-algorithm testing
    where you want to isolate the control logic from a specific chiller/
    coil model)

Usage:
    python scripts/patch_idf.py
Produces:
    models/controlled.idf
    models/two_zone_controlled.idf   (Blueprint 1.1, see below)

Point the AI-controlled EnergyPlusBridge instance at controlled.idf.
Keep baseline.idf as-is for whatever you use as the comparison instance
(see README.md for why you likely want a THIRD file - a schedule-only
baseline - rather than the literally-uncontrolled original, if you want
Feature 2's "baseline vs AI" comparison to mean anything energy-wise).

--- Blueprint 1.1: two-zone controlled model ---
The blueprint's suggested path was to lift two zone definitions out of
EnergyPlus's stock 5ZoneAirCooled.idf example. That file isn't available
in this environment (no EnergyPlus install/example-files directory here,
and it isn't fetchable over this network's allowed domains), so instead
this script SYNTHESIZES a second real zone: it duplicates ZONE ONE's
actual envelope geometry (same wall/floor/roof constructions, same
15.24m x 15.24m footprint) and translates every vertex +20m in X, so
"ZONE TWO" sits next to ZONE ONE as an independent, non-overlapping
thermal zone with its own real surfaces -- not a fake/relabeled zone.
Both zones get their own ZoneControl:Thermostat + IdealLoadsAirSystem
(patch_two_zone(), below), so EnergyPlusBridge can resolve independent
actuator/variable handles per zone and main.py's --multizone mode can
run one Strategist+SentinelGate+Failsafe stack per zone, in the same
process, on the same simulated clock -- proving the pattern generalizes
to independent, sometimes-conflicting per-zone proposals rather than
just a single-room demo.

--- Design-day patch: v2, regex-based ---
The original version of this script matched the SimulationControl block
with an exact multi-line string. That's brittle: any whitespace drift,
line-ending difference, or reformatting in baseline.idf (including ones
introduced by opening/resaving the file in some editors, or a different
EnergyPlus version's example-file export) makes the exact match silently
fail. It prints a console warning in that case, but a printed warning is
easy to miss, and the result is controlled.idf quietly keeping the
design-day sizing-period pollution with no hard failure.

This version instead:
  1. Finds the SimulationControl object as a whole (by its object header,
     case-insensitive, up to the terminating ';') rather than assuming
     exact formatting.
  2. Within that block, finds the specific field whose trailing IDF field
     comment says "Run Simulation for Sizing Periods" (case-insensitive,
     whitespace-tolerant) rather than matching the field's *value* text,
     since the comment is the stable identifier and the Yes/No value is
     exactly what we're changing.
  3. Reports how many fields it actually patched, so "0 found" and
     "found 2+ (unexpected duplicate object)" are both visible instead of
     silently doing nothing or the wrong thing.
"""

import os
import re

ZONE_NAME = "ZONE ONE"
MULTI_ZONE_NAMES = ["ZONE ONE", "ZONE TWO"]

# Shared by every zone: the schedules and the dual-setpoint object are
# schedule-driven, not zone-specific, so EnergyPlus is fine with multiple
# ZoneControl:Thermostat objects (one per zone) all pointing at the same
# ThermostatSetpoint:DualSetpoint. Written ONCE regardless of zone count.
SHARED_SCHEDULE_BLOCK = """
!-   ===========  ALL OBJECTS IN CLASS: SCHEDULETYPELIMITS ===========
ScheduleTypeLimits,
    Temperature,             !- Name
    -60,                     !- Lower Limit Value
    200,                     !- Upper Limit Value
    CONTINUOUS;              !- Numeric Type

ScheduleTypeLimits,
    Control Type,            !- Name
    0,                       !- Lower Limit Value
    4,                       !- Upper Limit Value
    DISCRETE;                !- Numeric Type

!-   ===========  ALL OBJECTS IN CLASS: SCHEDULE:COMPACT ===========
Schedule:Compact,
    HeatingSetpointSchedule, !- Name
    Temperature,             !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 20.0;      !- constant 20C heating setpoint

Schedule:Compact,
    CoolingSetpointSchedule, !- Name
    Temperature,             !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 26.0;      !- constant 26C cooling setpoint (overridden by actuator at runtime)

Schedule:Compact,
    ZoneControlTypeSched,    !- Name
    Control Type,            !- Schedule Type Limits Name
    Through: 12/31,
    For: AllDays,
    Until: 24:00, 4;         !- 4 = DualSetpoint control every hour

!-   ===========  ALL OBJECTS IN CLASS: THERMOSTATSETPOINT:DUALSETPOINT ===========
ThermostatSetpoint:DualSetpoint,
    ZoneDualSetpoint,        !- Name
    HeatingSetpointSchedule, !- Heating Setpoint Temperature Schedule Name
    CoolingSetpointSchedule; !- Cooling Setpoint Temperature Schedule Name
"""


def _per_zone_hvac_block(zone_name):
    """Everything that MUST be unique per zone: the thermostat, the air
    nodes, and the Ideal Loads unit. References the SHARED_SCHEDULE_BLOCK
    objects above by name, so that block must be written first/once."""
    return f"""
!-   ===========  ALL OBJECTS IN CLASS: ZONECONTROL:THERMOSTAT ({zone_name}) ===========
ZoneControl:Thermostat,
    {zone_name} Thermostat,  !- Name
    {zone_name},             !- Zone or ZoneList Name
    ZoneControlTypeSched,    !- Control Type Schedule Name
    ThermostatSetpoint:DualSetpoint,  !- Control 1 Object Type
    ZoneDualSetpoint;        !- Control 1 Name

!-   ===========  ALL OBJECTS IN CLASS: NODELIST / ZONE AIR NODES ({zone_name}) ===========
NodeList,
    {zone_name} Inlets,      !- Name
    {zone_name} Supply Inlet Node;  !- Node 1 Name

!-   ===========  ALL OBJECTS IN CLASS: ZONEHVAC:EQUIPMENTCONNECTIONS ({zone_name}) ===========
ZoneHVAC:EquipmentConnections,
    {zone_name},                       !- Zone Name
    {zone_name} Equipment,             !- Zone Conditioning Equipment List Name
    {zone_name} Inlets,                !- Zone Air Inlet Node or NodeList Name
    ,                                  !- Zone Air Exhaust Node or NodeList Name
    {zone_name} Zone Air Node,         !- Zone Air Node Name
    {zone_name} Return Outlet;         !- Zone Return Air Node or NodeList Name

!-   ===========  ALL OBJECTS IN CLASS: ZONEHVAC:EQUIPMENTLIST ({zone_name}) ===========
ZoneHVAC:EquipmentList,
    {zone_name} Equipment,             !- Name
    SequentialLoad,                    !- Load Distribution Scheme
    ZoneHVAC:IdealLoadsAirSystem,      !- Zone Equipment 1 Object Type
    {zone_name} Ideal Loads,           !- Zone Equipment 1 Name
    1,                                 !- Zone Equipment 1 Cooling Sequence
    1;                                 !- Zone Equipment 1 Heating Sequence

!-   ===========  ALL OBJECTS IN CLASS: ZONEHVAC:IDEALLOADSAIRSYSTEM ({zone_name}) ===========
!-   A "perfect" HVAC unit: satisfies the active thermostat setpoint exactly.
!-   Standard practice for control-algorithm testing so the demo measures
!-   the Strategist/Gate logic, not a specific chiller/coil model.
ZoneHVAC:IdealLoadsAirSystem,
    {zone_name} Ideal Loads,           !- Name
    ,                                  !- Availability Schedule Name
    {zone_name} Supply Inlet Node,     !- Zone Supply Air Node Name
    ,                                  !- Zone Exhaust Air Node Name
    ,                                  !- System Inlet Air Node Name
    50,                                !- Maximum Heating Supply Air Temperature {{C}}
    13,                                !- Minimum Cooling Supply Air Temperature {{C}}
    0.0156,                            !- Maximum Heating Supply Air Humidity Ratio {{kgWater/kgDryAir}}
    0.0077,                            !- Minimum Cooling Supply Air Humidity Ratio {{kgWater/kgDryAir}}
    NoLimit,                           !- Heating Limit
    autosize,                          !- Maximum Heating Air Flow Rate {{m3/s}}
    ,                                  !- Maximum Sensible Heating Capacity {{W}}
    NoLimit,                           !- Cooling Limit
    autosize,                          !- Maximum Cooling Air Flow Rate {{m3/s}}
    ,                                  !- Maximum Total Cooling Capacity {{W}}
    ,                                  !- Heating Availability Schedule Name
    ,                                  !- Cooling Availability Schedule Name
    ConstantSupplyHumidityRatio,       !- Dehumidification Control Type
    ,                                  !- Cooling Sensible Heat Ratio {{dimensionless}}
    ConstantSupplyHumidityRatio,       !- Humidification Control Type
    ,                                  !- Design Specification Outdoor Air Object Name
    ,                                  !- Outdoor Air Inlet Node Name
    ,                                  !- Demand Controlled Ventilation Type
    ,                                  !- Outdoor Air Economizer Type
    ,                                  !- Heat Recovery Type
    ,                                  !- Sensible Heat Recovery Effectiveness {{dimensionless}}
    ;                                  !- Latent Heat Recovery Effectiveness {{dimensionless}}
"""


# Preserves the original single-zone behavior/name for anything that
# imports HVAC_BLOCK directly: shared schedules + ZONE ONE's HVAC objects.
HVAC_BLOCK = SHARED_SCHEDULE_BLOCK + _per_zone_hvac_block(ZONE_NAME)


# ZONE TWO's real envelope geometry: ZONE ONE's actual Zone + 6
# BuildingSurface:Detailed objects from models/baseline.idf, every vertex
# (and the zone origin) translated +20m in X so it sits beside ZONE ONE
# without overlapping it. Same constructions (R13WALL / FLOOR / ROOF31),
# already defined once in baseline.idf and reused here -- this is a real,
# simulatable second zone, not a stub.
ZONE_TWO_GEOMETRY_BLOCK = """
!-   ===========  ZONE TWO: duplicated envelope, +20m X offset  ===========
Zone,
    ZONE TWO,                !- Name
    0,                       !- Direction of Relative North {deg}
    20,                      !- X Origin {m}
    0,                       !- Y Origin {m}
    0,                       !- Z Origin {m}
    1,                       !- Type
    1,                       !- Multiplier
    autocalculate,           !- Ceiling Height {m}
    autocalculate;           !- Volume {m3}

BuildingSurface:Detailed,
    Zn002:Wall001,           !- Name
    Wall,                    !- Surface Type
    R13WALL,                 !- Construction Name
    ZONE TWO,                !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    0.5000000,               !- View Factor to Ground
    4,                       !- Number of Vertices
    20,0,4.572000,  !- X,Y,Z ==> Vertex 1 {m}
    20,0,0,  !- X,Y,Z ==> Vertex 2 {m}
    35.24000,0,0,  !- X,Y,Z ==> Vertex 3 {m}
    35.24000,0,4.572000;  !- X,Y,Z ==> Vertex 4 {m}

BuildingSurface:Detailed,
    Zn002:Wall002,           !- Name
    Wall,                    !- Surface Type
    R13WALL,                 !- Construction Name
    ZONE TWO,                !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    0.5000000,               !- View Factor to Ground
    4,                       !- Number of Vertices
    35.24000,0,4.572000,  !- X,Y,Z ==> Vertex 1 {m}
    35.24000,0,0,  !- X,Y,Z ==> Vertex 2 {m}
    35.24000,15.24000,0,  !- X,Y,Z ==> Vertex 3 {m}
    35.24000,15.24000,4.572000;  !- X,Y,Z ==> Vertex 4 {m}

BuildingSurface:Detailed,
    Zn002:Wall003,           !- Name
    Wall,                    !- Surface Type
    R13WALL,                 !- Construction Name
    ZONE TWO,                !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    0.5000000,               !- View Factor to Ground
    4,                       !- Number of Vertices
    35.24000,15.24000,4.572000,  !- X,Y,Z ==> Vertex 1 {m}
    35.24000,15.24000,0,  !- X,Y,Z ==> Vertex 2 {m}
    20,15.24000,0,  !- X,Y,Z ==> Vertex 3 {m}
    20,15.24000,4.572000;  !- X,Y,Z ==> Vertex 4 {m}

BuildingSurface:Detailed,
    Zn002:Wall004,           !- Name
    Wall,                    !- Surface Type
    R13WALL,                 !- Construction Name
    ZONE TWO,                !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    0.5000000,               !- View Factor to Ground
    4,                       !- Number of Vertices
    20,15.24000,4.572000,  !- X,Y,Z ==> Vertex 1 {m}
    20,15.24000,0,  !- X,Y,Z ==> Vertex 2 {m}
    20,0,0,  !- X,Y,Z ==> Vertex 3 {m}
    20,0,4.572000;  !- X,Y,Z ==> Vertex 4 {m}

BuildingSurface:Detailed,
    Zn002:Flr001,            !- Name
    Floor,                   !- Surface Type
    FLOOR,                   !- Construction Name
    ZONE TWO,                !- Zone Name
    ,                        !- Space Name
    Adiabatic,               !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    NoSun,                   !- Sun Exposure
    NoWind,                  !- Wind Exposure
    1.000000,                !- View Factor to Ground
    4,                       !- Number of Vertices
    35.24000,0.000000,0.0,  !- X,Y,Z ==> Vertex 1 {m}
    20.000000,0.000000,0.0,  !- X,Y,Z ==> Vertex 2 {m}
    20.000000,15.24000,0.0,  !- X,Y,Z ==> Vertex 3 {m}
    35.24000,15.24000,0.0;  !- X,Y,Z ==> Vertex 4 {m}

BuildingSurface:Detailed,
    Zn002:Roof001,           !- Name
    Roof,                    !- Surface Type
    ROOF31,                  !- Construction Name
    ZONE TWO,                !- Zone Name
    ,                        !- Space Name
    Outdoors,                !- Outside Boundary Condition
    ,                        !- Outside Boundary Condition Object
    SunExposed,              !- Sun Exposure
    WindExposed,             !- Wind Exposure
    0,                       !- View Factor to Ground
    4,                       !- Number of Vertices
    20.000000,15.24000,4.572,  !- X,Y,Z ==> Vertex 1 {m}
    20.000000,0.000000,4.572,  !- X,Y,Z ==> Vertex 2 {m}
    35.24000,0.000000,4.572,  !- X,Y,Z ==> Vertex 3 {m}
    35.24000,15.24000,4.572;  !- X,Y,Z ==> Vertex 4 {m}
"""

# Matches the whole SimulationControl,...; object, however its internal
# whitespace/line-endings are formatted. DOTALL so '.' spans newlines;
# non-greedy up to the first ';' so it doesn't swallow the next object.
_SIM_CONTROL_RE = re.compile(
    r"SimulationControl\s*,.*?;",
    re.IGNORECASE | re.DOTALL,
)

# Matches a single IDF field line ending in the
# "!- Run Simulation for Sizing Periods" comment, capturing:
#   prefix  - leading whitespace (preserved so indentation doesn't shift)
#   value   - the current Yes/No token
#   sep     - the trailing comma/semicolon plus whitespace before '!-'
#   comment - the field comment itself (preserved verbatim)
# Matching on the comment (the stable identifier) rather than the value
# (the thing we're changing) is what makes this robust to the value
# already being "No", already being "Yes", or having odd spacing.
_SIZING_FIELD_RE = re.compile(
    r"""
    (?P<prefix>[ \t]*)
    (?P<value>Yes|No)
    (?P<sep>\s*[,;]\s*)
    (?P<comment>!-\s*Run\s+Simulation\s+for\s+Sizing\s+Periods)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _patch_sizing_periods(content):
    """
    Finds every SimulationControl object in `content` and flips its
    'Run Simulation for Sizing Periods' field value to 'No', matched by
    the stable trailing field comment rather than an exact string block.
    Returns (new_content, num_fields_patched).
    """
    patched_count = 0

    def _replace_field(field_match):
        nonlocal patched_count
        patched_count += 1
        return f"{field_match.group('prefix')}No{field_match.group('sep')}{field_match.group('comment')}"

    def _replace_block(block_match):
        block = block_match.group(0)
        return _SIZING_FIELD_RE.sub(_replace_field, block)

    new_content = _SIM_CONTROL_RE.sub(_replace_block, content)
    return new_content, patched_count


def patch(src="models/baseline.idf", dst="models/controlled.idf"):
    with open(src, "r", encoding="latin-1") as f:
        content = f.read()

    # Design-day sizing periods (e.g. Denver's 99% annual heating design
    # day) were being fully simulated -- burning control steps and Groq
    # calls on synthetic extreme-weather days that aren't part of the
    # real annual run -- even though Do Zone/System/Plant Sizing
    # Calculation are already "No" and never consume that sizing data.
    # Flip this one field so only the real .epw weather-file period runs.
    content, patched_count = _patch_sizing_periods(content)

    if patched_count == 0:
        print("WARNING: no 'Run Simulation for Sizing Periods' field was found "
              "inside any SimulationControl object in models/baseline.idf -- "
              "design-day sizing periods were NOT patched. Open "
              "models/controlled.idf, find the SimulationControl object by "
              "hand, and set that field to 'No'. Then check whether "
              "baseline.idf's SimulationControl object is formatted "
              "differently than expected (e.g. the field comment text "
              "itself was changed) and update _SIZING_FIELD_RE above to match.")
    elif patched_count > 1:
        print(f"WARNING: patched {patched_count} 'Run Simulation for Sizing "
              f"Periods' fields (expected exactly 1) -- models/baseline.idf "
              f"appears to contain more than one SimulationControl object. "
              f"Verify models/controlled.idf by hand before relying on it.")
    else:
        print("Design-day sizing periods disabled (SimulationControl patched, "
              "1 field matched).")

    with open(dst, "w", encoding="latin-1") as f:
        f.write(content)
        f.write("\n")
        f.write(HVAC_BLOCK)
    print(f"Wrote {dst} ({os.path.getsize(dst)} bytes). "
          f"Original {src} left untouched.")


def patch_two_zone(src="models/baseline.idf", dst="models/two_zone_controlled.idf"):
    """Blueprint 1.1. Same sizing-period patch as patch(), but appends
    ZONE TWO's duplicated geometry plus independent HVAC objects for
    BOTH zones, so EnergyPlusBridge(zone_names=MULTI_ZONE_NAMES, ...) has
    two real, independently-actuatable thermal zones to drive."""
    with open(src, "r", encoding="latin-1") as f:
        content = f.read()

    content, patched_count = _patch_sizing_periods(content)
    if patched_count != 1:
        print(f"WARNING (two-zone build): expected to patch exactly 1 "
              f"sizing-periods field, patched {patched_count}. See the "
              f"warning above from patch() for the same root cause.")

    per_zone_blocks = "".join(_per_zone_hvac_block(z) for z in MULTI_ZONE_NAMES)

    with open(dst, "w", encoding="latin-1") as f:
        f.write(content)
        f.write("\n")
        f.write(ZONE_TWO_GEOMETRY_BLOCK)
        f.write(SHARED_SCHEDULE_BLOCK)
        f.write(per_zone_blocks)
    print(f"Wrote {dst} ({os.path.getsize(dst)} bytes) with zones "
          f"{MULTI_ZONE_NAMES}. Original {src} left untouched.")


# --- Audit fix: short RunPeriod for fast iteration ---
# calibrate_ccs_sweep.py's own docstring math: full year / cadence_steps
# 12 ~= 2,920 Strategist calls per `python main.py` run; paced at the
# policy default of 25 req/min that's already ~4h of rate-limiter
# waiting alone, before EnergyPlus compute time -- and
# calibrate_ccs_sweep.py runs that 9x sequentially, generate_evidence.py
# chains baseline + AI + the 9-point sweep + multizone on top of that.
# This lets main.py / calibrate_ccs_sweep.py / generate_evidence.py
# (via --run-period-days / --quick-days) swap in a short-duration copy
# of an IDF for iteration, without ever touching the original file --
# same pattern as patch()/patch_two_zone() above: find the RunPeriod
# object as a whole, then patch its End Month / End Day of Month fields
# by their stable trailing field comment (not by position), so it's
# robust to reformatting.
_RUN_PERIOD_RE = re.compile(r"RunPeriod\s*,.*?;", re.IGNORECASE | re.DOTALL)

_END_MONTH_FIELD_RE = re.compile(
    r"""
    (?P<prefix>[ \t]*)
    (?P<value>\d+)
    (?P<sep>\s*,\s*)
    (?P<comment>!-\s*End\s+Month)
    """,
    re.IGNORECASE | re.VERBOSE,
)
_END_DAY_FIELD_RE = re.compile(
    r"""
    (?P<prefix>[ \t]*)
    (?P<value>\d+)
    (?P<sep>\s*,\s*)
    (?P<comment>!-\s*End\s+Day\s+of\s+Month)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def patch_run_period(src, dst, end_month, end_day):
    """Write a copy of `src` to `dst` with its RunPeriod object's 'End
    Month' / 'End Day of Month' fields set to (end_month, end_day),
    leaving Begin Month/Day (assumed Jan 1, matching every .idf in this
    project) and everything else byte-identical. Raises RuntimeError
    instead of silently no-op'ing or double-patching if the RunPeriod
    object isn't found in exactly the expected shape, since this is
    called programmatically at run time (main.py --run-period-days) and
    a silent failure here would mean "asked for a 2-week test run, but
    it quietly ran the full year instead" -- exactly the kind of
    surprise that cost hours in the first place.
    """
    with open(src, "r", encoding="latin-1") as f:
        content = f.read()

    patched = {"month": 0, "day": 0}

    def _repl_month(m):
        patched["month"] += 1
        return f"{m.group('prefix')}{end_month}{m.group('sep')}{m.group('comment')}"

    def _repl_day(m):
        patched["day"] += 1
        return f"{m.group('prefix')}{end_day}{m.group('sep')}{m.group('comment')}"

    def _replace_block(block_match):
        block = block_match.group(0)
        block = _END_MONTH_FIELD_RE.sub(_repl_month, block)
        block = _END_DAY_FIELD_RE.sub(_repl_day, block)
        return block

    new_content = _RUN_PERIOD_RE.sub(_replace_block, content)

    if patched["month"] != 1 or patched["day"] != 1:
        raise RuntimeError(
            f"Expected exactly 1 'End Month' and 1 'End Day of Month' field "
            f"inside a RunPeriod object in {src}, found {patched['month']} and "
            f"{patched['day']}. Refusing to write {dst} to avoid silently "
            f"running the wrong period length. Check that {src}'s RunPeriod "
            f"object still uses the standard field comments."
        )

    with open(dst, "w", encoding="latin-1") as f:
        f.write(new_content)
    return dst


if __name__ == "__main__":
    patch()
    patch_two_zone()