"""
core/failsafe_controller.py

Zero-network-dependency rule-based setback controller. Used whenever the
Groq call times out, errors, or the Sentinel Gate rejects a corrected
proposal a second time. Holds the building in-bounds indefinitely.

BUGFIX (savings erosion / reversal in cold weather -- see chat): target_low_c
/ target_high_c and every `setpoint` this project passes around (AI
proposals, the baseline's fixed 22.0C, this controller's own return value)
are all in "cooling-setpoint" framing -- core/energyplus_bridge.py always
applies `heating_setpoint = setpoint - deadband_c/2` before actuating, so
the REAL achieved indoor temp normally runs ~deadband_c/2 (1.0C by default)
BELOW whatever setpoint value is being reasoned about. `current_temp` here
is that real achieved temp (straight from the EnergyPlus sensor), so
comparing it directly against target_low_c=21.0 meant a healthy,
by-design winter reading of ~20.0-20.8C was misread as "too cold" on
essentially every tick -- tripping Failsafe, which then jumped the
setpoint straight up to baseline-equivalent (21.0 + setback 1.0 = 22.0,
i.e. the SAME value baseline uses). That's what was quietly erasing (and
eventually reversing) the AI's savings as the season got colder and
Failsafe/rejections fired more often: it wasn't a real comfort emergency,
it was the low-side threshold being off by one deadband-width. Fix:
translate current_temp back into the same cooling-setpoint framing
(current_temp + deadband_c/2) before comparing it to low/high, so a
building that's exactly on-target under the deadband design reads as
on-target here too, instead of perpetually "too cold."
"""


class FailsafeController:
    def __init__(self, policy: dict):
        fs = policy["failsafe"]
        self.low = float(fs["target_low_c"])
        self.high = float(fs["target_high_c"])
        self.setback = float(fs["setback_c"])
        # Same deadband_c the bridge uses to convert an approved setpoint
        # into an actuated heating target -- see comfort.deadband_c in
        # config/building_policy.yaml and EnergyPlusBridge's docstring.
        self.half_deadband = float(policy.get("comfort", {}).get("deadband_c", 2.0)) / 2.0

    def decide(self, current_temp):
        """Simple rule-based logic. Returns a setpoint, never raises."""
        # Translate the real achieved temp back into cooling-setpoint
        # framing before comparing against low/high, since low/high (and
        # this method's return value) live in that framing, not raw
        # achieved-temp space. See module docstring.
        effective_temp = current_temp + self.half_deadband
        if effective_temp > self.high:
            return round(self.high - self.setback, 2)   # cool it down
        if effective_temp < self.low:
            return round(self.low + self.setback, 2)    # heat it up
        return round(effective_temp, 2)                  # hold steady