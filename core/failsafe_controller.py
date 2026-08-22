"""
core/failsafe_controller.py

Zero-network-dependency rule-based setback controller. Used whenever the
Groq call times out, errors, or the Sentinel Gate rejects a corrected
proposal a second time. Holds the building in-bounds indefinitely.


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
