"""
core/baseline_controller.py

The "baseline-schedule instance" from the blueprint's Feature 2 (live
counterfactual overlay). This is deliberately NOT the AI: no Strategist,
no Sentinel Gate, no self-correction. It just holds a fixed setpoint
from config/building_policy.yaml the way an un-optimized building would
run on a simple thermostat schedule. Its whole job is to be the honest
"what would this building do without the AI" comparison line on the
dashboard and in the energy-savings number.
"""


class BaselineController:
    def __init__(self, policy: dict):
        baseline = policy.get("baseline", {})
        self.setpoint = float(baseline.get("schedule_setpoint_c", 22.0))

    def decide(self, current_temp):
        """Always the same scheduled target. Never raises."""
        return self.setpoint