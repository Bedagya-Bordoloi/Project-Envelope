
# Technical Architecture: Project Envelope

## 1. System Overview
Project Envelope is built on a **Supervisory Control** architecture. It does not replace the local Building Management System (BMS); instead, it acts as an intelligent "wrapper" that optimizes setpoints based on a simulated weather forecast (the EnergyPlus run's own known future `.epw` data, used as a forecast proxy), a simulated grid carbon-intensity signal, and thermal comfort physics.

The system follows a strict **SENSE → REASON → VERIFY → ACT** control loop.

---

## 2. Core Components

### A. The Simulator (EnergyPlus Bridge)
Unlike traditional approaches that use batch-mode simulations (editing an IDF and restarting), Project Envelope utilizes the **EnergyPlus Python Runtime API** (`pyenergyplus`).
*   **Runtime Callbacks:** Our logic is injected into the `callback_begin_system_timestep_before_predictor` hook. This ensures a true closed-loop where the AI reacts to the simulation state as it evolves.
*   **Dual-Instance Orchestration:** `main.py` manages two parallel simulation threads:
    1. **AI-Gated:** The optimized instance.
    2. **Baseline:** A "Counterfactual" instance running a fixed 22°C schedule.
    *This allows for the 1:1 "Live Savings" measurement shown on the dashboard.*

### B. The Strategist (Groq-Llama Cognitive Engine)
The reasoning engine is powered by **Llama-3.1-8B via Groq**. 
*   **Tool Calling:** The Strategist uses the **Model Context Protocol (MCP)** to interact with the building tools (`get_state`, `set_hvac`).
*   **Forward Reasoning:** By consuming the simulation’s `.epw` weather file as a forecast proxy, the Strategist can perform **proactive pre-cooling** before outdoor heat spikes, rather than reacting after the indoor temp rises.

### C. The Governor (Sentinel Gate)
The Governor (`core/sentinel_gate.py`) provides the "Safety Envelope." Every LLM proposal is scored against the **Critical Control Score (CCS)**.

**The CCS Formula:**
$$CCS = (w_{v} \cdot \text{Comfort}) + (w_{r} \cdot \text{Stability}) + (w_{c} \cdot \text{LLM\_Conf}) + (w_{ca} \cdot \text{Carbon})$$

*   **Comfort (ASHRAE-55):** We calculate the **Predicted Mean Vote (PMV)** using the `pythermalcomfort` library. If a proposal pushes PMV outside the $\pm0.5$ range, the gate triggers an immediate **REJECT**.
*   **Stability:** Prevents rapid setpoint oscillations (jitter) that would cause mechanical wear on HVAC actuators.

---

## 3. Agentic Autonomy: The Self-Correction Loop
A key requirement of PS1 is "Self-Correction." Project Envelope implements this via a **Recursive Feedback Loop**:
1.  **Rejection:** If the Sentinel Gate rejects a proposal (e.g., "Setpoint 19°C violates comfort limits"), the rejection reason is captured.
2.  **Context Injection:** This reason is fed back into the Strategist's "Correction Context."
3.  **Refinement:** The LLM re-evaluates its strategy and submits a revised proposal (e.g., "21.5°C") within the same timestep.
4.  **Verification:** The Gate re-evaluates. If it fails again, the system defaults to a **Local Rule-Based Failsafe**.

---

## 4. Resilience & Failsafes
To meet the **30% System Integration** requirement, the system must survive network loss:
*   **Local Failsafe:** If the Groq API exceeds the 10s timeout, the `FailsafeController` takes over. It uses a zero-dependency rule-based setback (e.g., Heat to 21°C / Cool to 23°C) to keep the building "in-bounds" until connectivity is restored.
*   **State Deduplication:** The `EnergyPlusBridge` includes a fix for "Step Inflation," ensuring that EnergyPlus internal sub-timesteps do not cause duplicate AI calls or incorrect energy metering.

---

## 5. Hardware Integration Path (2.1)
Project Envelope has been validated exclusively in a high-fidelity EnergyPlus digital twin, with a defined integration path to BACnet-based BMS hardware -- it is not itself running against a physical building.

That path is implemented, not just described: `integrations/bacnet_adapter.py` provides a `BACnetAdapter` that mirrors every Sentinel-Gate-approved setpoint to a real BACnet/IP `analogValue` point via a standard `WriteProperty` request, using the same call shape (`write_setpoint(value_c)`) as the write already applied to the EnergyPlus actuator in `core/energyplus_bridge.py`'s callback -- so swapping the digital twin for a physical BMS point means changing a config value (`bacnet.device_ip` in `building_policy.yaml`), not the control logic.

Because no physical hardware or vendor gateway is available for this build, the adapter is verified against a free local **virtual BACnet point** (`integrations/virtual_bacnet_point.py`) rather than real equipment -- a second local BACnet/IP device exposing one commandable `analogValue` point, standing in for a real BMS point the way BAC0's own virtual-device mode or a tool like YABE would. `scripts/test_bacnet_adapter.py` runs the full loop end-to-end: start the virtual point, write a setpoint through the adapter, read it back over the network, and assert the two match -- producing `logs/bacnet_adapter_test.json` as an evidence artifact.

---

## 6. Justification of Thresholds
Our **0.65 CCS Threshold** started from `scripts/calibrate_ccs.py`, which analyzes the distribution of CCS scores actually produced by ONE completed run:
*   **Mean Score:** 0.79
*   **Standard Deviation ($\sigma$):** 0.07
*   **Logic:** `calibrate_ccs.py` recommends **Mean − 1$\sigma$** (≈0.72 for the run above) as a safety floor -- accepting the agent's typical performance while filtering the bottom tail.

**Caveat, honestly stated:** the numbers above describe the *distribution of scores seen at whatever threshold was active during that run* -- they say nothing about how approval rate, energy savings, or comfort violations would actually change if the threshold moved. That's a materially different (and more useful) question, and it's why `scripts/calibrate_ccs_sweep.py` now exists: it re-runs the full pipeline once per candidate threshold (0.50 through 0.90) and plots approval rate, savings %, and violation count against threshold directly, so the "knee" in the trade-off curve can be pointed to on a chart instead of inferred from one run's score distribution. **Before presenting this section, re-run `scripts/calibrate_ccs_sweep.py` against your current build and swap in its `calibration_report.csv` / plot** -- the numbers above predate this round's step-inflation and energy-metering fixes and should not be presented as current without being reproduced.

---

## 7. Multi-Zone Generalization (1.1)
The single-zone build proves the SENSE→REASON→VERIFY→ACT loop works for one room; it doesn't by itself prove the pattern *generalizes* across zones with independent, sometimes-conflicting proposals. `scripts/patch_idf.py`'s `patch_two_zone()` produces `models/two_zone_controlled.idf`: a real second thermal zone ("ZONE TWO"), built by duplicating ZONE ONE's actual envelope geometry (same wall/floor/roof constructions, same footprint) and translating it +20m in X so it sits beside ZONE ONE without overlapping -- not a relabeled stub.

`python main.py --multizone` drives both zones in one process. Critically, each zone gets its **own** `SentinelGate`, `Strategist`, and `FailsafeController` instance (`main.py`'s `MultiZoneOrchestrator`), so on a single simulated tick one zone's proposal can be `APPROVED` while the other's is `REJECTED` -- independent hysteresis state, independent CCS scores, independent decisions, logged separately per zone under `logs/multizone/`. `scripts/summarize_multizone.py` produces the comparison evidence (per-zone approval/violation rates, an overlaid temperature-trace chart) referenced in Tier 0's "Verify" requirement.

**Known simplification:** facility-level energy meters (used for the AI-vs-Baseline savings comparison in Section 6 of this doc / Tier 0.3) are building-wide, not per-zone, so `cumulative_kwh` stays a single combined total across both zones even in `--multizone` mode. The bridge additionally attempts to read each zone's own Ideal Loads energy output variable for a genuine per-zone breakdown where the installed EnergyPlus Python API supports requesting it at runtime; where it doesn't, per-zone energy just isn't available and only the combined figure is meaningful. This is disclosed in code (`core/energyplus_bridge.py`'s fix note #5) and should be disclosed the same way in any deck slide using per-zone numbers.

**2.1's BACnet mirroring stays single-zone.** With two zones proposing independent setpoints there's no longer one "the" setpoint to mirror to a single BACnet point, so `--multizone` runs pass `bacnet_adapter=None`. A real per-zone BACnet point mapping is a natural next step but wasn't in scope for this round.

---

## 8. Technical Stack
*   **Language:** Python 3.11+
*   **Physics:** EnergyPlus API v24.1.0
*   **LLM:** Llama-3.1-8B (Groq)
*   **Communication:** Model Context Protocol (MCP)
*   **Comfort Model:** ASHRAE-55 (PMV/PPD)
*   **Dashboard:** Streamlit + Plotly (Real-time JSONL streaming)
*   **Hardware Bridge:** BAC0 / BACnet-IP `WriteProperty` (validated against a local virtual point; no physical BMS hardware tested)
