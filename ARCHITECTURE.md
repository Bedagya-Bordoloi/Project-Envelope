# Technical Architecture: Project Envelope

## 1. System Overview

Project Envelope is built on a **Supervisory Control** architecture. It does not replace the local Building Management System (BMS); instead, it acts as an intelligent safety-and-optimization layer that proposes and verifies HVAC setpoints before they reach the actuator layer.

The controller operates against a high-fidelity **EnergyPlus digital twin** and uses three main information sources for decision-making:

*   Live building state from the EnergyPlus Runtime API.
*   The simulation's future `.epw` weather data as a **forecast proxy** for forward-looking control.
*   A simulated grid carbon-intensity signal combined with thermal-comfort physics.

The system follows a strict **SENSE → REASON → VERIFY → ACT** control loop. EnergyPlus provides the physical state, the Strategist proposes an action, the Sentinel Gate verifies it, and only an approved action is applied to the HVAC actuator or the configured BACnet integration path.

---

## 2. Core Components

### A. The Simulator — EnergyPlus Bridge

Unlike approaches based on repeated batch simulations, Project Envelope uses the **EnergyPlus Python Runtime API** (`pyenergyplus`) to operate inside the simulation as it evolves.

*   **Runtime Callback:** Control logic is attached to the `callback_begin_system_timestep_before_predictor` hook so the controller can observe and influence the simulation during the live run.
*   **State Deduplication:** The bridge contains explicit handling for EnergyPlus callback/sub-timestep behavior so internal sub-timesteps do not inflate the application's control-step counter, duplicate AI calls, or corrupt energy comparisons.
*   **AI vs. Baseline:** The normal demo runs an AI-gated EnergyPlus instance alongside a counterfactual baseline instance using a fixed **22.0°C** schedule. This provides the shared-step comparison used by the dashboard's live savings metric.
*   **Multi-Zone Mode:** `python main.py --multizone` creates two independently controlled zones in one process, each with its own Strategist, Sentinel Gate, FailsafeController, trigger state, and decision cache.

The baseline schedule is defined in `config/building_policy.yaml` and is intentionally kept separate from the AI controller so the counterfactual remains reproducible.

### B. The Strategist — GPT-OSS-20B via Groq

The primary reasoning engine is **OpenAI GPT-OSS-20B served through Groq**. The current configuration uses:

```text
openai/gpt-oss-20b
```

The previous Llama-3.1-8B configuration has been replaced in the current build.

*   **Tool Calling:** The Strategist interacts with building tools through the project's FastMCP layer, including state/weather/HVAC operations.
*   **Forward Reasoning:** The Strategist can consume the simulation's `.epw` forecast proxy and look ahead for meaningful outdoor-temperature changes, enabling proactive pre-cooling or pre-heating behavior.
*   **Confidence:** The model's reported confidence contributes to the Sentinel Gate's CCS calculation.
*   **Correction Context:** Rejected proposals can be returned to the Strategist as explicit correction context so a revised proposal can be attempted within the same reasoning cycle.
*   **Provider Resilience:** The current implementation uses a provider pool with the Groq provider first and supports a separately-owned secondary provider plus optional additional OpenAI-compatible providers. Provider-specific cooldown and rate-budget state are maintained independently.

### C. The Governor — Sentinel Gate

The Governor (`core/sentinel_gate.py`) provides the system's **Safety Envelope**. Every proposal that reaches the verification stage is checked against hard safety constraints, comfort, stability, confidence, carbon conditions, and baseline-direction rules before actuation.

The current CCS calculation is conceptually:

$$CCS = w_v \cdot Comfort + w_r \cdot Stability + w_c \cdot LLM\_Confidence + w_{ca} \cdot Carbon$$

The current configured weights are:

```text
Violation / Comfort : 0.40
Rate Penalty        : 0.15
LLM Confidence      : 0.15
Override Rate       : 0.10
Carbon              : 0.20
CCS Threshold       : 0.65
```

The implementation also tracks a rolling rejection/override rate, rather than treating that component as a fixed constant.

*   **Comfort (ASHRAE-55):** PMV is calculated through `pythermalcomfort`. The configured PMV band is `±0.5`.
*   **Hard Safety Band:** The current policy bounds measured/proposed temperature between **18°C and 28°C** and limits a single-step setpoint change to **2°C**.
*   **Stability:** A minimum setpoint delta of **0.3°C** and minimum dwell logic prevent unnecessary HVAC jitter.
*   **Baseline Direction:** The current policy structurally rejects proposals that move in the wrong energy direction relative to the 22°C baseline schedule, rather than relying only on an LLM instruction to save energy.
*   **Carbon:** The carbon term is driven by a simulated signal rather than a hardcoded score. The current policy cycles Low/Medium/High carbon conditions and maps them to numeric CCS scores.

The current dashboard and presentation identify **0.65** as the CCS approval threshold. fileciteturn0file0L117-L133

---

## 3. Control Loop: SENSE → REASON → VERIFY → ACT

The complete control path is:

```text
                         ┌─────────────────────┐
                         │      EnergyPlus      │
                         │  Live Building State │
                         │  Weather / Forecast  │
                         │   Carbon Simulation  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                              ┌───────────┐
                              │   SENSE   │
                              └─────┬─────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │       REASON       │
                         │  GPT-OSS-20B/Groq  │
                         │      + MCP Tools   │
                         └─────────┬──────────┘
                                   │
                                   ▼
                         ┌────────────────────┐
                         │      VERIFY        │
                         │   Sentinel Gate    │
                         │    CCS + Safety    │
                         └───────┬───────┬────┘
                                 │       │
                         APPROVED│       │REJECTED
                                 │       │
                                 ▼       ▼
                         ┌──────────┐  ┌───────────────┐
                         │   ACT    │  │ SELF-CORRECT  │
                         │ Energy+  │  │ / Fallback    │
                         │ BACnet   │  └───────┬───────┘
                         └──────────┘          │
                                               └──► REASON

                         HARD SAFETY EXCURSION
                                   │
                                   ▼
                              FAILSAFE
```

The key architectural principle is that **LLM reasoning is never the final authority**. The Sentinel Gate and the local failsafe remain in the control path regardless of whether a proposal comes from a fresh LLM call, a cached decision, or a fallback provider.

---

## 4. Agentic Autonomy: Self-Correction Loop

Project Envelope implements self-correction through a bounded feedback loop rather than allowing the LLM to directly actuate HVAC equipment.

1.  **Proposal:** The Strategist produces a structured HVAC setpoint proposal.
2.  **Verification:** Sentinel Gate evaluates the proposal against safety, comfort, stability, confidence, carbon, and baseline-direction constraints.
3.  **Rejection:** If the proposal fails, the Gate produces an explicit reason such as a PMV, stability, safety, or baseline-direction violation.
4.  **Context Injection:** The rejection reason is added to the Strategist's correction context.
5.  **Refinement:** The Strategist may produce a revised proposal within the same control cycle.
6.  **Re-Verification:** The revised proposal is sent through the Sentinel Gate again.
7.  **Failsafe:** If correction is unavailable, skipped because the rejected move is too small to justify another LLM call, or fails again, the system falls back to the local rule-based controller.

The current policy uses `correction_skip_delta_c: 0.15`. This avoids spending a second LLM call on very small rejected changes that are unlikely to benefit from another reasoning pass.

---

## 5. Fast Safety Surveillance & Event-Triggered Reasoning

The current architecture separates **continuous physical safety monitoring** from the slower, more expensive LLM reasoning loop.

### A. Fast Safety Loop

`SentinelGate.quick_check()` is evaluated on every physical EnergyPlus step.

It performs a deliberately lightweight check of the **currently measured indoor temperature** against the configured hard safety band:

```text
Minimum indoor temperature: 18°C
Maximum indoor temperature: 28°C
```

If a hard excursion is detected, the system can enter the failsafe path immediately without waiting for the next Strategist call and without spending an LLM request.

This is intentionally separate from CCS scoring: `quick_check()` is a physical safety net, not a scored AI decision.

### B. Event-Triggered Strategist Scheduling

The current `TriggerEngine` replaces reliance on a blind fixed-cadence reasoning schedule when enabled.

A slow reasoning cycle can be triggered by:

*   **Indoor deviation:** indoor temperature moves beyond the configured deviation deadband.
*   **Forecast shift:** outdoor temperature changes materially relative to the last decision.
*   **Schedule boundary:** day/night or seasonal classification changes.
*   **Maximum staleness:** a hard upper bound ensures the controller eventually re-reasons even when conditions remain quiet.
*   **Cadence ceiling:** the existing cadence remains as a backstop.

The current strategist cadence configuration is **12 EnergyPlus steps**, equivalent to approximately **3 hours** for a 15-minute simulation timestep, while event triggers can cause earlier re-planning.

The trigger engine also debounces closely spaced events so multiple simultaneous trigger conditions result in one reasoning cycle rather than multiple redundant LLM calls.

---

## 6. Decision Cache & LLM Call Efficiency

The current build includes `core/decision_cache.py`, a state-bin cache designed to reduce unnecessary LLM calls without weakening the safety architecture.

The cache bins decisions using:

*   Indoor temperature
*   Outdoor temperature
*   Time of day
*   Season

An approved proposal can be reused when the current state falls into a sufficiently similar and still-valid bin.

**The cache never bypasses the Sentinel Gate.** A cached proposal is re-evaluated against the current physical state before it can be approved for actuation.

Only **post-gate approved proposals** are stored. Rejected proposals are never cached, and a cached proposal that later fails verification is invalidated.

This makes the cache an **LLM-call optimization**, not a safety bypass.

---

## 7. Resilience, Provider Pool & Failsafes

Project Envelope is designed to continue operating safely when cloud reasoning becomes unavailable.

### A. Multi-Provider LLM Resilience

The current `ProviderPool` generalizes the earlier binary Groq→secondary fallback into an N-provider architecture.

The pool can maintain independently:

*   Provider name and model
*   Requests-per-minute limits
*   Optional requests-per-day budgets
*   Provider cooldown state
*   Rate-limit exhaustion state

The primary provider is Groq using GPT-OSS-20B. A separately-owned secondary provider is supported through an OpenAI-compatible endpoint, and additional providers can be declared under `strategist.provider_pool.additional_providers`.

A provider failure therefore does not automatically mean that the entire control system must fail. If no provider can produce a valid proposal, the local failsafe remains authoritative.

### B. Local Failsafe

If the active LLM provider times out or fails, the `FailsafeController` takes over using a local rule-based target.

Current configured targets are:

```text
Heat target: 21°C
Cool target: 23°C
Setback:     1°C
```

The failsafe is intentionally local and deterministic so network availability is not a prerequisite for maintaining the configured safety behavior.

### C. Step Inflation / State Deduplication

The EnergyPlus bridge includes explicit fixes for internal callback/sub-timestep behavior. The goal is to keep application-level control steps aligned between the AI and baseline processes and prevent:

*   Duplicate AI calls
*   Artificially inflated step counts
*   Incorrect energy-meter accumulation
*   Misaligned AI-vs-baseline comparisons

---

## 8. Hardware Integration Path — BACnet/IP

Project Envelope has been validated in a high-fidelity **EnergyPlus digital twin**. It is **not being presented as a controller currently operating a physical building**.

A defined BACnet/IP integration path is implemented in:

```text
integrations/bacnet_adapter.py
```

The architecture is:

```text
Sentinel Gate
     │
     │ Approved Setpoint
     ▼
BACnetAdapter
     │
     │ BACnet/IP WriteProperty
     ▼
AnalogValue Point
     │
     ▼
Physical BMS / HVAC Controller
```

The adapter mirrors the same approved setpoint that is applied to the EnergyPlus actuator. The integration therefore keeps the supervisory control logic separate from the physical transport layer.

### Virtual BACnet Verification

Because no physical BMS hardware or vendor gateway is available for this build, the adapter is verified against the project's local virtual BACnet point:

```text
integrations/virtual_bacnet_point.py
```

The end-to-end test is:

1.  Start the virtual BACnet point.
2.  Connect the `BACnetAdapter`.
3.  Write a setpoint through the adapter.
4.  Read the point back over BACnet/IP.
5.  Assert that the written and read values match.
6.  Store the resulting evidence in the test log.

BACnet mirroring remains **single-zone** in the current build. Multi-zone mode intentionally passes `bacnet_adapter=None` because independent zones require independent physical point mappings, which are outside the current scope.

---

## 9. Multi-Zone Generalization

The single-zone build demonstrates the supervisory loop for one thermal zone. The multi-zone implementation extends the same architecture to two independently actuated zones.

`scripts/patch_idf.py` creates:

```text
models/two_zone_controlled.idf
```

The second zone is a genuine thermal zone constructed from the controlled building geometry rather than a renamed placeholder.

Running:

```bash
python main.py --multizone
```

creates a separate control stack for each zone.

Each zone receives its own:

*   Strategist
*   Sentinel Gate
*   FailsafeController
*   TriggerEngine
*   DecisionCache
*   Control log

Consequently, on the same simulated tick, one zone can be **APPROVED** while the other is **REJECTED**, with independent hysteresis state and CCS calculations.

`scripts/summarize_multizone.py` generates per-zone approval/hold/violation summaries and temperature-trace evidence.

### Known Multi-Zone Limitation

The facility-level energy meter used for the AI-vs-Baseline savings comparison is building-wide. Therefore, `cumulative_kwh` remains a combined total in multi-zone mode.

The bridge also attempts to request zone-specific Ideal Loads energy variables where the installed EnergyPlus Runtime API supports them. If those variables are unavailable, only the combined building-level energy figure should be treated as authoritative.

This limitation should be disclosed whenever per-zone results are presented.

---

## 10. Threshold Calibration & Safety Policy

The current policy configures:

```text
CCS threshold:             0.65
PMV comfort band:          ±0.5
Hard temperature band:     18°C – 28°C
Maximum step delta:        2°C
Minimum setpoint delta:    0.3°C
Minimum dwell:             3 strategist ticks
Baseline setpoint:         22°C
```

The repository contains two complementary calibration approaches:

### `scripts/calibrate_ccs.py`

Analyzes the CCS distribution produced by a completed run and provides a statistical reference for selecting a safety floor.

### `scripts/calibrate_ccs_sweep.py`

Runs the pipeline across candidate CCS thresholds and compares:

*   Approval rate
*   Savings percentage
*   Violation count

This sweep is the more appropriate evidence source when discussing the trade-off between safety strictness and performance.

**Important:** a single run's mean and standard deviation should not be presented as proof that a particular threshold is optimal. Current threshold claims should be reproduced against the current build using the calibration sweep before being used as formal experimental evidence.

---

## 11. Explainability & Evidence

Explainability is a first-class part of the architecture rather than a dashboard-only feature.

Each meaningful control decision can expose:

*   Simulation step
*   Decision source
*   Setpoint
*   Approval/HOLD/REJECT status
*   CCS score
*   PMV and clothing information where available
*   Rejection reason
*   Trigger reason
*   Cache-hit/fallback state

The project dashboard surfaces these decisions in human-readable form. The current presentation specifically demonstrates the Sentinel Gate's stability hysteresis through repeated **HOLD** decisions and explicit **APPROVED** decisions with CCS/PMV details. fileciteturn0file0L134-L143

---

## 12. Technical Stack

*   **Language:** Python 3.11+
*   **Physics Engine:** EnergyPlus API v24.1.0
*   **Runtime Interface:** `pyenergyplus` / EnergyPlus Python Runtime API
*   **Primary LLM:** GPT-OSS-20B via Groq
*   **LLM Resilience:** ProviderPool with secondary/additional OpenAI-compatible providers
*   **Tool Communication:** Model Context Protocol (FastMCP)
*   **Comfort Model:** ASHRAE-55 PMV/PPD via `pythermalcomfort`
*   **Safety Governor:** Sentinel Gate / Critical Control Score
*   **Dashboard:** Streamlit + Plotly
*   **Logging:** JSONL control logs + generated evidence artifacts
*   **Hardware Bridge:** BACnet/IP `WriteProperty` through `BACnetAdapter`
*   **Hardware Verification:** Local virtual BACnet point; no physical BMS hardware tested

---

## 13. Architectural Principles

Project Envelope is intentionally built around five principles:

1.  **Physics remains authoritative:** EnergyPlus provides the building-state ground truth for the digital-twin experiment.
2.  **AI proposes, the Governor disposes:** The LLM never directly bypasses the Sentinel Gate.
3.  **Safety is continuous:** Hard physical safety checks run independently of LLM cadence.
4.  **Optimization is explainable:** Every meaningful control outcome has a traceable reason.
5.  **Cloud reasoning is optional:** Provider failures must degrade to deterministic local control rather than leaving the building without a safety action.

The resulting architecture is a **supervisory, safety-gated AI controller**: AI supplies adaptive reasoning, EnergyPlus supplies the physics, and the Sentinel Gate remains the final authority over what is allowed to actuate.
