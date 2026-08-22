# Project Envelope — AI-Gated BMS
### A Self-Correcting, Explainable Safety Governor for an EnergyPlus + LLM Closed-Loop BMS

*A closed-loop AI building-optimization system: safety-gated HVAC setpoint control validated in a high-fidelity EnergyPlus digital twin.*

---

## 🎯 Project Vision

Project Envelope is a closed-loop Building Management System where **EnergyPlus runs the real physics**, a Groq-hosted **GPT-OSS-20B** model proposes energy-saving HVAC setpoints through real tool-calling, and a calibrated safety gate (**Sentinel Gate**) decides whether to trust each proposal.

Unlike "black box" AI controllers, Envelope is **explainable by design**: it rejects unsafe proposals with plain-language reasons, feeds those reasons back to the LLM for self-correction, and falls back to a local rule-based controller if the cloud disappears.

The system operates as a strict **SENSE → REASON → VERIFY → ACT** supervisory-control loop around the local BMS rather than replacing it. EnergyPlus provides live building state and forecast information, the Strategist reasons about HVAC actions, the Sentinel Gate verifies every proposal, and only approved actions reach the simulated or connected actuator layer.

## 🚀 Key Differentiators

*   **Reflective Self-Correction:** When the Governor rejects a proposal, the rejection reason is injected into the next prompt, allowing the LLM to learn from its "mistake" and submit a corrected setpoint within the same control step.
*   **ASHRAE-55 PMV Comfort:** We use the `pythermalcomfort` library to score comfort based on temperature, humidity, and metabolic rates (clo), rather than simple static temperature bounds.
*   **Live Counterfactual Overlay:** The system runs two EnergyPlus instances in parallel — a **Baseline** schedule and the **AI-Gated** controller — plotting both curves live on a Streamlit dashboard.
*   **Model Context Protocol (MCP):** A FastMCP server standardizes building tools such as `get_state`, `set_hvac`, and `get_weather`.
*   **Sentinel Gate Safety Governor:** Every AI proposal is evaluated using a **Critical Control Score (CCS)** incorporating comfort, stability, LLM confidence, and carbon considerations. The current approval threshold is **0.65**.
*   **Fast Safety Surveillance:** A lightweight safety check runs on every physical EnergyPlus step before the slower reasoning loop, allowing hard safety excursions to trigger an immediate failsafe response.
*   **Event-Triggered Reasoning:** The controller no longer depends solely on blind fixed-cadence LLM calls. Reasoning can be triggered by meaningful indoor deviation, forecast shifts, schedule boundaries, or maximum staleness.
*   **Decision Cache:** Recently approved decisions can be reused for sufficiently similar state bins, reducing unnecessary LLM calls while still re-verifying the cached proposal against current measurements.
*   **Multi-Provider Resilience:** The primary Groq provider can be supplemented by independently-owned secondary/additional providers through the `ProviderPool`, improving resilience against provider-specific rate limits and outages.
*   **Defined BACnet Integration Path:** A `BACnetAdapter` (`integrations/bacnet_adapter.py`) mirrors every approved setpoint to a BACnet/IP analogValue point and can be tested end-to-end against the local virtual BACnet point.
*   **Multi-Zone Generalization:** `python main.py --multizone` runs two independently-actuated zones in one process, each with its own Strategist + Sentinel Gate + Failsafe stack.

---

## 🛠️ Setup & Installation

### Prerequisites

*   **Python 3.11+** (Tested on Python 3.13)
*   **EnergyPlus v24.1.0**
*   **Groq API Key** (Required for the primary Strategist provider)
*   Optional secondary/additional LLM provider keys if multi-provider fallback is enabled

### 1. Installation

```bash
# Clone and enter the directory
git clone <your-repo-link>
cd Envelope

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root.

```env
# Primary provider
GROQ_API_KEY=gsk_your_key_here

# EnergyPlus installation
EPLUS_DIR=C:\EnergyPlusV24-1-0

# Optional secondary provider
SECONDARY_LLM_API_KEY=your_secondary_provider_key_here

# Optional additional providers
CEREBRAS_API_KEY=your_cerebras_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

The primary model is configured as:

```text
openai/gpt-oss-20b
```

via Groq. The project also contains a configurable provider-pool architecture for independently-owned fallback providers.

### 3. Prepare the Model

The baseline IDF file needs HVAC actuators attached. Run the patch script once:

```bash
python scripts/patch_idf.py
```

This prepares:

```text
models/controlled.idf
models/two_zone_controlled.idf
```

for active HVAC control.

---

## 🏃 Running the Simulation

For the full demo experience, open three terminals with the virtual environment activated.

### Terminal 1: The AI-Gated Instance

```bash
python main.py
```

### Terminal 2: The Baseline Instance

```bash
python main.py --baseline
```

### Terminal 3: The Live Dashboard

```bash
streamlit run ui/app.py
```

The dashboard provides live visibility into:

* AI indoor temperature
* Outdoor temperature
* AI energy
* Baseline energy
* Live savings
* AI-vs-baseline building-physics traces
* Explainable Sentinel Gate decisions
* Setpoints and decision reasons

---

## 🧩 Running the Multi-Zone Demo

To prove that the control pattern generalizes beyond a single room, `main.py` can drive **two independent zones** in one process, each with its own Strategist + Sentinel Gate + Failsafe stack:

```bash
# 1. Generate the controlled single-zone and two-zone models
python scripts/patch_idf.py

# 2. Run both zones
python main.py --multizone

# 3. Summarize and chart the two zones' logs
python scripts/summarize_multizone.py
```

This produces per-zone logs and a summary of approval, hold, and violation behavior.

Because each zone has its own gate instance, one zone's proposal may be approved while another zone's proposal is rejected on the same control tick.

---

## 🧠 How the Control Loop Works

Project Envelope follows a four-stage supervisory loop:

```text
┌─────────────────────────────────────────────────────────────┐
│                         ENERGYPLUS                          │
│          Live Building Physics + Weather Forecast           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                         ┌─────────┐
                         │  SENSE  │
                         └────┬────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │     REASON       │
                    │ Strategist       │
                    │ GPT-OSS-20B      │
                    │ via Groq         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     VERIFY       │
                    │  Sentinel Gate   │
                    │   CCS ≥ 0.65     │
                    └────────┬─────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                 APPROVED           REJECTED
                    │                 │
                    ▼                 ▼
             ┌─────────────┐   ┌───────────────┐
             │    ACT      │   │ SELF-CORRECT  │
             │ EnergyPlus  │   │ Feed reason   │
             │ / BACnet    │   │ back to LLM   │
             └─────────────┘   └───────┬───────┘
                                       │
                                       └──────► REASON

                    HARD SAFETY EXCURSION
                             │
                             ▼
                       FAILSAFE
```

The current implementation additionally performs a lightweight safety surveillance check on every physical EnergyPlus step before the slower reasoning loop.

---

## 🛡️ The Sentinel Gate

The Sentinel Gate is the project's safety governor.

The **Critical Control Score (CCS)** combines multiple control-quality dimensions:

```text
CCS =
    wᵥ · Comfort
  + wᵣ · Stability
  + wᶜ · LLM Confidence
  + wᶜₐ · Carbon
```

### Comfort

ASHRAE-55 PMV is used to evaluate thermal comfort. A PMV excursion outside the configured comfort band can trigger rejection.

### Stability

The Gate prevents unnecessary setpoint jitter that could waste energy or repeatedly actuate HVAC equipment.

### LLM Confidence

The Strategist's confidence in its own proposal contributes to the safety decision.

### Carbon

The controller incorporates a simulated grid carbon-intensity signal into the decision score.

### Current CCS Threshold

```text
CCS approval threshold: 0.65
```

The threshold is calibrated from the observed score distribution and is used to determine whether an AI proposal is safe enough to reach the actuator layer.

---

## 🔄 Reflective Self-Correction

A rejected AI proposal does not immediately terminate the control cycle.

Instead:

```text
LLM Proposal
     │
     ▼
Sentinel Gate
     │
     ├── APPROVED ──► Actuate
     │
     └── REJECTED
             │
             ▼
       Explain rejection
             │
             ▼
       Feed reason back
             │
             ▼
       LLM correction
             │
             ▼
       Sentinel Gate
```

This makes the system **self-correcting and explainable**, rather than simply accepting or silently discarding an AI decision.

---

## ⚡ Fast Safety Loop & Event-Triggered Reasoning

The current implementation separates fast physical safety surveillance from slower LLM reasoning.

### Fast Loop

A lightweight `quick_check()` runs on **every physical EnergyPlus step**.

It can immediately trigger the `FailsafeController` when a hard safety boundary is crossed.

### Slow Reasoning Loop

The Strategist is invoked when meaningful events occur, including:

* Indoor-temperature deviation
* Significant outdoor/forecast change
* Schedule boundaries
* Maximum decision staleness
* Forecast-driven pre-cooling or pre-heating opportunities

This avoids unnecessary LLM calls while maintaining continuous physical safety surveillance.

---

## 💾 Decision Cache

The current controller includes a state-bin decision cache.

A recently approved proposal can be reused when the current building state is sufficiently similar to a previously evaluated state.

Importantly, **the cache does not bypass the Sentinel Gate**.

Cached proposals are re-verified against the current:

* Indoor temperature
* Outdoor temperature
* Humidity
* Carbon signal
* Season
* Baseline direction

This reduces unnecessary LLM calls while preserving the safety-governor architecture.

---

## 🔌 Multi-Provider Resilience

The Strategist uses a configurable `ProviderPool`.

The primary provider is:

```text
Groq
└── openai/gpt-oss-20b
```

Optional secondary and additional providers can be configured independently.

The ProviderPool:

* Tracks provider-specific rate limits
* Maintains independent cooldowns
* Tracks optional daily request budgets
* Selects available providers in round-robin order
* Avoids repeatedly hammering an exhausted provider
* Keeps provider-specific failure state

This provides a more robust fallback architecture than relying on multiple API keys from the same provider.

---

## 📂 Directory Structure

```text
Envelope/
├── agents/
│   └── strategist.py              # LLM reasoning, tool calling & self-correction
├── bms_mcp/
│   ├── server.py                  # FastMCP server implementation
│   └── tools.py                   # Building tools
├── config/
│   └── building_policy.yaml       # Bounds, CCS threshold, triggers & failsafes
├── core/
│   ├── energyplus_bridge.py       # EnergyPlus Runtime API & control loop
│   ├── comfort.py                 # ASHRAE-55 PMV calculations
│   ├── sentinel_gate.py           # CCS safety governor
│   ├── failsafe_controller.py     # Rule-based safety fallback
│   ├── baseline_controller.py     # Baseline schedule controller
│   ├── provider_pool.py            # Multi-provider LLM resilience
│   ├── trigger_engine.py           # Event-triggered reasoning scheduler
│   ├── decision_cache.py           # State-bin approved-decision cache
│   └── seasonality.py              # Seasonal/baseline control logic
├── models/
│   ├── baseline.idf               # Original EnergyPlus model
│   ├── controlled.idf             # Controlled single-zone model
│   ├── two_zone_controlled.idf    # Two-zone controlled model
│   └── baseline.epw                # Weather file
├── scripts/
│   ├── patch_idf.py               # HVAC actuator/model preparation
│   ├── calibrate_ccs.py           # CCS calibration
│   ├── calibrate_ccs_sweep.py     # Threshold/savings/violation sweep
│   ├── compare_models.py          # Model comparison experiments
│   ├── find_forecast_spike.py     # Forecast-driven control-window search
│   ├── summarize_multizone.py     # Multi-zone summary and charts
│   ├── generate_evidence.py       # Evidence generation
│   ├── verify_savings_evidence.py # Savings-evidence verification
│   ├── verify_decision_cache.py   # Decision-cache verification
│   ├── check_llm_connectivity.py  # LLM connectivity check
│   ├── check_both_providers.py    # Provider availability check
│   ├── check_secondary_provider.py# Secondary provider check
│   ├── check_local_provider.py    # Local provider check
│   ├── check_css.py               # CSS/CCS verification
│   ├── test_phase0_quick_check.py # Fast safety-loop verification
│   └── test_bacnet_adapter.py     # BACnet adapter test
├── demos/
│   └── ev_charging_demo.py        # Sentinel Gate reuse demonstration
├── integrations/
│   ├── __init__.py
│   ├── bacnet_adapter.py          # BACnet/IP integration adapter
│   └── virtual_bacnet_point.py    # Local BACnet test point
├── ui/
│   └── app.py                     # Streamlit live dashboard
├── main.py                        # Main closed-loop orchestrator
├── .env.example                   # Environment-variable template
├── ARCHITECTURE.md                # Technical architecture
├── DEMO_REHEARSAL.md              # Demo procedure
├── ROUND3_FIXES.md                # Earlier implementation fixes
├── ROUND6_FIXES.md                # Round 6 implementation fixes
├── ROUND7_FIXES.md                # Round 7 implementation fixes
└── requirements.txt               # Python dependencies
```

---

## 📈 Measured Results

Project Envelope tracks AI-gated energy performance against a parallel baseline EnergyPlus simulation.

### Current Live Dashboard Snapshot

The latest captured dashboard run shows:

| Metric | AI-Gated Instance | Baseline Instance |
| :--- | :--- | :--- |
| **Simulation Step** | 2736 | 2736 |
| **Cumulative Energy** | **2299.58 kWh** | **2303.23 kWh** |
| **Live Savings** | **0.2%** | -- |
| **Indoor Temperature** | **20.20°C** | -- |
| **Outdoor Temperature** | **-8.30°C** | -- |

The dashboard is explicitly comparing the two instances at their shared simulation step.

### Latest Explainable Decision Log

A later captured dashboard state shows the Sentinel Gate continuing to stabilize the controller around a **21.2°C** setpoint:

| Step | Source | Setpoint | Reason / Status |
| :--- | :--- | :--- | :--- |
| 2760 | AI (Stabilized) (Fallback) | 21.2°C | **HOLD:** Change of 0.00°C is too small. Staying steady. |
| 2772 | AI (Stabilized) (Fallback) | 21.2°C | **HOLD:** Change of 0.00°C is too small. Staying steady. |
| 2784 | AI (Stabilized) (Fallback) | 21.2°C | **HOLD:** Change of 0.00°C is too small. Staying steady. |
| 2796 | AI (Fallback) | 21.2°C | **APPROVED:** CCS 0.93 \| PMV -0.02 (Clo 1.20) |
| 2808 | AI (Stabilized) (Fallback) | 21.2°C | **HOLD:** Change of 0.00°C is too small. Staying steady. |

This demonstrates the current **stability hysteresis + fallback + explainability** behavior visible in the live dashboard.

### Earlier Representative Performance Snapshot

For comparison, the project presentation also contains an earlier representative high-fidelity control snapshot at **Step 4392**:

| Metric | AI Instance | Baseline Instance |
| :--- | :--- | :--- |
| **Cumulative Energy** | 3,551.10 kWh | 3,694.90 kWh |
| **Measured Savings %** | **3.9%** | -- |
| **Indoor Temperature** | 20.00°C | 20.15°C |

The presentation also reports:

*   **Peak Observed Savings:** ~10.2% during transient peak-shaving events
*   **Steady-State Savings Range:** 1.5%–4.5%
*   **Comfort Reliability:** 100% in the captured representative run, with zero observed ASHRAE-55 PMV boundary violations

These figures are **run-specific measurements**, not universal guarantees. Fresh runs should be used when presenting current performance.

---

## 📊 Project Dashboard

The Streamlit dashboard provides live visibility into the interaction between EnergyPlus, the AI Strategist, the Sentinel Gate, and the baseline controller.

### Dashboard Overview

![Dashboard Overview](assests/dashboard_overview.jpg)

*Figure 1: Real-time alignment of AI vs. Baseline energy curves and indoor-temperature behavior.*

### Explainable Decision Log

![Decision Log](assests/decision_log.jpg)

*Figure 2: Explainable AI decision log showing Sentinel Gate approvals, holds, fallbacks, setpoints, and reasons.*

---

## 🎥 Working Demo

The following demo presents the live working system:

[Click here to watch the demo video](https://drive.google.com/file/d/1mr0sxNNNSCR1TodRuRMqV35PI54Au_KI/view?usp=sharing)

---

## 🛡️ Safety & Failsafes

Project Envelope is designed to **fail safe rather than fail silently**.

*   **Network / LLM Failure:** If the Groq API or active provider fails, times out, or becomes unavailable, the system falls back to the local `FailsafeController`.
*   **Hard Safety Override:** The fast `quick_check()` loop can immediately override the current trajectory when a hard physical safety condition is detected.
*   **Gate Override:** If the LLM cannot produce a safe setpoint after self-correction, the Governor forces a **Hold Steady** or safe fallback action.
*   **Stability Protection:** Small unnecessary setpoint changes are rejected or held to prevent HVAC control jitter.
*   **State Deduplication:** EnergyPlus sub-timestep handling prevents duplicate AI calls and incorrect energy accounting.
*   **Provider Resilience:** Provider-specific cooldowns and daily-budget tracking prevent the controller from continuously retrying an exhausted provider.
*   **Failsafe Setback:** The rule-based controller maintains the configured heating/cooling safety targets when cloud reasoning is unavailable.

---

## 🔬 Verification & Testing

The repository includes dedicated scripts for validating major parts of the architecture:

```bash
# LLM connectivity
python scripts/check_llm_connectivity.py

# Provider availability
python scripts/check_both_providers.py
python scripts/check_secondary_provider.py
python scripts/check_local_provider.py

# Fast safety loop
python scripts/test_phase0_quick_check.py

# Decision cache
python scripts/verify_decision_cache.py

# Savings evidence
python scripts/verify_savings_evidence.py

# Multi-zone behavior
python scripts/summarize_multizone.py

# BACnet integration
python scripts/test_bacnet_adapter.py

# CCS verification
python scripts/check_css.py
```

---

## 🏗️ Hardware Integration Path

Project Envelope includes a defined path from the EnergyPlus digital twin to a real BMS environment through BACnet/IP.

```text
Sentinel Gate
      │
      │ Approved Setpoint
      ▼
BACnetAdapter
      │
      ▼
BACnet/IP Analog Value
      │
      ▼
Real BMS / HVAC Controller
```

A local virtual BACnet point is included for safe end-to-end testing before connecting to physical equipment.

---

## 🌐 Beyond HVAC

The Sentinel Gate is designed as a reusable safety-governor pattern rather than an HVAC-only controller.

The repository includes:

```text
demos/ev_charging_demo.py
```

which demonstrates reuse of the Sentinel Gate architecture in a non-HVAC control domain.

---

## 📚 Project Documentation

Additional implementation documentation is available in:

*   `ARCHITECTURE.md` — technical architecture and supervisory-control design
*   `DEMO_REHEARSAL.md` — demonstration procedure
*   `ROUND3_FIXES.md` — implementation fixes
*   `ROUND6_FIXES.md` — implementation fixes
*   `ROUND7_FIXES.md` — provider/safety/control-loop fixes

---

## 🚀 Project Status

Project Envelope currently demonstrates:

*   High-fidelity EnergyPlus Runtime API control
*   GPT-OSS-20B reasoning through Groq
*   MCP-based building tool calling
*   Sentinel Gate CCS safety verification
*   Reflective LLM self-correction
*   ASHRAE-55 PMV comfort scoring
*   Fast per-step safety surveillance
*   Event-triggered reasoning
*   Decision caching with re-verification
*   Multi-provider LLM resilience
*   AI-vs-baseline live counterfactual comparison
*   Explainable decision logging
*   BACnet/IP integration path
*   Two-zone control
*   Cross-domain Sentinel Gate demonstration

**Project Envelope's core principle:**

> **Let AI optimize the building — but never let AI bypass the safety governor.**