
# Project Envelope — AI-Gated BMS
### A Self-Correcting, Explainable Safety Governor for an EnergyPlus + LLM Closed-Loop BMS

*A closed-loop AI building-optimization system: safety-gated HVAC setpoint control validated in a high-fidelity EnergyPlus digital twin.*

---

## 🎯 Project Vision
Project Envelope is a closed-loop Building Management System where **EnergyPlus runs the real physics**, a Groq-hosted LLM (**Llama-3.1-8B**) proposes energy-saving HVAC setpoints via real tool-calling, and a calibrated safety gate (**Sentinel Gate**) decides whether to trust each proposal. 

Unlike "black box" AI controllers, Envelope is **explainable by design**: it rejects unsafe proposals with plain-language reasons, feeds those reasons back to the LLM for self-correction, and falls back to a local rule-based controller if the cloud disappears.

## 🚀 Key Differentiators
*   **Reflective Self-Correction:** When the Governor rejects a proposal, the rejection reason is injected into the next prompt, allowing the LLM to learn from its "mistake" and submit a corrected setpoint within the same control step.
*   **ASHRAE-55 PMV Comfort:** We use the `pythermalcomfort` library to score comfort based on temperature, humidity, and metabolic rates (clo), rather than simple static temperature bounds.
*   **Live Counterfactual Overlay:** The system runs two EnergyPlus instances in parallel—a **Baseline** schedule and the **AI-Gated** controller—plotting both curves live on a Streamlit dashboard.
*   **Model Context Protocol (MCP):** Implementation of a FastMCP server to standardize building tools (`get_state`, `set_hvac`, `get_weather`).
*   **Defined BACnet Integration Path:** A `BACnetAdapter` (`integrations/bacnet_adapter.py`) mirrors every approved setpoint to a real BACnet/IP analogValue point over the network, verified end-to-end against a free local virtual point (`python scripts/test_bacnet_adapter.py`) -- see "Hardware Integration Path" below.
*   **Multi-Zone Generalization:** `python main.py --multizone` runs two independently-actuated zones in one process, each with its own Strategist + Sentinel Gate + Failsafe stack -- proving the pattern handles independent, sometimes-conflicting per-zone proposals, not just a single room.

---

## 🛠️ Setup & Installation

### Prerequisites
*   **Python 3.11+** (Tested on 3.13)
*   **EnergyPlus v24.1.0** (Official installer or Docker)
*   **Groq API Key** (Required for the Strategist's reasoning)

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
Create a `.env` file in the project root:
```env
GROQ_API_KEY=gsk_your_key_here
EPLUS_DIR=C:\EnergyPlusV24-1-0  # Path to your installation
```

### 3. Prepare the Model
The baseline IDF file needs HVAC actuators attached. Run the patch script once:
```bash
python scripts/patch_idf.py
```
*This generates `models/controlled.idf`, which the simulation uses for active control.*

---

## 🏃 Running the Simulation

For the full demo experience, open three terminals (with `venv` activated):

**Terminal 1: The AI Instance**
```bash
python main.py
```

**Terminal 2: The Baseline Instance**
```bash
python main.py --baseline
```

**Terminal 3: The Live Dashboard**
```bash
streamlit run ui/app.py
```

---

## 🧩 Running the Multi-Zone Demo (1.1)

To prove the control pattern generalizes beyond a single room, `main.py` can drive **two independent zones** in one process, each with its own Strategist + Sentinel Gate + Failsafe stack:

```bash
# 1. Generate models/two_zone_controlled.idf (in addition to controlled.idf)
python scripts/patch_idf.py

# 2. Run both zones -- AI-gated only, no --baseline counterpart for this mode
python main.py --multizone

# 3. After (or during) the run, summarize + chart the two zones' logs
python scripts/summarize_multizone.py
```
This writes `logs/multizone/zone_one.jsonl` and `logs/multizone/zone_two.jsonl`, and the summary script produces `logs/multizone/multizone_summary.csv` (approval/hold/violation rate per zone) plus `multizone_temps.png` (both zones' indoor-temp traces overlaid). Because each zone gets its own gate instance, it's normal -- and worth showing -- for one zone's proposal to be approved while the other's is rejected on the same tick. See `ARCHITECTURE.md` §7 for the known limitation on per-zone energy metering.

---

## 📂 Directory Structure
```text
Envelope/
├── agents/
│   └── strategist.py        # Groq/Llama prompting & self-correction logic
├── bms_mcp/
│   ├── server.py            # FastMCP Server implementation
│   └── tools.py             # Tool definitions (get_state, set_hvac, etc.)
├── config/
│   └── building_policy.yaml # Hard bounds, gate weights, and failsafe targets
├── core/
│   ├── energyplus_bridge.py # Runtime API callbacks & dual-instance logic
│   ├── comfort.py           # ASHRAE-55 PMV math via pythermalcomfort
│   ├── sentinel_gate.py     # The "Governor" (CCS Scoring & Rejection)
│   ├── failsafe_controller.py # Rule-based setback fallback
│   └── baseline_controller.py # Fixed-schedule logic for comparison
├── models/
│   ├── baseline.idf         # Original simulation file
│   ├── controlled.idf       # Patched file with exposed actuators (1 zone)
│   └── two_zone_controlled.idf # Patched file with 2 independent zones (1.1)
├── scripts/
│   ├── patch_idf.py         # Automates HVAC object injection (single + two-zone)
│   ├── calibrate_ccs.py     # Single-run CCS score distribution (superseded by the sweep below for threshold justification)
│   ├── calibrate_ccs_sweep.py # Blueprint 1.2: re-runs the pipeline per threshold, plots approval/savings/violations
│   ├── compare_models.py    # Blueprint 1.4: 8B vs 70B latency/CCS-pass-rate comparison
│   ├── find_forecast_spike.py # Blueprint 1.3: locates + charts a real pre-cool/pre-heat demo window
│   ├── summarize_multizone.py # Blueprint 1.1: per-zone approval/violation summary + temp chart
│   └── test_bacnet_adapter.py # Blueprint 2.1: end-to-end adapter <-> virtual point test
├── demos/
│   └── ev_charging_demo.py  # Tier 3: SentinelGate reused, unmodified, for a non-HVAC domain
├── integrations/
│   ├── bacnet_adapter.py    # Blueprint 2.1: mirrors approved setpoints to a BACnet/IP point
│   └── virtual_bacnet_point.py # Local virtual BACnet point for adapter testing
├── ui/
│   └── app.py               # Streamlit dashboard
└── main.py                  # Closed-loop orchestrator (single-zone, --baseline, and --multizone)
```

## 📈 Measured Results
Project Envelope is designed for long-term stability. The table below is a **captured snapshot from one specific run**, kept here as an example of the format/shape of result we track -- it is **not guaranteed to reflect the current build**. In particular, it predates this round's step-inflation and energy-metering fixes to `core/energyplus_bridge.py`, so treat the exact figures as illustrative rather than current. Before presenting these numbers to judges, regenerate them:

```bash
python scripts/patch_idf.py          # if not already done
python main.py &                     # AI instance
python main.py --baseline &          # Baseline instance
# let both run for a real simulated period, then:
python scripts/calibrate_ccs_sweep.py   # for the threshold-justification numbers
```
and read the fresh `cumulative_kwh` values straight out of `logs/ai/control_log.jsonl` / `logs/baseline/control_log.jsonl`'s last rows -- don't carry old numbers forward.

*   **Peak Observed Savings:** ~10.2% (During transient peak-shaving events)
*   **Steady-State Savings Range:** 1.5% – 4.5%
*   **Comfort Reliability:** 100% (Zero ASHRAE-55 PMV boundary violations observed in this captured run)

### Representative Performance Snapshot (Step 4392)
The following data reflects the building state during a high-fidelity control period as shown in the project dashboard.

| Metric | AI Instance | Baseline Instance |
| :--- | :--- | :--- |
| **Cumulative Energy** | 3,551.10 kWh | 3,694.90 kWh |
| **Measured Savings %** | **3.9%** | -- |
| **Indoor Temp (AI)** | 20.00°C | 20.15°C |

### Explainable Decision Log (Actual Output)
The following sequence from `control_log.jsonl` demonstrates the **Sentinel Gate's Stability Hysteresis**—preventing energy-wasting setpoint "jitter" while maintaining a precise comfort score (CCS).

| Step | Source | Setpoint | Reason / Status |
| :--- | :--- | :--- | :--- |
| 9000 | AI (Stabilized) | 21.00°C | **HOLD:** Change of 0.10°C is too small. |
| 9600 | AI (Stabilized) | 21.00°C | **HOLD:** Change of 0.20°C is too small. |
| 9900 | AI | 21.15°C | **APPROVED:** CCS 0.85 \| PMV -0.21 (Clo 1.09) |
| 10200| AI | 22.00°C | **APPROVED:** CCS 0.80 \| PMV -0.02 (Clo 1.05) |


## 📊 Project Dashboard
The following screenshots show the live interaction between the EnergyPlus physics engine and the AI Strategist.

![Dashboard Overview](assests/dashboard_overview.jpg)
*Figure 1: Real-time alignment of AI vs. Baseline energy curves and indoor temperatures.*

![Decision Log](assests/decision_log.jpg)
*Figure 2: The Explainable AI decision log showing the Sentinel Gate in action.*

## Working demo
The following demo presents the live working demo of the system:
[Click here to watch the demo video](https://drive.google.com/file/d/1mr0sxNNNSCR1TodRuRMqV35PI54Au_KI/view?usp=sharing)

## 🛡️ Safety & Failsafes
*   **Network Loss:** If the Groq API fails/timeouts, the system instantly switches to the `FailsafeController`, maintaining the building between 21°C and 23°C.
*   **Gate Override:** If the AI fails to produce a safe setpoint even after correction, the Governor forces a "Hold Steady" command to prevent equipment damage.
