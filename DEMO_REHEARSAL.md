# Tier 4 — Presentation & Demo Readiness Checklist

This is a rehearsal script and pre-flight checklist, not a slide deck. Work through it in order; the live-failure sequence in Part 2 is the single highest-value thing to rehearse multiple times before judges see it.

## Part 0 — Before you touch the deck

Do NOT rebuild slide numbers from memory or from an older submission. Every number that goes on a slide should trace back to a file you can open on demand:

| Claim | Regenerate with | Evidence file |
|---|---|---|
| Energy savings % | `python main.py` + `python main.py --baseline` (let both run a real simulated period) | `logs/ai/control_log.jsonl`, `logs/baseline/control_log.jsonl` (last row's `cumulative_kwh`) |
| CCS threshold justification | `python scripts/calibrate_ccs_sweep.py` | `calibration_report.csv` + its plot |
| Pre-cool/pre-heat demo | `python scripts/find_forecast_spike.py` | the before/after setpoint chart |
| 8B vs 70B model choice | `python scripts/compare_models.py` | latency/CCS-pass-rate table CSV |
| Multi-zone independence | `python main.py --multizone` + `python scripts/summarize_multizone.py` | `logs/multizone/multizone_summary.csv`, `multizone_temps.png` |
| Generalization beyond HVAC | `python demos/ev_charging_demo.py` | its own decision log |
| BACnet integration path | `python scripts/test_bacnet_adapter.py` | `logs/bacnet_adapter_test.json` |
| Live failsafe demo | manual (see Part 2) | a captured log segment showing AI → FAILSAFE → AI |

If a slide number doesn't have a matching row above with a file you personally regenerated this round, either regenerate it or pull the claim from the deck. **A smaller, fully-verified, honestly-presented system beats a bigger one with claims you can't defend live.**

## Part 1 — Strip before this goes near judges

- [ ] Confirm no `honeywell*` filenames remain (`assests/dashboard_overview.jpg` / `decision_log.jpg` already renamed this round — re-check any *new* screenshots you take).
- [ ] Search the deck itself (not just the repo) for "Honeywell" / "Eco-Loop" / "PS1" and remove or replace, unless your competition explicitly wants that framing kept.
- [ ] Search README.md, ARCHITECTURE.md, and the deck for "production-ready" / "deployed system" / similar deployment-readiness language. The accurate framing already in `ARCHITECTURE.md` §5 is the template: *"validated in a high-fidelity EnergyPlus digital twin, with a defined integration path to BACnet-based BMS hardware."*
- [ ] Confirm `.env` (with your real Groq key) was never committed: `git log --all -p -- .env` should return nothing. Rotate the key if it ever was.
- [ ] Re-read the deck once as a skeptical judge: does any slide claim more than the evidence file next to it in Part 0 actually shows?

## Part 2 — Rehearse this exact live sequence

Run this end-to-end, multiple times, before presenting. Time it. The goal is a demo that survives judge Q&A, not a smooth read-through of a script you've never actually executed.

1. **Normal operation.** Start `python main.py` and `streamlit run ui/app.py`. Let it run long enough to show a few AI decisions in the dashboard's decision log.
2. **A rejected proposal.** Point out (or wait for) a `REJECTED: Violation` row — narrate what the Gate is protecting against.
3. **Self-correction.** Show the next row for the same tick: `AI (Corrected)` with a revised setpoint, and the reason string that changed.
4. **Approval.** Show a clean `APPROVED` row with its CCS score.
5. **Kill the Groq connection live** — blank/rename `GROQ_API_KEY` in `.env` or block `api.groq.com`, without stopping `main.py`.
6. **Show the failsafe holding** — `current_source` flips to `FAILSAFE`, indoor temp stays within `failsafe.target_low_c`–`target_high_c` (21–23°C) in `config/building_policy.yaml`.
7. **Restore the key** and confirm it reconnects and resumes AI decisions without a crash or stuck state.

Capture the log segment spanning steps 5–7 (`AI → FAILSAFE → AI`, all in-bounds) as your evidence artifact for this row of Part 0's table — this is your single best live-demo moment, but only if you've tested it end-to-end beforehand rather than assuming it works because the code looks right.

## Part 3 — Slide additions

- [ ] Add one slide showing the CCS formula (`core/sentinel_gate.py`'s `compute_ccs()`), paired with the calibration sweep plot from `scripts/calibrate_ccs_sweep.py` — the formula alone, without the plot, is naming the metric rather than justifying it.
- [ ] If presenting the multi-zone result, use `multizone_temps.png` plus the `multizone_summary.csv` approval-rate columns side by side — the point to make is that the two zones' decisions diverge, not just that two traces exist.
- [ ] Caveat any per-zone energy number with the note in `ARCHITECTURE.md` §7 (facility meters are building-wide; per-zone kWh is best-effort and may be unavailable depending on the installed EnergyPlus Python API version).

## Part 4 — Dry run

- [ ] Do a full dry run with someone unfamiliar with the project watching.
- [ ] Time the live-failure demo (Part 2) specifically — this is the part most likely to run long or go sideways under pressure.
- [ ] Have a fallback: if the live Groq-kill demo fails to reconnect cleanly during rehearsal, decide now whether you'll show a pre-captured log segment instead, and say so explicitly rather than letting a broken live demo speak for itself.

## Cut list (if time is short before presenting)

Matches the blueprint's own priority order — if something has to give, cut from the bottom:
1. Reproduce/verify real numbers (Part 0) — never cut.
2. Dual-instance comparison actually working end-to-end.
3. CCS calibration sweep.
4. One working generalization proof point (EV demo).
5. The pre-cool demo moment.
6. Multi-zone — now implemented, but still time-expensive to rehearse live; a pre-captured `multizone_temps.png` + summary CSV is an acceptable fallback if you don't have time to run it live.
7. The BACnet bridge — same fallback logic: `logs/bacnet_adapter_test.json` from a pre-run of `test_bacnet_adapter.py` is fine to show instead of a live write.
