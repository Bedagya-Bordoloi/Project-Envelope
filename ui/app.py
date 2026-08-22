import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import time
import os

# --- PAGE SETUP ---
st.set_page_config(layout="wide", page_title="Eco-Loop: AI vs Baseline")

AI_LOG = "logs/ai/control_log.jsonl"
BASELINE_LOG = "logs/baseline/control_log.jsonl"

def get_last_valid_json(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0: return None
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            offset = min(size, 4096)
            f.seek(size - offset)
            lines = f.read().decode(errors='ignore').splitlines()
            for line in reversed(lines):
                try: return json.loads(line)
                except: continue
    except: return None

@st.cache_data(ttl=2) 
def load_and_align_data():
    if not os.path.exists(AI_LOG) or not os.path.exists(BASELINE_LOG):
        return pd.DataFrame(), pd.DataFrame(), {}

    # 1. Load AI and Baseline into DataFrames
    ai_raw = []
    with open(AI_LOG, 'r') as f:
        for line in f:
            try: ai_raw.append(json.loads(line))
            except: continue
    
    base_raw = []
    with open(BASELINE_LOG, 'r') as f:
        for line in f:
            try: base_raw.append(json.loads(line))
            except: continue

    if not ai_raw or not base_raw: return pd.DataFrame(), pd.DataFrame(), {}

    df_ai = pd.DataFrame(ai_raw)
    df_base = pd.DataFrame(base_raw)

    # 2. THE "TRAILING MATCH" LOGIC (Scientific & Accurate)
    # Find the last step that BOTH files have recorded
    max_common_step = min(df_ai['step'].max(), df_base['step'].max())

    # Filter both to this common point
    ai_synced = df_ai[df_ai['step'] <= max_common_step]
    base_synced = df_base[df_base['step'] <= max_common_step]

    # Get metrics at this exact shared point
    last_ai_row = ai_synced.iloc[-1]
    last_base_row = base_synced.iloc[-1]

    metrics = {
        "shared_step": max_common_step,
        "ai_full_step": df_ai['step'].max(),
        "base_full_step": df_base['step'].max(),
        "t_in": last_ai_row['t_in'],
        "t_out": last_ai_row['t_out'],
        "ai_kwh": last_ai_row['cumulative_kwh'],
        "base_kwh": last_base_row['cumulative_kwh']
    }

    # Only return the last 1000 shared steps for the chart (for speed)
    return ai_synced.tail(1000), base_synced.tail(1000), metrics

# --- UI RENDER ---
st.title("Project Envelope: AI-Gated BMS")

ai_df, base_df, stats = load_and_align_data()

if not stats:
    st.info("Awaiting simulation sync...")
    time.sleep(5); st.rerun()

# 1. SIDEBAR
with st.sidebar:
    st.header("Real-time Stats")
    st.metric("Indoor Temp (AI)", f"{stats['t_in']:.2f} °C")
    st.metric("Outdoor Temp", f"{stats['t_out']:.2f} °C")
    st.divider()
    st.write(f"**AI Logic at Step:** {int(stats['ai_full_step'])}")
    st.write(f"**Baseline at Step:** {int(stats['base_full_step'])}")
    st.success(f"Comparing at shared Step: {int(stats['shared_step'])}")

# 2. ENERGY COMPARISON 
st.header(f"Performance at Step {int(stats['shared_step'])}")
c1, c2, c3 = st.columns(3)
c1.metric("AI Energy", f"{stats['ai_kwh']:.2f} kWh")
c2.metric("Baseline Energy", f"{stats['base_kwh']:.2f} kWh")

savings = (stats['base_kwh'] - stats['ai_kwh']) / stats['base_kwh'] * 100 if stats['base_kwh'] > 0 else 0
c3.metric("Live Savings", f"{savings:.1f}%", delta=f"{savings:.1f}%")

# 3. PHYSICS CHART 
st.subheader("Building Physics: Aligned View")
fig = go.Figure()
fig.add_trace(go.Scatter(x=ai_df['step'], y=ai_df['t_in'], name="AI Indoor", line=dict(color='orange')))
fig.add_trace(go.Scatter(x=base_df['step'], y=base_df['t_in'], name="Baseline Indoor", line=dict(color='hotpink', dash='dash')))
fig.add_trace(go.Scatter(x=ai_df['step'], y=ai_df['t_out'], name="Outdoor", line=dict(color='deepskyblue', dash='dot')))
fig.update_layout(height=400, template="plotly_dark", xaxis_title="Simulation Step")
st.plotly_chart(fig, use_container_width=True)

# 4. DECISIONS
st.header("Explainable AI Decisions")
display_df = ai_df[ai_df['source'].str.contains('AI|FAILSAFE', na=False)].tail(5).copy()

# A cache hit takes display priority over the raw trigger_reason: the
# slow loop still fired for one of the real trigger reasons, but the
# more useful fact for this column is that it was served WITHOUT a fresh
# LLM call. quick_check trips (Blueprint 5.1) are a separate, always-on
# path outside the slow loop entirely -- trigger_reason is None for those
# rows by design (see main.py), so they're labeled from `source` instead.
def _trigger_label(row):
    if bool(row.get("cache_hit")):
        return "cache_hit"
    trigger_reason = row.get("trigger_reason")

    if pd.notna(trigger_reason) and trigger_reason:
        return trigger_reason
    source = row.get("source") or ""
    if "QuickCheck" in source:
        return "quick_check"
    return None

for col in ("trigger_reason", "cache_hit"):
    if col not in display_df.columns:
        display_df[col] = None
display_df["trigger"] = display_df.apply(_trigger_label, axis=1)

cols = ['step', 'source', 'trigger', 'setpoint', 'reason']
if display_df['trigger'].isna().all():
    cols.remove('trigger')  # nothing to show yet (e.g. very start of a run) -- don't render a blank column
if 'fallback_error' in display_df.columns and display_df['fallback_error'].notna().any():
    cols.append('fallback_error')
st.dataframe(display_df[cols], use_container_width=True, hide_index=True)

time.sleep(3); st.rerun()
