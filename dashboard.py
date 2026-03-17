"""
AI Intrusion Detection System — Wireshark-Style Live Simulation Dashboard
Real-time network packet stream with animated graph, ML classification,
attack hover details, and full analysis tables.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import sys
import joblib
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.simulate import get_stream_data, normal_stream, attack_stream, mixed_stream, _encode_row, realistic_mixed_stream, normal_burst
from src.predict import predict_single, predict_batch, CATEGORY_NAMES

# ── Page Config ──
st.set_page_config(page_title="AI IDS — Live Monitor", page_icon="🛡️", layout="wide")

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ════════════════════════════════════════════════
   GLOBAL RESET & BASE
════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    background-color: #060b14 !important;
    color: #c8d4e8 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Main app wrapper */
.stApp, .stApp > div, .main, .block-container {
    background-color: #060b14 !important;
}

/* Remove default top padding/gap */
.block-container { padding-top: 1rem !important; }

/* ════════════════════════════════════════════════
   STREAMLIT HEADER / TOOLBAR (top bar)
════════════════════════════════════════════════ */
header[data-testid="stHeader"],
header[data-testid="stHeader"] * {
    background-color: #060b14 !important;
    border-bottom: 1px solid rgba(0,255,136,0.08) !important;
}

/* Deploy button area */
.stDeployButton, [data-testid="stToolbar"] {
    background: #060b14 !important;
    border: none !important;
}
[data-testid="stToolbar"] button {
    background: rgba(0,200,255,0.06) !important;
    color: #8892a8 !important;
    border: 1px solid rgba(0,200,255,0.1) !important;
    border-radius: 6px !important;
}

/* ════════════════════════════════════════════════
   SIDEBAR — full dark
════════════════════════════════════════════════ */
div[data-testid="stSidebar"],
div[data-testid="stSidebar"] > div,
div[data-testid="stSidebar"] section {
    background: #06090f !important;
    border-right: 1px solid rgba(0,255,136,0.08) !important;
}

/* Sidebar collapse button */
div[data-testid="stSidebarCollapseButton"] button,
button[kind="header"] {
    background: #06090f !important;
    color: #4a5580 !important;
    border: none !important;
}
div[data-testid="stSidebarCollapseButton"] svg { fill: #4a5580 !important; }

/* ════════════════════════════════════════════════
   ALL FORM INPUTS — dark backgrounds
════════════════════════════════════════════════ */

/* Selectbox */
div[data-testid="stSelectbox"] > div > div,
div[data-baseweb="select"] > div,
div[data-baseweb="select"] input {
    background-color: #0a1020 !important;
    border-color: rgba(0,200,255,0.15) !important;
    color: #c8d4e8 !important;
}
div[data-baseweb="select"] * { color: #c8d4e8 !important; }
/* Selectbox dropdown popup */
div[data-baseweb="popover"] > div,
ul[data-baseweb="menu"],
li[data-baseweb="menu-item"] {
    background-color: #0d1626 !important;
    color: #c8d4e8 !important;
    border-color: rgba(0,200,255,0.15) !important;
}
li[data-baseweb="menu-item"]:hover { background: rgba(0,200,255,0.08) !important; }

/* Radio buttons */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] span { color: #8892a8 !important; }
div[data-testid="stRadio"] > label { font-size: 0.82rem !important; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap: 0.2rem; }

/* Slider */
div[data-testid="stSlider"] label { color: #8892a8 !important; font-size: 0.82rem !important; }
div[data-testid="stSlider"] div[data-baseweb="slider"] div {
    background-color: rgba(0,200,255,0.08) !important;
}
div[data-testid="stSlider"] div[data-baseweb="slider"] [role="slider"] {
    background-color: #00ff88 !important;
    border-color: #00ff88 !important;
}

/* Toggle */
div[data-testid="stToggle"] label,
div[data-testid="stToggle"] p { color: #c8d4e8 !important; }
div[data-testid="stToggle"] div[role="checkbox"] {
    background-color: rgba(0,200,255,0.08) !important;
    border-color: rgba(0,200,255,0.2) !important;
}

/* General labels and help text */
.stSelectbox label, .stSlider label, .stRadio label,
div[data-testid="stWidgetLabel"] label,
div[data-testid="stWidgetLabel"] p,
small, .caption { color: #8892a8 !important; }

/* Markdown text inside sidebar */
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] small,
div[data-testid="stSidebar"] div { color: #8892a8; }
div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 { color: #c8d4e8 !important; }

/* ════════════════════════════════════════════════
   TABS
════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 4px;
    border-bottom: 1px solid rgba(0,200,255,0.1);
}
.stTabs [data-baseweb="tab"] {
    background: rgba(0,200,255,0.04) !important;
    border: 1px solid rgba(0,200,255,0.1) !important;
    border-radius: 8px 8px 0 0 !important;
    color: #8892a8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,200,255,0.1) !important;
    border-color: rgba(0,200,255,0.35) !important;
    color: #00c8ff !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 0.5rem !important;
}

/* ════════════════════════════════════════════════
   METRICS & DATAFRAMES
════════════════════════════════════════════════ */
div[data-testid="stMetric"] {
    background: rgba(8,17,31,0.8) !important;
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 8px;
    padding: 0.6rem 0.8rem !important;
}
div[data-testid="stMetric"] label { color: #556 !important; font-size: 0.75rem !important; }
div[data-testid="stMetricValue"] { color: #00ff88 !important; }
div[data-testid="stMetricDelta"] { color: #8892a8 !important; }

/* Dataframe */
div[data-testid="stDataFrame"] iframe,
div[data-testid="stDataFrameContainer"],
.stDataFrame { background: #08111f !important; }

/* ════════════════════════════════════════════════
   PLOTLY CHART CONTAINERS
   The white box in pie chart area is plotly iframe bg
════════════════════════════════════════════════ */
div[data-testid="stPlotlyChart"],
div[data-testid="stPlotlyChart"] > div,
div[data-testid="stPlotlyChart"] iframe {
    background: transparent !important;
}
.js-plotly-plot, .plotly, .plotly .bg {
    background: transparent !important;
    background-color: transparent !important;
}

/* ════════════════════════════════════════════════
   ALERTS / INFO / WARNING BOXES
════════════════════════════════════════════════ */
div[data-testid="stAlert"],
div[data-baseweb="notification"] {
    background: rgba(0,200,255,0.05) !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
    color: #c8d4e8 !important;
    border-radius: 8px !important;
}

/* ════════════════════════════════════════════════
   SPINNER / LOADING
════════════════════════════════════════════════ */
div[data-testid="stSpinner"] > div {
    border-top-color: #00ff88 !important;
    background: rgba(6,11,20,0.9) !important;
}

/* ════════════════════════════════════════════════
   IFRAME (components.v1.html) — dark scrollbar
════════════════════════════════════════════════ */
iframe {
    border: none !important;
    background: transparent !important;
    color-scheme: dark;
}

/* ════════════════════════════════════════════════
   BUTTONS
════════════════════════════════════════════════ */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s !important;
    border: 1px solid rgba(0,200,255,0.2) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00c865, #00a855) !important;
    color: #060b14 !important;
    border-color: transparent !important;
}
.stButton > button:hover {
    box-shadow: 0 0 16px rgba(0,255,136,0.2) !important;
}

/* ════════════════════════════════════════════════
   SCROLLBARS — everywhere
════════════════════════════════════════════════ */
* {
    scrollbar-width: thin;
    scrollbar-color: rgba(0,200,255,0.2) #060b14;
}
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #060b14; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,200,255,0.4); }

/* ════════════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
════════════════════════════════════════════════ */
.ids-header {
    background: linear-gradient(135deg, rgba(0,255,136,0.04) 0%, rgba(0,140,255,0.04) 100%);
    border: 1px solid rgba(0,255,136,0.12);
    border-radius: 14px;
    padding: 1.2rem 2rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.ids-header-title {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00ff88, #00c8ff, #9b59ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
    margin: 0;
}
.ids-header-sub { color: #556; font-size: 0.82rem; margin-top: 0.2rem; font-family: 'JetBrains Mono'; }
.ids-header-badge {
    background: rgba(0,255,136,0.08);
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-family: 'JetBrains Mono';
    font-size: 0.8rem;
    color: #00ff88;
}

/* ---- COMPUTER NETWORK VIZ ---- */
.net-panel {
    background: linear-gradient(135deg, #0a1020 0%, #080e1c 100%);
    border: 1px solid rgba(0,140,255,0.18);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
    position: relative;
}
.net-panel-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #00c8ff;
    font-weight: 600;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.net-panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.3), transparent);
}

/* ---- METRICS ROW ---- */
.kpi-row { display: flex; gap: 0.8rem; margin-bottom: 1rem; }
.kpi-card {
    flex: 1;
    background: linear-gradient(135deg, rgba(10,16,32,0.95), rgba(8,14,28,0.95));
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    text-align: center;
    transition: border-color 0.3s;
}
.kpi-card:hover { border-color: rgba(0,200,255,0.35); }
.kpi-val { font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono'; }
.kpi-lbl { color: #556; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0.15rem; }
.c-green  { color: #00ff88; }
.c-red    { color: #ff4757; }
.c-yellow { color: #ffa502; }
.c-blue   { color: #00c8ff; }
.c-purple { color: #9b59ff; }

/* ---- SIMULATION PANEL ---- */
.sim-panel {
    background: #08111f;
    border: 1px solid rgba(0,255,136,0.15);
    border-radius: 14px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}

/* ---- LIVE FEED TABLE ---- */
.feed-table-wrapper {
    background: #08111f;
    border: 1px solid rgba(0,140,255,0.15);
    border-radius: 10px;
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
    margin-bottom: 1rem;
}
.feed-header {
    display: grid;
    grid-template-columns: 90px 60px 80px 80px 80px 80px 120px 90px 90px;
    background: rgba(0,200,255,0.06);
    padding: 0.5rem 0.8rem;
    color: #00c8ff;
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid rgba(0,200,255,0.12);
    font-weight: 600;
}
.feed-row {
    display: grid;
    grid-template-columns: 90px 60px 80px 80px 80px 80px 120px 90px 90px;
    padding: 0.35rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    align-items: center;
    transition: background 0.15s;
}
.feed-row:hover { background: rgba(0,200,255,0.04); }
.feed-row.is-attack { background: rgba(255,71,87,0.04); border-left: 2px solid #ff4757; }
.feed-row.is-normal { border-left: 2px solid rgba(0,255,136,0.3); }
.tag {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.68rem;
    font-weight: 600;
}
.tag-normal  { background: rgba(0,255,136,0.12); color: #00ff88; border: 1px solid rgba(0,255,136,0.25); }
.tag-dos     { background: rgba(255,71,87,0.12);  color: #ff4757; border: 1px solid rgba(255,71,87,0.25); }
.tag-probe   { background: rgba(255,165,2,0.12);  color: #ffa502; border: 1px solid rgba(255,165,2,0.25); }
.tag-r2l     { background: rgba(255,107,157,0.12);color: #ff6b9d; border: 1px solid rgba(255,107,157,0.25); }
.tag-u2r     { background: rgba(155,89,255,0.12); color: #9b59ff; border: 1px solid rgba(155,89,255,0.25); }
.tag-attack  { background: rgba(255,71,87,0.12);  color: #ff4757; border: 1px solid rgba(255,71,87,0.25); }
.tag-anomaly { background: rgba(255,165,2,0.12);  color: #ffa502; border: 1px solid rgba(255,165,2,0.25); }

/* ---- ALERT LOG ---- */
.alerts-panel {
    background: #08111f;
    border: 1px solid rgba(255,71,87,0.15);
    border-radius: 10px;
    padding: 0.9rem;
    max-height: 220px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.73rem;
}
.alert-line {
    padding: 0.3rem 0.5rem;
    margin: 0.2rem 0;
    border-radius: 4px;
    border-left: 3px solid #ff4757;
    background: rgba(255,71,87,0.06);
    color: #ff6b7a;
}
.alert-ok {
    border-left: 3px solid #00ff88;
    background: rgba(0,255,136,0.04);
    color: #00ff88;
    padding: 0.5rem 0.8rem;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}

/* ---- SECTION HEADERS ---- */
.sec-hdr {
    font-size: 0.75rem;
    font-weight: 700;
    color: #00c8ff;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1rem 0 0.6rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.sec-hdr::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,200,255,0.3), transparent);
}

/* ---- SIDEBAR ---- */
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b14, #070d1a) !important;
    border-right: 1px solid rgba(0,255,136,0.08);
}
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stRadio label { color: #8892a8 !important; font-size: 0.82rem; }

/* ---- BUTTONS ---- */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.25s !important;
}

/* ---- MODEL BADGE INLINE ---- */
.model-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    background: rgba(0,200,255,0.1);
    border: 1px solid rgba(0,200,255,0.3);
    color: #00c8ff;
    font-family: 'JetBrains Mono';
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #08111f; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,200,255,0.4); }

.stTabs [data-baseweb="tab-list"] { background: transparent; gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(0,200,255,0.04);
    border: 1px solid rgba(0,200,255,0.1);
    border-radius: 8px 8px 0 0;
    color: #8892a8;
    font-size: 0.82rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,200,255,0.1) !important;
    border-color: rgba(0,200,255,0.35) !important;
    color: #00c8ff !important;
}
</style>
""", unsafe_allow_html=True)


# ── Cache Resources ──
@st.cache_resource
def load_all_resources():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, 'models')
    df, scaler, label_encoders, feature_cols = get_stream_data()
    metrics = joblib.load(os.path.join(models_dir, 'metrics.pkl'))
    feat_imp = joblib.load(os.path.join(models_dir, 'feature_importance.pkl'))
    return df, scaler, label_encoders, feature_cols, metrics, feat_imp


# ── Color helpers ──
ATTACK_COLORS = {
    'Normal':  '#00ff88',
    'DoS':     '#ff4757',
    'Probe':   '#ffa502',
    'R2L':     '#ff6b9d',
    'U2R':     '#9b59ff',
    'Attack':  '#ff4757',
    'Anomaly': '#ffa502',
}

TAG_CLASS = {
    'Normal':  'tag-normal',
    'DoS':     'tag-dos',
    'Probe':   'tag-probe',
    'R2L':     'tag-r2l',
    'U2R':     'tag-u2r',
    'Attack':  'tag-attack',
    'Anomaly': 'tag-anomaly',
}

MODEL_DISPLAY = {
    'random_forest':       '🌳 Random Forest (Binary)',
    'mlp':                 '🧠 MLP Neural Network',
    'isolation_forest':    '🔍 Isolation Forest',
    'random_forest_multi': '🌳 Random Forest (Multi-class)',
}


# ── Chart builders ──
def build_live_graph(history_df):
    """
    Wireshark-style animated traffic timeline.
    Each point = one packet. Y-axis = confidence.
    Color = prediction type.  Hover = full packet details.
    Background shaded red zones where attacks occurred.
    """
    if history_df.empty:
        fig = go.Figure()
        fig.update_layout(
            height=340,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(8,17,31,0.6)',
            xaxis=dict(showgrid=False, color='#334',
                       title='Packet Index', zeroline=False),
            yaxis=dict(gridcolor='rgba(0,200,255,0.06)', color='#334',
                       title='Confidence', range=[0, 1.05], zeroline=False),
            annotations=[dict(
                text='▶ Press <b>Run Simulation</b> to start live traffic capture',
                x=0.5, y=0.5, xref='paper', yref='paper',
                showarrow=False, font=dict(color='#334', size=14, family='Inter')
            )],
            margin=dict(l=50, r=20, t=20, b=40),
        )
        return fig

    fig = go.Figure()

    # Add shaded attack regions (band across full confidence range)
    is_attack_mask = history_df['prediction'] != 'Normal'
    attack_indices = history_df.index[is_attack_mask].tolist()
    if attack_indices:
        # Group consecutive attack indices into spans
        groups = []
        start = attack_indices[0]
        prev = attack_indices[0]
        for idx in attack_indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                groups.append((start, prev))
                start = idx
                prev = idx
        groups.append((start, prev))

        for s, e in groups:
            fig.add_vrect(
                x0=s - 0.5, x1=e + 0.5,
                fillcolor='rgba(255,71,87,0.06)',
                layer='below', line_width=0
            )

    # Baseline 0 line
    fig.add_hline(y=0.5, line=dict(color='rgba(0,200,255,0.15)', width=1, dash='dot'))

    # Per-category traces with markers
    for cat in history_df['prediction'].unique():
        mask = history_df['prediction'] == cat
        sub = history_df[mask]
        color = ATTACK_COLORS.get(cat, '#00c8ff')
        is_normal = (cat == 'Normal')

        # Build hover text
        hover_texts = []
        for _, row in sub.iterrows():
            ht = (
                f"<b style='color:{color}'>{row['prediction']}</b><br>"
                f"<span style='color:#8892a8'>Confidence:</span> <b>{row['confidence']:.1%}</b><br>"
                f"<span style='color:#8892a8'>Protocol:</span> {row['protocol']}<br>"
                f"<span style='color:#8892a8'>Service:</span> {row['service']}<br>"
                f"<span style='color:#8892a8'>Src Bytes:</span> {int(row['src_bytes']):,}<br>"
                f"<span style='color:#8892a8'>Dst Bytes:</span> {int(row['dst_bytes']):,}<br>"
                f"<span style='color:#8892a8'>Duration:</span> {row['duration']}s<br>"
                f"<span style='color:#8892a8'>Model:</span> {row['model']}<br>"
                f"<span style='color:#8892a8'>Time:</span> {row['timestamp']}"
            )
            hover_texts.append(ht)

        fig.add_trace(go.Scatter(
            x=sub.index,
            y=sub['confidence'],
            mode='lines+markers',
            name=cat,
            line=dict(
                color=color,
                width=1.5 if is_normal else 2,
                dash='dot' if is_normal else 'solid',
            ),
            marker=dict(
                color=color,
                size=5 if is_normal else 9,
                symbol='circle' if is_normal else 'diamond',
                line=dict(color='rgba(255,255,255,0.25)', width=1),
                opacity=0.7 if is_normal else 1.0,
            ),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts,
        ))

    fig.update_layout(
        height=340,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(8,17,31,0.7)',
        margin=dict(l=50, r=20, t=15, b=40),
        xaxis=dict(
            title='Packet Index',
            color='#4a5580',
            gridcolor='rgba(0,200,255,0.05)',
            zeroline=False,
            showspikes=True,
            spikecolor='rgba(0,200,255,0.3)',
            spikethickness=1,
        ),
        yaxis=dict(
            title='Confidence',
            range=[0, 1.05],
            color='#4a5580',
            gridcolor='rgba(0,200,255,0.07)',
            zeroline=False,
            tickformat='.0%',
        ),
        legend=dict(
            orientation='h', yanchor='bottom', y=1.02,
            xanchor='right', x=1,
            font=dict(color='#8892a8', size=10, family='JetBrains Mono'),
            bgcolor='rgba(0,0,0,0)',
        ),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='#0d1b2e',
            bordercolor='rgba(0,200,255,0.4)',
            font=dict(color='#c8d4e8', size=11, family='JetBrains Mono'),
        ),
        font=dict(family='Inter'),
    )
    return fig


def build_pie_chart(history_df):
    if history_df.empty:
        return go.Figure()
    counts = history_df['prediction'].value_counts()
    colors = [ATTACK_COLORS.get(c, '#00c8ff') for c in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=colors, line=dict(color='#060b14', width=2)),
        textfont=dict(color='white', size=10, family='Inter'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>',
        hole=0.6,
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(font=dict(color='#8892a8', size=9), orientation='h',
                    yanchor='bottom', y=-0.2, xanchor='center', x=0.5,
                    bgcolor='rgba(0,0,0,0)'),
        annotations=[dict(text='Traffic<br>Types', x=0.5, y=0.5,
                          font=dict(size=11, color='#556'), showarrow=False)],
    )
    return fig


def build_model_comparison(metrics):
    display_map = {
        'random_forest': 'RF Binary', 'rf_multi': 'RF Multi',
        'mlp': 'MLP NN', 'isolation_forest': 'Iso Forest',
    }
    model_names, accs, f1s, precs = [], [], [], []
    for name, m in metrics.items():
        model_names.append(display_map.get(name, name))
        accs.append(m['accuracy'])
        f1s.append(m['f1'])
        precs.append(m['precision'])

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=model_names, y=[a*100 for a in accs],
                         marker_color='#00ff88', text=[f'{a:.1%}' for a in accs],
                         textposition='outside', textfont=dict(size=10)))
    fig.add_trace(go.Bar(name='F1 Score', x=model_names, y=[f*100 for f in f1s],
                         marker_color='#00c8ff', text=[f'{f:.1%}' for f in f1s],
                         textposition='outside', textfont=dict(size=10)))
    fig.add_trace(go.Bar(name='Precision', x=model_names, y=[p*100 for p in precs],
                         marker_color='#ffa502', text=[f'{p:.1%}' for p in precs],
                         textposition='outside', textfont=dict(size=10)))
    fig.update_layout(
        barmode='group', height=310,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,17,31,0.6)',
        margin=dict(l=30, r=20, t=20, b=60),
        xaxis=dict(color='#4a5580', gridcolor='rgba(42,48,80,0.2)'),
        yaxis=dict(color='#4a5580', gridcolor='rgba(42,48,80,0.2)',
                   title='Score (%)', range=[0, 115]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(color='#8892a8', size=10), bgcolor='rgba(0,0,0,0)'),
        font=dict(family='Inter', size=11),
    )
    return fig


def build_feature_importance(feat_imp, top_n=12):
    sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_feats][::-1]
    values = [f[1] for f in sorted_feats][::-1]
    max_v = max(values) if values else 1
    colors = [f'rgba(0,{int(136+119*(v/max_v))},{int(88+167*(v/max_v))},0.85)' for v in values]
    fig = go.Figure(go.Bar(
        y=names, x=values, orientation='h',
        marker=dict(color=colors, line=dict(color='rgba(0,255,136,0.25)', width=1)),
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,17,31,0.6)',
        margin=dict(l=10, r=20, t=10, b=30),
        xaxis=dict(gridcolor='rgba(42,48,80,0.2)', title='Importance Score', color='#4a5580'),
        yaxis=dict(color='#8892a8'),
        font=dict(family='JetBrains Mono', size=10),
    )
    return fig


def build_confusion_matrix(cm, labels):
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0,'#060b14'],[0.3,'#0a2a4a'],[0.65,'#00557a'],[1,'#00ffcc']],
        text=cm, texttemplate='%{text}',
        textfont=dict(size=13, color='white', family='JetBrains Mono'),
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>',
    ))
    fig.update_layout(
        height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(8,17,31,0.6)',
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title='Predicted', color='#4a5580', side='bottom'),
        yaxis=dict(title='Actual', color='#4a5580', autorange='reversed'),
        font=dict(family='Inter', size=11),
    )
    return fig


def build_gauge(value, title, reverse=False):
    if reverse:
        bar_color = '#00ff88' if value > 0.6 else ('#ffa502' if value > 0.3 else '#ff4757')
    else:
        bar_color = '#ff4757' if value > 0.6 else ('#ffa502' if value > 0.3 else '#00ff88')
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={'suffix': '%', 'font': {'size': 26, 'color': '#c8d4e8', 'family': 'JetBrains Mono'}},
        title={'text': title, 'font': {'size': 11, 'color': '#556'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#1a2040', 'dtick': 25},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': '#0a1020',
            'borderwidth': 1, 'bordercolor': '#1a2040',
            'steps': [
                {'range': [0, 30], 'color': 'rgba(0,255,136,0.04)'},
                {'range': [30, 60], 'color': 'rgba(255,165,2,0.04)'},
                {'range': [60, 100], 'color': 'rgba(255,71,87,0.04)'},
            ],
            'threshold': {'line': {'color': '#ff4757', 'width': 2}, 'thickness': 0.75, 'value': 70},
        },
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': '#c8d4e8'},
    )
    return fig


# ── Feed Table (embedded HTML component) ──
def render_feed_table(history_df, max_rows=40):
    """Render a Wireshark-style packet table using st.components.v1.html."""
    TAG_STYLES = {
        'Normal':  ('rgba(0,255,136,0.12)', '#00ff88', 'rgba(0,255,136,0.25)'),
        'DoS':     ('rgba(255,71,87,0.12)',  '#ff4757', 'rgba(255,71,87,0.25)'),
        'Probe':   ('rgba(255,165,2,0.12)',  '#ffa502', 'rgba(255,165,2,0.25)'),
        'R2L':     ('rgba(255,107,157,0.12)','#ff6b9d', 'rgba(255,107,157,0.25)'),
        'U2R':     ('rgba(155,89,255,0.12)', '#9b59ff', 'rgba(155,89,255,0.25)'),
        'Attack':  ('rgba(255,71,87,0.12)',  '#ff4757', 'rgba(255,71,87,0.25)'),
        'Anomaly': ('rgba(255,165,2,0.12)',  '#ffa502', 'rgba(255,165,2,0.25)'),
    }

    if history_df.empty:
        no_data_html = """
        <div style="background:#08111f; border:1px solid rgba(0,140,255,0.15); border-radius:10px;
                    padding:2rem; text-align:center; color:#334;
                    font-family:'JetBrains Mono',monospace; font-size:0.8rem;">
            No packets captured yet &mdash; press <b style="color:#00c8ff">&#9654; Run Batch</b> in the sidebar
        </div>
        """
        components.html(no_data_html, height=80)
        return

    rows_html = ""
    for _, row in history_df.tail(max_rows).iloc[::-1].iterrows():
        pred = str(row['prediction'])
        bg, fg, border = TAG_STYLES.get(pred, TAG_STYLES['Attack'])
        conf_pct = f"{row['confidence']:.1%}"
        conf_color = '#00ff88' if pred == 'Normal' else '#ff4757'
        row_bg = 'rgba(255,71,87,0.04)' if pred != 'Normal' else 'transparent'
        row_border = '#ff4757' if pred != 'Normal' else 'rgba(0,255,136,0.3)'

        rows_html += f"""
        <div style="display:grid; grid-template-columns:90px 55px 75px 80px 80px 55px 110px 85px 80px;
                    padding:0.3rem 0.8rem; border-bottom:1px solid rgba(255,255,255,0.025);
                    background:{row_bg}; border-left:2px solid {row_border};
                    align-items:center; transition:background 0.15s;">
            <span style="color:#4a5580; font-size:0.68rem">{row['timestamp']}</span>
            <span style="color:#8892a8">{row['protocol']}</span>
            <span style="color:#8892a8">{row['service']}</span>
            <span style="color:#556">{int(row['src_bytes']):,}</span>
            <span style="color:#556">{int(row['dst_bytes']):,}</span>
            <span style="color:#556">{row['duration']}s</span>
            <span><span style="display:inline-block;padding:1px 7px;border-radius:4px;font-size:0.67rem;
                font-weight:700;background:{bg};color:{fg};border:1px solid {border};">{pred}</span></span>
            <span style="color:{conf_color}; font-weight:700">{conf_pct}</span>
            <span style="color:#4a5580; font-size:0.63rem">{row['actual']}</span>
        </div>
        """

    full_html = f"""
    <!DOCTYPE html><html><head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
        * {{ box-sizing:border-box; margin:0; padding:0; }}
        body {{ background:#08111f; font-family:'JetBrains Mono',monospace;
                font-size:0.72rem; color:#c8d4e8; overflow-x:hidden; }}
        ::-webkit-scrollbar {{ width:4px; }} ::-webkit-scrollbar-track {{ background:#08111f; }}
        ::-webkit-scrollbar-thumb {{ background:rgba(0,200,255,0.2); border-radius:3px; }}
    </style>
    </head><body>
    <div style="background:#08111f; border:1px solid rgba(0,140,255,0.15); border-radius:10px; overflow:hidden;">
        <div style="display:grid; grid-template-columns:90px 55px 75px 80px 80px 55px 110px 85px 80px;
                    padding:0.5rem 0.8rem; background:rgba(0,200,255,0.06);
                    border-bottom:1px solid rgba(0,200,255,0.12);
                    color:#00c8ff; font-size:0.64rem; text-transform:uppercase;
                    letter-spacing:1px; font-weight:700;">
            <span>TIME</span><span>PROTO</span><span>SERVICE</span>
            <span>SRC BYTES</span><span>DST BYTES</span><span>DUR</span>
            <span>ML LABEL</span><span>CONF</span><span>ACTUAL</span>
        </div>
        {rows_html}
    </div>
    </body></html>
    """

    n_rows = min(len(history_df), max_rows)
    height = min(520, 42 + n_rows * 33)
    components.html(full_html, height=height, scrolling=True)


# ── MAIN ──
def main():
    df, scaler, label_encoders, feature_cols, metrics, feat_imp = load_all_resources()

    # ── Header ──
    st.markdown("""
    <div class="ids-header">
        <div>
            <div class="ids-header-title">🛡️ AI INTRUSION DETECTION SYSTEM</div>
            <div class="ids-header-sub">Wireshark-Style Live Network Packet Classification | NSL-KDD Dataset</div>
        </div>
        <div class="ids-header-badge">● LIVE MONITOR</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session State ──
    if 'history' not in st.session_state:
        st.session_state.history = pd.DataFrame(columns=[
            'timestamp', 'prediction', 'confidence', 'actual', 'model',
            'src_bytes', 'dst_bytes', 'protocol', 'service', 'duration'
        ])
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
    if 'total_packets' not in st.session_state:
        st.session_state.total_packets = 0

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Simulation Controls")
        st.markdown("---")

        model_choice = st.selectbox(
            "🤖 Active ML Model",
            ['random_forest', 'mlp', 'isolation_forest', 'random_forest_multi'],
            format_func=lambda x: MODEL_DISPLAY[x],
            help="Switch between ML models to compare classification behavior in real-time.",
        )

        st.markdown("---")
        st.markdown("### 🌐 Traffic Mode")

        under_attack = st.toggle(
            "⚡ Simulate Attack Event",
            value=False,
            help="Injects a realistic mix of DoS, Probe, R2L and U2R attacks into the stream. The ML model auto-detects the type.",
        )

        if under_attack:
            st.markdown("""
            <div style='background:rgba(255,71,87,0.08); border:1px solid rgba(255,71,87,0.25);
                        border-radius:8px; padding:0.6rem 0.8rem; font-size:0.75rem;
                        color:#ff6b7a; font-family:JetBrains Mono; line-height:1.6;'>
            ☠️ <b>Attack event active</b><br>
            Injecting DoS · Probe · R2L · U2R<br>
            <span style='color:#8892a8'>Let the model figure it out.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:rgba(0,255,136,0.05); border:1px solid rgba(0,255,136,0.15);
                        border-radius:8px; padding:0.6rem 0.8rem; font-size:0.75rem;
                        color:#00ff88; font-family:JetBrains Mono; line-height:1.6;'>
            🟢 <b>Monitoring mode</b><br>
            Normal network traffic flowing.
            </div>
            """, unsafe_allow_html=True)

        n_samples = st.slider("📦 Packets per batch", 5, 60, 25)

        st.markdown("---")

        col_run, col_clear = st.columns(2)
        with col_run:
            run_sim = st.button("▶ Run Batch", use_container_width=True, type="primary")
        with col_clear:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.history = pd.DataFrame(columns=[
                    'timestamp', 'prediction', 'confidence', 'actual', 'model',
                    'src_bytes', 'dst_bytes', 'protocol', 'service', 'duration',
                    'detected_type',
                ])
                st.session_state.alerts = []
                st.session_state.total_packets = 0
                st.rerun()

        # Auto-run toggle
        st.markdown("---")
        auto_run = st.toggle("🔄 Auto-stream (live)", value=False,
                             help="Continuously inject packets at a set interval")
        auto_interval = st.slider("Interval (s)", 0.5, 5.0, 1.5, 0.5) if auto_run else 1.5

        st.markdown("---")
        st.markdown("### 🧠 Model Guide")
        model_guides = {
            'random_forest':       ('🌳', '#00ff88', 'Binary classifier. Flags attack vs normal with high accuracy. Attack type is enriched by the multi-class model.'),
            'mlp':                 ('🧠', '#00c8ff', 'Neural network. Learns non-linear boundaries. Attack type enriched by multi-class model.'),
            'isolation_forest':    ('🔍', '#ffa502', 'Unsupervised anomaly detection — no labels needed. Flags anything unusual. Type enriched by multi-class model.'),
            'random_forest_multi': ('🌳', '#9b59ff', 'Multi-class — directly classifies DoS, Probe, R2L, U2R or Normal in one step.'),
        }
        icon, color, desc = model_guides[model_choice]
        st.markdown(f"""
        <div style='background:rgba(0,0,0,0.3); border:1px solid {color}33; border-radius:8px;
                    padding:0.7rem; font-size:0.78rem; color:#8892a8; line-height:1.6;'>
            <b style='color:{color}'>{icon} {MODEL_DISPLAY[model_choice]}</b><br>{desc}
        </div>
        """, unsafe_allow_html=True)

    # ── Run Simulation ──
    if run_sim or (auto_run and st.session_state.sim_running):
        # Get samples based on mode (no user-picked attack type)
        if under_attack:
            samples = realistic_mixed_stream(df, n_samples, attack_ratio=0.38)
        else:
            samples = normal_burst(df, n_samples)

        new_rows = []
        for idx, row in samples.iterrows():
            try:
                features_scaled = _encode_row(row, label_encoders, scaler, feature_cols)

                # ── Primary model: confidence + binary flag ──
                result = predict_single(features_scaled, model_choice)

                # ── Type labeling: always run multi-class in parallel ──
                # This gives us the specific attack category (DoS/Probe/R2L/U2R)
                # regardless of which primary model is selected.
                if model_choice == 'random_forest_multi':
                    # Already gives specific type — use directly
                    detected_type = result.get('attack_type', 'Normal')
                else:
                    # Run multi-class as auxiliary to name the type
                    try:
                        mc_result = predict_single(features_scaled, 'random_forest_multi')
                        if result['binary'] == 1:
                            # Primary model says it's an attack — use multi-class type
                            detected_type = mc_result.get('attack_type', 'Attack')
                            if detected_type == 'Normal':
                                detected_type = 'Attack'  # Trust primary over multi-class
                        else:
                            detected_type = 'Normal'
                    except Exception:
                        detected_type = result.get('attack_type', result['label'])

                ts = datetime.now().strftime('%H:%M:%S') + f'.{np.random.randint(0,999):03d}'
                proto_val = str(row.get('protocol_type', 'tcp'))
                svc_val = str(row.get('service', 'http'))

                new_rows.append({
                    'timestamp': ts,
                    'prediction': detected_type,
                    'confidence': result['confidence'],
                    'actual': str(row.get('attack_category', 'unknown')),
                    'model': result['model'],
                    'src_bytes': row.get('src_bytes', 0),
                    'dst_bytes': row.get('dst_bytes', 0),
                    'protocol': proto_val,
                    'service': svc_val,
                    'duration': row.get('duration', 0),
                    'detected_type': detected_type,
                })

                if result['binary'] == 1:
                    st.session_state.alerts.append(
                        f"🚨 [{ts}] {detected_type} detected | Conf {result['confidence']:.1%} | "
                        f"{proto_val}/{svc_val} | {result['model']}"
                    )
            except Exception:
                continue

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            st.session_state.history = pd.concat(
                [st.session_state.history, new_df], ignore_index=True
            ).tail(300)
            st.session_state.total_packets += len(new_rows)

    # Activate auto-run state
    if auto_run:
        st.session_state.sim_running = True
    else:
        st.session_state.sim_running = False

    history = st.session_state.history

    # ── KPI Row ──
    total        = len(history)
    n_attacks    = len(history[history['prediction'] != 'Normal']) if total > 0 else 0
    n_safe       = total - n_attacks
    threat_pct   = n_attacks / total if total > 0 else 0
    safe_pct     = 1 - threat_pct
    avg_conf     = history['confidence'].mean() if total > 0 else 0

    correct = len(history[
        ((history['prediction'] == 'Normal') & (history['actual'] == 'normal')) |
        ((history['prediction'] != 'Normal') & (history['actual'] != 'normal'))
    ]) if total > 0 else 0
    live_acc = correct / total if total > 0 else 0

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpi_data = [
        (k1, str(total),            'Packets Captured', 'c-blue'),
        (k2, str(n_safe),           'Safe Traffic',     'c-green'),
        (k3, str(n_attacks),        'Threats Detected', 'c-red'),
        (k4, f'{threat_pct:.1%}',   'Threat Level',     'c-red' if threat_pct > 0.4 else 'c-yellow' if threat_pct > 0.2 else 'c-green'),
        (k5, f'{live_acc:.1%}',     'Live Accuracy',    'c-green' if live_acc > 0.8 else 'c-yellow'),
        (k6, f'{avg_conf:.1%}',     'Avg Confidence',   'c-blue'),
    ]
    for col, val, lbl, cls in kpi_data:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-val {cls}">{val}</div>
                <div class="kpi-lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── SIMULATION PANEL ──
    st.markdown('<div class="sec-hdr">📡 Live Network Traffic Monitor</div>', unsafe_allow_html=True)

    sim_col, pie_col = st.columns([3, 1])

    with sim_col:
        # Model badge
        m_color = {'random_forest':'#00ff88','mlp':'#00c8ff',
                   'isolation_forest':'#ffa502','random_forest_multi':'#9b59ff'}.get(model_choice,'#00c8ff')
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
            <span style="color:#4a5580; font-size:0.72rem; font-family:JetBrains Mono;">ACTIVE MODEL:</span>
            <span class="model-badge" style="border-color:{m_color}44; color:{m_color}; background:{m_color}11;">
                {MODEL_DISPLAY[model_choice]}
            </span>
            <span style="color:#4a5580; font-size:0.72rem; font-family:JetBrains Mono; margin-left:auto;">
                {total} packets | Hover over nodes for details
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Main live graph
        st.plotly_chart(build_live_graph(history), use_container_width=True, key="live_graph")

    with pie_col:
        st.markdown("""
        <div style="height:28px;"></div>
        """, unsafe_allow_html=True)
        st.plotly_chart(build_pie_chart(history), use_container_width=True, key="pie_chart")

        # Mini threat gauge
        st.plotly_chart(build_gauge(threat_pct, "Threat Level %", reverse=False),
                        use_container_width=True, key="threat_gauge")

    # ── Alert Log + Feed Table ──
    st.markdown('<div class="sec-hdr">📋 Live Packet Feed & Alerts</div>', unsafe_allow_html=True)

    feed_col, alert_col = st.columns([3, 1])

    with feed_col:
        # Live packet feed (rendered via components.html for proper styling)
        render_feed_table(history, max_rows=35)

    with alert_col:
        st.markdown("**🚨 Threat Alert Log**", unsafe_allow_html=True)
        alerts = st.session_state.alerts
        if alerts:
            alerts_html = "".join(
                f'<div class="alert-line">{a}</div>'
                for a in reversed(alerts[-25:])
            )
            st.markdown(f'<div class="alerts-panel">{alerts_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-ok">✅ No threats detected. Network nominal.</div>', unsafe_allow_html=True)

    # ── Analysis Tables ──
    st.markdown("---")
    st.markdown('<div class="sec-hdr">🔬 Model Analysis & Explainability</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Feature Importance",
        "📊 Model Comparison",
        "🧮 Confusion Matrix",
        "📋 Session Summary",
        "🧠 How AI Works",
    ])

    with tab1:
        st.markdown('<div class="sec-hdr">🔑 Top Features for Attack Detection</div>', unsafe_allow_html=True)
        t1a, t1b = st.columns([2, 1])
        with t1a:
            st.plotly_chart(build_feature_importance(feat_imp), use_container_width=True, key="feat_imp")
        with t1b:
            st.markdown("""
            <div style='padding:1rem; background:rgba(0,200,255,0.04); border:1px solid rgba(0,200,255,0.15);
                        border-radius:10px; font-size:0.82rem; color:#8892a8; line-height:1.7;'>
            <b style='color:#00c8ff;'>What this shows:</b><br>
            These are the network features the <b>Random Forest</b> model relies
            on most to distinguish attacks from normal traffic.<br><br>
            <b style='color:#00ff88;'>src_bytes</b> — volume of data sent<br>
            <b style='color:#00ff88;'>dst_host_srv_count</b> — service request patterns<br>
            <b style='color:#00ff88;'>service</b> — target service (FTP, HTTP…)<br><br>
            High-volume byte transfers and unusual service access patterns are the
            <b>strongest indicators</b> of DoS and Probe attacks.
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="sec-hdr">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(build_model_comparison(metrics), use_container_width=True, key="model_cmp")

        mcols = st.columns(len(metrics))
        dnames = {'random_forest': '🌳 RF Binary', 'rf_multi': '🌳 RF Multi',
                  'mlp': '🧠 MLP NN', 'isolation_forest': '🔍 IsoForest'}
        for col, (name, m) in zip(mcols, metrics.items()):
            with col:
                st.markdown(f"**{dnames.get(name, name)}**")
                st.metric("Accuracy",  f"{m['accuracy']:.2%}")
                st.metric("F1 Score",  f"{m['f1']:.2%}")
                st.metric("Recall",    f"{m['recall']:.2%}")
                st.metric("Precision", f"{m['precision']:.2%}")

    with tab3:
        st.markdown('<div class="sec-hdr">🧮 Confusion Matrix — RF Binary Classifier</div>', unsafe_allow_html=True)
        cm = metrics.get('random_forest', {}).get('confusion_matrix', np.zeros((2,2)))
        t3a, t3b = st.columns([1, 1])
        with t3a:
            st.plotly_chart(build_confusion_matrix(cm, ['Normal', 'Attack']),
                            use_container_width=True, key="cm")
        with t3b:
            tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
            total_cm = tn + fp + fn + tp
            st.markdown(f"""
            <div style='padding:1rem; font-size:0.82rem; color:#8892a8; line-height:2;'>
            <b style='color:#00ff88;'>True Negatives (TN):</b> {tn:,} — Correctly identified normal traffic<br>
            <b style='color:#00c8ff;'>True Positives (TP):</b> {tp:,} — Correctly detected attacks<br>
            <b style='color:#ffa502;'>False Positives (FP):</b> {fp:,} — Normal flagged as attack<br>
            <b style='color:#ff4757;'>False Negatives (FN):</b> {fn:,} — Missed attacks<br><br>
            <b style='color:#9b59ff;'>Detection Rate:</b> {tp/(tp+fn)*100 if (tp+fn)>0 else 0:.1f}%<br>
            <b style='color:#9b59ff;'>False Alarm Rate:</b> {fp/(fp+tn)*100 if (fp+tn)>0 else 0:.1f}%
            </div>
            """, unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="sec-hdr">📋 Current Session Summary</div>', unsafe_allow_html=True)
        if not history.empty:
            s1, s2, s3 = st.columns(3)
            with s1:
                attack_breakdown = history[history['prediction'] != 'Normal']['prediction'].value_counts()
                if not attack_breakdown.empty:
                    st.markdown("**Attack Breakdown**")
                    for atype, cnt in attack_breakdown.items():
                        pct = cnt / n_attacks * 100 if n_attacks > 0 else 0
                        color = ATTACK_COLORS.get(atype, '#00c8ff')
                        st.markdown(f"""<div style='display:flex; justify-content:space-between;
                            padding:0.3rem 0.5rem; margin:0.2rem 0;
                            background:{color}0d; border-radius:4px; font-size:0.82rem;'>
                            <span style='color:{color}'>{atype}</span>
                            <span style='color:#8892a8'>{cnt} ({pct:.1f}%)</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info("No attacks detected in this session.")

            with s2:
                st.markdown("**Protocol Distribution**")
                proto_counts = history['protocol'].value_counts()
                for proto, cnt in proto_counts.items():
                    pct = cnt / total * 100
                    st.markdown(f"""<div style='display:flex; justify-content:space-between;
                        padding:0.3rem 0.5rem; margin:0.2rem 0;
                        background:rgba(0,200,255,0.05); border-radius:4px; font-size:0.82rem;'>
                        <span style='color:#00c8ff'>{proto}</span>
                        <span style='color:#8892a8'>{cnt} ({pct:.1f}%)</span>
                    </div>""", unsafe_allow_html=True)

            with s3:
                st.markdown("**Session Statistics**")
                stats = [
                    ("Total Packets", total),
                    ("Safe Traffic", f"{n_safe} ({safe_pct:.1%})"),
                    ("Threats", f"{n_attacks} ({threat_pct:.1%})"),
                    ("Avg Confidence", f"{avg_conf:.2%}"),
                    ("Live Accuracy", f"{live_acc:.2%}"),
                    ("Active Model", history['model'].mode()[0] if total > 0 else '—'),
                ]
                for lbl, val in stats:
                    st.markdown(f"""<div style='display:flex; justify-content:space-between;
                        padding:0.3rem 0.5rem; margin:0.15rem 0;
                        background:rgba(155,89,255,0.04); border-radius:4px; font-size:0.82rem;'>
                        <span style='color:#8892a8'>{lbl}</span>
                        <span style='color:#9b59ff; font-family:JetBrains Mono'>{val}</span>
                    </div>""", unsafe_allow_html=True)

            st.markdown("**Last 10 Flagged Packets**")
            flagged = history[history['prediction'] != 'Normal'].tail(10)[
                ['timestamp','protocol','service','src_bytes','dst_bytes','prediction','confidence','actual']
            ].copy()
            flagged.columns = ['Time','Protocol','Service','Src Bytes','Dst Bytes','Prediction','Confidence','Actual']
            flagged['Confidence'] = flagged['Confidence'].apply(lambda x: f'{x:.1%}')
            st.dataframe(flagged, use_container_width=True, height=260)
        else:
            st.info("👆 Run the simulation to see session statistics here.")

    with tab5:
        st.markdown('<div class="sec-hdr">🧠 Why AI/ML for Intrusion Detection?</div>', unsafe_allow_html=True)
        w1, w2 = st.columns(2)
        with w1:
            st.markdown("""
            <div style='padding:1rem; background:rgba(255,71,87,0.05); border:1px solid rgba(255,71,87,0.2); border-radius:10px;'>
            <h4 style='color:#ff4757; margin-top:0;'>❌ Traditional Rule-Based IDS</h4>
            <ul style='color:#8892a8; font-size:0.85rem; line-height:1.8;'>
            <li>Only detects <b>known</b> attack signatures</li>
            <li>Requires constant manual rule updates</li>
            <li>Cannot adapt to new attack variants</li>
            <li>High false-negative rate for novel threats</li>
            <li>Static — no learning from past data</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        with w2:
            st.markdown("""
            <div style='padding:1rem; background:rgba(0,255,136,0.05); border:1px solid rgba(0,255,136,0.2); border-radius:10px;'>
            <h4 style='color:#00ff88; margin-top:0;'>✅ AI/ML-Powered IDS</h4>
            <ul style='color:#8892a8; font-size:0.85rem; line-height:1.8;'>
            <li>Learns <b>patterns</b>, not just signatures</li>
            <li>Detects <b>zero-day attacks</b> automatically</li>
            <li>Adapts and improves with more data</li>
            <li>Low false-negative rate — catches anomalies</li>
            <li>Isolation Forest needs <b>no attack labels</b></li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:1rem; padding:1rem; background:rgba(155,89,255,0.05); border:1px solid rgba(155,89,255,0.2); border-radius:10px;'>
        <h4 style='color:#9b59ff; margin-top:0;'>🔮 How Each Model Works</h4>
        <div style='color:#8892a8; font-size:0.83rem; line-height:1.7;'>
        <b style='color:#00ff88;'>Random Forest (Binary)</b> — Ensemble of decision trees that vote on whether traffic is normal or attack.
        Robust, interpretable, and fast.<br>
        <b style='color:#00c8ff;'>MLP Neural Network</b> — Multi-layer perceptron learns complex non-linear boundaries.
        Best for capturing subtle attack patterns.<br>
        <b style='color:#ffa502;'>Isolation Forest</b> — Unsupervised anomaly detector. Isolates outliers using random partitioning.
        Catches <i>completely new</i> attack types without any labels.<br>
        <b style='color:#9b59ff;'>RF Multi-class</b> — Classifies traffic into specific attack categories:
        DoS, Probe, R2L, U2R, or Normal. Shows the specific threat type.
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Auto-refresh for live streaming ──
    if auto_run:
        time.sleep(auto_interval)
        st.rerun()

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#1e2840; font-size:0.72rem; padding:0.4rem;
                font-family:JetBrains Mono;'>
    🛡️ AI IDS · Python · scikit-learn · Streamlit · Plotly · NSL-KDD Dataset
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
