"""
Week 4 Role C: Premium Streamlit dashboard — aviation disruption risk monitor.
Run from project root: streamlit run dashboard/app.py
"""

import json
import sys
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "eda" / "cache"
MODELS_DIR = ROOT / "models"

# bq_loader lives next to this file
sys.path.insert(0, str(Path(__file__).parent))
import bq_loader

st.set_page_config(
    page_title="AviRisk · Disruption Monitor",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "bg":     "#0e1117",
    "card":   "#1c1f26",
    "border": "#2d3139",
    "blue":   "#00b4d8",
    "green":  "#06d6a0",
    "amber":  "#ffd166",
    "red":    "#ef476f",
    "muted":  "#9ca3af",
    "text":   "#fafafa",
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
.block-container {{ padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }}
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stSidebar"] {{
    background-color: {C["card"]};
    border-right: 1px solid {C["border"]};
}}
[data-testid="stSidebar"] .block-container {{ padding: 1.5rem 1rem; }}
[data-testid="metric-container"] {{
    background: {C["card"]}; border: 1px solid {C["border"]};
    border-radius: 12px; padding: 1rem 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,.35);
}}
[data-testid="metric-container"] label {{
    color: {C["muted"]} !important; font-size: 0.78rem !important;
    text-transform: uppercase; letter-spacing: .06em;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.9rem !important; font-weight: 700 !important; color: {C["text"]} !important;
}}
[data-testid="stTabs"] button {{ font-size:.9rem; font-weight:500; padding:.45rem 1.1rem; border-radius:8px 8px 0 0; }}
[data-testid="stTabs"] button[aria-selected="true"] {{ color:{C["blue"]} !important; border-bottom:2px solid {C["blue"]} !important; }}
hr {{ border-color: {C["border"]} !important; margin: .5rem 0 1.2rem 0; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; }}
.stButton > button {{ background:{C["blue"]}; color:{C["bg"]}; border:none; border-radius:8px; font-weight:600; padding:.45rem 1.2rem; }}
.stButton > button:hover {{ background:#48cae4; color:{C["bg"]}; }}
.section-title {{ font-size:1rem; font-weight:600; color:{C["muted"]}; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.6rem; }}
.stat-row {{ display:flex; gap:1rem; margin-bottom:1rem; }}
.stat-card {{
    flex:1; background:{C["card"]}; border:1px solid {C["border"]};
    border-radius:14px; padding:1.1rem 1.3rem;
    box-shadow:0 2px 10px rgba(0,0,0,.3);
}}
.stat-card .sc-label {{ font-size:.75rem; color:{C["muted"]}; text-transform:uppercase; letter-spacing:.07em; margin-bottom:.3rem; }}
.stat-card .sc-value {{ font-size:1.85rem; font-weight:700; color:{C["text"]}; line-height:1; }}
.stat-card .sc-sub   {{ font-size:.78rem; color:{C["muted"]}; margin-top:.25rem; }}
.accent-blue  {{ border-top:3px solid {C["blue"]}; }}
.accent-green {{ border-top:3px solid {C["green"]}; }}
.accent-amber {{ border-top:3px solid {C["amber"]}; }}
.accent-red   {{ border-top:3px solid {C["red"]}; }}
.bq-badge-on  {{ background:rgba(6,214,160,.15); color:{C["green"]}; border:1px solid {C["green"]}; border-radius:999px; padding:.2rem .75rem; font-size:.78rem; font-weight:600; }}
.bq-badge-off {{ background:rgba(255,209,102,.15); color:{C["amber"]}; border:1px solid {C["amber"]}; border-radius:999px; padding:.2rem .75rem; font-size:.78rem; font-weight:600; }}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ───────────────────────────────────────────────────────────────
PL = dict(
    paper_bgcolor=C["card"], plot_bgcolor=C["card"],
    font=dict(color=C["text"], family="Inter, Segoe UI, sans-serif", size=12),
    title_font=dict(size=14, color=C["text"]),
    xaxis=dict(gridcolor=C["border"], linecolor=C["border"], tickcolor=C["muted"], color=C["muted"]),
    yaxis=dict(gridcolor=C["border"], linecolor=C["border"], tickcolor=C["muted"], color=C["muted"]),
    margin=dict(l=48, r=16, t=44, b=36),
    hoverlabel=dict(bgcolor=C["bg"], font_color=C["text"], bordercolor=C["border"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C["border"]),
)
COLORS = [C["blue"], C["green"], C["amber"], C["red"], "#a78bfa", "#fb923c", "#34d399"]

def sf(fig): fig.update_layout(**PL); return fig


# ── BQ connection check ────────────────────────────────────────────────────────
USE_BQ = bq_loader.is_available()


# ── Local parquet fallback ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_local_bts() -> pd.DataFrame:
    frames = []
    for f in ["bts_2024_1.parquet", "bts_2024_7.parquet", "bts_2024_10.parquet"]:
        p = CACHE / f
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "Cancelled" in df.columns:
        df = df[df["Cancelled"] != 1.0].copy()
    for col in ["CarrierDelay", "WeatherDelay", "NASDelay", "SecurityDelay", "LateAircraftDelay"]:
        if col in df.columns: df[col] = df[col].fillna(0)
    df["flight_month"]       = df["Month"].astype(int)
    df["scheduled_dep_hour"] = (df["CRSDepTime"].fillna(0).astype(int) // 100).clip(0, 23)
    df["airline_code"]       = df["Reporting_Airline"]
    df["DepDelay"]           = df["DepDelay"].fillna(0).clip(-720, 720)
    df["ArrDelay"]           = df["ArrDelay"].fillna(0).clip(-720, 720)
    df["is_delayed"]         = (df["DepDelay"] > 15).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_risk_scores() -> pd.DataFrame:
    p = MODELS_DIR / "risk_scores.parquet"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p)
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df

@st.cache_data(show_spinner=False)
def load_centrality() -> pd.DataFrame:
    p = MODELS_DIR / "centrality_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_airline_list() -> list:
    if USE_BQ:
        return bq_loader.load_airline_list()
    df = load_local_bts()
    return sorted(df["airline_code"].dropna().unique().tolist()) if not df.empty else []


# ── Load base data ─────────────────────────────────────────────────────────────
with st.spinner("Connecting…"):
    local_bts = load_local_bts()
    risk_df   = load_risk_scores()
    cent_df   = load_centrality()
    all_airlines = get_airline_list()

scores_ready = not risk_df.empty
TARGET_AIRPORTS = ["ATL", "DFW", "DEN", "ORD", "LAX", "JFK", "SFO", "SEA", "LAS", "MCO"]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='color:{C['blue']};font-size:1.3rem;font-weight:700;margin-bottom:.2rem'>✈ AviRisk</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['muted']};font-size:.78rem;margin-bottom:.8rem'>Aviation Disruption Monitor · IS3107</div>", unsafe_allow_html=True)

    if USE_BQ:
        st.markdown("<span class='bq-badge-on'>⚡ BigQuery Connected · Feb 2022 – Jan 2026</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='bq-badge-off'>📂 Local Cache · Jan / Jul / Oct 2024</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='section-title'>Filters</div>", unsafe_allow_html=True)

    if USE_BQ:
        month_opts = list(range(1, 13))
        sel_months = st.multiselect("Month", month_opts, default=list(range(1, 13)),
                                     format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"))
        year_opts  = [2022, 2023, 2024, 2025, 2026]
        sel_years  = st.multiselect("Year", year_opts, default=year_opts)
        # encode as (month, year) pairs — we'll handle year in a separate helper
    else:
        month_map  = {1: "Jan 2024", 7: "Jul 2024", 10: "Oct 2024"}
        sel_months = st.multiselect("Month", list(month_map.keys()), default=list(month_map.keys()),
                                     format_func=lambda x: month_map[x])
        sel_years  = [2024]

    sel_airlines = st.multiselect("Airline", all_airlines, default=[], placeholder="All airlines")
    sel_airports = st.multiselect("Origin Airport", TARGET_AIRPORTS, default=[], placeholder="All airports")

    if scores_ready:
        st.divider()
        st.markdown("<div class='section-title'>Risk Level</div>", unsafe_allow_html=True)
        sel_risk = st.multiselect("Show", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
    else:
        sel_risk = ["Low", "Medium", "High"]

    st.divider()
    st.markdown(f"<div style='color:{C['muted']};font-size:.72rem'>Model: Gradient Boosting · AUC 0.73<br>Network: 345 airports · 6 506 routes</div>",
                unsafe_allow_html=True)


# ── BQ filter helpers ─────────────────────────────────────────────────────────
def bq_months():  return sel_months if sel_months else None
def bq_airlines(): return sel_airlines if sel_airlines else None
def bq_airports(): return sel_airports if sel_airports else None

def local_filter(df: pd.DataFrame) -> pd.DataFrame:
    if sel_months and "flight_month" in df.columns:
        df = df[df["flight_month"].isin(sel_months)]
    if sel_airlines and "airline_code" in df.columns:
        df = df[df["airline_code"].isin(sel_airlines)]
    if sel_airports and "Origin" in df.columns:
        df = df[df["Origin"].isin(sel_airports)]
    return df

bts_f = local_filter(local_bts) if not local_bts.empty else local_bts

risk_f = risk_df.copy()
if scores_ready:
    if sel_months and "flight_month" in risk_f.columns:
        risk_f = risk_f[risk_f["flight_month"].isin(sel_months)]
    if sel_airlines and "airline_code" in risk_f.columns:
        risk_f = risk_f[risk_f["airline_code"].isin(sel_airlines)]
    if sel_airports and "Origin" in risk_f.columns:
        risk_f = risk_f[risk_f["Origin"].isin(sel_airports)]
    if "risk_label" in risk_f.columns:
        risk_f = risk_f[risk_f["risk_label"].isin(sel_risk)]


# ── KPI cards ─────────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='margin:0;font-size:1.6rem;font-weight:700'>Aviation Disruption Risk Monitor</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{C['muted']};margin-top:.2rem;font-size:.85rem'>{'BigQuery · Full historical dataset Feb 2022 – Jan 2026' if USE_BQ else 'Local cache · 3 sample months'}</p>", unsafe_allow_html=True)
st.divider()

if USE_BQ:
    @st.cache_data(ttl=3600, show_spinner=False)
    def _kpis(months, airlines, airports):
        return bq_loader.load_kpis(months or None, airlines or None, airports or None)
    kpi = _kpis(tuple(sel_months), tuple(sel_airlines), tuple(sel_airports))
    total_f    = kpi["total_flights"] if kpi else 0
    delay_rate = kpi["delay_rate"]    if kpi else 0
else:
    total_f    = len(bts_f)
    delay_rate = bts_f["is_delayed"].mean() if not bts_f.empty else 0

high_risk_n = int((risk_f["risk_score"] >= 0.6).sum()) if scores_ready else 0
n_airports  = len(cent_df) if not cent_df.empty else 0

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card accent-blue">
    <div class="sc-label">Total Flights</div>
    <div class="sc-value">{total_f:,}</div>
    <div class="sc-sub">{"Feb 2022 – Jan 2026" if USE_BQ else "Jan / Jul / Oct 2024"}</div>
  </div>
  <div class="stat-card accent-amber">
    <div class="sc-label">Delay Rate  (>15 min)</div>
    <div class="sc-value">{delay_rate:.1%}</div>
    <div class="sc-sub">Departure delays</div>
  </div>
  <div class="stat-card accent-red">
    <div class="sc-label">High-Risk Flights</div>
    <div class="sc-value">{"—" if not scores_ready else f"{high_risk_n:,}"}</div>
    <div class="sc-sub">{"Run scoring script first" if not scores_ready else f"Score ≥ 60%"}</div>
  </div>
  <div class="stat-card accent-green">
    <div class="sc-label">Airports in Network</div>
    <div class="sc-value">{n_airports if n_airports else "—"}</div>
    <div class="sc-sub">{"US domestic network" if n_airports else "Run scoring script first"}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈  Historical Trends", "🌐  Network Analysis", "⚠️  Risk Scores", "🤖  Model Performance"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HISTORICAL TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── helper: cache BQ queries keyed on filter selections ──────────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_hour(m, al, ap):   return bq_loader.load_delay_by_hour(m or None, al or None, ap or None)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_airline(m, ap):    return bq_loader.load_delay_by_airline(m or None, ap or None)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_airport(m, al):    return bq_loader.load_delay_by_airport(m or None, al or None)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_monthly(al, ap):   return bq_loader.load_monthly_trend(al or None, ap or None)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_causes(m, al, ap): return bq_loader.load_delay_causes(m or None, al or None, ap or None)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_hist(m, al, ap):   return bq_loader.load_delay_distribution_sample(m or None, al or None, ap or None)

    fk = (tuple(sel_months), tuple(sel_airlines), tuple(sel_airports))

    # Row 1 — delay distribution + hourly
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Departure Delay Distribution</div>", unsafe_allow_html=True)
        if USE_BQ:
            dist_df = bq_hist(*fk)
            delay_series = dist_df["DepDelay"] if dist_df is not None else pd.Series(dtype=float)
        else:
            delay_series = bts_f["DepDelay"].clip(-60, 180).dropna()
            if len(delay_series) > 80_000: delay_series = delay_series.sample(80_000, random_state=42)

        fig = px.histogram(delay_series, nbins=80, color_discrete_sequence=[C["blue"]],
                           labels={"value": "Departure Delay (min)"})
        fig.add_vline(x=15, line_dash="dash", line_color=C["red"],
                      annotation_text="15-min threshold", annotation_font_color=C["red"])
        fig = sf(fig); fig.update_layout(showlegend=False, title="Departure Delay Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Delay Rate by Hour of Day</div>", unsafe_allow_html=True)
        if USE_BQ:
            hourly = bq_hour(*fk)
            if hourly is not None: hourly.columns = ["hour", "rate", "n"]
        else:
            hourly = bts_f.groupby("scheduled_dep_hour")["is_delayed"].mean().reset_index()
            hourly.columns = ["hour", "rate"]

        if hourly is not None and not hourly.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly["hour"], y=hourly["rate"], mode="lines+markers",
                line=dict(color=C["amber"], width=2.5), marker=dict(size=6, color=C["amber"]),
                fill="tozeroy", fillcolor="rgba(255,209,102,.08)",
            ))
            fig.update_yaxes(tickformat=".0%")
            fig.update_xaxes(tickvals=list(range(0, 24, 2)))
            fig = sf(fig); fig.update_layout(title="P(delay > 15 min) by Departure Hour")
            st.plotly_chart(fig, use_container_width=True)

    # Row 2 — airline + airport
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Delay Rate by Airline</div>", unsafe_allow_html=True)
        if USE_BQ:
            al_df = bq_airline(tuple(sel_months), tuple(sel_airports))
            if al_df is not None:
                al_df = al_df.sort_values("delay_rate").tail(15)
        else:
            al_df = (bts_f.groupby("airline_code")
                     .agg(delay_rate=("is_delayed","mean"), n=("is_delayed","count"))
                     .query("n >= 500").sort_values("delay_rate").tail(15).reset_index()
                     .rename(columns={"airline_code":"airline_code"}))

        if al_df is not None and not al_df.empty:
            x_col = "delay_rate"
            y_col = "airline_code" if "airline_code" in al_df.columns else al_df.columns[0]
            fig = px.bar(al_df, x=x_col, y=y_col, orientation="h",
                         color=x_col, color_continuous_scale=["#06d6a0","#ffd166","#ef476f"],
                         labels={x_col:"Delay Rate", y_col:"Airline"})
            fig.update_xaxes(tickformat=".0%")
            fig = sf(fig); fig.update_layout(coloraxis_showscale=False, title="Delay Rate by Airline")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Delay Rate by Origin Airport</div>", unsafe_allow_html=True)
        if USE_BQ:
            ap_df = bq_airport(tuple(sel_months), tuple(sel_airlines))
            if ap_df is not None:
                ap_df = ap_df.sort_values("delay_rate").tail(20)
        else:
            ap_df = (bts_f.groupby("Origin")
                     .agg(delay_rate=("is_delayed","mean"), n=("is_delayed","count"))
                     .query("n >= 300").sort_values("delay_rate").tail(20).reset_index()
                     .rename(columns={"Origin":"airport"}))

        if ap_df is not None and not ap_df.empty:
            y_col = "airport" if "airport" in ap_df.columns else ap_df.columns[0]
            fig = px.bar(ap_df, x="delay_rate", y=y_col, orientation="h",
                         color="delay_rate", color_continuous_scale=["#06d6a0","#ffd166","#ef476f"],
                         labels={"delay_rate":"Delay Rate", y_col:"Airport"})
            fig.update_xaxes(tickformat=".0%")
            fig = sf(fig); fig.update_layout(coloraxis_showscale=False, title="Delay Rate by Origin Airport")
            st.plotly_chart(fig, use_container_width=True)

    # Row 3 — causes + monthly trend
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Delay Cause Breakdown</div>", unsafe_allow_html=True)
        if USE_BQ:
            cause_df = bq_causes(*fk)
        else:
            cause_map = {"CarrierDelay":"Carrier","WeatherDelay":"Weather",
                         "NASDelay":"NAS","SecurityDelay":"Security","LateAircraftDelay":"Late Aircraft"}
            rows = [{"Cause":lbl,"Total Min":bts_f[col].sum()}
                    for col,lbl in cause_map.items() if col in bts_f.columns]
            cause_df = pd.DataFrame(rows) if rows else None

        if cause_df is not None and not cause_df.empty:
            fig = px.pie(cause_df, values="Total Min", names="Cause",
                         color_discrete_sequence=COLORS, hole=0.45)
            fig.update_traces(textposition="outside", textinfo="percent+label",
                              textfont=dict(color=C["text"]))
            fig = sf(fig); fig.update_layout(showlegend=False, title="Share of Total Delay Minutes")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Monthly Delay Trend</div>", unsafe_allow_html=True)
        if USE_BQ:
            mo_df = bq_monthly(tuple(sel_airlines), tuple(sel_airports))
        else:
            month_map_label = {1:"Jan 2024", 7:"Jul 2024", 10:"Oct 2024"}
            mo_df = bts_f.groupby("flight_month")["is_delayed"].agg(delay_rate="mean", n="count").reset_index()
            mo_df["label"] = mo_df["flight_month"].map(month_map_label)

        if mo_df is not None and not mo_df.empty and "label" in mo_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mo_df["label"], y=mo_df["delay_rate"], mode="lines+markers",
                line=dict(color=C["blue"], width=2.5), marker=dict(size=7, color=C["blue"]),
                fill="tozeroy", fillcolor="rgba(0,180,216,.08)",
                hovertemplate="%{x}: %{y:.1%}<extra></extra>",
            ))
            fig.update_yaxes(tickformat=".0%")
            fig = sf(fig); fig.update_layout(title="Monthly Delay Rate Over Time")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NETWORK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if cent_df.empty:
        st.info("Run `python scripts/generate_risk_scores.py` to generate network metrics.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            top20 = cent_df.nlargest(20, "pagerank").sort_values("pagerank")
            fig = px.bar(top20, x="pagerank", y="airport", orientation="h",
                         color="pagerank", color_continuous_scale=["#1c3f6e", C["blue"]],
                         labels={"pagerank":"PageRank","airport":"Airport"},
                         title="Top 20 Hubs — PageRank (Hub Importance)")
            fig = sf(fig); fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            top20b = cent_df.nlargest(20, "betweenness").sort_values("betweenness")
            fig = px.bar(top20b, x="betweenness", y="airport", orientation="h",
                         color="betweenness", color_continuous_scale=["#3d1c6e","#a78bfa"],
                         labels={"betweenness":"Betweenness","airport":"Airport"},
                         title="Top 20 Hubs — Betweenness (Relay Criticality)")
            fig = sf(fig); fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        if not bts_f.empty:
            st.markdown("---")
            st.markdown("<div class='section-title'>Centrality vs Actual Delay Rate</div>", unsafe_allow_html=True)
            ap_d = (bts_f.groupby("Origin")["is_delayed"]
                    .agg(["mean","count"]).query("count >= 300").reset_index())
            ap_d.columns = ["airport","delay_rate","count"]
            merged = ap_d.merge(cent_df, on="airport")
            col1, col2 = st.columns(2)
            for col, xc, xl, title in [
                (col1, "pagerank",    "PageRank",    "PageRank vs Delay Rate"),
                (col2, "betweenness", "Betweenness", "Betweenness vs Delay Rate"),
            ]:
                with col:
                    fig = px.scatter(merged, x=xc, y="delay_rate", size="count",
                                     hover_name="airport", color="delay_rate",
                                     color_continuous_scale=["#06d6a0","#ffd166","#ef476f"],
                                     title=title, size_max=30,
                                     labels={xc:xl,"delay_rate":"Delay Rate"})
                    fig.update_yaxes(tickformat=".0%")
                    fig = sf(fig); fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Network Visualisations — Week 2 Analysis</div>", unsafe_allow_html=True)
    imgs = [
        ("network_graph.png",        "Aviation Network Graph · Top 30 Airports"),
        ("centrality_comparison.png","Centrality Comparison · PageRank / Betweenness / Degree"),
        ("centrality_correlation.png","Centrality Correlation Heatmap"),
        ("degree_distribution.png",   "Degree Distribution · Power-Law"),
    ]
    for i in range(0, len(imgs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(imgs):
                fname, caption = imgs[i+j]
                p = CACHE / fname
                if p.exists():
                    col.image(str(p), caption=caption, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK SCORES
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if not scores_ready:
        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {C["border"]};border-radius:14px;padding:2rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:.8rem'>⚡</div>
            <div style='font-size:1.1rem;font-weight:600;margin-bottom:.5rem'>Risk Scores Not Generated Yet</div>
            <div style='color:{C["muted"]};margin-bottom:1rem'>Run the scoring script from the project root.</div>
            <code style='background:{C["bg"]};padding:.4rem .9rem;border-radius:6px;font-size:.88rem'>python scripts/generate_risk_scores.py</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        total_r = len(risk_f)
        low_n = int((risk_f["risk_score"] < 0.3).sum())
        med_n = int(((risk_f["risk_score"] >= 0.3) & (risk_f["risk_score"] < 0.6)).sum())
        hi_n  = int((risk_f["risk_score"] >= 0.6).sum())

        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card accent-green">
            <div class="sc-label">Low Risk  &lt;30%</div>
            <div class="sc-value">{low_n:,}</div>
            <div class="sc-sub">{low_n/max(total_r,1):.1%} of filtered flights</div>
          </div>
          <div class="stat-card accent-amber">
            <div class="sc-label">Medium Risk  30–60%</div>
            <div class="sc-value">{med_n:,}</div>
            <div class="sc-sub">{med_n/max(total_r,1):.1%} of filtered flights</div>
          </div>
          <div class="stat-card accent-red">
            <div class="sc-label">High Risk  &gt;60%</div>
            <div class="sc-value">{hi_n:,}</div>
            <div class="sc-sub">{hi_n/max(total_r,1):.1%} of filtered flights</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='section-title'>Risk Score Distribution</div>", unsafe_allow_html=True)
            fig = px.histogram(risk_f, x="risk_score", nbins=70, color_discrete_sequence=[C["blue"]],
                               labels={"risk_score":"Risk Score"})
            fig.add_vline(x=0.3, line_dash="dot", line_color=C["amber"],
                          annotation_text="Low/Med", annotation_font_color=C["amber"])
            fig.add_vline(x=0.6, line_dash="dot", line_color=C["red"],
                          annotation_text="Med/High", annotation_font_color=C["red"])
            fig.add_vrect(x0=0.6, x1=1.0, fillcolor=C["red"], opacity=0.05, line_width=0)
            fig = sf(fig); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if "Origin" in risk_f.columns:
                st.markdown("<div class='section-title'>Avg Risk by Origin Airport</div>", unsafe_allow_html=True)
                ap_r = (risk_f.groupby("Origin")["risk_score"]
                        .agg(["mean","count"]).query("count >= 50")
                        .sort_values("mean").tail(20).reset_index())
                ap_r.columns = ["airport","avg_risk","count"]
                fig = px.bar(ap_r, x="avg_risk", y="airport", orientation="h",
                             color="avg_risk", color_continuous_scale=["#06d6a0","#ffd166","#ef476f"],
                             labels={"avg_risk":"Avg Risk Score","airport":"Airport"})
                fig = sf(fig); fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        if "scheduled_dep_hour" in risk_f.columns:
            st.markdown("<div class='section-title'>Avg Risk by Departure Hour</div>", unsafe_allow_html=True)
            hr = risk_f.groupby("scheduled_dep_hour")["risk_score"].mean().reset_index()
            hr.columns = ["hour","risk"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hr["hour"], y=hr["risk"], mode="lines+markers",
                line=dict(color=C["red"], width=2.5), marker=dict(size=6, color=C["red"]),
                fill="tozeroy", fillcolor="rgba(239,71,111,.08)",
            ))
            fig.add_hline(y=0.3, line_dash="dot", line_color=C["amber"], opacity=0.5)
            fig.add_hline(y=0.6, line_dash="dot", line_color=C["red"], opacity=0.5)
            fig.update_xaxes(tickvals=list(range(0, 24, 2)))
            fig = sf(fig); fig.update_layout(title="Average Risk by Hour of Day", height=280)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-title'>Top 50 Highest-Risk Flights</div>", unsafe_allow_html=True)
        disp = ["flight_date","airline_code","Origin","Dest","scheduled_dep_hour",
                "risk_score","risk_label","rolling_6_flight_origin_delay_avg",
                "lag_1_tail_arr_delay_mins","wind_speed_knots"]
        disp = [c for c in disp if c in risk_f.columns]
        top50 = risk_f.nlargest(50, "risk_score")[disp].reset_index(drop=True)
        if "flight_date" in top50.columns:
            top50["flight_date"] = top50["flight_date"].dt.strftime("%Y-%m-%d")
        if "risk_score" in top50.columns:
            top50["risk_score"] = top50["risk_score"].round(3)
        st.dataframe(top50.style.background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
                     use_container_width=True, height=420)

        img = CACHE / "risk_analysis_summary.png"
        if img.exists():
            st.markdown("---")
            st.markdown("<div class='section-title'>4-Panel Summary Chart (Report Export)</div>", unsafe_allow_html=True)
            st.image(str(img), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-title'>Model Comparison</div>", unsafe_allow_html=True)
        res = pd.DataFrame({
            "Model":    ["Logistic Regression","Random Forest","Gradient Boosting"],
            "Accuracy": [0.6633, 0.6926, 0.8024],
            "F1 Score": [0.4297, 0.4608, 0.2676],
            "ROC-AUC":  [0.6954, 0.7271, 0.7278],
        })
        st.dataframe(res.style.highlight_max(subset=["Accuracy","F1 Score","ROC-AUC"],
                                              color="#0d3d2e"),
                     use_container_width=True, hide_index=True)
        fig = go.Figure()
        for i, row in res.iterrows():
            fig.add_trace(go.Bar(
                name=row["Model"],
                x=["Accuracy","F1 Score","ROC-AUC"],
                y=[row["Accuracy"],row["F1 Score"],row["ROC-AUC"]],
                text=[f"{v:.3f}" for v in [row["Accuracy"],row["F1 Score"],row["ROC-AUC"]]],
                textposition="outside", marker_color=COLORS[i], marker_line_width=0,
            ))
        fig = sf(fig)
        fig.update_layout(barmode="group", title="Metrics Side-by-Side",
                          yaxis=dict(range=[0,1.08], **PL["yaxis"]),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Feature Importance · Gradient Boosting</div>", unsafe_allow_html=True)
        fi = pd.DataFrame({
            "feature": ["rolling_6_flight_origin_delay_avg","lag_1_tail_arr_delay_mins",
                        "scheduled_dep_hour","origin_betweenness","rolling_3_flight_weather_delay_flag",
                        "wind_speed_delta","Distance","origin_pagerank","dest_pagerank",
                        "wind_speed_knots","cloud_cover_total_pct","temperature_c",
                        "flight_month","day_of_week","precipitation_mm","cloud_cover_low_pct"],
            "importance": [0.285,0.198,0.142,0.087,0.071,0.052,0.038,
                           0.032,0.028,0.021,0.018,0.015,0.012,0.009,0.007,0.005],
        }).sort_values("importance")
        src = "Approximate values from Week 2 notebook"
        if (MODELS_DIR / "gradient_boosting.joblib").exists():
            try:
                m = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
                with open(MODELS_DIR / "feature_cols.json") as fh: fcols = json.load(fh)
                fi = pd.DataFrame({"feature": fcols, "importance": m.feature_importances_}).sort_values("importance")
                src = "Live values from trained Gradient Boosting model"
            except Exception: pass
        st.caption(src)
        fig = px.bar(fi, x="importance", y="feature", orientation="h",
                     color="importance", color_continuous_scale=["#1a3a5c", C["blue"]],
                     labels={"importance":"Importance","feature":"Feature"})
        fig = sf(fig); fig.update_layout(coloraxis_showscale=False, height=490)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='section-title'>Evaluation Charts — Week 2 Analysis</div>", unsafe_allow_html=True)
    eval_imgs = [("roc_curves.png","ROC Curves"),("confusion_matrices.png","Confusion Matrices"),
                 ("feature_importance.png","Feature Importance"),("model_comparison.png","Model Comparison")]
    for i in range(0, len(eval_imgs), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j < len(eval_imgs):
                fname, caption = eval_imgs[i+j]
                p = CACHE / fname
                if p.exists(): col.image(str(p), caption=caption, use_container_width=True)
