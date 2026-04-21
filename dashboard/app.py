"""
Week 4 Role C: Premium Streamlit dashboard — aviation disruption risk monitor.
Run from project root: streamlit run dashboard/app.py
"""

import json
import sys
import joblib
import numpy as np
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
from airport_coords import AIRPORT_COORDS

st.set_page_config(
    page_title="AviRisk · Disruption Monitor",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed",
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
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
details[data-testid="stExpander"] {{
    background: {C["card"]};
    border: 1px solid {C["border"]} !important;
    border-radius: 12px;
    margin-bottom: 1rem;
}}
details[data-testid="stExpander"] summary {{
    font-weight: 600; font-size: .92rem; color: {C["text"]} !important;
}}
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

@st.cache_resource(show_spinner=False)
def load_prediction_artifacts():
    """Load trained Gradient Boosting model + scaler + feature columns + medians.
    Cached across reruns. Returns None if any artifact is missing."""
    need = ["gradient_boosting.joblib", "feature_cols.json", "feature_medians.json"]
    if not all((MODELS_DIR / f).exists() for f in need):
        return None
    try:
        model = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
        with open(MODELS_DIR / "feature_cols.json") as fh:
            feat_cols = json.load(fh)
        with open(MODELS_DIR / "feature_medians.json") as fh:
            medians = json.load(fh)
        return {"model": model, "feat_cols": feat_cols, "medians": medians}
    except Exception as e:
        print(f"[predict] artifact load failed: {e}")
        return None


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

# ── Header band (brand + connection status, above filters) ───────────────────
import datetime as _dt
DATA_MIN = _dt.date(2022, 2, 1)
DATA_MAX = _dt.date(2026, 1, 31)

hc1, hc2, hc3 = st.columns([3, 2, 2])
with hc1:
    st.markdown(
        f"<div style='color:{C['blue']};font-size:1.3rem;font-weight:700'>✈ AviRisk</div>"
        f"<div style='color:{C['muted']};font-size:.78rem'>Aviation Disruption Monitor · IS3107</div>",
        unsafe_allow_html=True,
    )
with hc2:
    if USE_BQ:
        st.markdown("<span class='bq-badge-on'>⚡ BigQuery Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='bq-badge-off'>📂 Local Cache · Jan / Jul / Oct 2024</span>",
                    unsafe_allow_html=True)
with hc3:
    st.markdown(
        f"<div style='color:{C['muted']};font-size:.72rem;text-align:right'>"
        f"Model: Gradient Boosting · AUC 0.73<br>Network: 345 airports · 6 506 routes</div>",
        unsafe_allow_html=True,
    )


# ── Filters removed — default to full dataset on every view ──────────────────
sel_months   = list(range(1, 13)) if USE_BQ else [1, 7, 10]
sel_airlines = []
sel_airports = []
sel_risk     = ["Low", "Medium", "High"]
start_date, end_date = DATA_MIN, DATA_MAX
date_range = None


# ── BQ filter helpers ─────────────────────────────────────────────────────────
def bq_date_range():
    """Return (start_iso, end_iso) tuple only if user narrowed from the full range."""
    if not USE_BQ:
        return None
    if start_date == DATA_MIN and end_date == DATA_MAX:
        return None
    return (start_date.isoformat(), end_date.isoformat())


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
    # Primary time filter — date range (applies to both BQ and local risk scores table)
    if "flight_date" in risk_f.columns:
        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        risk_f = risk_f[(risk_f["flight_date"] >= start_ts) & (risk_f["flight_date"] < end_ts)]
    if sel_months and "flight_month" in risk_f.columns:
        risk_f = risk_f[risk_f["flight_month"].isin(sel_months)]
    if sel_airlines and "airline_code" in risk_f.columns:
        risk_f = risk_f[risk_f["airline_code"].isin(sel_airlines)]
    if sel_airports and "Origin" in risk_f.columns:
        risk_f = risk_f[risk_f["Origin"].isin(sel_airports)]
    if "risk_label" in risk_f.columns:
        risk_f = risk_f[risk_f["risk_label"].isin(sel_risk)]


# ── KPI cards ─────────────────────────────────────────────────────────────────
_dr_label = (
    f"{start_date.strftime('%b %Y')} – {end_date.strftime('%b %Y')}"
    if USE_BQ else "Jan / Jul / Oct 2024"
)
st.markdown(f"<h1 style='margin:0;font-size:1.6rem;font-weight:700'>Aviation Disruption Risk Monitor</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color:{C['muted']};margin-top:.2rem;font-size:.85rem'>"
    f"{'BigQuery · ' if USE_BQ else 'Local cache · '}{_dr_label}</p>",
    unsafe_allow_html=True,
)
st.divider()

if USE_BQ:
    @st.cache_data(ttl=3600, show_spinner=False)
    def _kpis(months, airlines, airports, dr):
        return bq_loader.load_kpis(months or None, airlines or None, airports or None, dr)
    kpi = _kpis(tuple(sel_months), tuple(sel_airlines), tuple(sel_airports), bq_date_range())
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
    <div class="sc-sub">{_dr_label}</div>
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
tab1, tab2, tab3, tab_weather, tab4, tab_predict = st.tabs(
    ["📈  Historical Trends", "🌐  Network Analysis", "⚠️  Risk Scores",
     "🌦  Weather Impact", "🤖  Model Performance", "🎯  Predict Flight Risk"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HISTORICAL TRENDS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── helper: cache BQ queries keyed on filter selections ──────────────────
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_hour(m, al, ap, dr):   return bq_loader.load_delay_by_hour(m or None, al or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_airline(m, ap, dr):    return bq_loader.load_delay_by_airline(m or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_airport(m, al, dr):    return bq_loader.load_delay_by_airport(m or None, al or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_monthly(al, ap, dr, m):return bq_loader.load_monthly_trend(al or None, ap or None, dr, m or None)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_causes(m, al, ap, dr): return bq_loader.load_delay_causes(m or None, al or None, ap or None, dr)
    @st.cache_data(ttl=3600, show_spinner=False)
    def bq_hist(m, al, ap, dr):   return bq_loader.load_delay_distribution_sample(m or None, al or None, ap or None, dr)

    fk = (tuple(sel_months), tuple(sel_airlines), tuple(sel_airports), bq_date_range())

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
            al_df = bq_airline(tuple(sel_months), tuple(sel_airports), bq_date_range())
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
            ap_df = bq_airport(tuple(sel_months), tuple(sel_airlines), bq_date_range())
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
            mo_df = bq_monthly(tuple(sel_airlines), tuple(sel_airports), bq_date_range(), tuple(sel_months))
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
        # ── US Map — airports sized by hub importance, colored by delay rate ──
        st.markdown("<div class='section-title'>US Aviation Network Map</div>", unsafe_allow_html=True)

        map_df = cent_df.copy()
        map_df["lat"] = map_df["airport"].map(lambda a: AIRPORT_COORDS.get(a, (None, None))[0])
        map_df["lon"] = map_df["airport"].map(lambda a: AIRPORT_COORDS.get(a, (None, None))[1])
        map_df = map_df.dropna(subset=["lat", "lon"]).copy()

        # Attach per-airport delay rate (from risk_scores if available, else local bts)
        delay_lookup = pd.Series(dtype=float)
        if scores_ready and "Origin" in risk_df.columns and "is_delayed" in risk_df.columns:
            delay_lookup = (risk_df.groupby("Origin")["is_delayed"]
                              .agg(["mean", "count"]).query("count >= 100")["mean"])
        elif not bts_f.empty:
            delay_lookup = (bts_f.groupby("Origin")["is_delayed"]
                              .agg(["mean", "count"]).query("count >= 100")["mean"])
        map_df["delay_rate"] = map_df["airport"].map(delay_lookup).fillna(0.0)

        # Limit to continental US for sensible framing, keep AK/HI only if selected
        cont_us = map_df[(map_df["lat"].between(24, 50)) & (map_df["lon"].between(-125, -66))].copy()

        if not cont_us.empty:
            # Route lines — top 40 busiest routes on screen (derived from bts or risk_df)
            route_lines = []
            route_src = risk_df if (scores_ready and {"Origin", "Dest"}.issubset(risk_df.columns)) else bts_f
            if not route_src.empty and {"Origin", "Dest"}.issubset(route_src.columns):
                top_routes = (route_src.groupby(["Origin", "Dest"]).size()
                              .nlargest(50).reset_index(name="n"))
                coord_set = set(cont_us["airport"])
                for _, r in top_routes.iterrows():
                    if r["Origin"] in coord_set and r["Dest"] in coord_set:
                        o = AIRPORT_COORDS[r["Origin"]]; d = AIRPORT_COORDS[r["Dest"]]
                        route_lines.append(dict(
                            lon=[o[1], d[1]], lat=[o[0], d[0]], n=int(r["n"]),
                        ))

            fig = go.Figure()
            for line in route_lines:
                fig.add_trace(go.Scattergeo(
                    lon=line["lon"], lat=line["lat"], mode="lines",
                    line=dict(width=0.6, color="rgba(0,180,216,.35)"),
                    hoverinfo="skip", showlegend=False,
                ))

            fig.add_trace(go.Scattergeo(
                lon=cont_us["lon"], lat=cont_us["lat"],
                text=cont_us.apply(
                    lambda r: f"<b>{r['airport']}</b><br>PageRank: {r['pagerank']:.4f}"
                              f"<br>Betweenness: {r['betweenness']:.3f}"
                              f"<br>Delay rate: {r['delay_rate']:.1%}", axis=1),
                mode="markers", hoverinfo="text",
                marker=dict(
                    size=(cont_us["pagerank"] * 900).clip(4, 40),
                    color=cont_us["delay_rate"],
                    colorscale=[[0, C["green"]], [0.5, C["amber"]], [1, C["red"]]],
                    cmin=0, cmax=max(0.35, cont_us["delay_rate"].max() or 0.35),
                    colorbar=dict(title="Delay<br>Rate", tickformat=".0%",
                                  tickfont=dict(color=C["muted"]), thickness=12, len=0.6),
                    line=dict(width=0.5, color=C["bg"]),
                ),
                showlegend=False,
            ))
            fig.update_geos(
                scope="usa",
                bgcolor=C["card"],
                landcolor="#151922",
                subunitcolor=C["border"],
                countrycolor=C["border"],
                lakecolor=C["card"],
                showlakes=True,
            )
            fig.update_layout(
                paper_bgcolor=C["card"],
                font=dict(color=C["text"], family="Inter, Segoe UI, sans-serif"),
                margin=dict(l=0, r=0, t=30, b=0),
                height=460,
                title="Hubs sized by PageRank · colored by delay rate · lines = top 50 routes",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

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
# TAB WEATHER — WEATHER IMPACT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_weather:
    if not scores_ready:
        st.info("Weather-delay analysis requires `models/risk_scores.parquet`. "
                "Run `python scripts/generate_risk_scores.py` first.")
    else:
        wx_cols = {"wind_speed_knots", "precipitation_mm", "temperature_c",
                   "cloud_cover_total_pct", "is_delayed"}
        if not wx_cols.issubset(risk_f.columns):
            st.warning("Weather columns missing from risk_scores.parquet — regenerate scores.")
        else:
            st.markdown("<div class='section-title'>How weather drives delay probability</div>",
                        unsafe_allow_html=True)
            st.caption("Based on Open-Meteo / METAR observations joined to flight records in "
                       "the dbt fact table. Each point is a binned mean — bin width chosen to keep "
                       "≥200 flights per bin.")

            wx_df = risk_f[list(wx_cols) + (["risk_score"] if "risk_score" in risk_f.columns else [])].copy()

            def binned_rate(df: pd.DataFrame, col: str, bins) -> pd.DataFrame:
                cut = pd.cut(df[col], bins=bins, include_lowest=True)
                g = df.groupby(cut, observed=True).agg(
                    delay_rate=("is_delayed", "mean"),
                    risk=("risk_score", "mean") if "risk_score" in df.columns else ("is_delayed", "mean"),
                    n=("is_delayed", "count"),
                ).reset_index()
                g["mid"] = g[col].apply(lambda iv: (iv.left + iv.right) / 2 if pd.notna(iv) else np.nan)
                return g[g["n"] >= 200]

            # Row 1 — wind + precip
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-title'>Wind Speed vs Delay Rate</div>",
                            unsafe_allow_html=True)
                bins = [0, 3, 6, 9, 12, 15, 18, 22, 26, 30, 40, 60]
                wg = binned_rate(wx_df, "wind_speed_knots", bins)
                if not wg.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=wg["mid"], y=wg["delay_rate"], mode="lines+markers",
                        line=dict(color=C["blue"], width=2.5),
                        marker=dict(size=8 + (wg["n"] / wg["n"].max()) * 14,
                                    color=C["blue"], line=dict(color=C["bg"], width=1)),
                        name="Delay rate",
                        hovertemplate="Wind %{x:.0f} kt<br>Delay %{y:.1%}<br>n=%{customdata:,}<extra></extra>",
                        customdata=wg["n"],
                    ))
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_xaxes(title="Wind speed (knots)")
                    fig = sf(fig)
                    fig.update_layout(title="Delay probability rises with wind speed",
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("<div class='section-title'>Precipitation vs Delay Rate</div>",
                            unsafe_allow_html=True)
                bins = [-0.01, 0.01, 0.5, 1.5, 3, 5, 8, 12, 20, 40, 100]
                pg = binned_rate(wx_df, "precipitation_mm", bins)
                if not pg.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=pg["mid"], y=pg["delay_rate"], mode="lines+markers",
                        line=dict(color=C["red"], width=2.5),
                        marker=dict(size=8 + (pg["n"] / pg["n"].max()) * 14,
                                    color=C["red"], line=dict(color=C["bg"], width=1)),
                        hovertemplate="Precip %{x:.1f} mm<br>Delay %{y:.1%}<br>n=%{customdata:,}<extra></extra>",
                        customdata=pg["n"],
                    ))
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_xaxes(title="Precipitation (mm/hr)")
                    fig = sf(fig)
                    fig.update_layout(title="Even light rain shifts the delay baseline",
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Row 2 — cloud cover + temperature
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<div class='section-title'>Cloud Cover vs Delay Rate</div>",
                            unsafe_allow_html=True)
                bins = [-0.01, 10, 25, 40, 55, 70, 85, 100]
                cg = binned_rate(wx_df, "cloud_cover_total_pct", bins)
                if not cg.empty:
                    fig = px.bar(cg, x="mid", y="delay_rate",
                                 color="delay_rate",
                                 color_continuous_scale=[C["green"], C["amber"], C["red"]],
                                 labels={"mid": "Cloud cover (%)", "delay_rate": "Delay rate"})
                    fig.update_yaxes(tickformat=".0%")
                    fig = sf(fig)
                    fig.update_layout(coloraxis_showscale=False,
                                      title="Overcast conditions correlate with higher delay")
                    st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("<div class='section-title'>Temperature vs Delay Rate</div>",
                            unsafe_allow_html=True)
                bins = [-30, -15, -5, 5, 15, 22, 28, 34, 40, 50]
                tg = binned_rate(wx_df, "temperature_c", bins)
                if not tg.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=tg["mid"], y=tg["delay_rate"], mode="lines+markers",
                        line=dict(color=C["amber"], width=2.5),
                        marker=dict(size=10, color=C["amber"], line=dict(color=C["bg"], width=1)),
                        fill="tozeroy", fillcolor="rgba(255,209,102,.08)",
                        hovertemplate="%{x:.0f}°C<br>Delay %{y:.1%}<extra></extra>",
                    ))
                    fig.update_yaxes(tickformat=".0%")
                    fig.update_xaxes(title="Temperature (°C)")
                    fig = sf(fig)
                    fig.update_layout(title="Cold & heat extremes both lift delay rates",
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Hour × Wind heatmap
            if "scheduled_dep_hour" in risk_f.columns:
                st.markdown("---")
                st.markdown("<div class='section-title'>Hour × Wind-speed Heatmap</div>",
                            unsafe_allow_html=True)
                heat_df = risk_f[["scheduled_dep_hour", "wind_speed_knots", "is_delayed"]].copy()
                heat_df["wind_bin"] = pd.cut(
                    heat_df["wind_speed_knots"],
                    bins=[0, 5, 10, 15, 20, 30, 60],
                    labels=["0-5", "5-10", "10-15", "15-20", "20-30", "30+"],
                    include_lowest=True,
                )
                piv = (heat_df.groupby(["wind_bin", "scheduled_dep_hour"], observed=True)["is_delayed"]
                       .mean().unstack("scheduled_dep_hour"))
                piv = piv.reindex(columns=list(range(0, 24)))
                fig = px.imshow(
                    piv, aspect="auto",
                    color_continuous_scale=[C["green"], C["amber"], C["red"]],
                    labels=dict(x="Departure hour", y="Wind speed (kt)", color="Delay rate"),
                    text_auto=".0%",
                )
                fig.update_xaxes(tickvals=list(range(0, 24, 2)))
                fig = sf(fig)
                fig.update_layout(title="Delay rate by hour and wind band "
                                        "— evening + high wind = worst",
                                  height=340)
                fig.update_traces(textfont=dict(size=9, color=C["text"]))
                st.plotly_chart(fig, use_container_width=True)

            # Weather feature importance callout
            st.markdown("---")
            artifacts = load_prediction_artifacts()
            if artifacts is not None:
                m = artifacts["model"]
                fcols = artifacts["feat_cols"]
                fi = pd.DataFrame({"feature": fcols, "importance": m.feature_importances_})
                weather_feats = ["wind_speed_knots", "precipitation_mm", "temperature_c",
                                  "cloud_cover_total_pct", "cloud_cover_low_pct",
                                  "wind_speed_delta", "rolling_3_flight_weather_delay_flag"]
                w_fi = fi[fi["feature"].isin(weather_feats)].sort_values("importance")
                total_weather = w_fi["importance"].sum()
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"""
                    <div class="stat-card accent-blue">
                      <div class="sc-label">Weather Features · Combined Importance</div>
                      <div class="sc-value">{total_weather:.1%}</div>
                      <div class="sc-sub">Share of Gradient Boosting<br>feature importance</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    fig = px.bar(w_fi, x="importance", y="feature", orientation="h",
                                 color="importance",
                                 color_continuous_scale=["#1a3a5c", C["blue"]],
                                 labels={"importance": "Importance", "feature": ""})
                    fig = sf(fig)
                    fig.update_layout(coloraxis_showscale=False,
                                      title="Weather-related feature importances")
                    st.plotly_chart(fig, use_container_width=True)


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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB PREDICT — WHAT-IF FLIGHT RISK SCORE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    artifacts = load_prediction_artifacts()
    if artifacts is None:
        st.markdown(f"""
        <div style='background:{C["card"]};border:1px solid {C["border"]};border-radius:14px;padding:2rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:.8rem'>🎯</div>
            <div style='font-size:1.1rem;font-weight:600;margin-bottom:.5rem'>Prediction model not found</div>
            <div style='color:{C["muted"]};margin-bottom:1rem'>
                Expected <code>gradient_boosting.joblib</code>, <code>feature_cols.json</code>,
                and <code>feature_medians.json</code> under <code>models/</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        model     = artifacts["model"]
        feat_cols = artifacts["feat_cols"]
        medians   = artifacts["medians"]

        airport_options = (sorted(cent_df["airport"].tolist())
                           if not cent_df.empty else
                           sorted(AIRPORT_COORDS.keys()))
        cent_lookup = cent_df.set_index("airport") if not cent_df.empty else None

        st.markdown("<div class='section-title'>Enter flight details — get a real-time disruption risk score</div>",
                    unsafe_allow_html=True)
        st.caption("All fields pre-filled with dataset medians. Change only what you care about. "
                   "Origin/destination auto-populate PageRank and Betweenness from the network model.")

        with st.form("predict_form"):
            # ── Row 1 — schedule ──
            st.markdown("<div class='section-title'>Schedule</div>", unsafe_allow_html=True)
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            with r1c1:
                in_month = st.selectbox(
                    "Month", list(range(1, 13)),
                    index=int(medians.get("flight_month", 7)) - 1,
                    format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%b"),
                )
            with r1c2:
                dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                in_dow = st.selectbox(
                    "Day of week", list(range(7)),
                    index=int(medians.get("day_of_week", 3)) % 7,
                    format_func=lambda d: dow_labels[d],
                )
            with r1c3:
                in_hour = st.slider("Scheduled dep. hour", 0, 23,
                                     value=int(medians.get("scheduled_dep_hour", 13)))
            with r1c4:
                in_dist = st.number_input("Distance (miles)", min_value=50, max_value=5000, step=50,
                                            value=int(medians.get("Distance", 679)))

            # ── Row 2 — route ──
            st.markdown("<div class='section-title'>Route</div>", unsafe_allow_html=True)
            r2c1, r2c2 = st.columns(2)
            default_o = "ATL" if "ATL" in airport_options else airport_options[0]
            default_d = "LAX" if "LAX" in airport_options else airport_options[min(1, len(airport_options) - 1)]
            with r2c1:
                in_origin = st.selectbox("Origin airport", airport_options,
                                           index=airport_options.index(default_o))
            with r2c2:
                in_dest = st.selectbox("Destination airport", airport_options,
                                         index=airport_options.index(default_d))

            # ── Row 3 — weather ──
            st.markdown("<div class='section-title'>Weather (Open-Meteo / METAR observations)</div>",
                        unsafe_allow_html=True)
            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1:
                in_temp = st.slider("Temperature (°C)", -30.0, 45.0,
                                     value=float(medians.get("temperature_c", 24.5)), step=0.5)
                in_precip = st.slider("Precipitation (mm/hr)", 0.0, 40.0,
                                       value=float(medians.get("precipitation_mm", 0.0)), step=0.1)
            with r3c2:
                in_wind = st.slider("Wind speed (knots)", 0.0, 60.0,
                                     value=float(medians.get("wind_speed_knots", 5.0)), step=0.5)
                in_wind_delta = st.slider("Wind-speed delta (hour-over-hour)", -30.0, 30.0,
                                           value=float(medians.get("wind_speed_delta", 0.0)), step=0.5)
            with r3c3:
                in_cloud_total = st.slider("Total cloud cover (%)", 0, 100,
                                            value=int(medians.get("cloud_cover_total_pct", 22)))
                in_cloud_low = st.slider("Low cloud cover (%)", 0, 100,
                                          value=int(medians.get("cloud_cover_low_pct", 0)))

            # ── Row 4 — operational state ──
            st.markdown("<div class='section-title'>Operational State</div>", unsafe_allow_html=True)
            r4c1, r4c2, r4c3 = st.columns(3)
            with r4c1:
                in_rolling6 = st.slider(
                    "Rolling 6-flight origin delay avg (min)",
                    -30.0, 120.0,
                    value=float(medians.get("rolling_6_flight_origin_delay_avg", 4.0)), step=1.0,
                    help="Average departure delay of the last 6 flights from this origin — "
                         "proxy for current airport congestion.",
                )
            with r4c2:
                in_lag1 = st.slider(
                    "Lag-1 tail arr. delay (min)",
                    -60.0, 180.0,
                    value=float(medians.get("lag_1_tail_arr_delay_mins", -6.0)), step=1.0,
                    help="Arrival delay of this tail number's previous flight — "
                         "captures aircraft propagation chains.",
                )
            with r4c3:
                in_wx_flag = st.checkbox(
                    "Weather delay flagged in last 3 flights",
                    value=bool(int(medians.get("rolling_3_flight_weather_delay_flag", 0))),
                    help="Any of the last 3 flights from this origin recorded weather-coded delay.",
                )

            submitted = st.form_submit_button("Compute Risk Score")

        # ── Prediction ──
        if submitted:
          try:
            def _get_cent(ap: str, col: str) -> float:
                if cent_lookup is not None and ap in cent_lookup.index:
                    return float(cent_lookup.loc[ap, col])
                return 0.0

            feat_values = {
                "flight_month":                       int(in_month),
                "day_of_week":                        int(in_dow),
                "scheduled_dep_hour":                 int(in_hour),
                "Distance":                           float(in_dist),
                "temperature_c":                      float(in_temp),
                "precipitation_mm":                   float(in_precip),
                "wind_speed_knots":                   float(in_wind),
                "cloud_cover_total_pct":              float(in_cloud_total),
                "cloud_cover_low_pct":                float(in_cloud_low),
                "rolling_6_flight_origin_delay_avg":  float(in_rolling6),
                "rolling_3_flight_weather_delay_flag": 1.0 if in_wx_flag else 0.0,
                "wind_speed_delta":                   float(in_wind_delta),
                "lag_1_tail_arr_delay_mins":          float(in_lag1),
                "origin_pagerank":                    _get_cent(in_origin, "pagerank"),
                "origin_betweenness":                 _get_cent(in_origin, "betweenness"),
                "dest_pagerank":                      _get_cent(in_dest, "pagerank"),
            }

            X_row = pd.DataFrame(
                [[float(feat_values.get(c, medians.get(c, 0))) for c in feat_cols]],
                columns=feat_cols, dtype=float,
            )
            prob = float(model.predict_proba(X_row.values)[0, 1])

            if prob < 0.3:
                band, band_color, band_icon = "Low", C["green"], "✓"
            elif prob < 0.6:
                band, band_color, band_icon = "Medium", C["amber"], "⚠"
            else:
                band, band_color, band_icon = "High", C["red"], "✕"

            st.markdown("---")
            st.markdown("<div class='section-title'>Predicted Disruption Risk</div>", unsafe_allow_html=True)

            k1, k2, k3 = st.columns([1.2, 1, 1])
            with k1:
                st.markdown(f"""
                <div class="stat-card" style="border-top:3px solid {band_color}">
                  <div class="sc-label">Risk Score · {in_origin} → {in_dest}</div>
                  <div class="sc-value" style="color:{band_color}">{prob:.1%}</div>
                  <div class="sc-sub">{band_icon} <b style="color:{band_color}">{band} Risk</b>
                    · P(delay &gt; 15 min)</div>
                </div>
                """, unsafe_allow_html=True)
            with k2:
                st.markdown(f"""
                <div class="stat-card accent-blue">
                  <div class="sc-label">Origin Hub Score</div>
                  <div class="sc-value">{feat_values['origin_pagerank']:.4f}</div>
                  <div class="sc-sub">PageRank · betweenness {feat_values['origin_betweenness']:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with k3:
                st.markdown(f"""
                <div class="stat-card accent-amber">
                  <div class="sc-label">Destination Hub Score</div>
                  <div class="sc-value">{feat_values['dest_pagerank']:.4f}</div>
                  <div class="sc-sub">PageRank (dest congestion proxy)</div>
                </div>
                """, unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"color": C["text"], "size": 34}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": C["muted"], "tickfont": {"color": C["muted"]}},
                    "bar": {"color": band_color, "thickness": 0.28},
                    "bgcolor": C["card"],
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30],  "color": "rgba(6,214,160,.22)"},
                        {"range": [30, 60], "color": "rgba(255,209,102,.22)"},
                        {"range": [60, 100], "color": "rgba(239,71,111,.22)"},
                    ],
                    "threshold": {"line": {"color": C["text"], "width": 2},
                                   "thickness": 0.75, "value": prob * 100},
                },
            ))
            fig.update_layout(paper_bgcolor=C["card"], font={"color": C["text"]},
                              height=260, margin=dict(l=30, r=30, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

            # Driver breakdown — deviation × global importance
            st.markdown("<div class='section-title'>Top contributing factors</div>",
                        unsafe_allow_html=True)
            st.caption("Approximate attribution: normalized deviation from dataset median × "
                       "global Gradient Boosting feature importance. Red = pushes risk up, "
                       "green = pushes risk down. Not a SHAP explanation.")

            importances = model.feature_importances_
            rows = []
            for i, f in enumerate(feat_cols):
                val   = float(X_row.iloc[0, i])
                med   = float(medians.get(f, val))
                denom = max(abs(med), 1.0)
                dev   = (val - med) / denom  # signed normalized deviation
                contrib = dev * float(importances[i])
                rows.append({"feature": f, "value": val, "median": med,
                              "importance": float(importances[i]),
                              "contribution": contrib})
            drv = pd.DataFrame(rows)
            drv["abs_contrib"] = drv["contribution"].abs()
            drv = drv.nlargest(10, "abs_contrib").sort_values("contribution")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=drv["contribution"], y=drv["feature"], orientation="h",
                marker=dict(color=[C["red"] if c > 0 else C["green"] for c in drv["contribution"]]),
                customdata=drv[["value", "median", "importance"]].values,
                hovertemplate="<b>%{y}</b><br>Input: %{customdata[0]:.2f}"
                              "<br>Median: %{customdata[1]:.2f}"
                              "<br>Importance: %{customdata[2]:.3f}"
                              "<br>Contribution: %{x:+.4f}<extra></extra>",
            ))
            fig.add_vline(x=0, line_color=C["border"], line_width=1)
            fig = sf(fig)
            fig.update_layout(title="Top 10 drivers (red = increases risk · green = decreases)",
                              xaxis=dict(title="Signed contribution (dev × importance)", **PL["xaxis"]),
                              height=420)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Full input feature vector"):
                show = pd.DataFrame({
                    "feature":  feat_cols,
                    "input":    [float(X_row.iloc[0][c]) for c in feat_cols],
                    "median":   [float(medians.get(c, 0)) for c in feat_cols],
                    "importance": [float(v) for v in importances],
                })
                st.dataframe(
                    show, use_container_width=True, hide_index=True,
                    column_config={
                        "input":      st.column_config.NumberColumn(format="%.3f"),
                        "median":     st.column_config.NumberColumn(format="%.3f"),
                        "importance": st.column_config.NumberColumn(format="%.4f"),
                    },
                )
          except Exception as e:
            st.error("Prediction failed — details below.")
            st.exception(e)
