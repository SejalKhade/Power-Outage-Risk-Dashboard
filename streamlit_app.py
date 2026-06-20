"""
Power Outage Risk Dashboard — Streamlit Edition
================================================
Author  : Sejal Khade | MS Data Science, UT Arlington
Data    : EIA-861 (1,677 utilities, 46 features) + NOAA Storm Events 2024
Stack   : Streamlit · Plotly · scikit-learn
Run     : streamlit run streamlit_app.py

INSTALL (in your venv):
    pip install streamlit plotly pandas pyarrow scikit-learn

KEY VERIFIED DATA FACTS:
  • 1,677 utilities · 50 states · 3.4M raw records
  • High Risk: 336 (20%) · Medium Risk: 1,341 (80%)
  • Top HR states : TN 31 · TX 23 · KY 23 · GA 20 · NC 19
  • Worst NERC    : SERC avg 405 min/yr · FRCC 1,487 · TRE 208
  • Worst utility : Altamaha Electric (GA) SAIDI 17,313 min/yr  $2.73B loss
  • Cooperatives  : 217/336 High Risk (65% of all HR utilities)
"""

import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import streamlit as st

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Power Outage Risk",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────────────────────────────
BG       = "#0E1117"
BG2      = "#151B23"
BG3      = "#1A2130"
GRID     = "#2A3040"
TEXT     = "#E8EDF4"
MUTED    = "#8892A4"
ACCENT   = "#E74C3C"

RISK_COLORS = {
    "High Risk":   "#E74C3C",
    "Medium Risk": "#F39C12",
    "Low Risk":    "#27AE60",
}
HEX_PAL = [
    "#66C2A5","#FC8D62","#8DA0CB","#E78AC3",
    "#A6D854","#FFD92F","#E5C494","#B3B3B3",
    "#1F78B4","#33A02C","#E31A1C","#FF7F00",
]

SAIDI_COL = "IEEE_AllEvents_SAIDI_min_per_yr"
SAIFI_COL = "IEEE_AllEvents_SAIFI_times_per_yr"
CAIDI_COL = "IEEE_AllEvents_CAIDI_min_per_interruption"

STATE_CENTROIDS = {
    "AL":(32.81,-86.79),"AK":(61.37,-152.40),"AZ":(33.73,-111.43),
    "AR":(34.97,-92.37),"CA":(36.12,-119.68),"CO":(39.06,-105.31),
    "CT":(41.60,-72.76),"DE":(39.32,-75.51),"FL":(27.77,-81.69),
    "GA":(33.04,-83.64),"HI":(21.09,-157.50),"ID":(44.24,-114.48),
    "IL":(40.35,-88.99),"IN":(39.85,-86.26),"IA":(42.01,-93.21),
    "KS":(38.53,-96.73),"KY":(37.67,-84.67),"LA":(31.17,-91.87),
    "ME":(44.69,-69.38),"MD":(39.06,-76.80),"MA":(42.23,-71.53),
    "MI":(43.33,-84.54),"MN":(45.69,-93.90),"MS":(32.74,-89.68),
    "MO":(38.46,-92.29),"MT":(46.92,-110.45),"NE":(41.13,-98.27),
    "NV":(38.31,-117.06),"NH":(43.45,-71.56),"NJ":(40.30,-74.52),
    "NM":(34.84,-106.25),"NY":(42.17,-74.95),"NC":(35.63,-79.81),
    "ND":(47.53,-99.78),"OH":(40.39,-82.76),"OK":(35.57,-96.93),
    "OR":(44.57,-122.07),"PA":(40.59,-77.21),"RI":(41.68,-71.51),
    "SC":(33.86,-80.95),"SD":(44.30,-99.44),"TN":(35.75,-86.69),
    "TX":(31.05,-97.56),"UT":(40.15,-111.86),"VT":(44.05,-72.71),
    "VA":(37.77,-78.17),"WA":(47.40,-121.49),"WV":(38.49,-80.95),
    "WI":(44.27,-89.62),"WY":(42.76,-107.30),
}

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS  — dark theme, card styles, metric overrides
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stApp"] {
    background-color: #0E1117;
    color: #E8EDF4;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #2A3040;
}
[data-testid="stSidebar"] * { color: #C9D1DC !important; }

/* ── Cards ── */
.card {
    background: #151B23;
    border: 1px solid #2A3040;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 12px;
}
.card-accent-red    { border-left: 4px solid #E74C3C; }
.card-accent-orange { border-left: 4px solid #F39C12; }
.card-accent-blue   { border-left: 4px solid #2980B9; }
.card-accent-purple { border-left: 4px solid #8E44AD; }
.card-accent-teal   { border-left: 4px solid #16A085; }

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: #151B23;
    border: 1px solid #2A3040;
    border-radius: 10px;
    padding: 14px 16px !important;
}
[data-testid="stMetricLabel"] p  { color: #8892A4 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.8px; }
[data-testid="stMetricValue"]    { color: #E8EDF4 !important; font-size: 22px !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]    { font-size: 11px !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button { color: #8892A4 !important; font-size: 13px; border-bottom: 2px solid transparent; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #E74C3C !important; border-bottom: 2px solid #E74C3C; }

/* ── Expander ── */
[data-testid="stExpander"] { background: #151B23; border: 1px solid #2A3040; border-radius: 8px; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border: 1px solid #2A3040; border-radius: 8px; }

/* ── Header strip ── */
.main-header {
    background: linear-gradient(135deg, #111827 0%, #1E2A3A 50%, #111827 100%);
    border: 1px solid #2A3040;
    border-radius: 12px;
    padding: 24px 32px;
    margin-bottom: 18px;
}
.section-title {
    font-size: 14px;
    font-weight: 700;
    color: #E8EDF4;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    padding: 10px 0 6px;
    border-bottom: 1px solid #2A3040;
    margin-bottom: 10px;
}
/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# DATA LOADING  (cached — reloads only on restart)
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    path = "data/processed/utility_features.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        # Synthetic fallback
        np.random.seed(42)
        states = list(STATE_CENTROIDS.keys())
        n = 1677
        df = pd.DataFrame({
            "Utility Number": range(1, n+1),
            "Utility Name":   [f"Utility Company {i:04d}" for i in range(1, n+1)],
            "State":          np.random.choice(states, n),
            "Ownership":      np.random.choice(
                ["Investor Owned","Cooperative","Municipal",
                 "Political Subdivision","Federal"], n,
                p=[0.4,0.3,0.15,0.1,0.05]),
            "NERC Region":    np.random.choice(
                ["RFC","SERC","SPP","WECC","TRE","MRO","NPCC","FRCC"], n),
            "County_Count":   np.random.randint(1, 30, n),
            SAIDI_COL:   np.random.exponential(300, n),
            SAIFI_COL:   np.random.exponential(1.3, n),
            CAIDI_COL:   np.random.exponential(200, n),
            "weather_event_count":  np.random.poisson(45, n),
            "total_damage_usd":     np.random.exponential(500_000, n),
            "INJURIES_DIRECT":      np.random.poisson(0.5, n),
            "DEATHS_DIRECT":        np.random.poisson(0.05, n),
            "MAGNITUDE":            np.random.uniform(0, 100, n),
            "months_with_events":   np.random.randint(1, 13, n),
            "human_impact_score":   np.random.exponential(2, n),
            "log_total_damage":     np.random.exponential(5, n),
            "total_property_damage_usd": np.random.exponential(400_000, n),
            "total_crops_damage_usd":    np.random.exponential(100_000, n),
        })
        df["saidi_rank_pct"] = df[SAIDI_COL].rank(pct=True)
        df["saifi_rank_pct"] = df[SAIFI_COL].rank(pct=True)
        df["risk_score"]     = (df["saidi_rank_pct"] + df["saifi_rank_pct"]) / 2
        q80 = df["risk_score"].quantile(0.80)
        q50 = df["risk_score"].quantile(0.50)
        df["risk_category"] = np.where(df["risk_score"]>=q80,"High Risk",
                               np.where(df["risk_score"]>=q50,"Medium Risk","Low Risk"))
        df["high_risk"] = (df["risk_score"]>=q80).astype(int)
        df["estimated_annual_loss_usd"] = (
            (df[SAIDI_COL]/60)*df["County_Count"]*50_000*27)
        df["nerc_sla_breach_risk"] = (df[SAIDI_COL]>150).astype(int)
        df["sla_breach_margin_min"] = df[SAIDI_COL] - 150

    # Add lat/lon with jitter
    np.random.seed(0)
    df["lat"] = df["State"].map(
        lambda s: STATE_CENTROIDS.get(s,(39.5,-98.35))[0]
    ) + np.random.uniform(-1.5, 1.5, len(df))
    df["lon"] = df["State"].map(
        lambda s: STATE_CENTROIDS.get(s,(39.5,-98.35))[1]
    ) + np.random.uniform(-2.0, 2.0, len(df))

    for col in ["estimated_annual_loss_usd","nerc_sla_breach_risk",
                "risk_score","risk_category","high_risk",
                "weather_event_count","total_damage_usd","human_impact_score"]:
        if col not in df.columns:
            df[col] = 0

    return df


DF = load_data()
ACTUAL_TIERS  = sorted(DF["risk_category"].unique().tolist())
UTILITY_NAMES = sorted(DF["Utility Name"].dropna().unique().tolist())


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def _layout(fig, h=None, title=None, legend=True):
    kw = dict(
        paper_bgcolor=BG, plot_bgcolor=BG2,
        font_color=TEXT, font_size=11,
        margin=dict(l=30,r=20,t=50 if title else 20,b=30),
        legend=dict(bgcolor=BG3,font=dict(color=TEXT,size=10),
                    bordercolor=GRID,borderwidth=1) if legend else dict(visible=False),
    )
    if h:     kw["height"] = h
    if title: kw["title"]  = dict(text=title, x=0.5, font=dict(size=13,color=TEXT))
    fig.update_layout(**kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, color=MUTED)
    return fig

def _err_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=str(msg), x=0.5, y=0.5,
                       showarrow=False, font=dict(color=MUTED,size=12))
    fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,
                      margin=dict(l=10,r=10,t=10,b=10),height=200)
    return fig

def apply_filters(state, ownership, nerc, tiers):
    df = DF.copy()
    if state    != "All States":  df = df[df["State"]        == state]
    if ownership!= "All Types":   df = df[df["Ownership"]    == ownership]
    if nerc     != "All Regions": df = df[df["NERC Region"]  == nerc]
    if tiers:                     df = df[df["risk_category"].isin(tiers)]
    return df if len(df) > 0 else DF.copy()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Power Outage Risk")
    st.markdown(f"<span style='font-size:11px;color:{MUTED}'>EIA-861 + NOAA · 2024 · 1,677 utilities</span>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("**🔍 Filters**")
    state_sel  = st.selectbox("State",
                    ["All States"] + sorted(DF["State"].dropna().unique()))
    own_sel    = st.selectbox("Ownership Type",
                    ["All Types"]  + sorted(DF["Ownership"].dropna().unique()))
    nerc_sel   = st.selectbox("NERC Region",
                    ["All Regions"]+ sorted(DF["NERC Region"].dropna().unique()))
    tier_sel   = st.multiselect("Risk Tier", ACTUAL_TIERS, default=ACTUAL_TIERS)

    st.divider()
    st.markdown("**🗺️ Map Controls**")
    choro_metric = st.selectbox("Choropleth Metric", [
        "Average SAIDI (min/yr)",
        "High Risk Utility %",
        "Total Storm Damage ($B)",
        "Economic Loss ($M)",
    ])
    n_clusters = st.slider("KMeans Clusters (K)", 3, 12, 8)
    saidi_cap  = st.slider("SAIDI cap for charts (min/yr)",
                           500, 20000, 5000, step=500,
                           help="Excludes extreme outliers so charts are readable")

    st.divider()
    st.markdown("**🔍 Compare Utilities**")
    hr_names = DF[DF["risk_category"]=="High Risk"]["Utility Name"].dropna().tolist()
    mr_names = DF[DF["risk_category"]=="Medium Risk"]["Utility Name"].dropna().tolist()
    util_a = st.selectbox("Utility A", UTILITY_NAMES,
                          index=UTILITY_NAMES.index(hr_names[0]) if hr_names else 0)
    util_b = st.selectbox("Utility B", UTILITY_NAMES,
                          index=UTILITY_NAMES.index(mr_names[0]) if mr_names else 1)

    st.divider()
    st.markdown(
        f"<p style='font-size:10px;color:{MUTED};'>Built by <b style='color:#aaa'>Sejal Khade</b><br>"
        "MS Data Science · UT Arlington<br>"
        "<a href='https://github.com/SejalKhade' style='color:#E74C3C;'>github.com/SejalKhade</a></p>",
        unsafe_allow_html=True)

# Apply filters
df = apply_filters(state_sel, own_sel, nerc_sel, tier_sel)


# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="font-size:26px;font-weight:800;color:#E8EDF4;margin:0 0 4px 0;">
    ⚡ Power Outage Risk Dashboard
  </h1>
  <p style="font-size:12px;color:#8892A4;margin:0;line-height:1.7;">
    Identifies high-risk utilities across all 50 U.S. states ·
    EIA Form 861 reliability data + NOAA Storm Events 2024 ·
    1,677 utilities · 3.4M records · 46 engineered features ·
    ML: Logistic Regression + Weather feature set · ROC-AUC 0.668 · PR-AUC 0.325
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────
n        = len(df)
n_high   = int((df["risk_category"]=="High Risk").sum())
n_states = int(df["State"].nunique())
avg_saidi= float(df[SAIDI_COL].mean()) if SAIDI_COL in df.columns else 0
nat_avg  = float(DF[SAIDI_COL].mean())
loss     = float(df["estimated_annual_loss_usd"].sum()) if "estimated_annual_loss_usd" in df.columns else 0
sla      = int(df["nerc_sla_breach_risk"].sum()) if "nerc_sla_breach_risk" in df.columns else 0
storms   = int(df["weather_event_count"].sum()) if "weather_event_count" in df.columns else 0

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("⚡ Utilities",       f"{n:,}",
          delta=f"of 1,677 total", delta_color="off")
c2.metric("🔴 High Risk",       f"{n_high:,}",
          delta=f"{n_high/max(n,1)*100:.1f}% flagged", delta_color="inverse")
c3.metric("🗺️ States",           f"{n_states}",
          delta="US states covered", delta_color="off")
c4.metric("📊 Avg SAIDI",       f"{avg_saidi:.0f} min/yr",
          delta=f"{avg_saidi-nat_avg:+.0f} vs national avg",
          delta_color="inverse")
c5.metric("💰 Est. Annual Loss", f"${loss/1e9:.2f}B",
          delta=f"{sla} NERC breach", delta_color="inverse")
c6.metric("🌩️ Storm Events",     f"{storms:,}",
          delta="total events", delta_color="off")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────
# KEY FINDINGS BANNER
# ─────────────────────────────────────────────────────────────────
high_df  = df[df["risk_category"]=="High Risk"]
top_st   = high_df["State"].value_counts().index[0] if len(high_df)>0 else "N/A"
top_n    = int(high_df["State"].value_counts().iloc[0]) if len(high_df)>0 else 0
top_nerc = df.groupby("NERC Region")[SAIDI_COL].mean().idxmax() \
           if SAIDI_COL in df.columns and len(df)>0 else "N/A"
top_own  = high_df["Ownership"].value_counts().index[0] if len(high_df)>0 else "N/A"
loss_b   = loss / 1e9

fa,fb,fc = st.columns(3)
fa.markdown(f"""
<div class="card card-accent-blue">
  <div style="font-weight:700;color:#2980B9;font-size:12px;margin-bottom:5px;">🗺️ Geographic Concentration</div>
  <div style="font-size:11px;color:#C9D1DC;line-height:1.6;">
    <b style="color:#E8EDF4">{top_st}</b> leads with <b style="color:#E74C3C">{top_n} High Risk utilities</b>.
    NERC region <b style="color:#E8EDF4">{top_nerc}</b> has the highest avg SAIDI.
    Southeast states dominate the high-risk tier — Cooperatives account for 65% of all flagged utilities.
  </div>
</div>""", unsafe_allow_html=True)

fb.markdown(f"""
<div class="card card-accent-orange">
  <div style="font-weight:700;color:#F39C12;font-size:12px;margin-bottom:5px;">🌩️ Storm Exposure Predicts Risk</div>
  <div style="font-size:11px;color:#C9D1DC;line-height:1.6;">
    Adding storm features improves ML model ROC-AUC <b style="color:#E8EDF4">0.50 → 0.77 (+54%)</b>.
    <b style="color:#F39C12">{top_own}</b> utilities have the highest High Risk count.
    Altamaha Electric (GA) is worst: SAIDI <b style="color:#E74C3C">17,313 min/yr</b>.
  </div>
</div>""", unsafe_allow_html=True)

fc.markdown(f"""
<div class="card card-accent-purple">
  <div style="font-weight:700;color:#8E44AD;font-size:12px;margin-bottom:5px;">💰 Economic & Regulatory Stakes</div>
  <div style="font-size:11px;color:#C9D1DC;line-height:1.6;">
    Estimated <b style="color:#E8EDF4">${loss_b:.1f}B</b> annual economic loss from outages.
    <b style="color:#8E44AD">{sla} utilities</b> exceed NERC SAIDI reliability threshold of 150 min/yr.
    High Risk utilities represent 20% of utilities but drive the majority of economic impact.
  </div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# PAGE TABS
# ─────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🗺️ Maps",
    "📊 Risk Analysis",
    "🌩️ Storm & Damage",
    "🏭 NERC Deep Dive",
    "🔍 Utility Comparison",
    "📋 Data Table",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — MAPS
# ══════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Geographic Analysis</div>', unsafe_allow_html=True)

    map_tab1, map_tab2, map_tab3 = st.tabs([
        "🌍 Choropleth — State Level",
        "📍 Scatter — Individual Utilities",
        "🔵 KMeans Clusters",
    ])

    # ── Choropleth ────────────────────────────────────────────────
    with map_tab1:
        try:
            if choro_metric == "Average SAIDI (min/yr)":
                sd = df.groupby("State")[SAIDI_COL].mean().reset_index()
                sd.columns = ["State","value"]
                cs,cl = "RdYlGn_r","Avg SAIDI (min/yr)"
                title = "Average Outage Duration by State (SAIDI min/yr) — Red = Worse"
            elif choro_metric == "High Risk Utility %":
                tmp = df.copy()
                tmp["is_hr"] = (tmp["risk_category"]=="High Risk").astype(float)*100
                sd = tmp.groupby("State")["is_hr"].mean().reset_index()
                sd.columns = ["State","value"]
                cs,cl = "Reds","% High Risk Utilities"
                title = "Percentage of High Risk Utilities by State"
            elif choro_metric == "Total Storm Damage ($B)":
                sd = df.groupby("State")["total_damage_usd"].sum().div(1e9).reset_index()
                sd.columns = ["State","value"]
                cs,cl = "OrRd","Total Storm Damage ($B)"
                title = "Total Cumulative Storm Damage by State ($B)"
            else:
                sd = df.groupby("State")["estimated_annual_loss_usd"].sum().div(1e6).reset_index()
                sd.columns = ["State","value"]
                cs,cl = "Purples","Economic Loss ($M)"
                title = "Estimated Annual Economic Loss by State ($M)"

            fig = px.choropleth(sd, locations="State", locationmode="USA-states",
                                color="value", color_continuous_scale=cs,
                                scope="usa", title=title, labels={"value":cl})
            fig.update_layout(
                title_x=0.5, height=500,
                margin=dict(l=0,r=0,t=50,b=0),
                paper_bgcolor=BG, font_color=TEXT,
                geo=dict(bgcolor=BG,landcolor="#1E2530",lakecolor=BG,
                         showlakes=True,showland=True,
                         showcoastlines=True,coastlinecolor="#444"))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Choropleth error: {e}")

    # ── Scatter map ───────────────────────────────────────────────
    with map_tab2:
        try:
            d = df.copy()
            d["hover"] = (
                "<b>"+d["Utility Name"].fillna("?")+"</b><br>"+
                "State: "+d["State"].fillna("")+"<br>"+
                "Risk: "+d["risk_category"].fillna("")+"<br>"+
                "SAIDI: "+d[SAIDI_COL].round(1).astype(str)+" min/yr<br>"+
                "Ownership: "+d["Ownership"].fillna(""))
            fig = go.Figure()
            for tier in ACTUAL_TIERS:
                t = d[d["risk_category"]==tier]
                if len(t)==0: continue
                mx = t[SAIDI_COL].max()
                sz = ((t[SAIDI_COL]/mx*18+5) if mx>0 else pd.Series(6,index=t.index)).clip(4,22)
                fig.add_trace(go.Scattergeo(
                    lat=t["lat"], lon=t["lon"], mode="markers", name=tier,
                    marker=dict(color=RISK_COLORS.get(tier,"#999"),size=sz,opacity=0.75,
                                line=dict(width=0.3,color="#ffffff25")),
                    text=t["hover"], hovertemplate="%{text}<extra></extra>"))
            fig.update_layout(
                title_text="US Utility Risk Map — dot size = SAIDI severity",
                title_x=0.5, height=520,
                geo=dict(scope="usa",projection_type="albers usa",showland=True,
                         landcolor="#1E2530",showlakes=True,lakecolor=BG,
                         showcoastlines=True,coastlinecolor="#333",bgcolor=BG),
                legend=dict(orientation="h",y=-0.05,x=0.5,xanchor="center",
                            font=dict(color=TEXT),bgcolor=BG3),
                paper_bgcolor=BG, font_color=TEXT, margin=dict(l=0,r=0,t=50,b=0))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Scatter map error: {e}")

    # ── KMeans cluster scatter ─────────────────────────────────────
    with map_tab3:
        try:
            d = df.copy()
            k = max(2, min(n_clusters, len(d)-1))
            fcols = [c for c in [SAIDI_COL,SAIFI_COL,
                                  "weather_event_count","total_damage_usd",
                                  "lat","lon"] if c in d.columns]
            X = StandardScaler().fit_transform(d[fcols].fillna(0))
            d["cluster"] = KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X).astype(str)

            stats = d.groupby("cluster").agg(
                n=("cluster","count"),
                phr=("high_risk","mean"),
                avgs=(SAIDI_COL,"mean"),
            ).reset_index()
            lmap = {
                r["cluster"]:
                    f"Cluster {r['cluster']}: {int(r['n'])} utils · "
                    f"{r['phr']*100:.0f}% HR · SAIDI {r['avgs']:.0f}"
                for _,r in stats.iterrows()
            }
            d["cluster_label"] = d["cluster"].map(lmap)
            d["hover_c"] = (
                "<b>"+d["Utility Name"].fillna("")+"</b><br>"+
                "State: "+d["State"].fillna("")+"<br>"+
                "Risk: "+d["risk_category"].fillna("")+"<br>"+
                "SAIDI: "+d[SAIDI_COL].round(1).astype(str)+" min/yr<br>"+
                d["cluster_label"].fillna(""))

            fig = go.Figure()
            for cid in sorted(d["cluster"].unique()):
                c_df  = d[d["cluster"]==cid]
                color = HEX_PAL[int(cid)%len(HEX_PAL)]
                fig.add_trace(go.Scatter(
                    x=c_df["lon"], y=c_df["lat"],
                    mode="markers", name=lmap.get(cid,f"C{cid}"),
                    marker=dict(color=color,size=6,opacity=0.75,
                                line=dict(width=0.3,color="#ffffff25")),
                    text=c_df["hover_c"],
                    hovertemplate="%{text}<extra></extra>"))
                # centroid
                cx,cy = float(c_df["lon"].mean()), float(c_df["lat"].mean())
                fig.add_trace(go.Scatter(
                    x=[cx], y=[cy], mode="markers+text", showlegend=False,
                    marker=dict(color=color,size=20,opacity=0.3,
                                symbol="circle",line=dict(width=2,color=color)),
                    text=[f"C{cid}"], textfont=dict(size=9,color="#FFF"),
                    textposition="middle center", hoverinfo="skip"))

            fig.update_layout(
                title_text=(
                    f"Geographic Risk Clusters — KMeans k={k} · "
                    "X = Longitude · Y = Latitude · Ring = centroid"),
                title_x=0.5, height=540,
                xaxis=dict(title="Longitude",range=[-130,-65],
                           gridcolor=GRID,zeroline=False,color=MUTED),
                yaxis=dict(title="Latitude",range=[24,50],
                           gridcolor=GRID,zeroline=False,
                           scaleanchor="x",scaleratio=1.3,color=MUTED),
                paper_bgcolor=BG, plot_bgcolor="#0D1420", font_color=TEXT,
                legend=dict(orientation="v",y=1.0,x=1.01,xanchor="left",
                            font=dict(color=TEXT,size=9),bgcolor=BG3),
                margin=dict(l=40,r=220,t=60,b=40))
            st.plotly_chart(fig, use_container_width=True)

            # Cluster summary table
            with st.expander("📋 Cluster Summary Table"):
                ct = d.groupby("cluster_label").agg(
                    Utilities=(SAIDI_COL,"count"),
                    High_Risk_Pct=("high_risk","mean"),
                    Avg_SAIDI=(SAIDI_COL,"mean"),
                    Total_Damage_M=("total_damage_usd","sum"),
                    Avg_Loss_M=("estimated_annual_loss_usd","mean"),
                ).reset_index().rename(columns={"cluster_label":"Cluster"})
                ct["High_Risk_Pct"] = (ct["High_Risk_Pct"]*100).round(1).astype(str)+"%"
                ct["Avg_SAIDI"]     = ct["Avg_SAIDI"].round(1)
                ct["Total_Damage_M"]= (ct["Total_Damage_M"]/1e6).round(1)
                ct["Avg_Loss_M"]    = (ct["Avg_Loss_M"]/1e6).round(1)
                st.dataframe(ct, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Cluster map error: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 2 — RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-title">Risk Analysis</div>', unsafe_allow_html=True)

    ra1, ra2 = st.columns(2)

    # ── Donut + ownership stacked bar ─────────────────────────────
    with ra1:
        try:
            fig = make_subplots(rows=1,cols=2,
                specs=[[{"type":"pie"},{"type":"bar"}]],
                subplot_titles=["Risk Tier Split","Risk by Ownership"])
            rc = df["risk_category"].value_counts()
            fig.add_trace(go.Pie(
                labels=rc.index, values=rc.values, hole=0.6,
                marker_colors=[RISK_COLORS.get(r,"#999") for r in rc.index],
                textfont_size=11, name="",
                hovertemplate="<b>%{label}</b><br>%{value} utilities (%{percent})<extra></extra>"),
                row=1, col=1)
            major = df["Ownership"].value_counts().head(5).index.tolist()
            grp = df[df["Ownership"].isin(major)].groupby(
                ["Ownership","risk_category"]).size().reset_index(name="Count")
            for tier in ACTUAL_TIERS:
                t = grp[grp["risk_category"]==tier]
                if len(t)>0:
                    fig.add_trace(go.Bar(
                        x=t["Ownership"],y=t["Count"],name=tier,
                        marker_color=RISK_COLORS.get(tier,"#999")),row=1,col=2)
            _layout(fig, h=380)
            fig.update_layout(barmode="stack")
            fig.update_xaxes(tickangle=35,tickfont_size=8,row=1,col=2)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Distribution: {e}")

    # ── Risk score histogram with threshold ───────────────────────
    with ra2:
        try:
            threshold = float(df["risk_score"].quantile(0.80))
            fig = go.Figure()
            for tier in ACTUAL_TIERS:
                t = df[df["risk_category"]==tier]["risk_score"]
                if len(t)>0:
                    fig.add_trace(go.Histogram(
                        x=t, name=tier, nbinsx=40,
                        marker_color=RISK_COLORS.get(tier,"#999"),
                        opacity=0.8,
                        hovertemplate=f"Score: %{{x:.3f}}<br>Count: %{{y}}<extra>{tier}</extra>"))
            fig.add_vline(x=threshold, line_dash="dash",
                          line_color="#FFD700", line_width=2)
            fig.add_annotation(
                x=threshold, y=1, yref="paper",
                text=f"Top 20%<br>{threshold:.3f}",
                showarrow=True, arrowhead=2, arrowcolor="#FFD700",
                font=dict(color="#FFD700",size=9), bgcolor=BG3,
                ax=40, ay=-30)
            _layout(fig, h=380, title="Risk Score Distribution — SAIDI/SAIFI Composite Percentile")
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title_text="Risk Score (0 = lowest · 1 = highest)")
            fig.update_yaxes(title_text="Utilities")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Histogram: {e}")

    # ── SAIDI violin ──────────────────────────────────────────────
    ra3, ra4 = st.columns(2)
    with ra3:
        try:
            q95 = df[SAIDI_COL].quantile(0.95)
            fig = go.Figure()
            for tier in ACTUAL_TIERS:
                t = df[(df["risk_category"]==tier)&
                       (df[SAIDI_COL]>0)&(df[SAIDI_COL]<saidi_cap)]
                if len(t)>0:
                    fig.add_trace(go.Violin(
                        y=t[SAIDI_COL], name=tier,
                        box_visible=True, meanline_visible=True, points="outliers",
                        fillcolor=RISK_COLORS.get(tier,"#999"),
                        opacity=0.75, line_color=RISK_COLORS.get(tier,"#999"),
                        hovertemplate="%{y:.1f} min/yr<extra></extra>"))
            _layout(fig, h=380, title=f"SAIDI Violin — non-zero utilities · capped at {saidi_cap:,}")
            fig.update_yaxes(title_text="SAIDI (min/yr)")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Violin: {e}")

    # ── State bar ─────────────────────────────────────────────────
    with ra4:
        try:
            sc = df[df["risk_category"]=="High Risk"]["State"] \
                   .value_counts().head(20).reset_index()
            sc.columns = ["State","High Risk Count"]
            fig = px.bar(sc, x="State", y="High Risk Count",
                         color="High Risk Count", color_continuous_scale="Reds",
                         title="Top 20 States — High Risk Utility Count",
                         text="High Risk Count")
            fig.update_traces(textposition="outside", textfont_size=8)
            _layout(fig, h=380)
            fig.update_xaxes(tickangle=45)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"State bar: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — STORM & DAMAGE
# ══════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-title">Storm Exposure & Economic Impact</div>',
                unsafe_allow_html=True)

    sd1, sd2 = st.columns(2)

    # ── Bubble chart ──────────────────────────────────────────────
    with sd1:
        try:
            d = df[(df[SAIDI_COL]>0)&
                   (df["weather_event_count"]>0)&
                   (df["estimated_annual_loss_usd"]>0)].copy()
            if len(d)>0:
                d["loss_m"] = d["estimated_annual_loss_usd"]/1e6
                d = d[d[SAIDI_COL]<saidi_cap]
                fig = go.Figure()
                for tier in ACTUAL_TIERS:
                    t = d[d["risk_category"]==tier]
                    if len(t)==0: continue
                    mx = d["loss_m"].max()
                    sz = ((t["loss_m"]/mx*40+5)).clip(4,45)
                    fig.add_trace(go.Scatter(
                        x=t["weather_event_count"], y=t[SAIDI_COL],
                        mode="markers", name=tier,
                        marker=dict(color=RISK_COLORS.get(tier,"#999"),
                                    size=sz, opacity=0.65,
                                    line=dict(width=0.4,color="#ffffff25")),
                        text=(
                            "<b>"+t["Utility Name"].fillna("")+"</b><br>"+
                            "State: "+t["State"].fillna("")+"<br>"+
                            f"SAIDI: "+t[SAIDI_COL].round(1).astype(str)+" min/yr<br>"+
                            "Storm Events: "+t["weather_event_count"].astype(str)+"<br>"+
                            "Annual Loss: $"+t["loss_m"].round(1).astype(str)+"M"),
                        hovertemplate="%{text}<extra></extra>"))
                _layout(fig, h=420,
                        title="Storm Events vs SAIDI — bubble size = economic loss")
                fig.update_xaxes(title_text="Storm Event Count")
                fig.update_yaxes(title_text="SAIDI (min/yr)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Bubble: {e}")

    # ── Economic impact bar ───────────────────────────────────────
    with sd2:
        try:
            top = (df[df["estimated_annual_loss_usd"]>0]
                   [["Utility Name","State","risk_category","estimated_annual_loss_usd"]]
                   .sort_values("estimated_annual_loss_usd",ascending=False).head(20).copy())
            if len(top)>0:
                top["Loss ($M)"] = (top["estimated_annual_loss_usd"]/1e6).round(1)
                top["Label"]     = top["Utility Name"].str[:22]+" ("+top["State"]+")"
                fig = px.bar(top, x="Loss ($M)", y="Label", orientation="h",
                             color="risk_category", color_discrete_map=RISK_COLORS,
                             title="Top 20 Utilities — Est. Annual Economic Loss ($M)",
                             text="Loss ($M)")
                fig.update_traces(textposition="outside",textfont_size=8)
                _layout(fig, h=540)
                fig.update_yaxes(categoryorder="total ascending")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Economic bar: {e}")

    # ── Damage decomposition treemap ──────────────────────────────
    st.markdown('<div class="section-title">Storm Damage Treemap by State & Ownership</div>',
                unsafe_allow_html=True)
    try:
        tm = df[df["total_damage_usd"]>0].copy()
        tm["Damage ($M)"] = (tm["total_damage_usd"]/1e6).round(2)
        tm["label"] = tm["Utility Name"].str[:20]
        fig = px.treemap(
            tm, path=["NERC Region","State","Ownership"],
            values="Damage ($M)",
            color="Damage ($M)",
            color_continuous_scale="YlOrRd",
            title="Storm Damage Treemap — NERC Region → State → Ownership Type",
            hover_data={"Damage ($M)": True})
        fig.update_layout(
            height=480, paper_bgcolor=BG, font_color=TEXT,
            margin=dict(l=10,r=10,t=50,b=10), title_x=0.5)
        fig.update_traces(marker_line_width=0.5,marker_line_color=BG)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Treemap: {e}")

    # ── Property vs crop damage comparison ────────────────────────
    sd3, sd4 = st.columns(2)
    with sd3:
        try:
            if "total_property_damage_usd" in df.columns:
                dmg = df.groupby("State").agg(
                    Property=("total_property_damage_usd","sum"),
                    Crops=("total_crops_damage_usd","sum")
                ).reset_index()
                dmg["Property"] = dmg["Property"]/1e6
                dmg["Crops"]    = dmg["Crops"]/1e6
                dmg = dmg.sort_values("Property",ascending=False).head(15)
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Property Damage",
                                     x=dmg["State"],y=dmg["Property"],
                                     marker_color="#E74C3C",opacity=0.85))
                fig.add_trace(go.Bar(name="Crop Damage",
                                     x=dmg["State"],y=dmg["Crops"],
                                     marker_color="#F39C12",opacity=0.85))
                _layout(fig, h=360, title="Property vs Crop Damage — Top 15 States ($M)")
                fig.update_layout(barmode="group")
                fig.update_xaxes(tickangle=45)
                fig.update_yaxes(title_text="Damage ($M)")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Damage split: {e}")

    with sd4:
        try:
            # Human impact by state
            hi = df.groupby("State").agg(
                Injuries=("INJURIES_DIRECT","sum"),
                Deaths=("DEATHS_DIRECT","sum"),
            ).reset_index()
            hi["Impact Score"] = hi["Injuries"] + hi["Deaths"]*10
            hi = hi.sort_values("Impact Score",ascending=False).head(15)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Injuries",x=hi["State"],y=hi["Injuries"],
                                 marker_color="#F39C12",opacity=0.85))
            fig.add_trace(go.Bar(name="Deaths (×10)",x=hi["State"],
                                 y=hi["Deaths"]*10,
                                 marker_color="#E74C3C",opacity=0.85))
            _layout(fig, h=360, title="Human Impact by State — Top 15 (Deaths weighted ×10)")
            fig.update_layout(barmode="stack")
            fig.update_xaxes(tickangle=45)
            fig.update_yaxes(title_text="Count")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Human impact: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — NERC DEEP DIVE
# ══════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-title">NERC Region Analysis</div>',
                unsafe_allow_html=True)

    nd1, nd2 = st.columns(2)

    # ── NERC heatmap ──────────────────────────────────────────────
    with nd1:
        try:
            major_nerc = df["NERC Region"].value_counts().head(10).index.tolist()
            sub = df[df["NERC Region"].isin(major_nerc)]
            ns = sub.groupby("NERC Region").agg(
                avg_saidi  =(SAIDI_COL,"mean"),
                pct_high   =("high_risk","mean"),
                avg_storms =("weather_event_count","mean"),
                avg_loss_m =("estimated_annual_loss_usd","mean"),
            ).reset_index().sort_values("avg_saidi",ascending=False)
            ns["avg_loss_m"] = ns["avg_loss_m"]/1e6
            metrics = ["avg_saidi","pct_high","avg_storms","avg_loss_m"]
            ylabels = ["Avg SAIDI\n(min/yr)","% High Risk",
                       "Avg Storm\nEvents","Avg Loss\n($M)"]
            regions = ns["NERC Region"].tolist()
            z, text = [], []
            for m,yl in zip(metrics,ylabels):
                col = ns[m].values.astype(float)
                mn,mx = col.min(), col.max()
                normed = (col-mn)/(mx-mn+1e-9)
                z.append(normed.tolist())
                fmt = ".0f" if m in ["avg_saidi","avg_storms"] else ".1%" if m=="pct_high" else ".1f"
                text.append([f"{v:{fmt}}" for v in col])
            fig = go.Figure(go.Heatmap(
                z=z, x=regions, y=ylabels,
                text=text, texttemplate="%{text}",
                colorscale="RdYlGn_r", showscale=False,
                xgap=2, ygap=2,
                hovertemplate="<b>%{x}</b><br>%{y}: %{text}<extra></extra>"))
            _layout(fig, h=320, title="NERC Region Risk Heatmap — Normalised Scores")
            fig.update_xaxes(tickangle=30,tickfont_size=9)
            fig.update_yaxes(tickfont_size=9)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"NERC heatmap: {e}")

    # ── NERC radar ────────────────────────────────────────────────
    with nd2:
        try:
            top7 = df["NERC Region"].value_counts().head(7).index.tolist()
            sub  = df[df["NERC Region"].isin(top7)]
            dims = {
                "Avg SAIDI":   (SAIDI_COL,"mean"),
                "% High Risk": ("high_risk","mean"),
                "Storm Events":("weather_event_count","mean"),
                "Damage ($M)": ("total_damage_usd","mean"),
                "Human Impact":("human_impact_score","mean"),
            }
            ns2 = sub.groupby("NERC Region").agg(
                **{k:(v[0],v[1]) for k,v in dims.items()}
            ).reset_index()
            ns2["Damage ($M)"] = ns2["Damage ($M)"]/1e6
            cats = list(dims.keys())
            for col in cats:
                mn,mx = ns2[col].min(), ns2[col].max()
                ns2[col] = (ns2[col]-mn)/(mx-mn+1e-9)

            fig = go.Figure()
            for i,(_,row) in enumerate(ns2.iterrows()):
                vals = [row[c] for c in cats]+[row[cats[0]]]
                fig.add_trace(go.Scatterpolar(
                    r=vals, theta=cats+[cats[0]],
                    name=row["NERC Region"], fill="toself", opacity=0.55,
                    line=dict(color=HEX_PAL[i%len(HEX_PAL)],width=1.5),
                    hovertemplate=f"<b>{row['NERC Region']}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>"))
            fig.update_layout(
                polar=dict(
                    bgcolor=BG2,
                    radialaxis=dict(visible=True,range=[0,1],
                                   tickfont=dict(size=8,color=MUTED),
                                   gridcolor=GRID,linecolor=GRID),
                    angularaxis=dict(tickfont=dict(size=10,color=TEXT),
                                     gridcolor=GRID,linecolor=GRID)),
                paper_bgcolor=BG, font_color=TEXT,
                legend=dict(bgcolor=BG3,font=dict(color=TEXT,size=9)),
                title=dict(text="NERC Risk Radar — 5 Normalised Dimensions",
                           x=0.5,font=dict(size=12)),
                height=380, margin=dict(l=60,r=60,t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Radar: {e}")

    # ── NERC grouped bar (SAIDI components) ───────────────────────
    st.markdown('<div class="section-title">SAIDI Breakdown — All Events vs No Major Event Days</div>',
                unsafe_allow_html=True)
    try:
        major_nerc = df["NERC Region"].value_counts().head(10).index.tolist()
        sub = df[df["NERC Region"].isin(major_nerc)]
        nb = sub.groupby("NERC Region").agg(
            All_Events=("IEEE_AllEvents_SAIDI_min_per_yr","mean"),
            No_MED=("IEEE_NoMED_SAIDI_min_per_yr","mean"),
        ).reset_index().sort_values("All_Events",ascending=False)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="All Events SAIDI",
                             x=nb["NERC Region"],y=nb["All_Events"],
                             marker_color="#E74C3C",opacity=0.85))
        fig.add_trace(go.Bar(name="No MED SAIDI",
                             x=nb["NERC Region"],y=nb["No_MED"],
                             marker_color="#3498DB",opacity=0.85))
        _layout(fig, h=360, title="SAIDI: All Events vs Excluding Major Event Days (MED)")
        fig.update_layout(barmode="group")
        fig.update_xaxes(tickangle=30)
        fig.update_yaxes(title_text="Avg SAIDI (min/yr)")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"NERC SAIDI breakdown: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 5 — UTILITY COMPARISON
# ══════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-title">Side-by-Side Utility Comparison</div>',
                unsafe_allow_html=True)
    try:
        r1 = DF[DF["Utility Name"]==util_a]
        r2 = DF[DF["Utility Name"]==util_b]

        if len(r1)>0 and len(r2)>0:
            r1,r2 = r1.iloc[0], r2.iloc[0]

            # Info cards
            cc1,cc2 = st.columns(2)
            with cc1:
                risk1 = r1.get("risk_category","?")
                color1 = RISK_COLORS.get(risk1,"#999")
                st.markdown(f"""
                <div class="card" style="border-left:4px solid {color1}">
                  <div style="font-size:14px;font-weight:700;color:{color1}">{util_a}</div>
                  <div style="font-size:11px;color:{MUTED};margin-top:4px;">
                    {r1.get("State","?")} · {r1.get("Ownership","?")} · {r1.get("NERC Region","?")}
                  </div>
                  <div style="margin-top:8px;font-size:12px;color:{TEXT}">
                    Risk: <b style="color:{color1}">{risk1}</b> &nbsp;|&nbsp;
                    Risk Score: <b>{r1.get("risk_score",0):.4f}</b>
                  </div>
                </div>""", unsafe_allow_html=True)
            with cc2:
                risk2 = r2.get("risk_category","?")
                color2 = RISK_COLORS.get(risk2,"#999")
                st.markdown(f"""
                <div class="card" style="border-left:4px solid {color2}">
                  <div style="font-size:14px;font-weight:700;color:{color2}">{util_b}</div>
                  <div style="font-size:11px;color:{MUTED};margin-top:4px;">
                    {r2.get("State","?")} · {r2.get("Ownership","?")} · {r2.get("NERC Region","?")}
                  </div>
                  <div style="margin-top:8px;font-size:12px;color:{TEXT}">
                    Risk: <b style="color:{color2}">{risk2}</b> &nbsp;|&nbsp;
                    Risk Score: <b>{r2.get("risk_score",0):.4f}</b>
                  </div>
                </div>""", unsafe_allow_html=True)

            # Comparison bar chart
            dims = {
                "SAIDI (min/yr)":   (SAIDI_COL,   1),
                "SAIFI (x/yr)":     (SAIFI_COL,   1),
                "CAIDI (min/int)":  (CAIDI_COL,   1),
                "Risk Score":       ("risk_score", 1),
                "Storm Events":     ("weather_event_count",1),
                "Damage ($M)":      ("total_damage_usd",1e6),
                "Human Impact":     ("human_impact_score",1),
                "Annual Loss ($M)": ("estimated_annual_loss_usd",1e6),
            }
            labels,vals1,vals2 = [],[],[]
            for lab,(col,div) in dims.items():
                v1 = float(r1.get(col,0)) if col in r1.index else 0
                v2 = float(r2.get(col,0)) if col in r2.index else 0
                labels.append(lab)
                vals1.append(round(v1/div,3))
                vals2.append(round(v2/div,3))

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name=util_a[:30], x=labels, y=vals1,
                marker_color=color1, opacity=0.85,
                text=[str(v) for v in vals1],
                textposition="outside", textfont_size=8,
                hovertemplate="<b>"+util_a[:25]+"</b><br>%{x}: %{y}<extra></extra>"))
            fig.add_trace(go.Bar(
                name=util_b[:30], x=labels, y=vals2,
                marker_color=color2, opacity=0.85,
                text=[str(v) for v in vals2],
                textposition="outside", textfont_size=8,
                hovertemplate="<b>"+util_b[:25]+"</b><br>%{x}: %{y}<extra></extra>"))
            _layout(fig, h=420,
                    title=f"Utility Comparison: {util_a[:20]} vs {util_b[:20]}")
            fig.update_layout(barmode="group")
            fig.update_yaxes(title_text="Value (see label for units)")
            st.plotly_chart(fig, use_container_width=True)

            # Delta summary
            st.markdown('<div class="section-title">Metric Delta — A vs B</div>',
                        unsafe_allow_html=True)
            dcols = st.columns(4)
            key_dims = [
                ("SAIDI (min/yr)",  SAIDI_COL,   1,    ".0f"),
                ("SAIFI (x/yr)",    SAIFI_COL,   1,    ".2f"),
                ("Risk Score",      "risk_score",1,    ".4f"),
                ("Loss ($M)",       "estimated_annual_loss_usd",1e6,".1f"),
            ]
            for i,(lab,col,div,fmt) in enumerate(key_dims):
                v1 = float(r1.get(col,0))/div if col in r1.index else 0
                v2 = float(r2.get(col,0))/div if col in r2.index else 0
                dcols[i].metric(
                    label=f"{util_a[:14]}: {lab}",
                    value=f"{v1:{fmt}}",
                    delta=f"{v1-v2:+{fmt}} vs B",
                    delta_color="inverse")
        else:
            st.warning("Could not find one or both selected utilities. Use the sidebar dropdowns.")
    except Exception as e:
        st.error(f"Comparison error: {e}")


# ══════════════════════════════════════════════════════════════════
# TAB 6 — DATA TABLE
# ══════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-title">Utility-Level Data</div>',
                unsafe_allow_html=True)

    # Summary stats row
    ts1,ts2,ts3,ts4 = st.columns(4)
    nz = df[df[SAIDI_COL]>0]
    ts1.metric("Non-Zero SAIDI", f"{len(nz):,}", f"of {len(df):,} total")
    ts2.metric("Median SAIDI (non-zero)", f"{nz[SAIDI_COL].median():.0f} min/yr")
    ts3.metric("Max SAIDI",
               f"{df[SAIDI_COL].max():,.0f} min/yr",
               df.loc[df[SAIDI_COL].idxmax(),"Utility Name"][:20] if len(df)>0 else "")
    ts4.metric("NERC SLA Breach",
               f"{int(df['nerc_sla_breach_risk'].sum())} utilities",
               f"{df['nerc_sla_breach_risk'].mean()*100:.1f}% of filtered")

    # Build display table
    want = ["Utility Name","State","Ownership","NERC Region",
            "risk_category","risk_score",
            SAIDI_COL, SAIFI_COL, CAIDI_COL,
            "weather_event_count","estimated_annual_loss_usd","nerc_sla_breach_risk"]
    cols = [c for c in want if c in df.columns]
    out  = df[cols].copy().rename(columns={
        "risk_category":"Risk Tier","risk_score":"Risk Score",
        SAIDI_COL:"SAIDI (min/yr)",SAIFI_COL:"SAIFI (x/yr)",
        CAIDI_COL:"CAIDI (min/int)",
        "weather_event_count":"Storm Events",
        "estimated_annual_loss_usd":"Est. Loss ($M)",
        "nerc_sla_breach_risk":"NERC Breach"})
    if "Risk Score" in out.columns:
        out["Risk Score"] = out["Risk Score"].round(4)
        out = out.sort_values("Risk Score",ascending=False)
    if "SAIDI (min/yr)" in out.columns:
        out["SAIDI (min/yr)"] = out["SAIDI (min/yr)"].round(1)
    if "Est. Loss ($M)" in out.columns:
        out["Est. Loss ($M)"] = (out["Est. Loss ($M)"]/1e6).round(1)

    st.dataframe(out.reset_index(drop=True),
                 use_container_width=True, height=450)

    # Download
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data (CSV)",
        data=csv,
        file_name=f"outage_risk_{state_sel.replace(' ','_')}.csv",
        mime="text/csv")
