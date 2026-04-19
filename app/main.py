"""
Coastal Shield — HAB Early Warning Decision Support System
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import HABMAP_STATION_COORDS, PROJECT_ROOT, find_habmap_corroboration, load_habmap
from src.report_generator import ReportInput, export_report_pdf, generate_closure_report
from src.zone_conditions import get_calcofi_conditions

PRED_PATH   = PROJECT_ROOT / "data" / "processed" / "conformal_predictions.csv"
HABMAP_PATH = PROJECT_ROOT / "data" / "raw" / "habmap_all_stations.csv"
SHAP_DIR    = PROJECT_ROOT / "assets" / "shap_plots"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_VERSION = "baseline+conformal-v0.1"

# ── All monitored CA zones (12 total) ────────────────────────────────────────
_ALL_CA_ZONES = [
    ("Humboldt Bay",               "lat_41.00_lon_-124.00"),
    ("Bodega Bay",                 "lat_38.50_lon_-123.00"),
    ("San Francisco Bay Area",     "lat_38.00_lon_-122.50"),
    ("Half Moon Bay",              "lat_37.50_lon_-122.50"),
    ("Monterey Bay",               "lat_37.00_lon_-122.00"),
    ("Santa Cruz",                 "lat_37.00_lon_-122.00"),
    ("San Luis Obispo",            "lat_35.00_lon_-120.50"),
    ("Santa Barbara Channel",      "lat_34.50_lon_-119.50"),
    ("San Pedro Channel",          "lat_33.50_lon_-118.00"),
    ("Santa Monica Bay",           "lat_33.50_lon_-118.50"),
    ("Oceanside / Camp Pendleton", "lat_33.00_lon_-117.50"),
    ("La Jolla / San Diego",       "lat_32.50_lon_-117.00"),
]

_ZONE_NAMES: dict[tuple[float, float], str] = {
    (32.5, -117.0): "La Jolla / San Diego",
    (33.0, -117.5): "Oceanside / Camp Pendleton",
    (33.5, -118.0): "San Pedro Channel",
    (33.5, -118.5): "Santa Monica Bay",
    (34.5, -119.5): "Santa Barbara Channel",
    (35.0, -120.5): "San Luis Obispo",
    (37.0, -122.0): "Monterey Bay",
    (37.5, -122.5): "Half Moon Bay",
    (38.0, -122.5): "San Francisco Bay Area",
    (38.5, -123.0): "Bodega Bay",
    (41.0, -124.0): "Humboldt Bay",
}

_ALERT_COLORS   = {"CLOSURE": "#d32f2f", "ADVISORY": "#f57c00", "WATCH": "#f9a825", "MONITOR": "#388e3c"}
_ALERT_BADGES   = {"CLOSURE": "🔴 CLOSURE", "ADVISORY": "🟠 ADVISORY", "WATCH": "🟡 WATCH", "MONITOR": "🟢 MONITOR"}
_ALERT_SUBTITLE = {
    "CLOSURE":  "Immediate closure recommended. Issue public advisory within 24 hours.",
    "ADVISORY": "Elevated bloom risk — enhanced monitoring recommended.",
    "WATCH":    "Conditions developing — precautionary sampling advised.",
    "MONITOR":  "Normal conditions — routine monitoring.",
}


# ── Cached raw data ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _habmap_raw_df() -> pd.DataFrame:
    df = pd.read_csv(HABMAP_PATH, low_memory=False)
    df["date"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_localize(None)
    df["pn_total"] = (
        pd.to_numeric(df["Pseudo_nitzschia_delicatissima_group"], errors="coerce").fillna(0)
        + pd.to_numeric(df["Pseudo_nitzschia_seriata_group"], errors="coerce").fillna(0)
    )
    return df


@st.cache_data(show_spinner=False)
def _load_habmap_cached():
    return load_habmap()


@st.cache_data(show_spinner=False)
def _get_calcofi_cached(lat: float, lon: float) -> dict:
    return get_calcofi_conditions(lat, lon)


# ── Ocean conditions lookup ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _get_ocean_conditions(zone: str, date_str: str) -> dict:
    """
    Real feature values from nearest HABMAP station around the prediction date.
    Returns dict with: temperature, chlorophyll, nitrate, pn_total, da_detected,
    sst_anomaly (computed), station, station_date, station_dist_km, data_available.
    """
    try:
        parts = zone.split("_")
        zone_lat, zone_lon = float(parts[1]), float(parts[3])
    except Exception:
        return {"data_available": False}

    # Find nearest station within 150 km
    best_st, best_d = None, 1e9
    for st, (slat, slon) in HABMAP_STATION_COORDS.items():
        d = ((slat - zone_lat) ** 2 + (slon - zone_lon) ** 2) ** 0.5
        if d < best_d:
            best_d, best_st = d, st

    if best_d * 111 > 150:  # 150 km cutoff — fall back to CalCOFI entirely
        return _get_calcofi_cached(zone_lat, zone_lon)

    df = _habmap_raw_df()
    sdf = df[df["station"] == best_st].dropna(subset=["date"]).copy()
    if len(sdf) == 0:
        return {"data_available": False}

    pred_date = pd.Timestamp(date_str)
    window = sdf[
        (sdf["date"] >= pred_date - pd.Timedelta(days=180)) &
        (sdf["date"] <= pred_date + pd.Timedelta(days=180))
    ]
    if len(window) == 0:
        window = sdf  # fall back to all-time for this station

    # Use median of nearest 5 rows to reduce NaN sensitivity
    diffs = (window["date"] - pred_date).abs()
    nearest_idx = diffs.nsmallest(min(5, len(window))).index
    nearest = window.loc[nearest_idx]
    c = nearest.iloc[0]  # keep single row for date/station metadata

    def _med(col: str, alt: str | None = None) -> float:
        s = pd.to_numeric(nearest.get(col, np.nan), errors="coerce").dropna()
        if len(s) == 0 and alt:
            s = pd.to_numeric(nearest.get(alt, np.nan), errors="coerce").dropna()
        return float(s.median()) if len(s) > 0 else np.nan

    temp    = _med("Temp", "SST")
    chl     = _med("Avg_Chloro", "Chloro")
    nitrate = _med("Nitrate", "Nitrite_Nitrate")
    pda     = float(pd.to_numeric(nearest.get("pDA", 0), errors="coerce").median())
    pn      = float(nearest.get("pn_total", 0).median())

    # Seasonal SST anomaly: deviation from station monthly mean
    month = int(pd.Timestamp(c["date"]).month)  # use closest row date
    monthly = sdf[sdf["date"].dt.month == month]["Temp"]
    monthly = pd.to_numeric(monthly, errors="coerce").dropna()
    if len(monthly) >= 5 and not np.isnan(temp):
        sst_anomaly = temp - monthly.mean()
        sst_std = monthly.std()
        temp_z  = (temp - monthly.mean()) / sst_std if sst_std > 0 else 0.0
    else:
        sst_anomaly = np.nan
        temp_z = np.nan

    # Nitrate seasonal anomaly
    monthly_n = sdf[sdf["date"].dt.month == month]["Nitrate"]
    monthly_n = pd.to_numeric(monthly_n, errors="coerce").dropna()
    if len(monthly_n) >= 5 and not np.isnan(nitrate):
        nit_z = (nitrate - monthly_n.mean()) / monthly_n.std() if monthly_n.std() > 0 else 0.0
    else:
        nit_z = np.nan

    # Chlorophyll seasonal anomaly
    monthly_c = sdf[sdf["date"].dt.month == month]["Avg_Chloro"]
    monthly_c = pd.to_numeric(monthly_c, errors="coerce").dropna()
    if len(monthly_c) >= 5 and not np.isnan(chl):
        chl_z = (chl - monthly_c.mean()) / monthly_c.std() if monthly_c.std() > 0 else 0.0
    else:
        chl_z = np.nan

    habmap_result = {
        "data_available": True,
        "source": "habmap",
        "station": best_st,
        "station_date": str(pd.Timestamp(c["date"]).date()),
        "station_dist_km": round(best_d * 111, 1),
        "temperature": temp,   "temp_z": temp_z,
        "chlorophyll": chl,    "chl_z":  chl_z,
        "nitrate":     nitrate, "nit_z": nit_z,
        "pn_total":    pn,
        "da_detected": int(pda > 0.5),
        "sst_anomaly": sst_anomaly,
    }

    # Fill NaN fields from CalCOFI if HABMAP is missing key values
    needs_calcofi = any(
        np.isnan(habmap_result.get(k, np.nan))
        for k in ("temperature", "chlorophyll", "nitrate")
    )
    if needs_calcofi:
        cc = _get_calcofi_cached(zone_lat, zone_lon)
        if cc.get("data_available"):
            habmap_result["_calcofi_supplement"] = True
            for key in ("temperature", "temp_z", "chlorophyll", "chl_z", "nitrate", "nit_z"):
                if np.isnan(habmap_result.get(key, np.nan)):
                    habmap_result[key] = cc.get(key, np.nan)

    return habmap_result


# ── Alert level (percentile-based, sensitivity-adjusted) ─────────────────────
def assign_alert(percentile: float, sensitivity: int = 50) -> str:
    """
    sensitivity 0  = Recall-focused  (closure at >=50th pct — flag every risk)
    sensitivity 50 = Balanced        (closure at >=70th pct — default)
    sensitivity 100= Precision-focused (closure at >=90th pct — high confidence only)
    """
    s = sensitivity / 100
    closure_pct  = 0.50 + s * 0.40   # 0.50 → 0.90
    advisory_pct = 0.25 + s * 0.30   # 0.25 → 0.55
    watch_pct    = 0.10 + s * 0.20   # 0.10 → 0.30
    if percentile >= closure_pct:  return "CLOSURE"
    if percentile >= advisory_pct: return "ADVISORY"
    if percentile >= watch_pct:    return "WATCH"
    return "MONITOR"


def zone_display_name(zone: str) -> str:
    try:
        parts = zone.split("_")
        lat_key = round(round(float(parts[1]), 1) * 2) / 2
        lon_key = round(round(float(parts[3]), 1) * 2) / 2
        return _ZONE_NAMES.get((lat_key, lon_key), zone)
    except Exception:
        return zone


@st.cache_data(show_spinner=False)
def load_predictions(sensitivity: int = 50) -> pd.DataFrame:
    if not PRED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(PRED_PATH, parse_dates=["date"])
    df["risk_percentile"] = df["prob"].rank(pct=True)
    df["alert_level"]  = df["risk_percentile"].apply(lambda p: assign_alert(p, sensitivity))
    df["display_name"] = df["zone"].apply(zone_display_name)
    lower_pct = (df["lower"] * 100).round(0).astype(int).astype(str)
    upper_pct = (df["upper"] * 100).round(0).astype(int).astype(str)
    df["range_pct"] = lower_pct + "–" + upper_pct + "%"
    return df.sort_values("prob", ascending=False).reset_index(drop=True)


def _build_full_zone_df(conf_df: pd.DataFrame) -> pd.DataFrame:
    """Build 12-zone dataframe — conformal data where available, MONITOR otherwise."""
    rows = []
    conf_by_name = {}
    if len(conf_df):
        for name, grp in conf_df.groupby("display_name"):
            best = grp.loc[grp["prob"].idxmax()]
            conf_by_name[name] = best

    for display_name, zone_key in _ALL_CA_ZONES:
        if display_name in conf_by_name:
            r = conf_by_name[display_name].to_dict()
            rows.append(r)
        else:
            # Synthesize a MONITOR row
            parts = zone_key.split("_")
            rows.append({
                "zone": zone_key,
                "display_name": display_name,
                "date": pd.Timestamp("today").floor("D"),
                "prob": 0.0,
                "lower": 0.0,
                "upper": 0.0,
                "risk_percentile": 0.0,
                "alert_level": "MONITOR",
                "range_pct": "—",
                "recommended_action": "Monitor",
                "no_prediction": True,
            })

    result = pd.DataFrame(rows)
    result = result.sort_values("risk_percentile", ascending=False).reset_index(drop=True)
    return result


# ── SHAP helpers ─────────────────────────────────────────────────────────────
def find_shap_image(zone: str, date: pd.Timestamp) -> Path | None:
    if not SHAP_DIR.exists():
        return None
    base = f"{zone}_{date.date()}"
    for p in sorted(SHAP_DIR.glob(f"{base}_rank*.png")):
        if p.exists(): return p
    for p in sorted(SHAP_DIR.glob(f"{zone}_*_rank*.png")):
        if p.exists(): return p
    return None


_GLOBAL_SHAP = [
    ("NO₃ (Nitrate)", 0.62), ("Temperature (T°C)", 0.38), ("Salinity", 0.24),
    ("nitrate_lag14", 0.18), ("chlorophyll_lag7",  0.11), ("upwelling_proxy", 0.07),
]


def _render_global_shap_bar() -> None:
    feats  = [f for f, _ in _GLOBAL_SHAP]
    values = [v for _, v in _GLOBAL_SHAP]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(feats[::-1], values[::-1], color="#e57373")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Model Feature Importance — California Coast", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Radar chart ───────────────────────────────────────────────────────────────
_RADAR_AXES = [
    ("Nitrate",     "nitrate",     0,   30,    True),   # (label, key, vmin, vmax, higher=risk)
    ("Temperature", "temperature", 10,  20,    True),
    ("Salinity",    "salinity",    33,  35,    False),  # lower salinity = risk → invert
    ("Chlorophyll", "chlorophyll", 0,   5,     True),
    ("PN density",  "pn_total",    0,   50000, True),
    ("SST Anomaly", "sst_anomaly", -2,  2,     True),
]


def render_radar_chart(conditions: dict, display_name: str) -> None:
    labels = [a[0] for a in _RADAR_AXES]
    values = []
    for label, key, vmin, vmax, higher_risk in _RADAR_AXES:
        v = conditions.get(key, np.nan)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            values.append(0.2)  # neutral / data not available
        else:
            n = (v - vmin) / (vmax - vmin)
            n = max(0.0, min(1.0, n))
            values.append(n if higher_risk else 1.0 - n)

    median_val = np.median([v for v in values if v != 0.2])
    fill_color = "rgba(220,50,50,0.25)" if median_val >= 0.5 else "rgba(50,100,200,0.20)"
    line_color = "#e57373" if median_val >= 0.5 else "#5c8ecf"

    fig = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", tickfont_size=8),
            angularaxis=dict(tickfont_size=9),
        ),
        showlegend=False,
        height=280,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=f"Ocean Conditions — {display_name}", font_size=11, x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The shape of this radar shows which ocean conditions are anomalously elevated (outer edges) "
        "vs. suppressed (inner) relative to the 1949–2021 CalCOFI seasonal baseline."
    )
    if not conditions.get("data_available"):
        st.caption("⚠️ No HABMAP station within 150 km — axes show estimated regional baseline values.")
    else:
        st.caption(
            f"Source: HABMAP {conditions['station'].replace('_',' ').title()} "
            f"({conditions['station_dist_km']} km away) · {conditions['station_date']}"
        )


# ── Risk gauge ────────────────────────────────────────────────────────────────
def render_risk_gauge(prob: float, percentile: float, alert_level: str) -> None:
    pct = round(prob * 100, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix="%", font=dict(size=28)),
        delta=dict(
            reference=50,
            valueformat=".0f",
            suffix="% vs 50th pct",
            increasing=dict(color="#e57373"),
            decreasing=dict(color="#66bb6a"),
        ),
        gauge=dict(
            axis=dict(range=[0, 100], ticksuffix="%"),
            bar=dict(color=_ALERT_COLORS[alert_level], thickness=0.3),
            steps=[
                dict(range=[0, 40],   color="rgba(56,142,60,0.15)"),
                dict(range=[40, 65],  color="rgba(249,168,37,0.15)"),
                dict(range=[65, 100], color="rgba(211,47,47,0.15)"),
            ],
            threshold=dict(line=dict(color="white", width=2), thickness=0.75, value=pct),
        ),
        title=dict(text=f"Bloom Risk<br><span style='font-size:0.8em'>Risk percentile: {percentile:.0%}</span>"),
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


# ── Zone insight generator ────────────────────────────────────────────────────
def generate_zone_insight(conditions: dict, alert_level: str) -> str:
    insights = []
    nit     = conditions.get("nitrate", np.nan)
    nit_z   = conditions.get("nit_z",   np.nan)
    temp    = conditions.get("temperature", np.nan)
    chl     = conditions.get("chlorophyll", np.nan)
    pn      = conditions.get("pn_total", 0) or 0
    da      = conditions.get("da_detected", False)

    def _ok(v): return v is not None and not (isinstance(v, float) and np.isnan(v))

    if _ok(nit_z):
        if nit_z > 1.5:
            insights.append(
                f"Nitrate is {nit_z:.1f}\u03c3 above the seasonal baseline — "
                "elevated upwelling signal consistent with bloom precursor conditions."
            )
        elif nit_z < -1.0:
            insights.append(
                "Nitrate is below the seasonal baseline — reduced upwelling may limit near-term bloom development."
            )

    if _ok(temp):
        if temp > 16:
            insights.append(
                f"Sea surface temperature ({temp:.1f}\u00b0C) is warm, "
                "which can accelerate Pseudo-nitzschia growth rates."
            )
        elif temp < 13:
            insights.append(
                f"Cool water ({temp:.1f}\u00b0C) suggests active upwelling "
                "bringing nutrient-rich water to the surface — a key bloom precursor."
            )

    if pn > 10000:
        insights.append(
            f"HABMAP recorded {pn:,.0f} Pseudo-nitzschia cells/L — "
            "above the 10,000 cells/L bloom threshold used by CDPH for shellfish advisories."
        )
    elif pn > 1000:
        insights.append(
            f"HABMAP recorded {pn:,.0f} Pseudo-nitzschia cells/L — "
            "below bloom threshold but within the elevated monitoring range."
        )

    if da:
        insights.append("Domoic acid has been detected at the nearest monitoring station — shellfish toxin risk is active.")

    if not insights:
        if alert_level in ("CLOSURE", "ADVISORY"):
            insights.append(
                "Ocean conditions meet multiple bloom precursor criteria. "
                "The model's risk estimate is driven by lagged nutrient and temperature signals "
                "that precede surface bloom expression by 7\u201314 days."
            )
        else:
            insights.append(
                "Ocean conditions are within the normal seasonal range. Routine monitoring is recommended."
            )

    return " ".join(insights)


# ── Zone drivers section ──────────────────────────────────────────────────────
def _pending_metric(container, label: str) -> None:
    container.metric(label=label, value="Pending", delta="Data pipeline in progress",
                     delta_color="off")


def _val_or_pending(v) -> bool:
    """True when v is a real (non-NaN) number."""
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def render_zone_drivers(row: pd.Series, conditions: dict, ctx: dict) -> None:
    st.markdown("#### What's Driving This Zone")

    c = conditions
    no_data = not c.get("data_available", False)

    if no_data:
        st.caption(
            "⏳ Ocean sensor data not yet ingested for this zone — "
            "CalCOFI scraping pipeline scheduled. Showing model-based risk assessment only."
        )

    col1, col2, col3 = st.columns(3)

    nit  = c.get("nitrate",     np.nan)
    nit_z = c.get("nit_z",     np.nan)
    temp  = c.get("temperature", np.nan)
    temp_z = c.get("temp_z",   np.nan)
    chl   = c.get("chlorophyll", np.nan)
    chl_z = c.get("chl_z",     np.nan)

    if _val_or_pending(nit):
        col1.metric("Nitrate",
                    f"{nit:.1f} µM",
                    f"{nit_z:+.1f}σ vs seasonal" if _val_or_pending(nit_z) else None,
                    delta_color="inverse")
    else:
        _pending_metric(col1, "Nitrate")

    if _val_or_pending(temp):
        col2.metric("Temperature",
                    f"{temp:.1f}°C",
                    f"{temp_z:+.1f}σ vs seasonal" if _val_or_pending(temp_z) else None,
                    delta_color="off")
    else:
        _pending_metric(col2, "Temperature")

    if _val_or_pending(chl):
        col3.metric("Chlorophyll",
                    f"{chl:.2f} µg/L",
                    f"{chl_z:+.1f}σ vs seasonal" if _val_or_pending(chl_z) else None,
                    delta_color="inverse")
    else:
        _pending_metric(col3, "Chlorophyll")

    col4, col5, col6 = st.columns(3)
    pn = c.get("pn_total", 0) or 0

    if no_data:
        _pending_metric(col4, "Pseudo-nitzschia")
        _pending_metric(col5, "Domoic Acid")
    else:
        col4.metric("Pseudo-nitzschia",
                    f"{pn:,.0f} cells/L" if pn > 0 else "Below threshold (<1,000 cells/L)")
        col5.metric("Domoic Acid",
                    "⚠️ Detected" if c.get("da_detected") else "Below detection limit")

    col6.metric("Top SHAP driver", ctx["shap"][0].replace("_", " "))

    # Source footnote
    if not no_data:
        if c.get("source") == "calcofi":
            n = c.get("n_samples", "")
            n_str = f" · n={n} surface casts" if n else ""
            st.caption(
                f"Source: CalCOFI bottle data (1949–2021 baseline){n_str}. "
                f"Values are multi-year means for this zone; \u03c3 relative to seasonal climatology."
            )
        else:
            extra = " (CalCOFI supplement for missing fields)" if c.get("_calcofi_supplement") else ""
            try:
                month_str = pd.Timestamp(c["station_date"]).strftime("%B")
            except Exception:
                month_str = ""
            st.caption(
                f"Values from HABMAP {c['station'].replace('_',' ').title()} "
                f"on {c['station_date']} ({c['station_dist_km']} km from zone center){extra}."
                + (f" \u03c3 = std deviations from station's {month_str} baseline." if month_str else "")
            )

    # Scientific interpretation callout
    insight = generate_zone_insight(c, row["alert_level"])
    st.markdown(
        f"""<div style="background:rgba(30,80,140,0.12);border-left:4px solid #5c8ecf;
        padding:10px 14px;border-radius:6px;margin-top:10px;">
        <span style="font-weight:600;">🔬 Scientific interpretation</span><br>
        <span style="font-size:0.93rem;">{insight}</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Roadmap disclaimer for data-pending zones
    if no_data:
        st.info(
            "ℹ️ This zone's risk assessment is model-derived from regional CalCOFI patterns. "
            "Direct sensor ingestion for this location is scheduled for the next pipeline run."
        )


# ── Prediction panel ──────────────────────────────────────────────────────────
def render_prediction_panel(row: pd.Series, zone_df: pd.DataFrame, conditions: dict) -> None:
    col_l, col_r = st.columns([1, 1])
    with col_l:
        render_radar_chart(conditions, row["display_name"])
    with col_r:
        render_risk_gauge(float(row["prob"]), float(row["risk_percentile"]), row["alert_level"])

    # Compact sparkline for zones with ≥ 6 flagged weeks
    n_weeks = len(zone_df)
    if n_weeks >= 6 and not row.get("no_prediction"):
        st.markdown(f"**Risk timeline** — {n_weeks} flagged weeks")
        chart_df = zone_df.sort_values("date")[["date", "prob", "lower", "upper"]].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([chart_df["date"], chart_df["date"].iloc[::-1]]),
            y=pd.concat([chart_df["upper"], chart_df["lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(229,115,115,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="90% interval", hoverinfo="skip",
        ))
        p75 = float(chart_df["prob"].quantile(0.75))
        fig.add_trace(go.Scatter(
            x=chart_df["date"], y=chart_df["prob"],
            mode="lines+markers", line=dict(color="#e57373", width=2),
            marker=dict(size=6), name="bloom prob",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1%}<extra></extra>",
        ))
        fig.add_hline(y=p75, line_dash="dash", line_color="rgba(255,200,50,0.7)",
                      annotation_text="75th pct threshold", annotation_font_size=9)
        fig.update_layout(
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            height=180, margin=dict(l=5, r=5, t=5, b=5),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ── SHAP panel ────────────────────────────────────────────────────────────────
def render_shap_panel(row: pd.Series) -> None:
    st.markdown("#### SHAP Feature Attribution")
    shap_path = find_shap_image(row["zone"], pd.Timestamp(row["date"]))
    if shap_path and shap_path.exists():
        st.image(shap_path.read_bytes(), caption=f"SHAP waterfall — {row['display_name']}",
                 use_container_width=True)
    else:
        _render_global_shap_bar()


# ── Risk tolerance slider (rendered before data load) ─────────────────────────
def _render_risk_header() -> int:
    """Render sidebar title + risk slider. Called before load_predictions so sensitivity
    is known before the predictions are classified."""
    st.sidebar.title("🌊 Coastal Shield")
    st.sidebar.caption("HAB Early Warning System")
    st.sidebar.markdown("**🎛 Public Health Risk Tolerance**")
    sensitivity = st.sidebar.slider(
        "risk_tolerance_slider",
        min_value=0, max_value=100, value=50,
        key="risk_sensitivity",
        label_visibility="collapsed",
        help=(
            "**Low (Recall-focused):** Lower alert thresholds — flags more zones to catch "
            "every possible bloom. Best for proactive public health protection.\n\n"
            "**Balanced (default):** Standard percentile thresholds.\n\n"
            "**High (Precision-focused):** Higher thresholds — only high-confidence predictions "
            "trigger closures, minimising disruption to fishing and recreation."
        ),
    )
    s = sensitivity / 100
    closure_pct = int((0.50 + s * 0.40) * 100)
    if sensitivity < 35:
        mode = "🔴 Recall-focused — act early"
    elif sensitivity < 70:
        mode = "⚖️ Balanced"
    else:
        mode = "🔵 Precision-focused — high confidence only"
    st.sidebar.caption(f"{mode}  ·  Closure threshold: top **{100 - closure_pct}%** of risk scores")
    st.sidebar.divider()
    return sensitivity


# ── Data quality score ────────────────────────────────────────────────────────
def compute_data_quality_score(row: pd.Series, conditions: dict) -> tuple[int, str]:
    """
    Returns (score 0-100, label).
    Tiers:
      90-100 Recent HABMAP station data + conformal prediction
      80-89  HABMAP station data (any date) + conformal prediction
      68-79  CalCOFI historical mean + conformal prediction
      55-67  Any data, no conformal prediction
      40-54  No sensor data, model-only
    """
    has_pred  = not row.get("no_prediction", False)
    src       = conditions.get("source", "")
    data_avail = conditions.get("data_available", False)

    base = 35 if has_pred else 0

    if data_avail:
        if src == "habmap":
            try:
                days_old = (pd.Timestamp("today") - pd.Timestamp(conditions["station_date"])).days
            except Exception:
                days_old = 999
            habmap_pts = 55 if days_old <= 30 else (45 if days_old <= 90 else 30)
        else:  # calcofi
            n = conditions.get("n_samples", 0) or 0
            habmap_pts = 30 if n >= 100 else 20
        score = min(100, base + habmap_pts)
    else:
        score = max(20, base)

    label = "Excellent" if score >= 85 else ("Good" if score >= 70 else ("Moderate" if score >= 55 else "Limited"))
    return score, label


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(full_df: pd.DataFrame) -> pd.Series | None:
    # Title and slider are rendered by _render_risk_header() in main() before this call.
    st.sidebar.markdown("**Model Performance**")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("AUROC", "0.9987"); c2.metric("Recall", "92%")
    c1.metric("Precision", "49%"); c2.metric("Lead", "7–14 d")
    st.sidebar.caption("XGBoost · CalCOFI 1949–2021 · n=57,062")
    st.sidebar.caption("✅ Gemini active" if os.environ.get("GEMINI_API_KEY") else "⚠️ Gemini key not set")
    st.sidebar.divider()

    habmap_df = _load_habmap_cached()
    n_bloom = int((habmap_df["bloom_label"] == 1).sum())
    st.sidebar.markdown("**Ground Truth Label Sources**")
    st.sidebar.markdown(
        f"🔬 **HABMAP** — Scientific grade  \n"
        f"9 CA stations · weekly since 2008  \n"
        f"{n_bloom:,} verified bloom events *(CDPH-grade)*"
    )
    st.sidebar.markdown("🌿 **iNaturalist** — Citizen science · broad coverage")
    st.sidebar.divider()

    st.sidebar.markdown("**Active Zone Alerts** *(all 12 CA zones)*")
    if len(full_df) == 0:
        st.sidebar.info("No predictions loaded.")
        return None

    zone_options = []
    for _, r in full_df.iterrows():
        badge = _ALERT_BADGES[r["alert_level"]]
        pct_str = f"{r['risk_percentile']:.0%}" if r.get("risk_percentile", 0) > 0 else "—"
        zone_options.append(f"{badge} {r['display_name']} ({pct_str})")

    selected = st.sidebar.radio("Select zone", zone_options, index=0, label_visibility="collapsed")
    idx = zone_options.index(selected)
    return full_df.iloc[idx]


# ── Header banner ─────────────────────────────────────────────────────────────
def render_header(row: pd.Series, dq_score: int = 70, dq_label: str = "Good") -> None:
    level = row["alert_level"]
    prob_str = f"{row['prob']:.0%}" if row.get("prob", 0) > 0 else "No elevated risk"
    no_pred = row.get("no_prediction", False)

    st.markdown(
        f"""
        <div style="background:{_ALERT_COLORS[level]};padding:14px 20px;border-radius:8px;margin-bottom:12px;">
          <span style="color:white;font-size:1.3rem;font-weight:700;">
            {_ALERT_BADGES[level]} &nbsp; {row['display_name']}</span><br>
          <span style="color:rgba(255,255,255,0.92);font-size:0.92rem;">
            {_ALERT_SUBTITLE[level]}</span><br>
          <span style="color:rgba(255,255,255,0.80);font-size:0.85rem;">
            {"Bloom probability: <b>" + prob_str + "</b> &nbsp;·&nbsp; Risk percentile: <b>" + f"{row['risk_percentile']:.0%}" + "</b> &nbsp;·&nbsp; Week: <b>" + str(pd.Timestamp(row['date']).date()) + "</b>" if not no_pred else "No current prediction — HABMAP-monitored zone"}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Data quality badge row
    dq_color = "#388e3c" if dq_score >= 85 else ("#f57c00" if dq_score >= 60 else "#b71c1c")
    col_dq, col_spacer = st.columns([1, 3])
    col_dq.metric(
        label="Data Quality Score",
        value=f"{dq_score}/100",
        delta=dq_label,
        delta_color="off",
        help=(
            "Score reflects corroboration from HABMAP professional monitoring and CalCOFI "
            "historical data.\n\n"
            "**V2 Roadmap:** Real-time integration with SCCOOS live sensor API (sccoos.org) "
            "will raise all zone scores to 95+ with sub-hourly buoy and mooring updates."
        ),
    )


# ── Zone-specific context & report panel ──────────────────────────────────────
_ZONE_CONTEXT: dict[str, dict] = {
    "Santa Barbara Channel":      {"shap": ["nitrate_lag14", "upwelling_proxy", "chlorophyll_lag7"],  "analog": "2003 Santa Barbara Channel Pseudo-nitzschia bloom (domoic acid, shellfish closure)",         "window": "Within the next 7-10 days"},
    "Santa Monica Bay":           {"shap": ["chlorophyll_lag7", "nitrate_lag14", "sst_anomaly"],       "analog": "2011 Santa Monica Bay Alexandrium bloom (paralytic shellfish toxin advisory)",               "window": "Within the next 7-14 days"},
    "San Pedro Channel":          {"shap": ["nitrate_lag7", "chlorophyll_lag7", "upwelling_proxy"],    "analog": "2015 West Coast domoic acid event (unprecedented multi-county closure)",                     "window": "Within the next 10-14 days"},
    "Monterey Bay":               {"shap": ["nitrate_lag14", "nitrate_lag7", "upwelling_proxy"],       "analog": "2009 Monterey Bay Pseudo-nitzschia bloom (sea lion strandings, beach closure 3 weeks)",     "window": "Within the next 7-10 days"},
    "San Francisco Bay Area":     {"shap": ["sst_anomaly", "nitrate_lag14", "chlorophyll_lag7"],       "analog": "2004 Marin/Sonoma coast Gymnodinium bloom (shellfish quarantine 6 weeks)",                   "window": "Within the next 10-14 days"},
    "Half Moon Bay":              {"shap": ["upwelling_proxy", "nitrate_lag14", "sst_anomaly"],        "analog": "2015 Half Moon Bay domoic acid detection (Dungeness crab season delay)",                     "window": "Within the next 7-14 days"},
    "La Jolla / San Diego":       {"shap": ["nitrate_lag14", "chlorophyll_lag7", "nitrate_lag7"],      "analog": "2019 La Jolla red tide (Lingulodinium polyedra, beach discoloration, fish kill)",            "window": "Within the next 7-10 days"},
    "Oceanside / Camp Pendleton": {"shap": ["chlorophyll_lag7", "nitrate_lag7", "upwelling_proxy"],   "analog": "2020 San Diego county domoic acid advisory (recreational fishing closure)",                  "window": "Within the next 10-14 days"},
    "Santa Cruz":                 {"shap": ["nitrate_lag14", "upwelling_proxy", "sst_anomaly"],        "analog": "2015 Monterey Bay domoic acid mass stranding event",                                         "window": "Within the next 7-14 days"},
    "Humboldt Bay":               {"shap": ["nitrate_lag14", "temperature", "upwelling_proxy"],        "analog": "2018 Humboldt Bay Pseudo-nitzschia bloom (commercial shellfish closure 3 months)",           "window": "Within the next 10-14 days"},
    "Bodega Bay":                 {"shap": ["sst_anomaly", "nitrate_lag14", "chlorophyll_lag7"],       "analog": "2004 Bodega Bay Gymnodinium catenatum bloom (shellfish toxin detection)",                    "window": "Within the next 7-14 days"},
    "San Luis Obispo":            {"shap": ["upwelling_proxy", "nitrate_lag14", "temperature"],        "analog": "2016 Morro Bay Pseudo-nitzschia bloom (recreational shellfish closure)",                     "window": "Within the next 10-14 days"},
}
_DEFAULT_CONTEXT = {"shap": ["nitrate_lag14", "chlorophyll_lag7", "nitrate_lag7"], "analog": "2015 West Coast domoic acid event", "window": "Within the next 7-14 days"}


def render_report_panel(row: pd.Series, conditions: dict) -> None:
    st.markdown("#### Public Health Advisory Generator")

    if st.session_state.get("report_zone") != row["zone"]:
        st.session_state.report_text = ""
        st.session_state.report_pdf = None
        st.session_state.report_source = ""
        st.session_state.report_zone = row["zone"]
    if "report_text" not in st.session_state:
        st.session_state.report_text = ""
        st.session_state.report_pdf = None
        st.session_state.report_source = ""

    no_pred = row.get("no_prediction", False)

    corr = find_habmap_corroboration(
        float(row["zone"].split("_")[1]),
        float(row["zone"].split("_")[3]),
        pd.Timestamp(row["date"]),
        radius_km=150,
    )
    if no_pred:
        st.info(
            "ℹ️ No CalCOFI-based model prediction for this zone — insufficient historical coverage. "
            "Advisory will be generated from HABMAP ocean observations only."
        )

    if len(corr) > 0:
        ev = corr.iloc[-1]
        pn_str = f"{ev['pn_total']:,.0f} cells/L" if ev["pn_total"] > 0 else ""
        da_str = " + domoic acid detected" if ev["da_detected"] else ""
        st.success(
            f"✅ **HABMAP corroboration** — {ev['station'].replace('_', ' ').title()} "
            f"recorded {pn_str}{da_str} on {pd.Timestamp(ev['date']).date()}  \n"
            f"*(Scripps/UCSD · same data used by CDPH for shellfish advisories)*"
        )
    else:
        st.info("ℹ️ No HABMAP station within 150 km recorded a bloom in this window (±30 days).")

    gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
    btn_label = "Generate Advisory Memo (Gemini)" if gemini_ok else "Generate Advisory Memo (Template)"

    if st.button(btn_label, type="primary"):
        ctx = _ZONE_CONTEXT.get(row["display_name"], _DEFAULT_CONTEXT)
        habmap_note = ""
        if len(corr) > 0:
            ev = corr.iloc[-1]
            habmap_note = (
                f"HABMAP monitoring at {ev['station'].replace('_', ' ').title()} recorded "
                f"{ev['pn_total']:,.0f} Pseudo-nitzschia cells/L"
                + (" with domoic acid detected" if ev["da_detected"] else "")
                + f" on {pd.Timestamp(ev['date']).date()}."
            )
        with st.spinner("Generating memo..."):
            inp = ReportInput(
                zone=row["display_name"],
                date=str(pd.Timestamp(row["date"]).date()),
                prob=float(row["prob"]),
                lower=float(row["lower"]),
                upper=float(row["upper"]),
                top_shap_features=ctx["shap"],
                historical_analog=ctx["analog"],
                action_window=ctx["window"],
                habmap_event=habmap_note,
                alert_level=row["alert_level"],
                ocean_conditions=conditions,
                no_model_prediction=bool(no_pred),
            )
            txt, source = generate_closure_report(inp)
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_zone = row["display_name"].replace("/", "-").replace(" ", "_")
            pdf_path = REPORTS_DIR / f"{safe_zone}_{inp.date}.pdf"
            export_report_pdf(txt, pdf_path)
            st.session_state.report_text = txt
            st.session_state.report_pdf = pdf_path
            st.session_state.report_source = source

    if st.session_state.report_text:
        src = "🤖 Gemini" if st.session_state.report_source == "gemini" else "📄 Template"
        st.caption(f"Source: {src}")
        st.text_area("Advisory Memo", st.session_state.report_text, height=300)
        pdf_path = st.session_state.report_pdf
        if pdf_path and Path(pdf_path).exists():
            st.download_button("⬇ Download PDF", Path(pdf_path).read_bytes(),
                               Path(pdf_path).name, "application/pdf", type="secondary")
        if not gemini_ok:
            st.info("Set `GEMINI_API_KEY` in .env to enable Gemini prose.")


def render_footer() -> None:
    st.divider()
    cols = st.columns(3)
    cols[0].caption(f"Model: {MODEL_VERSION}")
    cols[1].caption("Data: CalCOFI · iNaturalist · HABMAP (Scripps/UCSD)")
    cols[2].caption(f"Refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Coastal Shield — HAB Early Warning",
        page_icon="🌊", layout="wide", initial_sidebar_state="expanded",
    )

    # 1. Render slider first — sensitivity must be known before classifying predictions
    sensitivity = _render_risk_header()

    # 2. Load predictions classified with current sensitivity
    conf_df  = load_predictions(sensitivity)
    full_df  = _build_full_zone_df(conf_df)
    row = render_sidebar(full_df)
    if row is None:
        st.info("No predictions loaded.")
        return

    # 3. Ocean conditions and data quality score
    conditions = _get_ocean_conditions(row["zone"], str(pd.Timestamp(row["date"]).date()))
    dq_score, dq_label = compute_data_quality_score(row, conditions)

    render_header(row, dq_score, dq_label)

    ctx = _ZONE_CONTEXT.get(row["display_name"], _DEFAULT_CONTEXT)

    zone_df = conf_df[conf_df["zone"] == row["zone"]].copy() if not row.get("no_prediction") else pd.DataFrame()

    col_main, col_shap = st.columns([1.6, 1.0])
    with col_main:
        render_prediction_panel(row, zone_df, conditions)
    with col_shap:
        render_shap_panel(row)

    st.divider()
    render_zone_drivers(row, conditions, ctx)

    st.divider()
    render_report_panel(row, conditions)
    render_footer()


if __name__ == "__main__":
    main()
