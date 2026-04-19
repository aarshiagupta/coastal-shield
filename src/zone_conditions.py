"""
CalCOFI-based zone conditions fallback.
Used when a zone has no nearby HABMAP station or HABMAP values are NaN.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import DEFAULT_CALCOFI_BOTTLE, DEFAULT_CALCOFI_CAST

_BOTTLE_COLS = ["Cst_Cnt", "Depthm", "T_degC", "Salnty", "O2ml_L", "ChlorA", "NO3uM"]
_CAST_COLS   = ["Cst_Cnt", "Lat_Dec", "Lon_Dec", "Date", "Month", "Year"]


@functools.lru_cache(maxsize=1)
def _load_calcofi_surface() -> pd.DataFrame:
    """Load bottle+cast merged, surface only (≤30 m). Cached in memory."""
    if not Path(DEFAULT_CALCOFI_BOTTLE).exists():
        return pd.DataFrame()

    bottle = pd.read_csv(
        DEFAULT_CALCOFI_BOTTLE, low_memory=False, encoding="latin-1",
        usecols=[c for c in _BOTTLE_COLS if c in
                 pd.read_csv(DEFAULT_CALCOFI_BOTTLE, nrows=0, encoding="latin-1").columns],
    )
    cast = pd.read_csv(
        DEFAULT_CALCOFI_CAST, low_memory=False, encoding="latin-1",
        usecols=[c for c in _CAST_COLS if c in
                 pd.read_csv(DEFAULT_CALCOFI_CAST, nrows=0, encoding="latin-1").columns],
    )
    merged = bottle.merge(cast, on="Cst_Cnt", how="inner")
    merged = merged[pd.to_numeric(merged.get("Depthm", np.nan), errors="coerce") <= 30]
    merged["Date"] = pd.to_datetime(merged.get("Date", ""), errors="coerce")
    merged["month"] = merged["Date"].dt.month
    return merged


def get_calcofi_conditions(zone_lat: float, zone_lon: float, radius_deg: float = 1.0) -> dict:
    """
    Return ocean condition means and seasonal z-scores from CalCOFI for a zone.

    Returns dict with keys: temperature, temp_z, chlorophyll, chl_z, nitrate, nit_z,
    salinity, data_available, source, n_samples.
    """
    df = _load_calcofi_surface()
    if len(df) == 0:
        return {"data_available": False, "source": "calcofi"}

    zone_df = df[
        (pd.to_numeric(df.get("Lat_Dec", np.nan), errors="coerce").between(
            zone_lat - radius_deg, zone_lat + radius_deg))
        & (pd.to_numeric(df.get("Lon_Dec", np.nan), errors="coerce").between(
            zone_lon - radius_deg, zone_lon + radius_deg))
    ].copy()

    if len(zone_df) < 5:
        return {"data_available": False, "source": "calcofi"}

    # Recent slice: post-2016; fall back to last 50 rows if sparse
    recent = zone_df[zone_df["Date"] >= "2016-01-01"]
    if len(recent) < 5:
        recent = zone_df.tail(50)

    months = recent["month"].dropna().unique()

    def _mean_and_z(col: str) -> tuple[float, float]:
        vals = pd.to_numeric(recent.get(col, pd.Series(dtype=float)), errors="coerce").dropna()
        if len(vals) == 0:
            return np.nan, np.nan
        mean_val = float(vals.mean())
        hist = pd.to_numeric(
            zone_df[zone_df["month"].isin(months)].get(col, pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        if len(hist) < 5:
            return mean_val, np.nan
        z = (mean_val - hist.mean()) / hist.std() if hist.std() > 0 else 0.0
        return mean_val, float(z)

    temp,    temp_z = _mean_and_z("T_degC")
    chl,     chl_z  = _mean_and_z("ChlorA")
    nitrate, nit_z  = _mean_and_z("NO3uM")
    salinity, _     = _mean_and_z("Salnty")

    return {
        "data_available": True,
        "source": "calcofi",
        "n_samples": len(recent),
        "temperature": temp,  "temp_z":  temp_z,
        "chlorophyll": chl,   "chl_z":   chl_z,
        "nitrate":     nitrate, "nit_z": nit_z,
        "salinity":    salinity,
        # fields not in CalCOFI:
        "pn_total":    0,
        "da_detected": False,
        "sst_anomaly": np.nan,
        "station":     f"CalCOFI grid ({zone_lat:.1f}N, {abs(zone_lon):.1f}W)",
        "station_date": "1949–2021 mean",
        "station_dist_km": 0.0,
    }
