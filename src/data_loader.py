"""
Coastal Shield — load Argo, CalCOFI, iNaturalist and align on a 0.5° grid with weekly bins.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Project root: coastal_shield/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALCOFI_BOTTLE = (
    PROJECT_ROOT / "data" / "raw"
    / "CalCOFI_Database_194903-202105_csv_16October2023"
    / "CalCOFI_Database_194903-202105_csv_16October2023"
    / "194903-202105_Bottle.csv"
)
DEFAULT_CALCOFI_CAST = (
    PROJECT_ROOT / "data" / "raw"
    / "CalCOFI_Database_194903-202105_csv_16October2023"
    / "CalCOFI_Database_194903-202105_csv_16October2023"
    / "194903-202105_Cast.csv"
)
DEFAULT_INATURALIST = PROJECT_ROOT / "data" / "raw" / "inaturalist_hab.csv"
DEFAULT_INATURALIST_SPECIES_DIR = PROJECT_ROOT / "data" / "raw" / "species"
DEFAULT_HABMAP = PROJECT_ROOT / "data" / "raw" / "habmap_all_stations.csv"

HABMAP_STATION_COORDS: dict[str, tuple[float, float]] = {
    "scripps_pier": (32.867, -117.257),
    "santa_monica": (34.008, -118.500),
    "newport_pier": (33.610, -117.930),
    "santa_cruz":   (36.962, -122.022),
    "monterey":     (36.603, -121.893),
    "bodega_bay":   (38.330, -123.050),
    "cal_poly":     (35.166, -120.742),
    "stearns":      (34.408, -119.687),
    "humboldt":     (40.800, -124.180),
}


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance in km between arrays of (lat1,lon1) and a single (lat2,lon2)."""
    r = 6371.0
    p = np.pi / 180.0
    a = (
        0.5
        - np.cos((lat2 - lat1) * p) / 2
        + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
    )
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def grid_half_degree(lat: pd.Series, lon: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Snap to nearest 0.5°; return grid_lat, grid_lon (bin identifiers)."""
    g_lat = (lat * 2).round() / 2
    g_lon = (lon * 2).round() / 2
    return g_lat, g_lon


def grid_center_lat_lon(grid_lat: float, grid_lon: float) -> tuple[float, float]:
    """Center of the 0.5° cell for distance checks."""
    return grid_lat + 0.25, grid_lon + 0.25


class DataLoader:
    """Load ocean and observation datasets; align spatiotemporally."""

    def __init__(
        self,
        calcofi_bottle_path: Optional[Path | str] = None,
        calcofi_cast_path: Optional[Path | str] = None,
        inaturalist_path: Optional[Path | str] = None,
        date_start: str = "2015-01-01",
        date_end: str = "2027-01-01",
        ca_lat: tuple[float, float] = (32.0, 38.0),
        ca_lon: tuple[float, float] = (-125.0, -117.0),
    ) -> None:
        self.calcofi_bottle_path = Path(calcofi_bottle_path) if calcofi_bottle_path else DEFAULT_CALCOFI_BOTTLE
        self.calcofi_cast_path = Path(calcofi_cast_path) if calcofi_cast_path else DEFAULT_CALCOFI_CAST
        self.inaturalist_path = Path(inaturalist_path) if inaturalist_path else DEFAULT_INATURALIST
        self.date_start = pd.Timestamp(date_start)
        self.date_end = pd.Timestamp(date_end)
        self.ca_lat = ca_lat
        self.ca_lon = ca_lon

    @staticmethod
    def summarize(df: pd.DataFrame, name: str) -> None:
        print(f"\n=== {name} ===")
        print(f"shape: {df.shape}")
        if len(df) == 0:
            print("(empty)")
            return
        print("null counts (top 15):")
        nc = df.isna().sum().sort_values(ascending=False).head(15)
        print(nc.to_string())
        dt_cols = [c for c in df.columns if "date" in c.lower() or c in ("week_start", "time")]
        for c in dt_cols:
            if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c]):
                print(f"date range [{c}]: {df[c].min()} → {df[c].max()}")

    def load_argo(
        self,
        cache_parquet: Optional[Path | str] = None,
        region: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """
        Fetch Argo via argopy (or read cache). Columns: date, lat, lon, temperature, salinity, depth.
        """
        cache_parquet = Path(cache_parquet) if cache_parquet else PROJECT_ROOT / "data" / "raw" / "argo_ca_2015_2025.parquet"
        if cache_parquet.exists():
            df = pd.read_parquet(cache_parquet)
            return self._normalize_argo_df(df)

        if region is None:
            region = [-125, -117, 32, 38, 0, 500, "2015-01-01", "2025-01-01"]

        try:
            from argopy import DataFetcher as ArgoDataFetcher
        except ImportError as e:
            raise ImportError("Install argopy: pip install argopy") from e

        warnings.filterwarnings("ignore", category=UserWarning)
        loader = ArgoDataFetcher(src="erddap")
        ds = loader.region(region).to_xarray()
        df = ds.to_dataframe().reset_index()

        # Normalize common argopy / ERDDAP column names
        colmap: dict[str, str] = {}
        lower = {c.lower(): c for c in df.columns}
        for want, candidates in [
            ("date", ["time", "date"]),
            ("lat", ["latitude", "lat"]),
            ("lon", ["longitude", "lon"]),
            ("temperature", ["temp", "temperature"]),
            ("salinity", ["psal", "salinity"]),
            ("depth", ["pres", "pressure", "depth"]),
        ]:
            for cand in candidates:
                if cand in lower:
                    colmap[lower[cand]] = want
                    break
        df = df.rename(columns=colmap)

        if "date" not in df.columns:
            for c in df.columns:
                if "time" in c.lower():
                    df = df.rename(columns={c: "date"})
                    break

        df = self._normalize_argo_df(df)
        try:
            cache_parquet.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_parquet, index=False)
        except Exception:
            pass
        return df

    def _normalize_argo_df(self, df: pd.DataFrame) -> pd.DataFrame:
        need = ["date", "lat", "lon", "temperature", "salinity", "depth"]
        for c in need:
            if c not in df.columns:
                # try case-insensitive
                match = next((x for x in df.columns if x.lower() == c), None)
                if match:
                    df = df.rename(columns={match: c})
        if "depth" not in df.columns and "pres" in df.columns:
            df = df.rename(columns={"pres": "depth"})
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
        df = df.dropna(subset=["date", "lat", "lon"])
        df = df[(df["lat"] >= self.ca_lat[0]) & (df["lat"] <= self.ca_lat[1])]
        df = df[(df["lon"] >= self.ca_lon[0]) & (df["lon"] <= self.ca_lon[1])]
        df = df[(df["date"] >= self.date_start) & (df["date"] < self.date_end)]
        out = df[["date", "lat", "lon", "temperature", "salinity", "depth"]].copy()
        return out

    def load_calcofi(self) -> pd.DataFrame:
        """Load bottle + cast CSVs; key columns; parse dates; drop nulls on core fields."""
        bottle = pd.read_csv(self.calcofi_bottle_path, low_memory=False, encoding="latin-1")
        cast = pd.read_csv(self.calcofi_cast_path, low_memory=False, encoding="latin-1")

        cast_sub = cast[
            ["Cst_Cnt", "Date", "Lat_Dec", "Lon_Dec", "Year"]
        ].copy()
        cast_sub["Date"] = pd.to_datetime(cast_sub["Date"], errors="coerce")

        bottle_sub = bottle[
            [
                "Cst_Cnt",
                "Depthm",
                "T_degC",
                "Salnty",
                "ChlorA",
                "NO3uM",
            ]
        ].copy()

        merged = bottle_sub.merge(cast_sub, on="Cst_Cnt", how="inner")
        merged = merged.rename(
            columns={
                "Lat_Dec": "lat",
                "Lon_Dec": "lon",
                "T_degC": "temperature",
                "Salnty": "salinity",
                "ChlorA": "chlorophyll",
                "NO3uM": "nitrate",
            }
        )
        merged = merged.rename(columns={"Date": "date"})

        merged = merged[
            (merged["lat"] >= self.ca_lat[0])
            & (merged["lat"] <= self.ca_lat[1])
            & (merged["lon"] >= self.ca_lon[0])
            & (merged["lon"] <= self.ca_lon[1])
        ]
        merged = merged[(merged["date"] >= self.date_start) & (merged["date"] < self.date_end)]

        merged = merged.dropna(subset=["date", "lat", "lon"])
        # require at least one of the chemical/bio signals
        merged = merged.dropna(subset=["chlorophyll", "nitrate"], how="all")

        return merged

    def load_inaturalist(self) -> pd.DataFrame:
        """
        Load iNaturalist export. Expects columns including date + lat/lon.
        Recognizes: observed_on, eventDate, observation_date, date, latitude, longitude.
        """
        if not self.inaturalist_path.exists():
            warnings.warn(
                f"iNaturalist file not found: {self.inaturalist_path}. "
                "Returning empty frame; bloom_label will be all zeros.",
                stacklevel=2,
            )
            return pd.DataFrame(columns=["date", "lat", "lon", "species", "id"])

        raw = pd.read_csv(self.inaturalist_path, low_memory=False, encoding="utf-8")
        col_lower = {c.lower(): c for c in raw.columns}

        def pick(*names: str) -> Optional[str]:
            for n in names:
                if n in col_lower:
                    return col_lower[n]
                if n.capitalize() in raw.columns:
                    return n.capitalize()
            return None

        c_date = pick("observed_on", "eventdate", "observation_date", "date", "time_observed_at")
        c_lat = pick("latitude", "lat")
        c_lon = pick("longitude", "lon")
        if not all([c_date, c_lat, c_lon]):
            raise ValueError(
                f"Could not infer date/lat/lon columns from {self.inaturalist_path}. "
                f"Columns: {list(raw.columns)}"
            )

        out = pd.DataFrame(
            {
                "date": pd.to_datetime(raw[c_date], errors="coerce", utc=True).dt.tz_localize(None),
                "lat": pd.to_numeric(raw[c_lat], errors="coerce"),
                "lon": pd.to_numeric(raw[c_lon], errors="coerce"),
            }
        )
        sp_col = pick("scientific_name", "species_guess")
        if sp_col:
            out["species"] = raw[sp_col].astype(str)
        else:
            out["species"] = ""
        id_col = pick("id", "observation_uuid")
        if id_col:
            out["id"] = raw[id_col]
        else:
            out["id"] = np.arange(len(out))

        out = out.dropna(subset=["date", "lat", "lon"])
        out = out[
            (out["lat"] >= self.ca_lat[0])
            & (out["lat"] <= self.ca_lat[1])
            & (out["lon"] >= self.ca_lon[0])
            & (out["lon"] <= self.ca_lon[1])
        ]
        out = out[(out["date"] >= self.date_start) & (out["date"] < self.date_end)]
        return out

    def _append_inat_grid_weeks(self, merged: pd.DataFrame, df_inat: pd.DataFrame) -> pd.DataFrame:
        """
        CalCOFI has gaps in space/time; iNat blooms may fall in weeks with no bottle sample.
        Add sparse rows for (grid, week) at each observation and prior 2 week-starts so
        ``_bloom_labels`` can attach positives when an observation falls in ``[week, week+14d]``.
        """
        if len(df_inat) == 0 or len(merged) == 0:
            return merged

        def norm_week(ws: Any) -> pd.Timestamp:
            return pd.Timestamp(ws).normalize()

        def key_set(m: pd.DataFrame) -> set[tuple[float, float, pd.Timestamp]]:
            out: set[tuple[float, float, pd.Timestamp]] = set()
            for _, rr in m.iterrows():
                out.add((float(rr["grid_lat"]), float(rr["grid_lon"]), norm_week(rr["week_start"])))
            return out

        keys = key_set(merged)
        extra: list[dict[str, Any]] = []

        for _, r in df_inat.iterrows():
            glat_s, glon_s = grid_half_degree(pd.Series([float(r["lat"])]), pd.Series([float(r["lon"])]))
            glat, glon = float(glat_s.iloc[0]), float(glon_s.iloc[0])
            ts = pd.Timestamp(r["date"])
            base = pd.Timestamp(ts.to_period("W-SUN").start_time)
            for week_back in (0, 1, 2):
                ws = base - pd.Timedelta(weeks=week_back)
                nk = norm_week(ws)
                k = (glat, glon, nk)
                if k in keys:
                    continue
                row = {c: np.nan for c in merged.columns}
                row["grid_lat"] = glat
                row["grid_lon"] = glon
                row["week_start"] = nk
                extra.append(row)
                keys.add(k)

        if not extra:
            return merged

        add = pd.DataFrame(extra)
        return pd.concat([merged, add], ignore_index=True)

    def _weekly_aggregate(
        self,
        df: pd.DataFrame,
        value_cols: list[str],
        date_col: str = "date",
    ) -> pd.DataFrame:
        df = df.copy()
        df["grid_lat"], df["grid_lon"] = grid_half_degree(df["lat"], df["lon"])
        df["week_start"] = df[date_col].dt.to_period("W-SUN").dt.start_time
        agg_dict: dict[str, Any] = {c: "mean" for c in value_cols if c in df.columns}
        g = df.groupby(["grid_lat", "grid_lon", "week_start"], as_index=False).agg(agg_dict)
        return g

    def _bloom_labels(
        self,
        grid_weeks: pd.DataFrame,
        obs: pd.DataFrame,
        horizon_days: int = 14,
        radius_km: float = 50.0,
    ) -> pd.Series:
        """For each row (grid_lat, grid_lon, week_start), 1 if any obs within radius in (week, week+horizon]."""
        if len(obs) == 0:
            return pd.Series(0, index=grid_weeks.index, dtype=int)

        o_dates = obs["date"].values.astype("datetime64[ns]")
        o_lat = obs["lat"].to_numpy()
        o_lon = obs["lon"].to_numpy()

        labels: list[int] = []
        for _, row in grid_weeks.iterrows():
            glat, glon = float(row["grid_lat"]), float(row["grid_lon"])
            wk = pd.Timestamp(row["week_start"])
            t0 = wk
            t1 = wk + pd.Timedelta(days=horizon_days)
            clat, clon = grid_center_lat_lon(glat, glon)

            time_ok = (o_dates >= np.datetime64(t0)) & (o_dates <= np.datetime64(t1))
            if not np.any(time_ok):
                labels.append(0)
                continue
            d_km = haversine_km(o_lat[time_ok], o_lon[time_ok], clat, clon)
            labels.append(int(np.any(d_km <= radius_km)))
        return pd.Series(labels, index=grid_weeks.index, dtype=int)

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling anomalies, lags, upwelling proxy on unified weekly grid."""
        df = df.sort_values(["grid_lat", "grid_lon", "week_start"]).reset_index(drop=True)

        def roll_anom(s: pd.Series) -> pd.Series:
            m = s.rolling(window=5, min_periods=1).mean()
            return s - m

        out = df.copy()
        if "temperature" in out.columns:
            out["sst_anomaly"] = out.groupby(["grid_lat", "grid_lon"], group_keys=False)["temperature"].transform(
                roll_anom
            )
        else:
            out["sst_anomaly"] = np.nan

        if "nitrate" in out.columns:
            out["nitrate_anomaly"] = out.groupby(["grid_lat", "grid_lon"], group_keys=False)["nitrate"].transform(
                roll_anom
            )
        else:
            out["nitrate_anomaly"] = np.nan

        # Weekly bins ≈ 7-day steps: lag 1 week ≈ 7d, lag 2 weeks ≈ 14d
        if "chlorophyll" in out.columns:
            out["chlorophyll_lag7"] = out.groupby(["grid_lat", "grid_lon"])["chlorophyll"].shift(1)
        if "nitrate" in out.columns:
            out["nitrate_lag7"] = out.groupby(["grid_lat", "grid_lon"])["nitrate"].shift(1)
            out["nitrate_lag14"] = out.groupby(["grid_lat", "grid_lon"])["nitrate"].shift(2)

        if "upwelling_proxy" not in out.columns:
            out["upwelling_proxy"] = np.nan

        return out

    def align_datasets(
        self,
        df_argo: Optional[pd.DataFrame] = None,
        df_calcofi: Optional[pd.DataFrame] = None,
        df_inat: Optional[pd.DataFrame] = None,
        load_argo_if_missing: bool = True,
    ) -> pd.DataFrame:
        """
        Spatial join on 0.5° grid, weekly aggregation, merge sources, engineer features and bloom_label.
        """
        if df_calcofi is None:
            df_calcofi = self.load_calcofi()

        if df_inat is None:
            df_inat = self.load_inaturalist()

        if df_argo is None and load_argo_if_missing:
            try:
                df_argo = self.load_argo()
            except Exception as e:
                warnings.warn(f"Argo load failed ({e}); continuing without Argo columns.", stacklevel=2)
                df_argo = pd.DataFrame()
        elif df_argo is None:
            df_argo = pd.DataFrame()

        # CalCOFI weekly (surface bottles for SST/chl/nitrate)
        surf = df_calcofi[df_calcofi["Depthm"] <= 30].copy()
        deep = df_calcofi[(df_calcofi["Depthm"] >= 45) & (df_calcofi["Depthm"] <= 55)].copy()

        c_surf = self._weekly_aggregate(
            surf,
            ["temperature", "salinity", "chlorophyll", "nitrate"],
        )

        if len(deep):
            d_deep = self._weekly_aggregate(deep, ["temperature"])
            d_deep = d_deep.rename(columns={"temperature": "temperature_deep"})
            c_surf = c_surf.merge(
                d_deep[["grid_lat", "grid_lon", "week_start", "temperature_deep"]],
                on=["grid_lat", "grid_lon", "week_start"],
                how="left",
            )
            c_surf["upwelling_proxy"] = c_surf["temperature"] - c_surf["temperature_deep"]
        else:
            c_surf["upwelling_proxy"] = np.nan

        merged = c_surf

        # Argo weekly — optional columns for gap-fill and cross-check
        if len(df_argo):
            a = df_argo.copy()
            shallow = a[a["depth"] <= 20]
            mid = a[(a["depth"] >= 40) & (a["depth"] <= 60)]
            aw_s = self._weekly_aggregate(shallow, ["temperature", "salinity"])
            aw_s = aw_s.rename(columns={"temperature": "argo_sst", "salinity": "argo_salinity"})
            if len(mid):
                aw_m = self._weekly_aggregate(mid, ["temperature"])
                aw_m = aw_m.rename(columns={"temperature": "argo_temp_50m"})
                aw_s = aw_s.merge(
                    aw_m[["grid_lat", "grid_lon", "week_start", "argo_temp_50m"]],
                    on=["grid_lat", "grid_lon", "week_start"],
                    how="outer",
                )
            merged = merged.merge(aw_s, on=["grid_lat", "grid_lon", "week_start"], how="outer")
            if "argo_sst" in merged.columns and "argo_temp_50m" in merged.columns:
                merged["upwelling_proxy"] = merged["upwelling_proxy"].combine_first(
                    merged["argo_sst"] - merged["argo_temp_50m"]
                )

        if "argo_sst" in merged.columns:
            merged["temperature"] = merged["temperature"].fillna(merged["argo_sst"])

        merged = self._append_inat_grid_weeks(merged, df_inat)

        merged = merged.sort_values(["grid_lat", "grid_lon", "week_start"])
        merged = self._engineer_features(merged)

        inat_labels = self._bloom_labels(merged, df_inat)
        merged["bloom_label"] = inat_labels
        merged["bloom_label_inat"] = inat_labels

        # HABMAP: merge confirmed positives and negatives on grid + week
        df_habmap = load_habmap()
        merged["lat_grid"] = merged["grid_lat"]
        merged["lon_grid"] = merged["grid_lon"]
        merged["week"] = merged["week_start"].dt.to_period("W")

        merged = merged.merge(
            df_habmap[["lat_grid", "lon_grid", "week", "bloom_label", "confirmed_negative", "source"]],
            on=["lat_grid", "lon_grid", "week"],
            how="left",
            suffixes=("", "_habmap"),
        )

        # HABMAP positive overrides
        merged.loc[merged["bloom_label_habmap"] == 1, "bloom_label"] = 1
        # HABMAP confirmed negative: keep explicit 0 where iNat also has no positive
        merged.loc[
            (merged["confirmed_negative"] == 1) & (merged["bloom_label"] == 0),
            "bloom_label",
        ] = 0

        merged = merged.drop(columns=["lat_grid", "lon_grid", "week", "source_habmap"], errors="ignore")

        return merged


def load_habmap(path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load HABMAP station data with exact confirmed column names.
    Returns training-ready frame with lat_grid, lon_grid, week (Period),
    bloom_label, confirmed_negative, source, confidence.
    """
    p = Path(path) if path else DEFAULT_HABMAP
    if not p.exists():
        warnings.warn(f"HABMAP file not found: {p}. Returning empty frame.", stacklevel=2)
        return pd.DataFrame(
            columns=["date", "week", "year", "month", "Lat_Dec", "Lon_Dec",
                     "lat_grid", "lon_grid", "pn_total", "bloom_label",
                     "confirmed_negative", "confidence", "source"]
        )

    df = pd.read_csv(p, low_memory=False)

    df["date"] = pd.to_datetime(df["time"], errors="coerce")
    df["week"] = df["date"].dt.to_period("W")
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    df = df.rename(columns={"latitude": "Lat_Dec", "longitude": "Lon_Dec"})

    pn_del = pd.to_numeric(df["Pseudo_nitzschia_delicatissima_group"], errors="coerce").fillna(0)
    pn_ser = pd.to_numeric(df["Pseudo_nitzschia_seriata_group"], errors="coerce").fillna(0)
    df["pn_total"] = pn_del + pn_ser

    da_cols = [c for c in df.columns if "domoic" in c.lower() or c in ("pDA", "tDA", "dDA")]
    df["bloom_label"] = (df["pn_total"] > 10_000).astype(int)
    if da_cols:
        da = pd.to_numeric(df[da_cols[0]], errors="coerce").fillna(0)
        df.loc[da > 0.5, "bloom_label"] = 1

    df["confirmed_negative"] = (df["pn_total"] < 1000).astype(int)
    df["source"] = "habmap"
    df["confidence"] = 0.9

    df["lat_grid"] = (df["Lat_Dec"] / 0.5).round() * 0.5
    df["lon_grid"] = (df["Lon_Dec"] / 0.5).round() * 0.5

    print(f"HABMAP loaded: {len(df)} rows")
    print(f"Bloom positive (pn>10k or DA): {df['bloom_label'].sum()}")
    print(f"Confirmed negative (pn<1k): {df['confirmed_negative'].sum()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df[["date", "week", "year", "month",
               "Lat_Dec", "Lon_Dec", "lat_grid", "lon_grid",
               "pn_total", "bloom_label", "confirmed_negative",
               "confidence", "source"]]


def find_habmap_corroboration(
    zone_lat: float,
    zone_lon: float,
    week_start: str | pd.Timestamp,
    radius_km: float = 100.0,
    window_days: int = 30,
    habmap_path: Optional[Path | str] = None,
) -> pd.DataFrame:
    """
    Return HABMAP bloom events near (zone_lat, zone_lon) within window_days of week_start.
    Used by the dashboard and report generator — operates on raw lat/lon/date.
    """
    p = Path(habmap_path) if habmap_path else DEFAULT_HABMAP
    if not p.exists():
        return pd.DataFrame()

    raw = pd.read_csv(p, low_memory=False)
    raw["date"] = pd.to_datetime(raw["time"], errors="coerce").dt.tz_localize(None)
    raw = raw.dropna(subset=["date"])
    pn = (
        pd.to_numeric(raw["Pseudo_nitzschia_delicatissima_group"], errors="coerce").fillna(0)
        + pd.to_numeric(raw["Pseudo_nitzschia_seriata_group"], errors="coerce").fillna(0)
    )
    da_cols = [c for c in raw.columns if "domoic" in c.lower() or c in ("pDA", "tDA", "dDA")]
    da = pd.to_numeric(raw[da_cols[0]], errors="coerce").fillna(0) if da_cols else pd.Series(0, index=raw.index)
    raw["pn_total"] = pn
    raw["da_detected"] = (da > 0.5).astype(int)
    raw["bloom_label"] = ((pn > 10_000) | (da > 0.5)).astype(int)
    raw = raw.rename(columns={"latitude": "lat", "longitude": "lon"})

    blooms = raw[raw["bloom_label"] == 1].copy()
    if len(blooms) == 0:
        return blooms

    wk = pd.Timestamp(week_start)
    blooms = blooms[
        (blooms["date"] >= wk - pd.Timedelta(days=window_days)) &
        (blooms["date"] <= wk + pd.Timedelta(days=window_days))
    ]
    if len(blooms) == 0:
        return blooms

    clat = zone_lat + 0.25
    clon = zone_lon + 0.25
    dist = haversine_km(blooms["lat"].to_numpy(), blooms["lon"].to_numpy(), clat, clon)
    mask = dist <= radius_km
    blooms = blooms[mask].copy()
    blooms["distance_km"] = dist[mask]
    return blooms[["date", "station", "lat", "lon", "pn_total", "da_detected", "distance_km"]].sort_values("date")


def print_all_summaries(df_argo: pd.DataFrame, df_calcofi: pd.DataFrame, df_inat: pd.DataFrame) -> None:
    DataLoader.summarize(df_argo, "Argo")
    DataLoader.summarize(df_calcofi, "CalCOFI (merged)")
    DataLoader.summarize(df_inat, "iNaturalist")


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, encoding="latin-1")


def merge_inaturalist_species_exports(
    species_dir: Optional[Path | str] = None,
    out_path: Optional[Path | str] = None,
) -> Path:
    """
    Concatenate one or more iNaturalist export CSVs (e.g. one taxon per file) from ``data/raw/species/``,
    dedupe by observation ``id``, and write ``data/raw/inaturalist_hab.csv`` for :meth:`DataLoader.load_inaturalist`.
    """
    species_dir = Path(species_dir) if species_dir else DEFAULT_INATURALIST_SPECIES_DIR
    out_path = Path(out_path) if out_path else DEFAULT_INATURALIST

    paths = sorted(species_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files in {species_dir}. Add exports (e.g. pseudo-nitzschia.csv) and run again."
        )

    parts: list[pd.DataFrame] = []
    for p in paths:
        chunk = _read_csv_flexible(p)
        if len(chunk):
            parts.append(chunk)

    if not parts:
        raise ValueError(f"All CSVs under {species_dir} are empty.")

    combined = pd.concat(parts, ignore_index=True)

    col_lower = {c.lower(): c for c in combined.columns}
    id_col = col_lower.get("id")
    if id_col:
        before = len(combined)
        combined = combined.drop_duplicates(subset=[id_col], keep="first")
        dropped = before - len(combined)
        if dropped:
            print(f"Dropped {dropped} duplicate observation id(s).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Wrote {len(combined)} rows to {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Merge iNaturalist species/*.csv into inaturalist_hab.csv")
    p.add_argument(
        "--species-dir",
        type=Path,
        default=DEFAULT_INATURALIST_SPECIES_DIR,
        help=f"Folder with one CSV per taxon export (default: {DEFAULT_INATURALIST_SPECIES_DIR})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_INATURALIST,
        help=f"Output CSV path (default: {DEFAULT_INATURALIST})",
    )
    args = p.parse_args()
    merge_inaturalist_species_exports(species_dir=args.species_dir, out_path=args.out)
