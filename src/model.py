"""
Coastal Shield — XGBoost baseline classifier for bloom risk (Step 2).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from src.data_loader import DataLoader


def _fit_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    scale_pos_weight: float,
) -> tuple[Any, str]:
    """Prefer XGBoost; fall back to sklearn if libxgboost cannot load (e.g. missing libomp on macOS)."""
    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
        )
        clf.fit(X_train, y_train)
        return clf, "xgboost"
    except Exception:
        sw = compute_sample_weight("balanced", y_train)
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_state,
        )
        clf.fit(X_train, y_train, sample_weight=sw)
        return clf, "sklearn_hist_gbt"

DEFAULT_FEATURE_COLS = [
    "sst_anomaly",
    "nitrate_anomaly",
    "upwelling_proxy",
    "chlorophyll_lag7",
    "nitrate_lag7",
    "nitrate_lag14",
    "temperature",
    "chlorophyll",
    "nitrate",
    "salinity",
]


def build_chronos_sequences(
    df: pd.DataFrame,
    lookback_days: int = 30,
    horizon_days: tuple[int, int] = (7, 14),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build per-grid weekly sequences for Chronos-style experiments.
    Input sequence uses [nitrate, chlorophyll, sst_anomaly] over lookback_days.
    Targets are bloom_label at +7 and +14 days (approximated as +1 and +2 weekly steps).
    """
    seq_cols = [c for c in ["nitrate", "chlorophyll", "sst_anomaly"] if c in df.columns]
    if len(seq_cols) < 3:
        raise ValueError("Need nitrate, chlorophyll, and sst_anomaly columns for Chronos sequences.")

    lookback_steps = max(2, int(round(lookback_days / 7)))
    h_steps = [max(1, int(round(h / 7))) for h in horizon_days]

    d = df.sort_values(["grid_lat", "grid_lon", "week_start"]).copy()
    d[seq_cols] = d[seq_cols].astype(float)
    d[seq_cols] = d.groupby(["grid_lat", "grid_lon"])[seq_cols].transform(lambda s: s.ffill().bfill())

    X, y = [], []
    for _, g in d.groupby(["grid_lat", "grid_lon"]):
        g = g.reset_index(drop=True)
        arr = g[seq_cols].to_numpy()
        lbl = g["bloom_label"].astype(int).to_numpy()
        max_h = max(h_steps)
        for i in range(lookback_steps - 1, len(g) - max_h):
            win = arr[i - lookback_steps + 1 : i + 1]
            if np.isnan(win).any():
                continue
            targets = [lbl[i + hs] for hs in h_steps]
            X.append(win)
            y.append(targets)

    if not X:
        raise ValueError("No valid sequence windows produced for Chronos.")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def train_chronos_hybrid(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
    use_wandb: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Hackathon-friendly Step 3:
    1) Build Chronos input sequences
    2) Try loading Chronos checkpoint (zero-shot readiness check)
    3) Flatten sequence features and train a fast classifier as hybrid baseline
    """
    X_seq_all, y_seq_all = build_chronos_sequences(df)
    cut = max(1, int(len(X_seq_all) * (1.0 - test_ratio)))
    X_seq_train, X_seq_test = X_seq_all[:cut], X_seq_all[cut:]
    y_seq_train, y_seq_test = y_seq_all[:cut], y_seq_all[cut:]
    if len(X_seq_test) == 0:
        raise ValueError("Chronos hybrid split produced empty test window set.")

    # +14d target as primary
    y_train = y_seq_train[:, 1].astype(int)
    y_test = y_seq_test[:, 1].astype(int)

    # Flatten sequence into tabular hybrid features
    X_train = X_seq_train.reshape(X_seq_train.shape[0], -1)
    X_test = X_seq_test.reshape(X_seq_test.shape[0], -1)
    X_train_df = pd.DataFrame(X_train)
    X_test_df = pd.DataFrame(X_test)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale = min((neg / pos) if pos > 0 else 1.0, 1e6)

    clf, backend = _fit_classifier(X_train_df, pd.Series(y_train), random_state, scale)
    y_proba = clf.predict_proba(X_test_df)[:, 1]
    y_pred = (y_proba >= 0.3).astype(int)

    out: dict[str, Any] = {
        "backend": f"chronos_hybrid_{backend}",
        "n_train": len(y_train),
        "n_test": len(y_test),
        "positive_rate_train": float(y_train.mean()) if len(y_train) else 0.0,
        "positive_rate_test": float(y_test.mean()) if len(y_test) else 0.0,
        "f1_at_0.3": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    if len(np.unique(y_test)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        out["pr_auc"] = float(average_precision_score(y_test, y_proba))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    chronos_ok = False
    chronos_err = ""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        _ = AutoTokenizer.from_pretrained("amazon/chronos-t5-small")
        _ = AutoModelForSeq2SeqLM.from_pretrained("amazon/chronos-t5-small")
        chronos_ok = True
    except Exception as e:
        chronos_err = str(e)
    out["chronos_checkpoint_loaded"] = chronos_ok
    out["chronos_load_error"] = chronos_err

    if use_wandb is None:
        use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb

        wandb.init(project="coastal-shield", job_type="chronos_hybrid")
        wandb.log(
            {
                "backend": out["backend"],
                "roc_auc": out["roc_auc"],
                "pr_auc": out["pr_auc"],
                "f1_at_0.3": out["f1_at_0.3"],
                "n_train": out["n_train"],
                "n_test": out["n_test"],
                "chronos_checkpoint_loaded": int(chronos_ok),
            }
        )
        wandb.finish()

    out["model"] = clf
    out["y_test"] = y_test
    out["y_proba"] = y_proba
    return out


def chronological_split_by_week(
    df: pd.DataFrame,
    date_col: str = "week_start",
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last ``test_ratio`` of distinct weeks → test; avoids mixing future weeks into train."""
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    weeks = sorted(d[date_col].unique())
    if len(weeks) < 3:
        raise ValueError("Need at least 3 distinct weeks for a time split.")
    cut_i = max(1, int(len(weeks) * (1.0 - test_ratio)))
    cut_week = weeks[cut_i]
    train = d[d[date_col] < cut_week]
    test = d[d[date_col] >= cut_week]
    return train, test


def prepare_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "bloom_label",
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Require target; features may contain NaN (filled later with train medians)."""
    use = [c for c in feature_cols if c in df.columns]
    if not use:
        raise ValueError("No feature columns present in dataframe.")
    sub = df[use + [target_col]].dropna(subset=[target_col])
    X = sub[use].astype(float)
    y = sub[target_col].astype(int)
    return X, y, use


def train_xgb_baseline(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    target_col: str = "bloom_label",
    test_ratio: float = 0.2,
    threshold: float = 0.3,
    random_state: int = 42,
    use_wandb: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Chronological train/test split, XGBoost, AUROC / PR-AUC / F1 at ``threshold``.
    If ``use_wandb`` is None, W&B runs when ``WANDB_API_KEY`` is set.
    """
    feature_cols = feature_cols or DEFAULT_FEATURE_COLS
    train_df, test_df = chronological_split_by_week(df, test_ratio=test_ratio)

    X_train, y_train, cols = prepare_xy(train_df, feature_cols, target_col)
    X_test, y_test, _ = prepare_xy(test_df, feature_cols, target_col)

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Empty train ({len(X_train)}) or test ({len(X_test)}) after chronological split. "
            "Check feature columns and date range."
        )

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=cols, index=X_test.index)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale = min((neg / pos) if pos > 0 else 1.0, 1e6)

    clf, backend = _fit_classifier(X_train, y_train, random_state, scale)

    proba = clf.predict_proba(X_test)[:, 1]
    pred_t = (proba >= threshold).astype(int)

    out: dict[str, Any] = {
        "feature_columns": cols,
        "backend": backend,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "positive_rate_train": float(y_train.mean()) if len(y_train) else 0.0,
        "positive_rate_test": float(y_test.mean()) if len(y_test) else 0.0,
    }

    if len(np.unique(y_test)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_test, proba))
        out["pr_auc"] = float(average_precision_score(y_test, proba))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    out[f"f1_at_{threshold}"] = float(f1_score(y_test, pred_t, zero_division=0))

    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        imp = np.ones(len(cols)) / len(cols)
    fi = pd.Series(imp, index=cols).sort_values(ascending=False)
    out["feature_importance"] = fi

    if use_wandb is None:
        use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb

        wandb.init(project="coastal-shield", job_type="xgb_baseline")
        wandb.log(
            {
                "backend": backend,
                "roc_auc": out["roc_auc"],
                "pr_auc": out["pr_auc"],
                "f1_at_threshold": out[f"f1_at_{threshold}"],
                "n_train": out["n_train"],
                "n_test": out["n_test"],
                "threshold": threshold,
            }
        )
        for name, val in fi.items():
            wandb.log({f"feature_importance/{name}": float(val)})
        wandb.finish()

    out["model"] = clf
    out["X_test"] = X_test
    out["y_test"] = y_test
    out["y_proba"] = proba
    return out


def run_baseline_from_loader(load_argo_if_missing: bool = False) -> dict[str, Any]:
    """Load aligned data via :class:`DataLoader`, train baseline, return metrics dict."""
    loader = DataLoader()
    df = loader.align_datasets(load_argo_if_missing=load_argo_if_missing)
    return train_xgb_baseline(df)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train Step 2/3 bloom models")
    p.add_argument("--argo", action="store_true", help="Try to load Argo (network/cache)")
    p.add_argument(
        "--mode",
        choices=["baseline", "chronos_hybrid"],
        default="baseline",
        help="Modeling mode: Step 2 baseline or Step 3 Chronos hybrid.",
    )
    args = p.parse_args()
    loader = DataLoader()
    df = loader.align_datasets(load_argo_if_missing=args.argo)

    if args.mode == "baseline":
        r = train_xgb_baseline(df)
        print("Mode:", args.mode, "Backend:", r["backend"])
        print("Train:", r["n_train"], " Test:", r["n_test"])
        print("ROC-AUC:", r["roc_auc"], " PR-AUC:", r["pr_auc"])
        print("F1 @0.3:", r["f1_at_0.3"])
        print("\nFeature importance:\n", r["feature_importance"])
    else:
        r = train_chronos_hybrid(df)
        print("Mode:", args.mode, "Backend:", r["backend"])
        print("Train:", r["n_train"], " Test:", r["n_test"])
        print("ROC-AUC:", r["roc_auc"], " PR-AUC:", r["pr_auc"])
        print("F1 @0.3:", r["f1_at_0.3"])
        print("Chronos checkpoint loaded:", r["chronos_checkpoint_loaded"])
        if not r["chronos_checkpoint_loaded"] and r["chronos_load_error"]:
            print("Chronos load error:", r["chronos_load_error"])
