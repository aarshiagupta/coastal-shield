"""
Coastal Shield — Step 4 conformal prediction wrapper.

Generates calibrated probability intervals for bloom risk and recommended actions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Repo root (coastal_shield/) so `python src/conformal.py` works without PYTHONPATH
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.data_loader import DataLoader, PROJECT_ROOT
from src.model import DEFAULT_FEATURE_COLS, _fit_classifier, prepare_xy


def _chronological_three_way_split(
    df: pd.DataFrame,
    date_col: str = "week_start",
    train_ratio: float = 0.6,
    cal_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    weeks = sorted(d[date_col].unique())
    if len(weeks) < 6:
        raise ValueError("Need at least 6 distinct weeks for train/cal/test split.")

    i_train = max(1, int(len(weeks) * train_ratio))
    i_cal = max(i_train + 1, int(len(weeks) * (train_ratio + cal_ratio)))
    i_cal = min(i_cal, len(weeks) - 1)

    w_train = weeks[i_train]
    w_cal = weeks[i_cal]
    train = d[d[date_col] < w_train]
    cal = d[(d[date_col] >= w_train) & (d[date_col] < w_cal)]
    test = d[d[date_col] >= w_cal]
    return train, cal, test


def recommended_action(prob: float) -> str:
    if prob < 0.3:
        return "Monitor"
    if prob < 0.6:
        return "Advisory"
    return "Closure Recommended"


def _zone_name(row: pd.Series) -> str:
    return f"lat_{row['grid_lat']:.2f}_lon_{row['grid_lon']:.2f}"


def _split_conformal_interval(
    y_cal: np.ndarray,
    p_cal: np.ndarray,
    p_test: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Distribution-free split conformal interval on probabilities:
    score = |y - p|, q = (1-alpha)-quantile(score), interval = [p-q, p+q].
    """
    scores = np.abs(y_cal.astype(float) - p_cal.astype(float))
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration set is empty; cannot build conformal interval.")

    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(scores, q_level, method="higher"))
    lower = np.clip(p_test - q_hat, 0.0, 1.0)
    upper = np.clip(p_test + q_hat, 0.0, 1.0)
    return lower, upper, q_hat


def run_conformal_pipeline(
    df: pd.DataFrame,
    alpha: float = 0.1,
    feature_cols: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    feature_cols = feature_cols or DEFAULT_FEATURE_COLS
    train_df, cal_df, test_df = _chronological_three_way_split(df)

    X_train, y_train, cols = prepare_xy(train_df, feature_cols, "bloom_label")
    X_cal, y_cal, _ = prepare_xy(cal_df, feature_cols, "bloom_label")
    X_test, y_test, _ = prepare_xy(test_df, feature_cols, "bloom_label")

    if min(len(X_train), len(X_cal), len(X_test)) == 0:
        raise ValueError("Empty train/cal/test after preprocessing; adjust date ranges.")

    imputer = SimpleImputer(strategy="median")
    X_train_i = pd.DataFrame(imputer.fit_transform(X_train), columns=cols, index=X_train.index)
    X_cal_i = pd.DataFrame(imputer.transform(X_cal), columns=cols, index=X_cal.index)
    X_test_i = pd.DataFrame(imputer.transform(X_test), columns=cols, index=X_test.index)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    scale = min((neg / pos) if pos > 0 else 1.0, 1e6)
    clf, backend = _fit_classifier(X_train_i, y_train, random_state=random_state, scale_pos_weight=scale)

    p_cal = clf.predict_proba(X_cal_i)[:, 1]
    p_test = clf.predict_proba(X_test_i)[:, 1]

    lower, upper, q_hat = _split_conformal_interval(
        y_cal=y_cal.to_numpy(),
        p_cal=p_cal,
        p_test=p_test,
        alpha=alpha,
    )

    # Empirical coverage of true labels by predicted intervals on test set.
    y_test_np = y_test.to_numpy().astype(float)
    coverage = float(np.mean((y_test_np >= lower) & (y_test_np <= upper)))

    test_meta = test_df.loc[X_test.index, ["grid_lat", "grid_lon", "week_start"]].copy()
    out_df = pd.DataFrame(
        {
            "zone": test_meta.apply(_zone_name, axis=1),
            "date": pd.to_datetime(test_meta["week_start"]),
            "prob": p_test,
            "lower": lower,
            "upper": upper,
        }
    )
    out_df["recommended_action"] = out_df["prob"].apply(recommended_action)

    return {
        "backend": backend,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "empirical_coverage": coverage,
        "q_hat": q_hat,
        "predictions": out_df.sort_values(["date", "zone"]).reset_index(drop=True),
    }


def save_conformal_predictions(
    pred_df: pd.DataFrame,
    out_path: Path | str | None = None,
) -> Path:
    out_path = Path(out_path) if out_path else PROJECT_ROOT / "data" / "processed" / "conformal_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run conformal prediction wrapper for bloom risk.")
    p.add_argument("--alpha", type=float, default=0.1, help="Miscoverage level (default: 0.1 => 90%% nominal).")
    p.add_argument("--argo", action="store_true", help="Try loading Argo data for alignment.")
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "conformal_predictions.csv",
        help="Output CSV path for per-zone conformal predictions.",
    )
    args = p.parse_args()

    loader = DataLoader()
    unified = loader.align_datasets(load_argo_if_missing=args.argo)
    res = run_conformal_pipeline(unified, alpha=args.alpha)
    out_csv = save_conformal_predictions(res["predictions"], out_path=args.out)

    print("Backend:", res["backend"])
    print("Nominal coverage:", f"{res['nominal_coverage']:.3f}")
    print("Empirical coverage:", f"{res['empirical_coverage']:.3f}")
    print("Conformal q_hat:", f"{res['q_hat']:.4f}")
    print("Rows written:", len(res["predictions"]))
    print("Output:", out_csv)
