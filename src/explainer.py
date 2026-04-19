"""
Coastal Shield — Step 5 SHAP explainability artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_loader import DataLoader, PROJECT_ROOT
from src.model import train_xgb_baseline

SHAP_OUT_DIR = PROJECT_ROOT / "assets" / "shap_plots"


def _zone_from_row(row: pd.Series) -> str:
    return f"lat_{row['grid_lat']:.2f}_lon_{row['grid_lon']:.2f}"


def compute_shap_for_top_predictions(
    unified: pd.DataFrame,
    top_k: int = 3,
    out_dir: Path | str = SHAP_OUT_DIR,
) -> dict[str, Any]:
    """
    Train baseline model, find top-k high-risk test predictions, and save SHAP waterfall PNGs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = train_xgb_baseline(unified)
    model = res["model"]
    X_test: pd.DataFrame = res["X_test"]
    y_proba = np.asarray(res["y_proba"])

    if len(X_test) == 0:
        raise ValueError("No test rows available for SHAP plotting.")

    # Recover metadata rows aligned with X_test index.
    meta = unified.loc[X_test.index, ["grid_lat", "grid_lon", "week_start"]].copy()
    ranked = pd.DataFrame(
        {
            "idx": X_test.index,
            "prob": y_proba,
            "zone": meta.apply(_zone_from_row, axis=1).values,
            "week_start": pd.to_datetime(meta["week_start"]).values,
        }
    ).sort_values("prob", ascending=False)

    top = ranked.head(top_k).reset_index(drop=True)

    # Generic explainer handles both tree and non-tree fallback models.
    bg = shap.sample(X_test, min(100, len(X_test)), random_state=42)
    explainer = shap.Explainer(model.predict_proba, bg)
    shap_values = explainer(X_test)

    saved: list[Path] = []
    for i, row in top.iterrows():
        idx = row["idx"]
        # position in X_test / shap arrays
        pos = X_test.index.get_loc(idx)
        single = shap_values[pos, :, 1] if shap_values.values.ndim == 3 else shap_values[pos]

        fig = plt.figure(figsize=(10, 5))
        shap.waterfall_plot(single, max_display=10, show=False)
        fname = f"{row['zone']}_{pd.Timestamp(row['week_start']).date()}_rank{i+1}.png"
        fpath = out_dir / fname
        fig.savefig(fpath, dpi=140, bbox_inches="tight")
        plt.close(fig)
        saved.append(fpath)

    return {
        "backend": res["backend"],
        "top_predictions": top,
        "saved_plots": saved,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate SHAP waterfall plots for top-risk zones.")
    p.add_argument("--top-k", type=int, default=3, help="Number of top-risk predictions to explain.")
    p.add_argument("--argo", action="store_true", help="Try loading Argo data for alignment.")
    args = p.parse_args()

    loader = DataLoader()
    unified_df = loader.align_datasets(load_argo_if_missing=args.argo)
    out = compute_shap_for_top_predictions(unified_df, top_k=args.top_k)

    print("Backend:", out["backend"])
    print("Top predictions:")
    print(out["top_predictions"][["zone", "week_start", "prob"]].to_string(index=False))
    print("Saved SHAP plots:")
    for pth in out["saved_plots"]:
        print(pth)
