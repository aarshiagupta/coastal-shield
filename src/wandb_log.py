"""
Coastal Shield — Step 9: log baseline, Chronos hybrid, conformal, and validation to Weights & Biases.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from src.conformal import run_conformal_pipeline
from src.data_loader import DataLoader, PROJECT_ROOT
from src.model import train_chronos_hybrid, train_xgb_baseline

HIST_VAL = PROJECT_ROOT / "data" / "processed" / "historical_validation.csv"


def _fig_to_image(fig: plt.Figure) -> Any:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    try:
        import wandb

        return wandb.Image(buf)
    except Exception:
        return None


def run_wandb_dashboard(
    load_argo: bool = False,
    project: str = "coastal-shield",
    run_name: str | None = None,
    alpha: float = 0.1,
) -> None:
    import wandb

    # Keep W&B cache/staging inside the repo (CI, sandboxes, shared machines).
    wb_home = PROJECT_ROOT / ".wandb"
    wb_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DATA_DIR", str(wb_home))
    os.environ.setdefault("WANDB_DIR", str(wb_home))

    loader = DataLoader()
    df = loader.align_datasets(load_argo_if_missing=load_argo)

    baseline = train_xgb_baseline(df, use_wandb=False)

    chronos: dict[str, Any] = {}
    try:
        chronos = train_chronos_hybrid(df, use_wandb=False)
    except Exception as e:
        chronos = {"error": str(e), "roc_auc": float("nan"), "pr_auc": float("nan"), "f1_at_0.3": float("nan")}

    conf = run_conformal_pipeline(df, alpha=alpha)

    wandb.init(project=project, name=run_name, job_type="step9_dashboard")

    # --- Scalars ---
    wandb.log(
        {
            "baseline/roc_auc": baseline.get("roc_auc", float("nan")),
            "baseline/pr_auc": baseline.get("pr_auc", float("nan")),
            "baseline/f1_at_0.3": baseline.get("f1_at_0.3", float("nan")),
            "baseline/n_train": baseline.get("n_train", 0),
            "baseline/n_test": baseline.get("n_test", 0),
            "baseline/backend": baseline.get("backend", ""),
            "chronos_hybrid/roc_auc": chronos.get("roc_auc", float("nan")),
            "chronos_hybrid/pr_auc": chronos.get("pr_auc", float("nan")),
            "chronos_hybrid/f1_at_0.3": chronos.get("f1_at_0.3", float("nan")),
            "chronos_hybrid/backend": chronos.get("backend", ""),
            "conformal/nominal_coverage": conf["nominal_coverage"],
            "conformal/empirical_coverage": conf["empirical_coverage"],
            "conformal/q_hat": conf["q_hat"],
            "conformal/alpha": alpha,
        }
    )

    # Precision–recall curve (baseline test set)
    y_test = baseline["y_test"]
    y_proba = baseline["y_proba"]
    if len(np.unique(y_test)) > 1:
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc_plot = auc(rec, prec)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rec, prec, color="steelblue", lw=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Baseline PR curve (AUC={pr_auc_plot:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        img = _fig_to_image(fig)
        if img is not None:
            wandb.log({"charts/pr_curve_baseline": img})
        wandb.log({"baseline/pr_auc_curve_integral": float(pr_auc_plot)})

    # Feature importance bar chart
    fi = baseline.get("feature_importance")
    if fi is not None and len(fi):
        fig, ax = plt.subplots(figsize=(8, 5))
        fi.head(15).plot(kind="barh", ax=ax, color="darkslategray")
        ax.set_xlabel("Importance")
        ax.set_title("Baseline feature importance (top 15)")
        ax.invert_yaxis()
        img = _fig_to_image(fig)
        if img is not None:
            wandb.log({"charts/feature_importance_baseline": img})

    # Calibration: nominal vs empirical coverage
    fig, ax = plt.subplots(figsize=(5, 4))
    nom = conf["nominal_coverage"]
    emp = conf["empirical_coverage"]
    ax.bar(["Nominal (1-α)", "Empirical (test)"], [nom, emp], color=["#2c3e50", "#e67e22"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_title("Conformal interval coverage")
    img = _fig_to_image(fig)
    if img is not None:
        wandb.log({"charts/calibration_coverage": img})

    # Ablation table (hackathon-style)
    ablation = wandb.Table(
        columns=["model", "roc_auc", "pr_auc", "f1_at_0.3", "notes"],
        data=[
            [
                "baseline_tabular",
                baseline.get("roc_auc", np.nan),
                baseline.get("pr_auc", np.nan),
                baseline.get("f1_at_0.3", np.nan),
                "XGBoost or sklearn fallback on engineered features",
            ],
            [
                "chronos_hybrid",
                chronos.get("roc_auc", np.nan),
                chronos.get("pr_auc", np.nan),
                chronos.get("f1_at_0.3", np.nan),
                "Sequence-flatten + same classifier; Chronos checkpoint optional",
            ],
            [
                "baseline_plus_conformal",
                baseline.get("roc_auc", np.nan),
                baseline.get("pr_auc", np.nan),
                baseline.get("f1_at_0.3", np.nan),
                f"Same base model; intervals: nominal {nom:.2f}, empirical {emp:.2f}",
            ],
        ],
    )
    wandb.log({"tables/ablation": ablation})

    if HIST_VAL.exists():
        vdf = pd.read_csv(HIST_VAL)
        hit_rate = float(vdf["fired_14d_prior"].mean()) if len(vdf) and "fired_14d_prior" in vdf.columns else 0.0
        wandb.log({"validation/historical_hit_rate": hit_rate, "validation/n_events": len(vdf)})
        vt = wandb.Table(
            columns=list(vdf.columns),
            data=vdf.values.tolist(),
        )
        wandb.log({"tables/historical_validation": vt})

    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Log Coastal Shield metrics and charts to W&B (Step 9).")
    p.add_argument("--argo", action="store_true", help="Load Argo when aligning data")
    p.add_argument("--project", type=str, default="coastal-shield")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--alpha", type=float, default=0.1, help="Conformal miscoverage level")
    args = p.parse_args()

    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        print("Tip: set WANDB_API_KEY, or run with WANDB_MODE=offline for a local run.")

    run_wandb_dashboard(
        load_argo=args.argo,
        project=args.project,
        run_name=args.run_name,
        alpha=args.alpha,
    )
    print("W&B run finished.")
