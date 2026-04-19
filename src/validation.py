"""
Coastal Shield — Step 8 historical validation helper.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.data_loader import PROJECT_ROOT

EVENTS_CSV = PROJECT_ROOT / "data" / "raw" / "historical_events.csv"
PRED_CSV = PROJECT_ROOT / "data" / "processed" / "conformal_predictions.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "historical_validation.csv"


def validate_historical_events(
    events_csv: Path | str = EVENTS_CSV,
    pred_csv: Path | str = PRED_CSV,
    lookback_days: int = 14,
) -> pd.DataFrame:
    events = pd.read_csv(events_csv, parse_dates=["event_date"])
    preds = pd.read_csv(pred_csv, parse_dates=["date"])

    rows = []
    for _, e in events.iterrows():
        zone = e["zone"]
        d0 = pd.Timestamp(e["event_date"]) - pd.Timedelta(days=lookback_days)
        d1 = pd.Timestamp(e["event_date"])

        cand = preds[
            (preds["zone"] == zone)
            & (preds["date"] >= d0)
            & (preds["date"] <= d1)
        ].sort_values("date")

        if len(cand) == 0:
            rows.append(
                {
                    "event_id": e["event_id"],
                    "event_name": e["event_name"],
                    "zone": zone,
                    "event_date": e["event_date"],
                    "fired_14d_prior": False,
                    "prediction_date": pd.NaT,
                    "days_before_event": None,
                    "prob": None,
                    "lower": None,
                    "upper": None,
                    "recommended_action": "No prediction",
                    "source": e.get("source", ""),
                }
            )
            continue

        best = cand.sort_values("prob", ascending=False).iloc[0]
        pred_date = pd.Timestamp(best["date"])
        rows.append(
            {
                "event_id": e["event_id"],
                "event_name": e["event_name"],
                "zone": zone,
                "event_date": e["event_date"],
                "fired_14d_prior": bool(best["prob"] >= 0.3),
                "prediction_date": pred_date,
                "days_before_event": int((pd.Timestamp(e["event_date"]) - pred_date).days),
                "prob": float(best["prob"]),
                "lower": float(best["lower"]),
                "upper": float(best["upper"]),
                "recommended_action": best["recommended_action"],
                "source": e.get("source", ""),
            }
        )

    out = pd.DataFrame(rows).sort_values(["event_date", "event_id"]).reset_index(drop=True)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Validate whether model would have fired ~14 days before known events.")
    p.add_argument("--events", type=Path, default=EVENTS_CSV)
    p.add_argument("--predictions", type=Path, default=PRED_CSV)
    p.add_argument("--out", type=Path, default=OUT_CSV)
    p.add_argument("--lookback-days", type=int, default=14)
    args = p.parse_args()

    out = validate_historical_events(
        events_csv=args.events,
        pred_csv=args.predictions,
        lookback_days=args.lookback_days,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("Historical validation rows:", len(out))
    print(out[["event_name", "event_date", "zone", "fired_14d_prior", "prob", "recommended_action"]].to_string(index=False))
    print("Output:", args.out)
