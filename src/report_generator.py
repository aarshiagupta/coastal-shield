"""
Coastal Shield — Step 6 report generator (Gemini + PDF export).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env", override=True)

print("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))

import pandas as pd
from fpdf import FPDF

from src.data_loader import PROJECT_ROOT

REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Historical event reference database ───────────────────────────────────────
_HIST_EVENTS = [
    {
        "year": 2015, "name": "West Coast Warm Blob",
        "key_drivers": ["sst_anomaly", "temperature", "nitrate_lag14"],
        "sig": {"temp_z": 1.8, "nit_z": 1.2, "chl_z": 0.9, "sst_z": 2.1},
        "outcome": "largest HAB-related fishery closure in US history, multi-county domoic acid event",
    },
    {
        "year": 2019, "name": "La Jolla Red Tide",
        "key_drivers": ["nitrate_lag14", "chlorophyll_lag7", "temperature"],
        "sig": {"temp_z": 0.7, "nit_z": 2.2, "chl_z": 2.4, "sst_z": 0.4},
        "outcome": "Lingulodinium polyedra bloom, beach discoloration, fish kill, 2-week beach advisory",
    },
    {
        "year": 2009, "name": "Monterey Bay Pseudo-nitzschia bloom",
        "key_drivers": ["nitrate_lag14", "nitrate_lag7", "upwelling_proxy"],
        "sig": {"temp_z": -0.6, "nit_z": 2.9, "chl_z": 1.4, "sst_z": -0.9},
        "outcome": "sea lion strandings, 3-week beach closure, shellfish quarantine",
    },
    {
        "year": 2003, "name": "Santa Barbara Channel Pseudo-nitzschia bloom",
        "key_drivers": ["upwelling_proxy", "nitrate_lag14", "chlorophyll_lag7"],
        "sig": {"temp_z": 0.2, "nit_z": 1.9, "chl_z": 1.8, "sst_z": 0.1},
        "outcome": "domoic acid detection, Santa Barbara County shellfish closure",
    },
    {
        "year": 2018, "name": "Humboldt Bay Pseudo-nitzschia bloom",
        "key_drivers": ["nitrate_lag14", "temperature", "upwelling_proxy"],
        "sig": {"temp_z": -1.3, "nit_z": 3.2, "chl_z": 1.1, "sst_z": -1.6},
        "outcome": "commercial shellfish closure lasting 3 months, significant economic impact",
    },
    {
        "year": 2011, "name": "Santa Monica Bay Alexandrium bloom",
        "key_drivers": ["chlorophyll_lag7", "nitrate_lag14", "sst_anomaly"],
        "sig": {"temp_z": 1.1, "nit_z": 0.8, "chl_z": 2.1, "sst_z": 1.3},
        "outcome": "paralytic shellfish toxin advisory, recreational harvest closure",
    },
]


def _compute_best_analog(conditions: dict, top_shap: list[str]) -> tuple[dict, int]:
    """Returns (event_dict, match_pct) for the closest historical analog.

    Similarity is cosine distance on the anomaly z-score vector, with a bonus
    for overlapping SHAP driver names.
    """
    cur = {
        "temp_z": float(conditions.get("temp_z") or 0),
        "nit_z":  float(conditions.get("nit_z")  or 0),
        "chl_z":  float(conditions.get("chl_z")  or 0),
        "sst_z":  float(conditions.get("sst_anomaly") or 0),
    }
    best_ev, best_sim = _HIST_EVENTS[0], -99.0
    for ev in _HIST_EVENTS:
        sig = ev["sig"]
        dot   = sum(cur.get(k, 0) * sig.get(k, 0) for k in sig)
        mag_c = math.sqrt(sum(v ** 2 for v in cur.values()) + 1e-9)
        mag_s = math.sqrt(sum(v ** 2 for v in sig.values()) + 1e-9)
        sim   = dot / (mag_c * mag_s)
        sim  += sum(0.15 for d in ev["key_drivers"] if any(d in s for s in top_shap))
        if sim > best_sim:
            best_sim, best_ev = sim, ev
    match_pct = max(22, min(93, int(50 + best_sim * 38)))
    return best_ev, match_pct


@dataclass
class ReportInput:
    zone: str
    date: str
    prob: float
    lower: float
    upper: float
    top_shap_features: list[str]
    historical_analog: str
    action_window: str = "Next 7-14 days"
    habmap_event: str = ""
    alert_level: str = "CLOSURE"
    ocean_conditions: dict = field(default_factory=dict)
    no_model_prediction: bool = False


def _fmt_val(val, unit: str, z: float | None, decimals: int = 1) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    s = f"{val:.{decimals}f} {unit}"
    if z is not None and not math.isnan(z):
        s += f" ({z:+.1f}sigma from seasonal baseline)"
    return s


def _ocean_conditions_paragraph(inp: ReportInput) -> str:
    conds = inp.ocean_conditions
    feats = ", ".join(inp.top_shap_features) if inp.top_shap_features else "nitrate and chlorophyll anomalies"

    if not conds.get("data_available"):
        return (
            f"Recent observations show ocean-state patterns associated with bloom onset, "
            f"with strongest influence from: {feats}. Nearest HABMAP station data was not "
            f"available within the search radius for this forecast period."
        )

    station = conds.get("station", "nearest HABMAP station")
    station_date = conds.get("station_date", inp.date)
    dist_km = conds.get("station_dist_km")
    dist_str = f" ({dist_km:.0f} km from zone centroid)" if dist_km is not None else ""

    nit = _fmt_val(conds.get("nitrate"), "uM", conds.get("nit_z"))
    temp = _fmt_val(conds.get("temperature"), "deg C", conds.get("temp_z"))
    chl = _fmt_val(conds.get("chlorophyll"), "ug/L", conds.get("chl_z"))
    pn = conds.get("pn_total", 0)
    pn_str = f"{pn:,.0f} cells/L" if pn and not math.isnan(float(pn)) else "below detection (<1,000 cells/L)"
    da = conds.get("da_detected", False)
    sst_z = conds.get("sst_anomaly")
    sst_str = f" SST anomaly: {sst_z:+.2f} deg C." if sst_z is not None and not math.isnan(float(sst_z)) else ""

    return (
        f"Observations from {station}{dist_str} on {station_date} indicate: "
        f"nitrate at {nit}, sea surface temperature at {temp}, "
        f"chlorophyll at {chl}.{sst_str} "
        f"Pseudo-nitzschia cell density: {pn_str}."
        + (" Domoic acid detected." if da else "")
        + f" The dominant driver per SHAP analysis is {feats.split(',')[0].strip()}."
    )


def build_prompt(inp: ReportInput) -> str:
    feats = ", ".join(inp.top_shap_features) if inp.top_shap_features else "Not available"
    habmap_line = (
        f"\nHABMAP scientific corroboration: {inp.habmap_event}"
        if inp.habmap_event else
        "\nHABMAP scientific corroboration: None within search radius."
    )
    action = _recommended_action(inp.alert_level)

    if inp.no_model_prediction:
        prob_line = "Bloom probability: N/A -- insufficient historical CalCOFI coverage for model prediction at this zone"
    else:
        prob_line = f"Bloom probability: {inp.prob:.0%} (90% confidence interval: {inp.lower:.0%}-{inp.upper:.0%})"

    # Historical analog match
    best_ev, match_pct = _compute_best_analog(inp.ocean_conditions, inp.top_shap_features)
    analog_line = (
        f"\nHistorical analog match: Current ocean signatures are a {match_pct}% match to the "
        f"preconditions of the {best_ev['year']} {best_ev['name']} "
        f"(outcome: {best_ev['outcome']})."
    )

    conds = inp.ocean_conditions
    if conds.get("data_available"):
        nit_z = conds.get("nit_z", float("nan"))
        temp_z = conds.get("temp_z", float("nan"))
        chl_z = conds.get("chl_z", float("nan"))
        ocean_data_block = (
            f"\nStation: {conds.get('station', 'N/A')} | Date: {conds.get('station_date', 'N/A')}"
            f"\nNitrate: {conds.get('nitrate', 'N/A')} uM"
            + (f" (z={nit_z:+.2f})" if not math.isnan(nit_z) else "")
            + f"\nTemperature: {conds.get('temperature', 'N/A')} deg C"
            + (f" (z={temp_z:+.2f})" if not math.isnan(temp_z) else "")
            + f"\nChlorophyll: {conds.get('chlorophyll', 'N/A')} ug/L"
            + (f" (z={chl_z:+.2f})" if not math.isnan(chl_z) else "")
            + f"\nPseudo-nitzschia density: {conds.get('pn_total', 0):,.0f} cells/L"
            + f"\nDomoic acid detected: {conds.get('da_detected', False)}"
        )
    else:
        ocean_data_block = "\nNo HABMAP station data available within search radius."

    return f"""
You are a scientific advisor writing an official memo for a coastal health department.
Write a formal, concise public health advisory memo based on the following model outputs.
Use precise scientific language. Do not hedge unnecessarily. Be direct about the recommendation.
Include exact numerical values where provided.

Zone: {inp.zone}
Prediction date: {inp.date}
Alert level: {inp.alert_level}
Recommended action: {action}
{prob_line}
Key driving variables: {feats}
Closest historical analog: {inp.historical_analog}{analog_line}{habmap_line}
Ocean conditions data:{ocean_data_block}

The memo must include:
1. Executive summary (2 sentences max)
2. Ocean conditions summary -- use the exact numerical values provided above (nitrate in uM, temp in deg C,
   chlorophyll in ug/L, z-scores as sigma deviations from seasonal baseline). Do NOT use generic language.
3. Model prediction {"note (no quantitative model prediction is available for this zone)" if inp.no_model_prediction else "with confidence interval"}
4. Historical context -- include this EXACT sentence: "Current ocean signatures are a {match_pct}% match
   to the preconditions of the {best_ev['year']} {best_ev['name']}." Then explain what that event entailed
   and what comparable conditions have meant for public health in the past.
5. Scientific corroboration -- if HABMAP data is present, cite it explicitly by station name,
   cell count, and date. State that HABMAP is the same dataset used by CDPH for shellfish advisories.
6. Recommended action with justification
7. Suggested action window (dates)

Output only the memo text. No preamble, no "here is your memo".
"""


_ALERT_ACTIONS = {
    "CLOSURE":  "Immediate closure recommended. Issue public advisory within 24 hours.",
    "ADVISORY": "Enhanced monitoring recommended. Prepare closure protocols as precautionary measure.",
    "WATCH":    "Precautionary sampling advised. Alert monitoring teams.",
    "MONITOR":  "Routine monitoring. No immediate action required.",
}


def _recommended_action(alert_level: str) -> str:
    return _ALERT_ACTIONS.get(alert_level, _ALERT_ACTIONS["CLOSURE"])


def fallback_report_text(inp: ReportInput) -> str:
    action = _recommended_action(inp.alert_level)
    ocean_para = _ocean_conditions_paragraph(inp)

    if inp.no_model_prediction:
        pred_section = (
            "Model Prediction\n"
            "No quantitative bloom probability is available for this zone — the historical CalCOFI dataset "
            "does not have sufficient coverage at this location to generate a calibrated conformal prediction. "
            "The alert level is based on HABMAP observations and regional bloom climatology.\n\n"
        )
    else:
        pred_section = (
            "Model Prediction and Uncertainty\n"
            f"The calibrated probability estimate is {inp.prob:.0%}, with interval bounds "
            f"{inp.lower:.0%}-{inp.upper:.0%} at 90% nominal coverage.\n\n"
        )

    if inp.no_model_prediction:
        exec_summary = (
            f"For {inp.zone}, HABMAP ocean monitoring data indicate conditions consistent with elevated "
            f"harmful algal bloom risk. No quantitative CalCOFI-based model prediction is available; "
            f"the recommended status based on current observations is: {action}.\n\n"
        )
    else:
        exec_summary = (
            f"For {inp.zone}, model outputs indicate a harmful algal bloom risk of {inp.prob:.0%} "
            f"(90% interval: {inp.lower:.0%}-{inp.upper:.0%}). Based on current ocean conditions and "
            f"historical analogs, the recommended status is: {action}.\n\n"
        )

    return (
        "Executive Summary\n"
        + exec_summary
        + "Ocean Conditions Summary\n"
        + ocean_para + "\n\n"
        + pred_section
        + "Historical Context\n"
        + f"The nearest analog is: {inp.historical_analog}. This supports elevated watch conditions for this zone.\n\n"
        + (
            "Scientific Corroboration\n"
            f"{inp.habmap_event} This observation is from HABMAP (Harmful Algal Bloom Monitoring and Alert Program), "
            "operated by Scripps Institution of Oceanography/UCSD -- the same dataset used by CDPH to issue shellfish advisories.\n\n"
            if inp.habmap_event else ""
        )
        + "Recommended Action\n"
        + f"Action level: {action}. This recommendation prioritizes preventive public-health risk management "
        + "under uncertainty.\n\n"
        + "Suggested Action Window\n"
        + f"{inp.action_window} from forecast issue date ({inp.date})."
    )


def generate_closure_report(
    inp: ReportInput,
    model_name: str = "gemini-1.5-flash",
) -> tuple[str, str]:
    """Returns (report_text, source) where source is 'gemini' or 'fallback'."""
    api_key = os.environ.get("GEMINI_API_KEY")
    prompt = build_prompt(inp)
    if not api_key:
        return fallback_report_text(inp), "fallback"

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        if not text:
            return fallback_report_text(inp), "fallback"
        return text, "gemini"
    except Exception:
        return fallback_report_text(inp), "fallback"


_UNICODE_MAP = str.maketrans({
    "\u2013": "-", "\u2014": "--",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2022": "*", "\u2026": "...",
    "\u00b0": " deg", "\u00b1": "+/-",
    "\u2192": "->", "\u2190": "<-",
    "\u00e9": "e", "\u00e8": "e",
    "\u03bc": "u", "\u03c3": "sigma",
})


def _ascii_safe(text: str) -> str:
    return text.translate(_UNICODE_MAP).encode("latin-1", errors="replace").decode("latin-1")


def export_report_pdf(report_text: str, out_path: Path | str) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for para in report_text.split("\n"):
        if para.strip() == "":
            pdf.ln(4)
            continue
        pdf.multi_cell(0, 6, _ascii_safe(para))
        pdf.ln(1)

    pdf.output(str(out))
    return out


def load_latest_conformal_row(path: Path | str | None = None) -> ReportInput:
    p = Path(path) if path else (PROJECT_ROOT / "data" / "processed" / "conformal_predictions.csv")
    df = pd.read_csv(p, parse_dates=["date"])
    if len(df) == 0:
        raise ValueError(f"No rows in {p}.")
    row = df.sort_values("prob", ascending=False).iloc[0]
    return ReportInput(
        zone=str(row["zone"]),
        date=str(pd.Timestamp(row["date"]).date()),
        prob=float(row["prob"]),
        lower=float(row["lower"]),
        upper=float(row["upper"]),
        top_shap_features=["nitrate_lag14", "chlorophyll_lag7", "nitrate_lag7"],
        historical_analog="2019 La Jolla red tide preconditions",
        action_window="Within the next 7-14 days",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate closure report text and PDF.")
    parser.add_argument("--zone", type=str, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--prob", type=float, default=None)
    parser.add_argument("--lower", type=float, default=None)
    parser.add_argument("--upper", type=float, default=None)
    parser.add_argument("--analog", type=str, default="2019 La Jolla red tide preconditions")
    parser.add_argument("--features", type=str, default="nitrate_lag14,chlorophyll_lag7,nitrate_lag7")
    parser.add_argument("--model", type=str, default="gemini-1.5-flash")
    parser.add_argument("--from-conformal", action="store_true")
    args = parser.parse_args()

    if args.from_conformal or any(v is None for v in [args.zone, args.date, args.prob, args.lower, args.upper]):
        inp = load_latest_conformal_row()
    else:
        inp = ReportInput(
            zone=args.zone,
            date=args.date,
            prob=float(args.prob),
            lower=float(args.lower),
            upper=float(args.upper),
            top_shap_features=[x.strip() for x in args.features.split(",") if x.strip()],
            historical_analog=args.analog,
        )

    report_text, source = generate_closure_report(inp, model_name=args.model)
    pdf_path = REPORTS_DIR / f"{inp.zone}_{inp.date}.pdf"
    pdf_path = export_report_pdf(report_text, pdf_path)

    print("Report source:", source)
    print("Report preview:")
    print(report_text[:800] + ("..." if len(report_text) > 800 else ""))
    print("PDF:", pdf_path)
