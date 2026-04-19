# 🌊 Coastal Shield

> **Beaches close after people get sick. Coastal Shield closes them before.**

Coastal Shield is a machine learning decision support system that predicts harmful algal bloom (HAB) risk along the California coast 14 days in advance, giving coastal health departments calibrated, audit-ready predictions with statistically guaranteed uncertainty bounds.

Built at **DataHacks 2026** · UC San Diego · Team: Aarshia Gupta & Ariha Shah

---

## The Problem

California beaches close reactively — after water testing confirms contamination, 48-72 hours after the bloom has already affected swimmers, shellfish, and marine mammals. The data to predict this 14 days early has existed since 1949.

The reason no predictive system exists for health officials isn't a data or model problem. It's a trust problem. A health official cannot defend a proactive beach closure based on "73% bloom risk." They can defend it based on a calibrated confidence interval, documented ocean variable attribution, and scientific corroboration — all in a format they can forward to their director. That's what Coastal Shield generates.

---

## What It Does

- **14-day HAB risk forecasts** per California coastal zone with conformal prediction confidence intervals (90% coverage guaranteed)
- **SHAP explainability** — which ocean variables drove each prediction and by how much
- **HABMAP corroboration** — cross-referenced against weekly Scripps/UCSD professional monitoring, the same data CDPH uses for shellfish advisories
- **Gemini-powered advisory memos** — formal PDF documents health officials can use to justify proactive closures
- **12-zone alert feed** — CLOSURE / ADVISORY / WATCH / MONITOR based on relative risk ranking

---

## Dashboard

The Streamlit dashboard shows all 12 monitored California coastal zones. Clicking any zone reveals:

- Ocean conditions radar chart normalized to the 70-year CalCOFI historical baseline
- Bloom risk gauge with conformal confidence interval
- SHAP waterfall feature attribution
- Key ocean metrics with σ anomalies vs seasonal baseline
- HABMAP scientific corroboration when available
- One-click Gemini advisory memo generation with PDF export

---

## ML Results

| Metric | Value |
|--------|-------|
| AUROC | 0.9987 |
| Recall | 92% |
| Precision | 49% |
| Training samples | 57,062 |
| Bloom base rate | 0.55% |
| Conformal coverage | 90% guaranteed |
| Top SHAP driver | NO₃ (Nitrate) |

---

## Datasets

All from **Scripps Institution of Oceanography**:

| Dataset | Coverage | Role |
|---------|----------|------|
| CalCOFI Bottle + Cast | 895,371 rows, 1949–2021 | Primary ML features + historical baseline |
| HABMAP | 9 CA stations, weekly since 2008 | Ground truth bloom labels + corroboration |
| iNaturalist | CA coast, research grade | Secondary bloom labels |

---

## Architecture

CalCOFI + HABMAP + iNaturalist
│
▼
Data Pipeline
Weekly 0.5° spatial grid
Feature engineering + lag features
│
▼
XGBoost Classifier
Chronological train/val/test split
│
▼
Conformal Prediction
90% coverage-guaranteed intervals
│
├──► SHAP Explainer → waterfall plots
├──► Gemini Memo Generator → PDF export
├──► W&B Experiment Tracking
└──► Streamlit Dashboard

---

## Installation

```bash
git clone https://github.com/your-username/coastal-shield
cd coastal-shield/coastal_shield
pip install -r requirements.txt
```

Download CalCOFI data from [calcofi.org](https://calcofi.org/data/oceanographic-data/bottle-database/) and place `194903-202105_Bottle.csv` and `194903-202105_Cast.csv` in `data/raw/calcofi/`.

Create a `.env` file in `coastal_shield/`:
GEMINI_API_KEY=your_key_here
WANDB_API_KEY=your_key_here

---

## Running the Pipeline

```bash
cd coastal_shield
export PYTHONPATH=.

python src/data_loader.py       # ingest + align all data
python src/model.py             # train XGBoost model
python src/conformal.py         # generate conformal predictions
python src/explainer.py         # generate SHAP plots
python src/wandb_log.py         # log to W&B
streamlit run app/main.py       # launch dashboard
```

---

## Project Structure

coastal_shield/
├── src/
│   ├── data_loader.py         # data ingestion + feature engineering
│   ├── model.py               # XGBoost + Chronos hybrid
│   ├── conformal.py           # split conformal prediction
│   ├── explainer.py           # SHAP waterfall generation
│   ├── report_generator.py    # Gemini memo + PDF export
│   ├── validation.py          # historical event backtest
│   └── wandb_log.py           # experiment tracking
├── app/
│   └── main.py                # Streamlit dashboard
├── data/
│   ├── raw/                   # CalCOFI, HABMAP, iNaturalist
│   └── processed/             # conformal predictions, validation
├── assets/shap_plots/         # zone-specific SHAP waterfalls
├── reports/                   # generated PDF memos
├── notebooks/01_eda.ipynb
└── requirements.txt

---

## Stack

Python · XGBoost · SHAP · Conformal Prediction · Streamlit · Plotly · Gemini API · Weights & Biases · fpdf2 · HuggingFace (Chronos) · CalCOFI · HABMAP · iNaturalist

---

## Built At

DS3 UC San Diego · 
---

*MIT License*
