# Loan Default Risk & Approval Assistant

An end-to-end credit-risk system trained on **real historical credit-bureau
data**: predicts probability of default, converts it into a FICO-style
credit score, recommends **Approve / Manual Review / Reject** using an
expected-loss economics framework (not arbitrary thresholds), explains every
decision with SHAP, audits itself for age-based disparate impact, and
includes a production drift-monitoring check — all served through a
Streamlit app.

## Why this version is different from a typical portfolio project

Most "loan default predictor" projects stop at training a classifier on
synthetic or lightly-cleaned data and calling `predict_proba`. This one
tries to cover the parts that actually matter in a real lending
operation, and is explicit about where it still falls short of
production-grade:

| Component | What it does |
|---|---|
| **Real data, not synthetic** | Trained on the ["Give Me Some Credit"](https://www.kaggle.com/competitions/GiveMeSomeCredit) dataset — 150,000 real, anonymized US credit-bureau records from a 2011 industry competition. AUC (~0.87) lines up closely with published benchmarks for this exact dataset, which is a good sign the pipeline isn't overfit to its own quirks. |
| **Real data cleaning** | The raw data has genuine messiness: sentinel error codes (96/98) in delinquency counters, an impossible age=0 row, extreme outliers in utilization/debt ratio, and ~20% missing income. `01_data_preparation.py` handles each explicitly, with missingness indicator flags rather than silent imputation. |
| **Proper validation** | 5-fold stratified cross-validation reported as mean ± std, not a single train/test split. Probability **calibration** (isotonic) is checked explicitly via Brier score, since raw probabilities feed directly into pricing. |
| **Scorecard-style score** | Converts model probability into a 300-900 score using points-to-double-odds (PDO) scaling — the same log-odds transform real bureaus use, not just `probability * 850`. |
| **Fairness / disparate-impact audit** | Age is a protected class under the US Equal Credit Opportunity Act. `03_fairness_audit.py` runs the four-fifths rule and checks for performance parity (AUC) across age groups — and it genuinely surfaces a borderline finding (see below), not a manufactured clean result. |
| **Cost-based decision thresholds** | Approve/Reject/Review cutoffs come from a PD × LGD × EAD expected-loss framework with risk-based pricing and a hard risk-appetite ceiling — not round numbers picked by feel. `04_threshold_optimization.py` also cross-checks this against a pure classification-cost-minimizing threshold sweep. |
| **Drift monitoring** | `05_monitoring.py` implements Population Stability Index (PSI), the standard credit-risk drift metric, including a simulated-drift worked example that proves the check actually fires (not just reporting "all stable" trivially). |
| **Honest limitations** | Called out explicitly below and in the code — this is not a finished production system, and pretending otherwise would undercut the parts that are genuinely solid. |

## Project structure

```
data/
  gmsc_raw.csv           real, uncleaned "Give Me Some Credit" data (150,000 rows)
  gmsc_clean.csv          after 01_data_preparation.py
artifacts/                 model, calibrator, SHAP explainer, thresholds, metadata
reports/                    fairness audit, PSI monitoring, threshold cost sweep

01_data_preparation.py     cleans raw data (sentinel codes, outliers, missing values)
02_train_model.py          trains XGBoost, 5-fold CV, isotonic calibration, SHAP, scorecard
03_fairness_audit.py       disparate-impact audit (four-fifths rule) + performance parity by age
04_threshold_optimization.py   PD x LGD x EAD expected-loss framework -> decision thresholds
05_monitoring.py           Population Stability Index drift monitoring + simulated-drift demo
app.py                     Streamlit app tying it all together (3 tabs)
```

## Running it

```bash
pip install streamlit scikit-learn xgboost shap pandas numpy plotly joblib

# Full pipeline, in order (only needed if you want to retrain from scratch —
# artifacts/ and reports/ are already populated so you can skip straight to the app):
python 01_data_preparation.py
python 02_train_model.py
python 03_fairness_audit.py
python 04_threshold_optimization.py
python 05_monitoring.py

streamlit run app.py
```

## The app has three tabs

1. **Assess Applicant** — enter an applicant's credit profile and requested
   loan; get default probability, a 300-900 score, risk tier, recommendation,
   SHAP-based explanation (both plain-language and a chart), and the
   expected-loss/revenue math for that specific loan.
2. **Model Governance & Fairness** — cross-validated performance metrics,
   the calibration improvement from isotonic scaling, and the full
   age-based disparate-impact audit (including the flagged borderline
   group, shown honestly rather than hidden).
3. **Business Impact & Monitoring** — how the decision thresholds were
   derived from unit economics, the expected population split, and the
   PSI drift-monitoring report (with a worked drift example).

## A genuine (not manufactured) finding worth highlighting

The fairness audit flags the **25-39 age group** at a four-fifths ratio of
**0.793** — just under the 0.80 threshold. This isn't a bug to "fix" by
tweaking the model; the real observed default rate for that group in this
data is genuinely higher (11.0%) than for, say, the 62+ group (2.5%). ECOA
allows disparate impact justified by legitimate business necessity, but
requires it to be **documented and monitored**, not quietly present without
anyone noticing. That's exactly the kind of nuanced, real-world finding a
synthetic dataset would never surface, and exactly the kind of thing a
fair-lending review would need to examine before deployment.

## Honest limitations (read before using this as "production-ready")

- **No true out-of-time validation.** This dataset has no origination-date
  field, so we can't backtest "train on 2022-23, test on 2024" the way a
  real deployment should. Cross-validation shows stability across random
  samples, not through an economic cycle. This is stated explicitly in the
  app and in `02_train_model.py`, not glossed over.
- **Fairness audit covers age only.** No race/gender/ethnicity fields exist
  in this dataset. A real compliance review needs those, run by people with
  fair-lending expertise — not just an engineering team running a script.
- **LGD, APR, and risk-appetite figures are illustrative assumptions**
  (see `04_threshold_optimization.py`), not this institution's actual cost
  of capital, collections recovery rates, or board-approved risk appetite.
  They're structured the way a real bank would structure them, but the
  numbers themselves would need to come from Finance/Treasury and Risk in
  a real deployment.
- **PSI monitoring is demonstrated on a proxy** (train vs. held-out test
  split, plus a simulated drift scenario) since there's no genuinely
  time-separated "new" population in this dataset. In production this same
  function would run monthly against real new-applicant batches.
- **This is a decision-support demo, not a real credit decision engine.**
  Don't use it to actually approve or reject loans without independent
  validation against your own data, sign-off from risk/compliance, and
  a full fair-lending review.

## Data source & attribution

["Give Me Some Credit"](https://www.kaggle.com/competitions/GiveMeSomeCredit),
a 2011 Kaggle competition dataset of anonymized US credit-bureau records,
used here for its historical default-outcome labels (`SeriousDlqin2yrs`:
90+ days past due within two years).
