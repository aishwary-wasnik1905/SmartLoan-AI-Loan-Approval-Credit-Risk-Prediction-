"""
01_data_preparation.py

Cleans the real "Give Me Some Credit" (GMSC) dataset — 150,000 anonymized
US credit-bureau records from a 2011 industry-run competition
(https://www.kaggle.com/competitions/GiveMeSomeCredit). This is real,
messy, historical credit data (not synthetic), which is why it needs the
cleaning steps below — the same kind of steps you'd do on a real bank's
loan book.

Target: SeriousDlqin2yrs — whether the borrower experienced 90+ days
past-due delinquency within two years. This is the industry-standard
definition of "default" for retail credit-risk modeling (Basel-style
90-days-past-due).

Known data-quality issues in this dataset (documented by the original
competition and confirmed by inspection here), and how we handle each:

1. Sentinel/error codes: the three "times past due" columns contain 96 and
   98 as values in a small number of rows (269 total) — these are far
   outside the plausible range (2 years / 30-60 day windows -> max ~12-24)
   and are almost certainly data-entry/system error codes, not real counts.
   We cap them at the 99th percentile of otherwise-valid values.
2. One row has age = 0, which is impossible for a loan applicant. We
   replace it with the median age.
3. RevolvingUtilizationOfUnsecuredLines and DebtRatio are meant to be
   ratios but contain extreme outliers (into the tens of thousands) from
   likely data errors or edge-case reporting. We winsorize (cap) at the
   99th percentile rather than drop the rows, to avoid losing real
   applicants at the edges of the distribution.
4. MonthlyIncome is missing for ~19.8% of rows and NumberOfDependents for
   ~2.6%. Rather than silently imputing (which would hide the fact that
   "income unknown" is itself informative -- e.g. self-reported income on
   thin-file applicants), we add explicit `_was_missing` indicator flags
   AND impute with the median, so the model can learn from missingness
   itself if it's predictive.

Output: data/gmsc_clean.csv, ready for train/test split and modeling.
"""
import numpy as np
import pandas as pd

RAW_PATH = "data/gmsc_raw.csv"
CLEAN_PATH = "data/gmsc_clean.csv"

DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]
SENTINEL_VALUES = {96, 98}


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path, index_col=0)
    return df


def clean(df):
    df = df.copy()
    report = {}

    # --- 1. Delinquency sentinel codes ---
    for col in DELINQUENCY_COLS:
        is_sentinel = df[col].isin(SENTINEL_VALUES)
        report[f"{col}_sentinel_rows"] = int(is_sentinel.sum())
        valid_cap = df.loc[~is_sentinel, col].quantile(0.99)
        df.loc[is_sentinel, col] = valid_cap
        # flag: applicant had a data-quality anomaly on this field (can itself be predictive
        # of thin-file / subprime applicants who are more likely to hit system edge cases)
        df[f"{col}_flagged_anomaly"] = is_sentinel.astype(int)

    df["any_delinquency_anomaly"] = df[[f"{c}_flagged_anomaly" for c in DELINQUENCY_COLS]].max(axis=1)
    for c in DELINQUENCY_COLS:
        df.drop(columns=[f"{c}_flagged_anomaly"], inplace=True)

    # --- 2. Impossible age ---
    bad_age = df["age"] < 18
    report["invalid_age_rows"] = int(bad_age.sum())
    df.loc[bad_age, "age"] = df.loc[~bad_age, "age"].median()

    # --- 3. Winsorize extreme ratio outliers ---
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        cap = df[col].quantile(0.99)
        n_capped = int((df[col] > cap).sum())
        report[f"{col}_capped_rows"] = n_capped
        df[col] = df[col].clip(upper=cap)

    # --- 4. Missing-value indicators + imputation ---
    for col in ["MonthlyIncome", "NumberOfDependents"]:
        missing = df[col].isna()
        report[f"{col}_missing_rows"] = int(missing.sum())
        df[f"{col}_was_missing"] = missing.astype(int)
        df[col] = df[col].fillna(df[col].median())

    # --- Derived features (business-friendly, and useful signal) ---
    df["annual_income"] = df["MonthlyIncome"] * 12
    df["total_past_due_events"] = df[DELINQUENCY_COLS].sum(axis=1)
    df["has_any_past_due"] = (df["total_past_due_events"] > 0).astype(int)
    df["credit_lines_per_dependent"] = df["NumberOfOpenCreditLinesAndLoans"] / (df["NumberOfDependents"] + 1)

    df.to_csv(CLEAN_PATH, index=False)
    return df, report


if __name__ == "__main__":
    df = load_raw()
    print(f"Raw shape: {df.shape}")
    clean_df, report = clean(df)
    print(f"Clean shape: {clean_df.shape}")
    print("\nData-quality issues found and handled:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nDefault rate (SeriousDlqin2yrs=1): {clean_df['SeriousDlqin2yrs'].mean():.4f}")
    print(f"\nSaved cleaned data to {CLEAN_PATH}")
