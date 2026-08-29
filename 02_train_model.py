"""
02_train_model.py

Trains a probability-of-default (PD) model on the cleaned GMSC dataset,
validates it properly (stratified k-fold with confidence intervals, not
just a single train/test split), calibrates probabilities, converts them
into a classic 300-850 "scorecard" score using points-to-double-odds (PDO)
scaling (the same transform real bureaus use to turn model odds into a
FICO-style number loan officers recognize), and fits a SHAP explainer.

Note on validation: this dataset has no origination-date/vintage field, so
a true out-of-time backtest (train on one period, test on a later one)
isn't possible here. In a production setting with real dated loan
vintages, you would replace the stratified k-fold below with a rolling
out-of-time split (e.g. train on 2022-2023 originations, test on 2024) to
catch performance degradation from economic-cycle drift that random
splits can hide. That's called out explicitly rather than faked.
"""
import json

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CLEAN_PATH = "data/gmsc_clean.csv"
TARGET = "SeriousDlqin2yrs"

FEATURES = [
    "age",
    "annual_income",
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "total_past_due_events",
    "has_any_past_due",
    "MonthlyIncome_was_missing",
    "NumberOfDependents_was_missing",
]

FRIENDLY_NAMES = {
    "age": "Age",
    "annual_income": "Annual income",
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilization ratio",
    "DebtRatio": "Debt-to-income ratio",
    "NumberOfOpenCreditLinesAndLoans": "Number of open credit lines/loans",
    "NumberRealEstateLoansOrLines": "Number of real-estate loans/lines",
    "NumberOfDependents": "Number of dependents",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 days past due",
    "NumberOfTimes90DaysLate": "Times 90+ days late",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 days past due",
    "total_past_due_events": "Total past-due events",
    "has_any_past_due": "Any past-due history",
    "MonthlyIncome_was_missing": "Income was unreported",
    "NumberOfDependents_was_missing": "Dependents was unreported",
}

# Scorecard (PDO) scaling — standard industry transform from odds to a
# recognizable score. score = OFFSET + FACTOR * ln(odds_good)
PDO = 20          # points to double the odds
BASE_SCORE = 600  # score at BASE_ODDS
BASE_ODDS = 50    # odds of good:bad at the base score (50:1 ~ "good" applicant)
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def prob_to_score(prob_default):
    prob_default = np.clip(prob_default, 1e-6, 1 - 1e-6)
    odds_good = (1 - prob_default) / prob_default
    score = OFFSET + FACTOR * np.log(odds_good)
    return np.clip(score, 300, 900)


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), FEATURES)]
    )
    base_model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        min_child_weight=8,
        eval_metric="logloss",
        random_state=42,
    )
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", base_model)])
    return pipe


def cross_validate(X, y, n_splits=5):
    """Stratified k-fold CV with bootstrapped-style spread across folds,
    reported as mean +/- std so a reviewer can see how stable the metric is,
    not just a single lucky split."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, pr_aucs, briers = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        pipe = build_pipeline()
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        proba = pipe.predict_proba(X.iloc[te_idx])[:, 1]
        auc = roc_auc_score(y.iloc[te_idx], proba)
        pr_auc = average_precision_score(y.iloc[te_idx], proba)
        brier = brier_score_loss(y.iloc[te_idx], proba)
        aucs.append(auc); pr_aucs.append(pr_auc); briers.append(brier)
        print(f"  Fold {fold}: AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  Brier={brier:.4f}")
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)), "pr_auc_std": float(np.std(pr_aucs)),
        "brier_mean": float(np.mean(briers)), "brier_std": float(np.std(briers)),
    }


def main():
    df = pd.read_csv(CLEAN_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    print("Running 5-fold stratified cross-validation for robust performance estimates...")
    cv_results = cross_validate(X, y, n_splits=5)
    print(f"\nCV summary: AUC = {cv_results['auc_mean']:.4f} +/- {cv_results['auc_std']:.4f}  "
          f"(vs. single-split estimates, this shows how much performance varies by chance)")

    # Final held-out split for calibration + SHAP + reporting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    base_pipe = build_pipeline()
    base_pipe.fit(X_train, y_train)

    # Probability calibration: XGBoost's raw outputs can be poorly calibrated
    # at the tails, which matters a lot here since we convert probabilities
    # directly into pricing/decisions downstream.
    calibrated = CalibratedClassifierCV(base_pipe, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    proba_test = calibrated.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)
    brier_before = brier_score_loss(y_test, base_pipe.predict_proba(X_test)[:, 1])
    brier_after = brier_score_loss(y_test, proba_test)

    print(f"\nHeld-out test set (calibrated model):")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"  Brier score before calibration: {brier_before:.4f}")
    print(f"  Brier score after calibration:  {brier_after:.4f}")

    # ---- SHAP explainer on the *uncalibrated* base model (TreeExplainer
    # needs raw tree output; calibration is a monotonic-ish wrapper on top
    # so SHAP's ranking/direction on the base model still explains the
    # calibrated decision faithfully in practice) ----
    preprocessor = base_pipe.named_steps["preprocess"]
    model = base_pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)

    X_train_t = preprocessor.transform(X_train)
    background_df = pd.DataFrame(X_train_t[:500], columns=FEATURES)

    # ---- Score distribution sanity check ----
    scores_test = prob_to_score(proba_test)
    print(f"\nScorecard-style score range on test set: {scores_test.min():.0f} - {scores_test.max():.0f} "
          f"(median {np.median(scores_test):.0f})")

    # ---- Save everything the app / other scripts need ----
    joblib.dump(base_pipe, "artifacts/base_pipeline.joblib")
    joblib.dump(calibrated, "artifacts/calibrated_model.joblib")
    joblib.dump(explainer, "artifacts/shap_explainer.joblib")
    background_df.to_csv("artifacts/shap_background.csv", index=False)

    # Save test set predictions for the fairness audit script
    test_export = X_test.copy()
    test_export["y_true"] = y_test.values
    test_export["proba_default"] = proba_test
    test_export["score"] = scores_test
    test_export.to_csv("artifacts/test_predictions.csv", index=False)

    metadata = {
        "features": FEATURES,
        "friendly_names": FRIENDLY_NAMES,
        "target": TARGET,
        "scorecard": {"pdo": PDO, "base_score": BASE_SCORE, "base_odds": BASE_ODDS,
                      "factor": FACTOR, "offset": OFFSET},
        "cv_metrics": cv_results,
        "test_metrics": {
            "roc_auc": round(float(auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "brier_before_calibration": round(float(brier_before), 4),
            "brier_after_calibration": round(float(brier_after), 4),
            "test_default_rate": round(float(y_test.mean()), 4),
        },
    }
    with open("artifacts/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved artifacts to ./artifacts/")


if __name__ == "__main__":
    main()
