"""
03_fairness_audit.py

Runs a disparate-impact audit on the model's decisions, using age as the
protected attribute. Age is an explicitly protected class under the US
Equal Credit Opportunity Act (ECOA) — lenders may not discriminate against
applicants on the basis of age (with narrow, specific exceptions for minors
and elderly-specific programs), so this is a real, non-optional compliance
check for a credit model, not just a "nice to have."

This dataset doesn't include race/gender, so a full protected-class audit
would need those fields in a production setting — that limitation is
called out explicitly below rather than glossed over.

Method: the "four-fifths rule" (a widely used EEOC/regulatory rule of
thumb) — the approval rate for any group should be at least 80% of the
approval rate for the most-favored group. A ratio below 0.8 is a common
trigger for further investigation, not automatically illegal discrimination,
but it's the standard first screen.

We also check for differential model *performance* (AUC) by age group,
because a model can be "fair" on approval rates yet still be systematically
worse at ranking risk within one subgroup (accuracy parity), which is a
different and equally important failure mode.
"""
import json

import joblib
import numpy as np
import pandas as pd

DECISION_THRESHOLDS_PATH = "artifacts/decision_thresholds.json"


def age_bucket(age):
    if age < 25:
        return "18-24"
    elif age < 40:
        return "25-39"
    elif age < 62:
        return "40-61"
    else:
        return "62+"


def four_fifths_ratio(approval_rates):
    max_rate = max(approval_rates.values())
    return {group: round(rate / max_rate, 4) if max_rate > 0 else None
            for group, rate in approval_rates.items()}


def main():
    df = pd.read_csv("artifacts/test_predictions.csv")

    try:
        with open(DECISION_THRESHOLDS_PATH) as f:
            thresholds = json.load(f)
        approve_max = thresholds["approve_max_prob"]
        reject_min = thresholds["reject_min_prob"]
    except FileNotFoundError:
        # Fallback defaults if run before 04_threshold_optimization.py
        approve_max, reject_min = 0.05, 0.30

    df["age_group"] = df["age"].apply(age_bucket)
    df["decision"] = np.select(
        [df["proba_default"] <= approve_max, df["proba_default"] >= reject_min],
        ["Approve", "Reject"], default="Manual Review"
    )

    report = {"protected_attribute": "age", "groups": {}}

    print("=" * 70)
    print("FAIRNESS / DISPARATE IMPACT AUDIT — protected attribute: AGE")
    print("=" * 70)

    approval_rates = {}
    for group, g in df.groupby("age_group"):
        approval_rate = (g["decision"] == "Approve").mean()
        reject_rate = (g["decision"] == "Reject").mean()
        review_rate = (g["decision"] == "Manual Review").mean()
        actual_default_rate = g["y_true"].mean()
        n = len(g)
        approval_rates[group] = approval_rate
        report["groups"][group] = {
            "n": int(n),
            "approval_rate": round(float(approval_rate), 4),
            "manual_review_rate": round(float(review_rate), 4),
            "reject_rate": round(float(reject_rate), 4),
            "actual_default_rate": round(float(actual_default_rate), 4),
        }
        print(f"\nAge group {group}  (n={n}):")
        print(f"  Approval rate:       {approval_rate:.1%}")
        print(f"  Manual review rate:  {review_rate:.1%}")
        print(f"  Reject rate:         {reject_rate:.1%}")
        print(f"  Actual default rate: {actual_default_rate:.1%}  (ground truth, for reference)")

    ratios = four_fifths_ratio(approval_rates)
    report["four_fifths_ratios"] = ratios
    print("\n--- Four-fifths rule check (approval rate vs. most-favored group) ---")
    flagged = []
    for group, ratio in ratios.items():
        flag = "⚠️  BELOW 0.80 THRESHOLD" if ratio < 0.8 else "OK"
        if ratio < 0.8:
            flagged.append(group)
        print(f"  {group}: ratio = {ratio}  [{flag}]")

    report["four_fifths_flagged_groups"] = flagged

    # ---- Performance parity: AUC by group ----
    from sklearn.metrics import roc_auc_score
    print("\n--- Model performance (AUC) by age group ---")
    auc_by_group = {}
    for group, g in df.groupby("age_group"):
        if g["y_true"].nunique() < 2:
            auc_by_group[group] = None
            print(f"  {group}: not enough positive/negative cases to compute AUC")
            continue
        auc = roc_auc_score(g["y_true"], g["proba_default"])
        auc_by_group[group] = round(float(auc), 4)
        print(f"  {group}: AUC = {auc:.4f}")
    report["auc_by_group"] = auc_by_group

    auc_values = [v for v in auc_by_group.values() if v is not None]
    auc_spread = round(max(auc_values) - min(auc_values), 4) if auc_values else None
    report["auc_spread_across_groups"] = auc_spread
    print(f"\nAUC spread across age groups: {auc_spread} "
          f"({'small — no major performance-parity concern' if auc_spread and auc_spread < 0.05 else 'notable — investigate further'})")

    print("\n--- Limitations of this audit ---")
    print("  This dataset contains no race/ethnicity/gender fields, so this audit")
    print("  covers age only. A production compliance review would also need to test")
    print("  race, gender, and proxy variables (e.g. zip code) using appropriately")
    print("  governed data, typically via fair-lending / disparate-impact specialists,")
    print("  not just an engineering team.")

    with open("reports/fairness_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved full report to reports/fairness_audit.json")


if __name__ == "__main__":
    main()
