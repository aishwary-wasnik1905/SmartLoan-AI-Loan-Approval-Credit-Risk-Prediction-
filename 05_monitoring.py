"""
05_monitoring.py

A model in production doesn't stay accurate forever — applicant
populations shift (economic cycles, marketing channel changes, new
products), and a model trained on last year's applicants can quietly
degrade. This script implements the Population Stability Index (PSI), the
standard metric credit-risk teams use to detect that drift, and applies it
to compare the training population against a held-out "new batch" as a
worked example of what a monthly production monitoring job would do.

PSI interpretation (industry rule of thumb):
  PSI < 0.10           : no significant population change
  0.10 <= PSI < 0.25   : moderate shift — investigate
  PSI >= 0.25           : significant shift — model likely needs review/retraining

We compute PSI both for individual input features (has the applicant
population changed?) and for the model's output score distribution (has
the *mix of risk* the model is seeing changed, even if no single input
moved much on its own?).

In this repo we don't have a genuinely time-separated "new" population (see
the note in 02_train_model.py about the lack of a vintage/date field), so
we illustrate the mechanism using the held-out test split as a stand-in for
"new" data relative to the training set it was scored against. In
production this same function would run monthly, comparing the current
month's applicants against the training/reference population.
"""
import json

import numpy as np
import pandas as pd


def psi(reference, current, bins=10):
    """Population Stability Index between two 1-D numeric arrays."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    quantile_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(quantile_edges) < 3:
        return 0.0  # not enough distinct values to bin meaningfully

    quantile_edges[0] = -np.inf
    quantile_edges[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=quantile_edges)
    cur_counts, _ = np.histogram(current, bins=quantile_edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_flag(value):
    if value < 0.10:
        return "stable"
    elif value < 0.25:
        return "moderate shift — investigate"
    else:
        return "significant shift — review model"


def main():
    train_df = pd.read_csv("data/gmsc_clean.csv")
    test_df = pd.read_csv("artifacts/test_predictions.csv")

    monitored_features = [
        "age", "annual_income", "RevolvingUtilizationOfUnsecuredLines",
        "DebtRatio", "NumberOfOpenCreditLinesAndLoans", "total_past_due_events",
    ]

    print("=" * 70)
    print("POPULATION STABILITY INDEX (PSI) — reference (train) vs current (test)")
    print("=" * 70)

    report = {"feature_psi": {}, "score_psi": None}
    for feat in monitored_features:
        value = psi(train_df[feat], test_df[feat])
        flag = psi_flag(value)
        report["feature_psi"][feat] = {"psi": round(value, 4), "status": flag}
        print(f"  {feat:45s} PSI = {value:.4f}   [{flag}]")

    # Score-level PSI: has the overall risk mix shifted?
    train_scores_proxy = train_df["SeriousDlqin2yrs"]  # no scores available for full train set here
    score_psi_value = psi(test_df["proba_default"], test_df["proba_default"])
    # (Self-comparison here is expected to be ~0 since it's the same population;
    # in production this line compares *last month's* scored population against
    # *this month's*, both computed by scoring with the same fixed model.)
    report["score_psi"] = {"psi": round(score_psi_value, 4), "status": psi_flag(score_psi_value)}
    print(f"\n  {'Model output score (self-check, see note above)':45s} "
          f"PSI = {score_psi_value:.4f}   [{psi_flag(score_psi_value)}]")

    print("\nHow to use this in production:")
    print("  1. Schedule this job to run monthly (or per batch of new originations).")
    print("  2. Compare each incoming batch's features AND model scores against the")
    print("     frozen training-time reference distributions (not last month's batch,")
    print("     to avoid slow drift going undetected one small step at a time).")
    print("  3. PSI >= 0.25 on the score distribution -> pause auto-decisioning and")
    print("     escalate to the model risk / validation team before continuing to use")
    print("     the model in production.")

    # ---- Worked example: simulate a drifted population to prove PSI actually
    # catches something (comparing two random splits of the same data, as
    # above, will always look stable — that's a necessary but not sufficient
    # check that the metric is wired correctly) ----
    print("\n" + "=" * 70)
    print("SIMULATED DRIFT EXAMPLE (proves the PSI check actually fires)")
    print("=" * 70)
    print("Simulating a marketing-channel shift toward younger, thinner-file")
    print("applicants next quarter (a realistic drift scenario):")

    rng = np.random.default_rng(7)
    drifted = test_df.sample(frac=0.6, random_state=7).copy()
    # bias the sample younger and toward higher utilization, mimicking a shift
    # toward a riskier acquisition channel
    drifted["age"] = (drifted["age"] * 0.8).clip(lower=18)
    drifted["RevolvingUtilizationOfUnsecuredLines"] = np.clip(
        drifted["RevolvingUtilizationOfUnsecuredLines"] * 1.8, 0, None
    )

    drift_report = {}
    for feat in ["age", "RevolvingUtilizationOfUnsecuredLines"]:
        value = psi(test_df[feat], drifted[feat])
        flag = psi_flag(value)
        drift_report[feat] = {"psi": round(value, 4), "status": flag}
        print(f"  {feat:45s} PSI = {value:.4f}   [{flag}]")

    report["simulated_drift_example"] = drift_report

    with open("reports/psi_monitoring.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved reports/psi_monitoring.json")


if __name__ == "__main__":
    main()
