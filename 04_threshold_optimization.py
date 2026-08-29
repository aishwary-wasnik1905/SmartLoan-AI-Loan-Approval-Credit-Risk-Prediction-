"""
04_threshold_optimization.py

Turns the model's raw probability-of-default (PD) output into actual
Approve / Manual Review / Reject decision thresholds — grounded in unit
economics, not picked as round numbers.

This uses the standard credit-risk Expected Loss framework (the same one
underlying Basel-style IRB capital models):

    Expected Loss (EL) = PD x LGD x EAD

    PD  = probability of default (from our model)
    LGD = Loss Given Default — the fraction of exposure the lender doesn't
          recover after a default (net of any collections/collateral).
          We use an illustrative 65% for unsecured personal loans (industry
          figures for unsecured retail credit commonly sit in the 55-75%
          range; recovery rates for unsecured consumer debt are low because
          there's no collateral to seize).
    EAD = Exposure at Default — approximated here as the loan amount,
          which is standard for a term loan (fully drawn, unlike a credit
          line where utilization can change).

We compare EL against the expected interest revenue for a given loan
(assuming it doesn't default), and only approve loans where expected
revenue clears expected loss by a required margin (a "hurdle rate") — this
is a simplified version of risk-based pricing.

The three-way policy is then:
  - Approve:        expected net margin comfortably positive
  - Reject:         expected net margin negative (loan loses money even
                     in expectation)
  - Manual Review:  borderline cases where a human should look at
                     context the model can't see (e.g. co-signers,
                     recent life events, alternative income proof)

We also sweep candidate PD thresholds against a classification cost matrix
(cost of a false-approval / missed default vs. cost of a false-rejection /
lost good customer) and plot the total-cost curve, so the manual-review
band width is itself a defensible, data-driven choice rather than a guess.
"""
import json

import numpy as np
import pandas as pd

LGD = 0.65               # Loss Given Default, illustrative for unsecured personal loans
HURDLE_MARGIN = 0.03     # required expected profit margin (of loan amount) to auto-approve

# Risk-based pricing: real lenders don't charge every applicant the same rate —
# riskier applicants are priced higher (up to a regulatory/usury ceiling), which is
# both fairer to low-risk borrowers and a truer reflection of how unit economics
# actually work in consumer lending.
BASE_APR = 0.10            # best-tier APR for near-zero-risk applicants
RISK_PREMIUM_SLOPE = 0.55  # additional APR per unit of PD
MAX_APR = 0.32             # regulatory / policy ceiling on APR regardless of risk

# Risk appetite ceiling: even if the math on a single loan looks profitable at a
# high enough interest rate, real institutions cap how much default risk they'll
# accept per loan — driven by regulatory capital requirements, portfolio
# concentration limits, and reputational risk, not just per-loan unit economics.
# This ceiling binds independently of the margin-based decision below.
MAX_ACCEPTABLE_PD = 0.20

# Classification cost assumptions (illustrative, in "loan amount" units) for the
# threshold sweep: cost of approving a defaulter vs. cost of rejecting a good payer.
COST_FALSE_APPROVAL = LGD          # lose LGD fraction of the loan on a missed default
COST_FALSE_REJECTION = 0.08        # lost profit + customer lifetime value from a wrongly rejected good applicant


def risk_based_apr(pd_):
    return min(BASE_APR + RISK_PREMIUM_SLOPE * pd_, MAX_APR)


def expected_loss(pd_, loan_amount, lgd=LGD):
    return pd_ * lgd * loan_amount


def expected_revenue(pd_, loan_amount, term_years=3):
    """Expected interest income if the loan performs, weighted by survival
    probability, priced at a risk-based APR (not a flat rate for everyone)."""
    apr = risk_based_apr(pd_)
    return (1 - pd_) * loan_amount * apr * term_years


# Manual-review buffer: applicants inside this fraction of the risk-appetite
# ceiling get referred to a human rather than auto-approved. Important design
# note: at a high enough interest rate, the expected-margin math above stays
# positive even for quite risky applicants (charge enough interest and almost
# any PD looks "profitable" in expectation) — which is exactly the reasoning
# that leads to predatory pricing. A responsible policy does NOT let margin
# math override the risk-appetite ceiling; the ceiling is a hard boundary set
# by capital/regulatory/reputational considerations, not something a high
# enough interest rate can buy past. So decisions here are band-based around
# the ceiling, and the margin economics are reported for pricing/monitoring
# purposes but do not on their own authorize approving high-PD applicants.
REVIEW_BUFFER_FRACTION = 0.6  # review band starts at 60% of the ceiling


def decision_from_economics(pd_, loan_amount, term_years=3):
    el = expected_loss(pd_, loan_amount)
    rev = expected_revenue(pd_, loan_amount, term_years)
    net_margin = (rev - el) / loan_amount

    review_start = REVIEW_BUFFER_FRACTION * MAX_ACCEPTABLE_PD
    if pd_ >= MAX_ACCEPTABLE_PD:
        return "Reject", net_margin
    elif pd_ >= review_start:
        return "Manual Review", net_margin
    else:
        return "Approve", net_margin


def sweep_pd_thresholds(df, thresholds=None):
    """Sweep candidate PD cutoffs and compute total classification cost at each,
    to find the cost-minimizing single cutoff (used as a sanity check alongside
    the economics-based policy above)."""
    if thresholds is None:
        thresholds = np.arange(0.02, 0.5, 0.01)

    results = []
    for t in thresholds:
        predicted_default = (df["proba_default"] >= t).astype(int)
        false_approvals = ((predicted_default == 0) & (df["y_true"] == 1)).sum()
        false_rejections = ((predicted_default == 1) & (df["y_true"] == 0)).sum()
        total_cost = (false_approvals * COST_FALSE_APPROVAL) + (false_rejections * COST_FALSE_REJECTION)
        results.append({"threshold": round(float(t), 3),
                         "false_approvals": int(false_approvals),
                         "false_rejections": int(false_rejections),
                         "total_cost": round(float(total_cost), 2)})
    return pd.DataFrame(results)


def main():
    df = pd.read_csv("artifacts/test_predictions.csv")

    # ---- Cost-minimizing single threshold (reference point) ----
    sweep = sweep_pd_thresholds(df)
    best_row = sweep.loc[sweep["total_cost"].idxmin()]
    print("Cost-minimizing single PD threshold (reference point):")
    print(f"  Threshold: {best_row['threshold']:.2f}  Total cost: {best_row['total_cost']:.1f} "
          f"(false approvals={int(best_row['false_approvals'])}, false rejections={int(best_row['false_rejections'])})")

    # ---- Economics-based three-way policy, applied at a representative loan size ----
    # We evaluate at a representative loan amount since EL/revenue scale linearly
    # with loan size for a fixed PD; the *rates* (net margin) are what matter for
    # setting PD cutoffs, not the specific dollar amount.
    representative_loan = 400_000  # for illustration in INR; policy is loan-size-invariant in rate terms
    sample_pds = np.arange(0.0, 0.5, 0.005)
    decisions = [decision_from_economics(p, representative_loan)[0] for p in sample_pds]

    approve_max_prob = max([p for p, d in zip(sample_pds, decisions) if d == "Approve"], default=0.0)
    reject_candidates = [p for p, d in zip(sample_pds, decisions) if d == "Reject"]
    reject_min_prob = min(reject_candidates) if reject_candidates else 0.5

    print(f"\nEconomics-based policy (LGD={LGD:.0%}, hurdle margin={HURDLE_MARGIN:.1%}, "
          f"risk-based APR {BASE_APR:.0%}-{MAX_APR:.0%}, risk-appetite ceiling PD<{MAX_ACCEPTABLE_PD:.0%}):")
    print(f"  Approve if PD <= {approve_max_prob:.3f}")
    print(f"  Reject  if PD >= {reject_min_prob:.3f}")
    print(f"  Manual Review in between")

    # sanity-check against test set
    approve_share = (df["proba_default"] <= approve_max_prob).mean()
    reject_share = (df["proba_default"] >= reject_min_prob).mean()
    review_share = 1 - approve_share - reject_share
    print(f"\nOn the held-out test population, this policy would:")
    print(f"  Approve {approve_share:.1%} of applicants")
    print(f"  Send {review_share:.1%} to Manual Review")
    print(f"  Reject {reject_share:.1%} of applicants")

    thresholds_out = {
        "approve_max_prob": round(float(approve_max_prob), 4),
        "reject_min_prob": round(float(reject_min_prob), 4),
        "assumptions": {
            "lgd": LGD,
            "hurdle_margin": HURDLE_MARGIN,
            "base_apr": BASE_APR,
            "risk_premium_slope": RISK_PREMIUM_SLOPE,
            "max_apr": MAX_APR,
            "max_acceptable_pd": MAX_ACCEPTABLE_PD,
            "cost_false_approval_per_unit_ead": COST_FALSE_APPROVAL,
            "cost_false_rejection_per_unit_ead": COST_FALSE_REJECTION,
        },
        "cost_minimizing_reference_threshold": round(float(best_row["threshold"]), 3),
        "expected_population_split": {
            "approve": round(float(approve_share), 4),
            "manual_review": round(float(review_share), 4),
            "reject": round(float(reject_share), 4),
        },
    }
    with open("artifacts/decision_thresholds.json", "w") as f:
        json.dump(thresholds_out, f, indent=2)

    sweep.to_csv("reports/threshold_cost_sweep.csv", index=False)
    print("\nSaved artifacts/decision_thresholds.json and reports/threshold_cost_sweep.csv")


if __name__ == "__main__":
    main()
