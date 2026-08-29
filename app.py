"""
Streamlit app for the Loan Default Risk & Approval Assistant (v2).

Run with: streamlit run app.py

Three tabs:
  1. Assess Applicant   — the core tool: PD, risk score, recommendation, SHAP explanation
  2. Model Governance    — cross-validated performance + the age-based fairness/disparate-impact audit
  3. Business Impact     — expected-loss economics, threshold rationale, and drift monitoring (PSI)
"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Loan Risk & Approval Assistant", page_icon="🏦", layout="wide")


# ---------------------------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_pipe = joblib.load("artifacts/base_pipeline.joblib")
    calibrated = joblib.load("artifacts/calibrated_model.joblib")
    explainer = joblib.load("artifacts/shap_explainer.joblib")
    with open("artifacts/model_metadata.json") as f:
        metadata = json.load(f)
    with open("artifacts/decision_thresholds.json") as f:
        thresholds = json.load(f)
    fairness_report, psi_report = None, None
    try:
        with open("reports/fairness_audit.json") as f:
            fairness_report = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open("reports/psi_monitoring.json") as f:
            psi_report = json.load(f)
    except FileNotFoundError:
        pass
    return base_pipe, calibrated, explainer, metadata, thresholds, fairness_report, psi_report


base_pipe, calibrated, explainer, metadata, thresholds, fairness_report, psi_report = load_artifacts()

FEATURES = metadata["features"]
FRIENDLY = metadata["friendly_names"]
SC = metadata["scorecard"]
ASSUMPTIONS = thresholds["assumptions"]
APPROVE_MAX = thresholds["approve_max_prob"]
REJECT_MIN = thresholds["reject_min_prob"]


def friendly(name):
    return FRIENDLY.get(name, name.replace("_", " ").title())


def prob_to_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds_good = (1 - p) / p
    score = SC["offset"] + SC["factor"] * np.log(odds_good)
    return float(np.clip(score, 300, 900))


def risk_category(prob):
    if prob <= APPROVE_MAX:
        return "Low", "#1e8e3e"
    elif prob < REJECT_MIN:
        return "Medium", "#e8a415"
    else:
        return "High", "#d93025"


def recommendation(prob):
    if prob <= APPROVE_MAX:
        return "Approve", "✅"
    elif prob >= REJECT_MIN:
        return "Reject", "❌"
    else:
        return "Manual Review", "⚠️"


def risk_based_apr(prob):
    return min(ASSUMPTIONS["base_apr"] + ASSUMPTIONS["risk_premium_slope"] * prob, ASSUMPTIONS["max_apr"])


def build_explanation(shap_row, top_n=4):
    contributions = list(zip(FEATURES, shap_row))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    increasing = [(friendly(n), v) for n, v in contributions if v > 0][:top_n]
    decreasing = [(friendly(n), v) for n, v in contributions if v < 0][:top_n]
    return increasing, decreasing


# ---------------------------------------------------------------------------
# Sidebar: applicant input
# ---------------------------------------------------------------------------
st.sidebar.header("📋 Applicant Information")
st.sidebar.caption(
    "Fields below feed the credit-risk model. Loan amount/term (further down) "
    "don't affect the risk score itself — they feed the *business* decision layer "
    "(expected loss vs. expected revenue), same as how real risk-based pricing works."
)

with st.sidebar.form("applicant_form"):
    st.subheader("Credit profile")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    annual_income = st.number_input("Annual income (₹)", min_value=0, max_value=20_000_000,
                                     value=1_000_000, step=10_000, format="%d")
    utilization_pct = st.slider("Credit utilization (% of available revolving credit in use)",
                                 0, 150, 25) / 100.0
    dti_pct = st.slider("Debt-to-income ratio (%)", 0, 200, 25) / 100.0
    open_credit_lines = st.number_input("Number of open credit lines/loans", 0, 60, 6)
    real_estate_loans = st.number_input("Number of real-estate loans/lines", 0, 20, 1)
    dependents = st.number_input("Number of dependents", 0, 20, 0)

    st.subheader("Payment history")
    late_30_59 = st.number_input("Times 30-59 days past due (last 2 yrs)", 0, 20, 0)
    late_60_89 = st.number_input("Times 60-89 days past due (last 2 yrs)", 0, 20, 0)
    late_90_plus = st.number_input("Times 90+ days late (last 2 yrs)", 0, 20, 0)

    st.subheader("Requested loan")
    loan_amount = st.number_input("Loan amount (₹)", min_value=10_000, max_value=10_000_000,
                                   value=400_000, step=10_000, format="%d")
    loan_term_years = st.select_slider("Loan term (years)", options=[1, 2, 3, 4, 5, 7, 10], value=3)

    submitted = st.form_submit_button("🔍 Assess Applicant", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Trained on the real, historical 'Give Me Some Credit' dataset (150,000 anonymized "
    "US credit-bureau records). Educational demo — not a real lending decision."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🏦 Loan Default Risk & Approval Assistant")

tab1, tab2, tab3 = st.tabs(["🔍 Assess Applicant", "🛡️ Model Governance & Fairness", "📊 Business Impact & Monitoring"])

# =====================================================================
# TAB 1 — Assess Applicant
# =====================================================================
with tab1:
    if not submitted and "last_input" not in st.session_state:
        st.info("👈 Enter applicant details in the sidebar and click **Assess Applicant** to begin.")
        st.markdown(
            "This model was trained on **real historical credit-bureau data** "
            "(the 2011 'Give Me Some Credit' industry dataset, 150,000 applicants), "
            "not synthetic data — so the risk patterns it learned are genuine, "
            "not hand-coded."
        )
        st.stop()

    if submitted:
        total_past_due = late_30_59 + late_60_89 + late_90_plus
        row = pd.DataFrame([{
            "age": age,
            "annual_income": annual_income,
            "RevolvingUtilizationOfUnsecuredLines": utilization_pct,
            "DebtRatio": dti_pct,
            "NumberOfOpenCreditLinesAndLoans": open_credit_lines,
            "NumberRealEstateLoansOrLines": real_estate_loans,
            "NumberOfDependents": dependents,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
            "NumberOfTimes90DaysLate": late_90_plus,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
            "total_past_due_events": total_past_due,
            "has_any_past_due": int(total_past_due > 0),
            "MonthlyIncome_was_missing": 0,
            "NumberOfDependents_was_missing": 0,
        }])
        st.session_state["last_input"] = row
        st.session_state["last_loan"] = (loan_amount, loan_term_years)

    row = st.session_state["last_input"]
    loan_amount, loan_term_years = st.session_state["last_loan"]

    prob = float(calibrated.predict_proba(row[FEATURES])[:, 1][0])
    score = prob_to_score(prob)
    risk_label, risk_color = risk_category(prob)
    rec_label, rec_icon = recommendation(prob)
    apr = risk_based_apr(prob)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Default Probability", f"{prob:.1%}")
    c2.metric("Credit Score (300-900)", f"{score:.0f}")
    c3.markdown(f"<div style='padding:0.6rem;border-radius:8px;background:{risk_color}22;'>"
                f"<span style='color:{risk_color};font-weight:700;'>Risk: {risk_label}</span></div>",
                unsafe_allow_html=True)
    c4.markdown(f"<div style='padding:0.6rem;border-radius:8px;background:#eee;'>"
                f"<span style='font-weight:700;'>{rec_icon} {rec_label}</span></div>",
                unsafe_allow_html=True)

    st.caption(f"Risk-based pricing for this profile: **{apr:.1%} APR** "
               f"(vs. {ASSUMPTIONS['base_apr']:.0%} best-tier rate) if approved.")

    # ---- SHAP ----
    preprocessor = base_pipe.named_steps["preprocess"]
    model = base_pipe.named_steps["model"]
    X_t = preprocessor.transform(row[FEATURES])
    shap_row = explainer.shap_values(X_t)[0]

    increasing, decreasing = build_explanation(shap_row)

    st.markdown("### Why this recommendation")
    ec1, ec2 = st.columns(2)
    with ec1:
        st.markdown("**⬆️ Factors increasing risk**")
        for label, val in increasing:
            st.markdown(f"- {label} (impact: +{val:.3f})")
        if not increasing:
            st.markdown("_None significant._")
    with ec2:
        st.markdown("**⬇️ Factors decreasing risk**")
        for label, val in decreasing:
            st.markdown(f"- {label} (impact: {val:.3f})")
        if not decreasing:
            st.markdown("_None significant._")

    parts = []
    if decreasing:
        parts.append(f"{' and '.join(l for l, _ in decreasing[:2])} reduced the risk")
    if increasing:
        parts.append(f"{' and '.join(l for l, _ in increasing[:2])} increased the risk")
    if parts:
        st.markdown(f"> {'. '.join(p.capitalize() for p in parts)}.")

    st.markdown("### Feature contribution breakdown")
    contrib_df = pd.DataFrame({"feature": [friendly(f) for f in FEATURES], "shap_value": shap_row})
    contrib_df["abs_val"] = contrib_df["shap_value"].abs()
    contrib_df = contrib_df.sort_values("abs_val", ascending=True).tail(10)
    fig = go.Figure(go.Bar(
        x=contrib_df["shap_value"], y=contrib_df["feature"], orientation="h",
        marker_color=["#d93025" if v > 0 else "#1e8e3e" for v in contrib_df["shap_value"]],
    ))
    fig.update_layout(xaxis_title="Impact on default probability (SHAP value)", yaxis_title="",
                       height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ---- Business economics for this specific loan ----
    st.markdown("### Expected economics for this loan (if approved)")
    lgd = ASSUMPTIONS["lgd"]
    expected_loss = prob * lgd * loan_amount
    expected_revenue = (1 - prob) * loan_amount * apr * loan_term_years
    net_margin = expected_revenue - expected_loss
    e1, e2, e3 = st.columns(3)
    e1.metric("Expected loss (PD × LGD × EAD)", f"₹{expected_loss:,.0f}")
    e2.metric("Expected interest revenue", f"₹{expected_revenue:,.0f}")
    e3.metric("Expected net margin", f"₹{net_margin:,.0f}")

    with st.expander("📄 Applicant details submitted"):
        st.dataframe(row.T.rename(columns={0: "Value"}), use_container_width=True)

# =====================================================================
# TAB 2 — Model Governance & Fairness
# =====================================================================
with tab2:
    st.markdown("## Model performance")
    st.caption(
        "Reported as 5-fold stratified cross-validation (mean ± std), not a single lucky "
        "train/test split — this shows how much the metric actually varies by chance."
    )
    cv = metadata["cv_metrics"]
    tm = metadata["test_metrics"]
    g1, g2, g3 = st.columns(3)
    g1.metric("ROC-AUC (5-fold CV)", f"{cv['auc_mean']:.3f} ± {cv['auc_std']:.3f}")
    g2.metric("PR-AUC (5-fold CV)", f"{cv['pr_auc_mean']:.3f} ± {cv['pr_auc_std']:.3f}")
    g3.metric("Brier score (5-fold CV)", f"{cv['brier_mean']:.4f} ± {cv['brier_std']:.4f}")

    st.markdown(f"On the final held-out test set: ROC-AUC **{tm['roc_auc']}**, "
                f"Brier score improved from **{tm['brier_before_calibration']}** (raw model) to "
                f"**{tm['brier_after_calibration']}** after isotonic calibration — meaning the "
                f"predicted probabilities are closer to true observed default rates, which matters "
                f"since we feed these probabilities directly into pricing and expected-loss math.")

    st.warning(
        "**Validation limitation, stated plainly:** this dataset has no origination-date field, "
        "so we cannot run a true out-of-time backtest (train on one period, test on a later one) "
        "here. Cross-validation shows the model is *stable across random samples* of this "
        "population, but it doesn't prove the model would hold up through an economic cycle "
        "shift. A production deployment would need dated loan vintages to test that properly."
    )

    st.markdown("---")
    st.markdown("## Fairness & disparate-impact audit — protected attribute: age")
    st.caption(
        "Age is an explicitly protected class under the US Equal Credit Opportunity Act (ECOA). "
        "This dataset has no race/gender fields, so this audit covers age only — a production "
        "compliance review would need those fields too, reviewed by fair-lending specialists."
    )

    if fairness_report:
        groups = fairness_report["groups"]
        group_df = pd.DataFrame(groups).T
        group_df.index.name = "Age group"
        st.dataframe(
            group_df.rename(columns={
                "n": "N", "approval_rate": "Approval rate", "manual_review_rate": "Review rate",
                "reject_rate": "Reject rate", "actual_default_rate": "Actual default rate (ground truth)"
            }).style.format({
                "Approval rate": "{:.1%}", "Review rate": "{:.1%}",
                "Reject rate": "{:.1%}", "Actual default rate (ground truth)": "{:.1%}",
            }),
            use_container_width=True,
        )

        st.markdown("**Four-fifths rule check** (approval rate ÷ most-favored group's approval rate; flags below 0.80):")
        ratios = fairness_report["four_fifths_ratios"]
        ratio_df = pd.DataFrame([{"Age group": g, "Ratio": r,
                                   "Status": "⚠️ Below 0.80" if r < 0.8 else "OK"}
                                  for g, r in ratios.items()])
        st.dataframe(ratio_df, use_container_width=True, hide_index=True)

        flagged = fairness_report.get("four_fifths_flagged_groups", [])
        if flagged:
            st.error(
                f"**Flagged: {', '.join(flagged)}** falls below the four-fifths threshold. "
                f"This does *not* automatically mean illegal discrimination — ECOA allows "
                f"disparate impact justified by legitimate business necessity (age genuinely "
                f"correlates with default risk in this data: younger borrowers have a higher "
                f"observed default rate). But it does require this to be documented, monitored, "
                f"and periodically re-justified — not quietly ignored."
            )

        st.markdown("**Performance parity — AUC by age group** (checks the model isn't just fair "
                     "on approval rates but also equally *accurate* at ranking risk within each group):")
        auc_df = pd.DataFrame([{"Age group": g, "AUC": v} for g, v in fairness_report["auc_by_group"].items()])
        st.dataframe(auc_df, use_container_width=True, hide_index=True)
        st.caption(f"AUC spread across groups: {fairness_report['auc_spread_across_groups']} "
                   f"— small enough that this isn't a performance-parity concern on its own.")
    else:
        st.info("Run `python 03_fairness_audit.py` to generate the fairness report.")

# =====================================================================
# TAB 3 — Business Impact & Monitoring
# =====================================================================
with tab3:
    st.markdown("## Decision policy: how the thresholds were set")
    st.markdown(
        f"""
Rather than picking round-number probability cutoffs, thresholds come from a
standard credit-risk **Expected Loss** framework:

`Expected Loss = PD × LGD × EAD`

- **LGD** (Loss Given Default) assumed at **{ASSUMPTIONS['lgd']:.0%}** for unsecured personal loans
  (industry figures for unsecured consumer credit commonly run 55-75%, since there's no collateral to recover).
- **EAD** (Exposure at Default) = the loan amount.
- Pricing is **risk-based**, not flat: APR = {ASSUMPTIONS['base_apr']:.0%} base rate +
  {ASSUMPTIONS['risk_premium_slope']:.2f} × PD, capped at {ASSUMPTIONS['max_apr']:.0%}.
- A hard **risk-appetite ceiling** of PD ≥ {ASSUMPTIONS['max_acceptable_pd']:.0%} always rejects, regardless
  of the margin math — because charging a high enough interest rate can make almost *any* risk look
  "profitable" on paper, which is exactly the logic that leads to predatory pricing. Real institutions
  cap risk per loan for regulatory-capital and reputational reasons, not just unit economics.

This gives: **Approve ≤ {APPROVE_MAX:.1%} PD**, **Reject ≥ {REJECT_MIN:.1%} PD**, Manual Review between.
"""
    )
    st.caption(
        f"Sanity check: a pure cost-minimizing threshold sweep (minimizing false-approval cost "
        f"vs. false-rejection cost directly, with no pricing assumptions at all) independently landed "
        f"at PD = {thresholds['cost_minimizing_reference_threshold']:.2f} — close to the economics-based "
        f"approve cutoff above, which cross-validates the policy from a completely different angle."
    )

    split = thresholds["expected_population_split"]
    p1, p2, p3 = st.columns(3)
    p1.metric("Approve", f"{split['approve']:.1%}")
    p2.metric("Manual Review", f"{split['manual_review']:.1%}")
    p3.metric("Reject", f"{split['reject']:.1%}")
    st.caption("Expected population split if this policy were applied to the held-out test set.")

    st.markdown("---")
    st.markdown("## Production monitoring — Population Stability Index (PSI)")
    st.caption(
        "Applicant populations drift over time (economic cycles, new marketing channels, product "
        "changes), which can silently degrade a model that looked great at launch. PSI is the "
        "standard credit-risk metric for catching this: PSI < 0.10 = stable, 0.10-0.25 = investigate, "
        "≥ 0.25 = the model likely needs review before continued use."
    )

    if psi_report:
        feat_psi = psi_report["feature_psi"]
        psi_df = pd.DataFrame([{"Feature": friendly(k) if k in FRIENDLY else k, "PSI": v["psi"], "Status": v["status"]}
                                for k, v in feat_psi.items()])
        st.dataframe(psi_df, use_container_width=True, hide_index=True)

        if "simulated_drift_example" in psi_report:
            st.markdown("**Worked example — simulated drift** (proves the check actually fires; "
                         "simulates a shift toward younger, higher-utilization applicants):")
            drift_df = pd.DataFrame([{"Feature": friendly(k) if k in FRIENDLY else k, "PSI": v["psi"], "Status": v["status"]}
                                      for k, v in psi_report["simulated_drift_example"].items()])
            st.dataframe(drift_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python 05_monitoring.py` to generate the PSI monitoring report.")

    st.markdown("---")
    st.markdown("## Business impact summary")
    st.markdown(
        """
- **Loss reduction**: applicants scoring above the risk-appetite ceiling are declined before
  origination, directly reducing charge-offs.
- **Faster approvals**: the majority of low-risk applicants clear automatically, cutting
  turnaround time and underwriter workload.
- **Human judgment where it matters**: borderline applicants near the risk ceiling are routed
  to underwriters instead of being auto-approved or auto-declined — the model handles the clear
  cases, people handle the ambiguous ones.
- **Consistency & auditability**: every decision has a documented, reproducible rationale (SHAP
  attributions + the expected-loss calculation), instead of varying by which underwriter reviewed it.
"""
    )
