"""
Phase 10 — Interactive scorecard demo.

Loads the LightGBM model and scorecard configuration saved by
notebooks/04_modelling.ipynb, scores a single applicant, and explains the
decision with per-applicant SHAP contributions.

Run with:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

ROOT = Path(__file__).parent
MODELS = ROOT / "outputs" / "models"

st.set_page_config(page_title="Credit Risk Scorecard", page_icon="📊", layout="wide")

# The feature engineering below must match notebooks/02_clean_features.ipynb
# exactly. Any divergence is training/serving skew — the model would be scoring
# inputs built differently from the ones it learned on.
CENSUS_REGION = {
    "NORTHEAST": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "MIDWEST": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "SOUTH": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY",
              "MS", "TN", "AR", "LA", "OK", "TX"],
    "WEST": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
STATE_TO_REGION = {s: r for r, states in CENSUS_REGION.items() for s in states}

PLAIN_NAMES = {
    "term_months": "length of loan term requested",
    "loan_to_income": "size of loan relative to your income",
    "payment_to_income": "monthly payment relative to your income",
    "revol_util": "how much of your available credit is already used",
    "dti": "existing debt relative to your income",
    "log_annual_inc": "reported annual income",
    "annual_inc": "reported annual income",
    "loan_amnt": "amount requested",
    "emp_length": "length of employment",
    "emp_length_missing": "employment history not provided",
    "total_acc": "number of credit accounts",
    "longest_credit_length": "length of your credit history",
    "acct_open_rate": "how quickly you have opened new accounts",
    "has_delinquency": "previous missed payments",
    "delinq_2yrs": "missed payments in the last 2 years",
    "over_limit": "balance above your credit limit",
    "purpose": "stated purpose of the loan",
    "home_ownership": "housing situation",
    "verification_status": "income verification status",
    "region": "location",
}


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS / "lightgbm.joblib")
    config = joblib.load(MODELS / "scorecard_config.joblib")
    return model, config, shap.TreeExplainer(model)


def build_features(inputs, config):
    """Replicate the Phase 3 feature engineering for one applicant."""
    d = dict(inputs)
    emp_missing = d.pop("emp_length_unknown")
    emp_length = np.nan if emp_missing else d["emp_length"]

    row = {
        "loan_amnt": d["loan_amnt"],
        "annual_inc": d["annual_inc"],
        "term_months": d["term_months"],
        "dti": d["dti"],
        "revol_util": d["revol_util"],
        "total_acc": d["total_acc"],
        "longest_credit_length": d["longest_credit_length"],
        "delinq_2yrs": d["delinq_2yrs"],
        "emp_length": emp_length,
        "emp_length_missing": int(emp_missing),
        "loan_to_income": d["loan_amnt"] / d["annual_inc"],
        "payment_to_income": (d["loan_amnt"] / d["term_months"]) / (d["annual_inc"] / 12),
        "log_annual_inc": np.log10(d["annual_inc"]),
        "acct_open_rate": d["total_acc"] / (d["longest_credit_length"] + 1),
        "has_delinquency": float(d["delinq_2yrs"] > 0),
        "over_limit": float(d["revol_util"] > 100),
        "purpose": d["purpose"],
        "home_ownership": d["home_ownership"],
        "verification_status": d["verification_status"],
        "region": STATE_TO_REGION[d["addr_state"]],
    }

    X = pd.DataFrame([row])[config["TREE_NUM"] + config["TREE_CAT"]]
    for col in config["TREE_CAT"]:
        X[col] = pd.Categorical(X[col], categories=config["tree_categories"][col])
    return X


def to_score(p, config):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return int(np.clip(config["OFFSET"] + config["FACTOR"] * np.log((1 - p) / p), 300, 850))


st.title("Credit Risk Scorecard")
st.caption(
    "LightGBM — AUC 0.690, KS 0.283. Trained **without** `int_rate`, which leaks "
    "Lending Club's own risk grade. Probabilities are calibrated (mean predicted "
    "18.25% vs actual 18.3%). See the README for the full rationale."
)

try:
    model, config, explainer = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `notebooks/04_modelling.ipynb` first.")
    st.stop()

with st.sidebar:
    st.header("Applicant details")
    loan_amnt = st.number_input("Loan amount ($)", 500, 35_000, 12_000, step=500)
    term_months = st.selectbox("Term", [36, 60], format_func=lambda m: f"{m} months")
    annual_inc = st.number_input("Annual income ($)", 4_000, 500_000, 60_000, step=1_000)
    purpose = st.selectbox("Loan purpose", sorted(config["tree_categories"]["purpose"]))
    home_ownership = st.selectbox("Home ownership", sorted(config["tree_categories"]["home_ownership"]))
    verification_status = st.selectbox(
        "Income verification", sorted(config["tree_categories"]["verification_status"])
    )
    addr_state = st.selectbox("State", sorted(STATE_TO_REGION), index=sorted(STATE_TO_REGION).index("CA"))

    st.divider()
    dti = st.slider("Debt-to-income ratio", 0.0, 40.0, 18.0, 0.1)
    revol_util = st.slider("Revolving utilisation (%)", 0.0, 150.0, 55.0, 0.1,
                           help="Values above 100% are genuine — see README.")
    total_acc = st.slider("Total credit accounts", 1, 120, 24)
    longest_credit_length = st.slider("Credit history (years)", 0, 65, 14)
    delinq_2yrs = st.slider("Delinquencies in last 2 years", 0, 15, 0)

    st.divider()
    emp_length_unknown = st.checkbox(
        "Employment history not provided",
        help="Applicants who do not state employment length default at 1.46x the rate of "
             "those who do — the model treats this as a risk signal rather than ignoring it.",
    )
    emp_length = st.slider("Employment length (years)", 0, 10, 6, disabled=emp_length_unknown)

X = build_features(
    dict(loan_amnt=loan_amnt, term_months=term_months, annual_inc=annual_inc, dti=dti,
         revol_util=revol_util, total_acc=total_acc, delinq_2yrs=delinq_2yrs,
         longest_credit_length=longest_credit_length, emp_length=emp_length,
         emp_length_unknown=emp_length_unknown, purpose=purpose,
         home_ownership=home_ownership, verification_status=verification_status,
         addr_state=addr_state),
    config,
)

prob = float(model.predict_proba(X)[0, 1])
score = to_score(prob, config)

left, right = st.columns([1, 1.4])

with left:
    st.subheader("Assessment")
    st.metric("Credit score", score, help="300-850 scale, 20 points to double the odds")
    st.metric("Estimated default risk", f"{prob:.1%}")

    # Bands are the score quintiles of the held-out test set, with the observed
    # default rate in each. Derived from the data, not chosen by eye.
    if score >= 551:
        st.success("Top 20% of the portfolio — observed default rate 6.7%")
    elif score >= 540:
        st.info("Above average (60-80th percentile) — observed 11.7%")
    elif score >= 530:
        st.info("Around portfolio average (40-60th) — observed 15.8%")
    elif score >= 517:
        st.warning("Below average (20-40th percentile) — observed 23.2%")
    else:
        st.error("Bottom 20% of the portfolio — observed default rate 34.5%")

    st.caption(
        "Scores span roughly 462-603 rather than the full 300-850. That is a "
        "consequence of AUC 0.69, not a bug — a narrow probability range maps to a "
        "narrow score range. Rescaling would imply accuracy the model lacks."
    )
    st.caption(
        "Probabilities are calibrated by design — no class reweighting — so the "
        "percentage can be read as a genuine default probability."
    )

with right:
    st.subheader("Why this score")
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    contrib = pd.Series(shap_values[0], index=X.columns).sort_values()

    plot_df = pd.DataFrame({
        "factor": [PLAIN_NAMES.get(f, f) for f in contrib.index],
        "effect": contrib.values,
    })
    plot_df = plot_df.reindex(plot_df.effect.abs().sort_values(ascending=False).index).head(8)

    st.bar_chart(plot_df.set_index("factor"), horizontal=True, color="#C44E52")
    st.caption("Positive values increase assessed risk; negative values reduce it.")

    worst = contrib[contrib > 0].sort_values(ascending=False).head(4)
    if len(worst):
        st.markdown("**Principal reasons counting against this application:**")
        for i, feat in enumerate(worst.index, 1):
            st.markdown(f"{i}. {PLAIN_NAMES.get(feat, feat)} — your value: `{X.iloc[0][feat]}`")
    else:
        st.markdown("No factors counted materially against this application.")

st.divider()
st.caption(
    "Portfolio demonstration only — not a lending decision. Trained on historical "
    "Lending Club data with no out-of-time validation."
)
