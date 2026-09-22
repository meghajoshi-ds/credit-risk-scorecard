"""
Phase 10 — Interactive scorecard demo.

Loads the LightGBM model and scorecard configuration saved by
notebooks/04_modelling.ipynb, scores a single applicant, and explains the
decision with per-applicant SHAP contributions.

Charts are built as inline HTML/SVG rather than with a plotting library, so
the runtime dependency list stays minimal for the hosted deploy.

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

st.set_page_config(
    page_title="Credit Risk Scorecard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- palette ---
# Values taken unchanged from the data-viz reference palette.
# Diverging pair (polarity: does a factor push risk up or down) is blue<->red
# with a neutral gray midpoint. The risk ramp is sequential red: light = low
# magnitude, dark = high. Every band carries a numeric label, so meaning is
# never colour-alone.
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
SURFACE = "#fcfcfb"
LINE = "#e6e5e1"
RISK_UP = "#e34948"     # diverging warm pole
RISK_DOWN = "#2a78d6"   # diverging cool pole
NEUTRAL = "#f0efec"     # diverging midpoint

# Sequential red ramp for the five score bands (low -> high risk).
BAND_RAMP = ["#0ca30c", "#8fbf3f", "#fab219", "#ec835a", "#d03b3b"]

# Score bands: the quintiles of the held-out test set, with the default rate
# observed in each. Derived from the data in notebook 04, not chosen by eye.
BANDS = [
    (440, 517, "Bottom 20%", 34.5, BAND_RAMP[4]),
    (517, 530, "20th–40th", 23.2, BAND_RAMP[3]),
    (530, 540, "40th–60th", 15.8, BAND_RAMP[2]),
    (540, 551, "60th–80th", 11.7, BAND_RAMP[1]),
    (551, 603, "Top 20%", 6.7, BAND_RAMP[0]),
]
SCORE_MIN, SCORE_MAX = 440, 610
BASE_RATE = 18.3

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
      #MainMenu, footer {{ visibility: hidden; }}

      .hero-title {{
        font-size: 2.05rem; font-weight: 700; letter-spacing: -0.02em;
        color: {INK}; margin: 0 0 .3rem 0; line-height: 1.15;
      }}
      .hero-sub {{ color: {INK_2}; font-size: 1rem; margin: 0 0 .9rem 0; }}

      .badge {{
        display: inline-block; padding: .22rem .6rem; margin: 0 .3rem .3rem 0;
        border: 1px solid {LINE}; border-radius: 999px; background: {SURFACE};
        font-size: .78rem; color: {INK_2};
      }}
      .badge b {{ color: {INK}; font-weight: 600; }}

      .card {{
        border: 1px solid {LINE}; border-radius: 12px; background: {SURFACE};
        padding: 1.1rem 1.25rem; margin-bottom: .9rem;
      }}
      .card-label {{
        font-size: .74rem; text-transform: uppercase; letter-spacing: .08em;
        color: {INK_3}; margin-bottom: .35rem; font-weight: 600;
      }}
      .score-hero {{
        font-size: 3.6rem; font-weight: 700; line-height: 1;
        letter-spacing: -0.03em; color: {INK};
      }}
      .score-unit {{ font-size: 1rem; color: {INK_3}; font-weight: 500; }}
      .risk-hero {{ font-size: 2.1rem; font-weight: 700; color: {INK}; line-height: 1.1; }}

      .verdict {{
        display: flex; align-items: center; gap: .55rem;
        padding: .7rem .9rem; border-radius: 10px; font-weight: 600;
        font-size: .95rem; border: 1px solid;
      }}
      .note {{ color: {INK_3}; font-size: .8rem; line-height: 1.5; }}
      .insight {{
        border-left: 3px solid {RISK_DOWN}; background: #f6f9fe;
        padding: .7rem .9rem; border-radius: 0 8px 8px 0;
        font-size: .88rem; color: {INK_2}; margin: .5rem 0 .9rem 0;
      }}
      .reason-row {{
        display: grid; grid-template-columns: 1fr 120px; gap: .6rem;
        align-items: center; padding: .42rem 0; border-bottom: 1px solid {LINE};
      }}
      .reason-name {{ font-size: .9rem; color: {INK}; }}
      .reason-val {{ font-size: .78rem; color: {INK_3}; }}
      h3 {{ font-size: 1.05rem !important; font-weight: 650 !important; color: {INK} !important; }}
      .stTabs [data-baseweb="tab"] {{ font-size: .95rem; font-weight: 550; }}
    </style>
    """,
    unsafe_allow_html=True,
)

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
    "term_months": "Loan term requested",
    "loan_to_income": "Loan size vs income",
    "payment_to_income": "Monthly payment vs income",
    "revol_util": "Credit utilisation",
    "dti": "Existing debt vs income",
    "log_annual_inc": "Annual income",
    "annual_inc": "Annual income",
    "loan_amnt": "Amount requested",
    "emp_length": "Length of employment",
    "emp_length_missing": "Employment history not given",
    "total_acc": "Number of credit accounts",
    "longest_credit_length": "Length of credit history",
    "acct_open_rate": "Rate of opening new accounts",
    "has_delinquency": "Previous missed payments",
    "delinq_2yrs": "Missed payments (2 yrs)",
    "over_limit": "Balance above credit limit",
    "purpose": "Loan purpose",
    "home_ownership": "Housing situation",
    "verification_status": "Income verification",
    "region": "Location",
}

# Some features encode the same underlying fact twice (raw and transformed, or
# a count and its binary flag). Showing both confuses a reader: "Annual income"
# would appear once as 60000 and again as 4.78, its log. SHAP values are
# additive, so contributions are summed within each group and reported against
# the human-readable value. Only genuine duplicates are merged.
MERGE_GROUPS = {
    "log_annual_inc": ("annual_inc", "Annual income"),
    "annual_inc": ("annual_inc", "Annual income"),
    "has_delinquency": ("delinq_2yrs", "Missed payments (last 2 yrs)"),
    "delinq_2yrs": ("delinq_2yrs", "Missed payments (last 2 yrs)"),
    "over_limit": ("revol_util", "Credit utilisation"),
    "revol_util": ("revol_util", "Credit utilisation"),
}


def merge_contributions(contrib, values):
    """Sum SHAP values within each duplicate group; keep the readable value."""
    summed, display = {}, {}
    for feat, val in contrib.items():
        key, label = MERGE_GROUPS.get(feat, (feat, PLAIN_NAMES.get(feat, feat)))
        summed[key] = summed.get(key, 0.0) + float(val)
        display[key] = label
    out = pd.Series(summed)
    shown = {k: values.get(k, "") for k in out.index}
    return out, display, shown


# One-click examples, so a visitor sees the model behave without fiddling.
PRESETS = {
    "Safe": dict(loan_amnt=6000, term_months=36, annual_inc=120000, dti=8.0,
                     revol_util=15.0, total_acc=28, longest_credit_length=22,
                     delinq_2yrs=0, emp_length=10, emp_length_unknown=False,
                     purpose="CREDIT_CARD", home_ownership="MORTGAGE",
                     verification_status="NOT VERIFIED", addr_state="CA"),
    "Typical": dict(loan_amnt=12000, term_months=36, annual_inc=60000, dti=18.0,
                    revol_util=55.0, total_acc=24, longest_credit_length=14,
                    delinq_2yrs=0, emp_length=6, emp_length_unknown=False,
                    purpose="DEBT_CONSOLIDATION", home_ownership="RENT",
                    verification_status="VERIFIED", addr_state="TX"),
    "Risky": dict(loan_amnt=32000, term_months=60, annual_inc=32000, dti=34.0,
                      revol_util=96.0, total_acc=9, longest_credit_length=5,
                      delinq_2yrs=2, emp_length=1, emp_length_unknown=True,
                      purpose="SMALL_BUSINESS", home_ownership="RENT",
                      verification_status="VERIFIED", addr_state="NV"),
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


def band_for(score):
    for lo, hi, label, rate, color in BANDS:
        if score < hi:
            return label, rate, color
    return BANDS[-1][2], BANDS[-1][3], BANDS[-1][4]


def gauge_svg(score):
    """Where this applicant sits among applicants.

    Bands are drawn at equal width because they are population quintiles: each
    holds 20% of the book. The marker is placed by interpolating the score
    within its own band, so position reads as percentile. Each band carries its
    observed default rate, so colour never carries meaning alone.
    """
    W, H = 760, 104
    pad_l, pad_r = 10, 10
    track_y, track_h = 46, 24
    inner = W - pad_l - pad_r
    seg = inner / len(BANDS)

    # Percentile position: which band, then how far through it.
    pos = pad_l
    for i, (lo, hi, *_rest) in enumerate(BANDS):
        if score < hi or i == len(BANDS) - 1:
            frac = np.clip((score - lo) / max(hi - lo, 1), 0, 1)
            pos = pad_l + (i + frac) * seg
            break

    parts = []
    for i, (lo, hi, label, rate, color) in enumerate(BANDS):
        x0 = pad_l + i * seg
        w = seg - 2                      # 2px surface gap between fills
        r = 5 if i in (0, len(BANDS) - 1) else 0
        parts.append(
            f'<rect x="{x0:.1f}" y="{track_y}" width="{w:.1f}" height="{track_h}" '
            f'rx="{r}" fill="{color}" opacity="0.92"/>'
        )
        parts.append(
            f'<text x="{x0 + seg / 2:.1f}" y="{track_y - 9}" text-anchor="middle" '
            f'font-size="10.5" fill="{INK_3}" font-family="system-ui,sans-serif">{label}</text>'
        )
        parts.append(
            f'<text x="{x0 + seg / 2:.1f}" y="{track_y + track_h + 16}" text-anchor="middle" '
            f'font-size="12" font-weight="600" fill="{INK_2}" '
            f'font-family="system-ui,sans-serif">{rate:.1f}%</text>'
        )
        parts.append(
            f'<text x="{x0 + seg / 2:.1f}" y="{track_y + track_h + 30}" text-anchor="middle" '
            f'font-size="9.5" fill="{INK_3}" font-family="system-ui,sans-serif">defaulted</text>'
        )

    parts.append(
        f'<polygon points="{pos:.1f},{track_y - 2} {pos - 6:.1f},{track_y - 11} '
        f'{pos + 6:.1f},{track_y - 11}" fill="{INK}"/>'
    )
    # 2px surface ring keeps the marker legible over any band colour
    parts.append(
        f'<rect x="{pos - 2.5:.1f}" y="{track_y - 3}" width="5" height="{track_h + 6}" '
        f'rx="2.5" fill="{INK}" stroke="{SURFACE}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{pos:.1f}" y="{track_y - 17}" text-anchor="middle" font-size="13" '
        f'font-weight="700" fill="{INK}" font-family="system-ui,sans-serif">{score}</text>'
    )
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" '
            f'role="img" aria-label="Score {score}, {band_for(score)[0]} of applicants">'
            + "".join(parts) + "</svg>")


def contributions_svg(contrib, values, labels, top_n=9):
    """Diverging bars: how much each factor moved this applicant's risk.
    Red pushes risk up, blue pulls it down, gray zero line at the centre."""
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(top_n)
    top = top.iloc[::-1]

    row_h, gap = 30, 6
    W = 760
    label_w, val_w = 250, 62
    plot_w = W - label_w - val_w - 16
    cx = label_w + plot_w / 2
    H = len(top) * (row_h + gap) + 26
    vmax = max(float(top.abs().max()), 1e-6)

    parts = [f'<line x1="{cx}" y1="14" x2="{cx}" y2="{H - 14}" stroke="{LINE}" stroke-width="1"/>']
    for i, (feat, val) in enumerate(top.items()):
        y = 14 + i * (row_h + gap)
        w = abs(val) / vmax * (plot_w / 2 - 10)
        up = val > 0
        color = RISK_UP if up else RISK_DOWN
        x = cx if up else cx - w
        raw = values.get(feat, "")
        if isinstance(raw, float):
            raw = f"{raw:,.2f}".rstrip("0").rstrip(".")
        parts.append(
            f'<text x="{label_w - 12}" y="{y + row_h / 2 + 4}" text-anchor="end" font-size="12.5" '
            f'fill="{INK}" font-family="system-ui,sans-serif">{labels.get(feat, feat)}</text>'
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y + 5}" width="{max(w, 2):.1f}" height="{row_h - 10}" '
            f'rx="4" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{W - 8}" y="{y + row_h / 2 + 4}" text-anchor="end" font-size="11.5" '
            f'fill="{INK_3}" font-family="system-ui,sans-serif">{raw}</text>'
        )
    parts.append(
        f'<text x="{cx - 8}" y="{H - 2}" text-anchor="end" font-size="10" fill="{RISK_DOWN}" '
        f'font-family="system-ui,sans-serif">&#9664; lowers risk</text>'
    )
    parts.append(
        f'<text x="{cx + 8}" y="{H - 2}" font-size="10" fill="{RISK_UP}" '
        f'font-family="system-ui,sans-serif">raises risk &#9654;</text>'
    )
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" '
            f'aria-label="Factor contributions to this score">' + "".join(parts) + "</svg>")


# ------------------------------------------------------------------- page ---
st.markdown('<div class="hero-title">Credit Risk Scorecard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Score a loan applicant, and see exactly which factors '
    'drove the decision.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="badge">Model <b>LightGBM</b></span>'
    '<span class="badge">AUC <b>0.690</b></span>'
    '<span class="badge">KS <b>0.283</b></span>'
    '<span class="badge">Trained on <b>163,987 loans</b></span>'
    '<span class="badge">Calibrated: predicts <b>18.25%</b> vs actual <b>18.3%</b></span>'
    '<span class="badge">Excludes <b>int_rate</b> (leakage)</span>',
    unsafe_allow_html=True,
)
st.write("")

try:
    model, config, explainer = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `notebooks/04_modelling.ipynb` first.")
    st.stop()

if "preset" not in st.session_state:
    st.session_state.update(PRESETS["Typical"])

with st.sidebar:
    st.markdown("### Try an example")
    cols = st.columns(3)
    for col, name in zip(cols, PRESETS):
        if col.button(name, use_container_width=True):
            st.session_state.update(PRESETS[name])
            st.session_state["preset"] = name
            st.rerun()

    st.markdown("### Applicant")
    loan_amnt = st.number_input("Loan amount ($)", 500, 35_000, step=500, key="loan_amnt")
    term_months = st.radio("Term", [36, 60], format_func=lambda m: f"{m} months",
                           horizontal=True, key="term_months")
    annual_inc = st.number_input("Annual income ($)", 4_000, 500_000, step=1_000, key="annual_inc")
    purpose = st.selectbox("Loan purpose", sorted(config["tree_categories"]["purpose"]),
                           key="purpose")
    home_ownership = st.selectbox("Home ownership",
                                  sorted(config["tree_categories"]["home_ownership"]),
                                  key="home_ownership")
    verification_status = st.selectbox("Income verification",
                                       sorted(config["tree_categories"]["verification_status"]),
                                       key="verification_status")
    addr_state = st.selectbox("State", sorted(STATE_TO_REGION), key="addr_state")

    st.markdown("### Credit profile")
    dti = st.slider("Debt-to-income ratio", 0.0, 40.0, step=0.5, key="dti")
    revol_util = st.slider("Credit utilisation (%)", 0.0, 150.0, step=1.0, key="revol_util",
                           help="Above 100% is genuine: interest and fees can push a "
                                "balance past its limit. Those borrowers default at 25.6%.")
    total_acc = st.slider("Total credit accounts", 1, 120, key="total_acc")
    longest_credit_length = st.slider("Credit history (years)", 0, 65, key="longest_credit_length")
    delinq_2yrs = st.slider("Missed payments (last 2 yrs)", 0, 15, key="delinq_2yrs")

    emp_length_unknown = st.checkbox(
        "Employment history not provided", key="emp_length_unknown",
        help="Applicants who don't state employment length default at 1.46x the rate "
             "of those who do. The model treats the absence as a signal.",
    )
    emp_length = st.slider("Employment length (years)", 0, 10, key="emp_length",
                           disabled=emp_length_unknown)

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
band_label, band_rate, band_color = band_for(score)

shap_values = explainer.shap_values(X)
if isinstance(shap_values, list):
    shap_values = shap_values[1]
contrib_raw = pd.Series(shap_values[0], index=X.columns)
values_raw = {c: X.iloc[0][c] for c in X.columns}
contrib, labels, values = merge_contributions(contrib_raw, values_raw)

tab_decision, tab_why, tab_model = st.tabs(["Decision", "Why this score", "About the model"])

with tab_decision:
    c1, c2, c3 = st.columns([1, 1, 1.5])
    with c1:
        st.markdown(
            f'<div class="card"><div class="card-label">Credit score</div>'
            f'<div class="score-hero">{score}</div>'
            f'<div class="score-unit">of 300–850</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="card"><div class="card-label">Default probability</div>'
            f'<div class="risk-hero">{prob:.1%}</div>'
            f'<div class="note" style="margin-top:.35rem">Portfolio average {BASE_RATE}%</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        vs = prob * 100 / BASE_RATE
        st.markdown(
            f'<div class="card"><div class="card-label">Portfolio position</div>'
            f'<div class="verdict" style="border-color:{band_color}; color:{INK}; '
            f'background:{band_color}18;">'
            f'<span style="width:10px;height:10px;border-radius:50%;background:{band_color};'
            f'display:inline-block"></span>{band_label} of applicants</div>'
            f'<div class="note" style="margin-top:.5rem">Loans in this band defaulted '
            f'<b>{band_rate}%</b> of the time. This applicant scores <b>{vs:.1f}x</b> '
            f'the portfolio average risk.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Where this applicant sits")
    st.markdown(gauge_svg(score), unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Each band holds 20% of the held-out test set, so they are drawn '
        'at equal width and position reads as percentile. The figure under each band is '
        'the default rate <i>actually observed</i> in it, not a prediction.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="insight"><b>Why the range is 462–603, not 300–850.</b> '
        'The model reaches AUC 0.69 on 14 coarse application fields, with no credit '
        'bureau score and no payment history. A narrow probability range maps to a '
        'narrow score range. Rescaling to fill 300–850 would look more familiar but '
        'would imply discriminating power the model does not have.</div>',
        unsafe_allow_html=True,
    )

with tab_why:
    st.markdown("### What moved this score")
    st.markdown(contributions_svg(contrib, values, labels), unsafe_allow_html=True)
    st.markdown(
        '<div class="note">SHAP values: each bar is that factor\'s contribution to '
        '<i>this</i> applicant\'s prediction, and they sum exactly to the model output. '
        'The applicant\'s own value is shown on the right.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### The decline reasons a lender would issue")
    st.markdown(
        '<div class="insight">Lenders must be able to justify a rejection, so a global '
        '"feature importance" chart is not enough. These are the top risk-increasing '
        'factors <i>for this applicant specifically</i>, which is what an adverse '
        'action notice requires.</div>',
        unsafe_allow_html=True,
    )
    worst = contrib[contrib > 0].sort_values(ascending=False).head(4)
    if len(worst):
        rows = []
        for i, (feat, _) in enumerate(worst.items(), 1):
            raw = values.get(feat, "")
            if isinstance(raw, float):
                raw = f"{raw:,.2f}".rstrip("0").rstrip(".")
            rows.append(
                f'<div class="reason-row"><div class="reason-name"><b>{i}.</b> '
                f'{labels.get(feat, feat)}</div>'
                f'<div class="reason-val" style="text-align:right">{raw}</div></div>'
            )
        st.markdown(f'<div class="card">{"".join(rows)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="card">No factors counted materially against this application.</div>',
            unsafe_allow_html=True,
        )

with tab_model:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### How it was built")
        st.markdown(
            """
- **163,987** Lending Club loans, 18.3% default rate
- **60/20/20** train / validation / test split
- **LightGBM**, AUC 0.690 · KS 0.283 · Brier 0.139
- Logistic regression baseline reaches 0.680, so the
  complex model wins by about **one AUC point**
            """
        )
        st.markdown(
            '<div class="insight"><b>Interest rate was excluded on purpose.</b> '
            'It was the strongest predictor in the data, worth 2.4 AUC points. But '
            'Lending Club <i>sets</i> the rate from its own risk model at origination, '
            'so it summarises a risk assessment that already happened. You could not '
            'use it to score a new applicant, because the rate does not exist until '
            'after the decision.</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("### What it cannot do")
        st.markdown(
            """
- **No out-of-time validation.** The data has no origination
  date, so only a random split was possible. Economic
  conditions shift; a random split flatters the model.
- **Accepted loans only.** Everyone Lending Club declined is
  absent, so this learned who defaults *among applicants
  already approved*. The fix is reject inference.
- **Modest discrimination.** AUC 0.69 reflects limited inputs:
  no bureau score, no payment history.
            """
        )
        st.markdown(
            '<div class="note">Demonstration only, not a lending decision. '
            'Full write-up, notebooks and code: '
            '<a href="https://github.com/meghajoshi-ds/credit-risk-scorecard">'
            'github.com/meghajoshi-ds/credit-risk-scorecard</a>.</div>',
            unsafe_allow_html=True,
        )
