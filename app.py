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
PAGE_BG = "#eef1f5"
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
      .stApp {{ background: {PAGE_BG}; }}
      .block-container {{ padding-top: 1.2rem; max-width: 1320px; }}
      #MainMenu, footer, header {{ visibility: hidden; }}

      /* Dark hero band, so the page does not open on a wall of white */
      .hero {{
        background: linear-gradient(135deg, #10243d 0%, #1b3b5f 55%, #24506f 100%);
        border-radius: 16px; padding: 1.5rem 1.8rem 1.35rem 1.8rem;
        margin-bottom: 1.1rem; color: #fff;
        box-shadow: 0 10px 28px rgba(16,36,61,.22);
      }}
      .hero-title {{
        font-size: 2.15rem; font-weight: 750; letter-spacing: -0.025em;
        margin: 0 0 .25rem 0; line-height: 1.12; color: #fff;
      }}
      .hero-sub {{ color: #bcd4ea; font-size: 1.02rem; margin: 0 0 1rem 0; }}
      .badge {{
        display: inline-block; padding: .3rem .7rem; margin: 0 .35rem .35rem 0;
        border: 1px solid rgba(255,255,255,.22); border-radius: 999px;
        background: rgba(255,255,255,.10); font-size: .79rem; color: #d7e6f5;
        backdrop-filter: blur(2px);
      }}
      .badge b {{ color: #fff; font-weight: 650; }}

      .card {{
        border: 1px solid {LINE}; border-radius: 14px; background: #fff;
        padding: 1.15rem 1.3rem; margin-bottom: .9rem;
        box-shadow: 0 2px 10px rgba(16,36,61,.06);
      }}
      .card-tinted {{ border: none; color: #fff; }}
      .card-label {{
        font-size: .72rem; text-transform: uppercase; letter-spacing: .09em;
        opacity: .75; margin-bottom: .4rem; font-weight: 700;
      }}
      .risk-hero {{ font-size: 2.9rem; font-weight: 750; line-height: 1; letter-spacing: -.02em; }}
      .big-sub {{ font-size: .86rem; opacity: .8; margin-top: .45rem; line-height: 1.5; }}

      .note {{ color: {INK_3}; font-size: .82rem; line-height: 1.55; }}
      .insight {{
        border-left: 4px solid {RISK_DOWN};
        background: linear-gradient(90deg, #eef5fe 0%, #f8fbff 100%);
        padding: .85rem 1.05rem; border-radius: 0 10px 10px 0;
        font-size: .89rem; color: {INK_2}; margin: .6rem 0 1rem 0;
      }}
      .reason-row {{
        display: grid; grid-template-columns: 34px 1fr auto; gap: .7rem;
        align-items: center; padding: .6rem .2rem; border-bottom: 1px solid {LINE};
      }}
      .rank {{
        width: 26px; height: 26px; border-radius: 8px; background: {RISK_UP}1a;
        color: {RISK_UP}; font-weight: 750; font-size: .82rem;
        display: flex; align-items: center; justify-content: center;
      }}
      .reason-name {{ font-size: .95rem; color: {INK}; font-weight: 500; }}
      .reason-val {{
        font-size: .82rem; color: {INK_2}; font-weight: 600;
        background: {NEUTRAL}; padding: .18rem .55rem; border-radius: 6px;
      }}
      h3 {{ font-size: 1.1rem !important; font-weight: 700 !important; color: {INK} !important; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
      .stTabs [data-baseweb="tab"] {{
        font-size: .95rem; font-weight: 600; background: #fff;
        border: 1px solid {LINE}; border-radius: 10px 10px 0 0; padding: .3rem 1rem;
      }}
      section[data-testid="stSidebar"] {{ background: #f7f9fb; border-right: 1px solid {LINE}; }}
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


def _luminance(hex_color):
    """WCAG relative luminance, used to pick legible text over a band colour."""
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ink_on(hex_color):
    """Text colour with adequate contrast against a filled band colour."""
    return "#ffffff" if _luminance(hex_color) < 0.42 else "#14212e"


def darken(hex_color, factor=0.62):
    """A darker step of a band colour, for text on a white surface."""
    h = hex_color.lstrip("#")
    rgb = [max(0, min(255, int(int(h[i:i + 2], 16) * factor))) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(rgb)


def percentile_of(score):
    """Position of a score across the five equal-population bands, 0..1."""
    for i, (lo, hi, *_rest) in enumerate(BANDS):
        if score < hi or i == len(BANDS) - 1:
            frac = float(np.clip((score - lo) / max(hi - lo, 1), 0, 1))
            return (i + frac) / len(BANDS)
    return 1.0


def dial_svg(score, prob, band_color, band_label):
    """Radial score dial: the arc is segmented by the five population bands and
    a marker sits at this applicant's position. The number is the headline; the
    arc supplies context that a bare number cannot."""
    W, H = 340, 215
    cx, cy, r = W / 2, 172.0, 128.0
    sweep = 180.0
    stroke = 20

    def pt(frac):
        a = np.radians(180 - sweep * frac)
        return cx + r * np.cos(a), cy - r * np.sin(a)

    def arc(f0, f1, color, width, opacity=1.0):
        x0, y0 = pt(f0)
        x1, y1 = pt(f1)
        large = 1 if (f1 - f0) > 0.5 else 0
        return (f'<path d="M {x0:.2f} {y0:.2f} A {r} {r} 0 {large} 1 {x1:.2f} {y1:.2f}" '
                f'fill="none" stroke="{color}" stroke-width="{width}" '
                f'stroke-linecap="butt" opacity="{opacity}"/>')

    parts = []
    n = len(BANDS)
    for i, (_lo, _hi, _label, _rate, color) in enumerate(BANDS):
        pad = 0.006
        parts.append(arc(i / n + pad, (i + 1) / n - pad, color, stroke, 0.95))

    frac = percentile_of(score)
    mx, my = pt(frac)
    parts.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="12" fill="#fff"/>')
    parts.append(f'<circle cx="{mx:.2f}" cy="{my:.2f}" r="8.5" fill="{band_color}"/>')

    parts.append(
        f'<text x="{cx}" y="{cy - 40}" text-anchor="middle" font-size="60" font-weight="750" '
        f'fill="{INK}" font-family="system-ui,sans-serif" letter-spacing="-2">{score}</text>'
    )
    parts.append(
        f'<text x="{cx}" y="{cy - 16}" text-anchor="middle" font-size="12.5" '
        f'fill="{INK_3}" font-family="system-ui,sans-serif">credit score &#183; {prob:.1%} risk</text>'
    )
    parts.append(
        f'<text x="{cx}" y="{cy + 8}" text-anchor="middle" font-size="13" font-weight="700" '
        f'fill="{darken(band_color)}" font-family="system-ui,sans-serif">{band_label.upper()}</text>'
    )
    lx, ly = pt(0)
    rx, ry = pt(1)
    parts.append(f'<text x="{lx - 2:.0f}" y="{ly + 20}" text-anchor="middle" font-size="10.5" '
                 f'fill="{INK_3}" font-family="system-ui,sans-serif">highest risk</text>')
    parts.append(f'<text x="{rx + 2:.0f}" y="{ry + 20}" text-anchor="middle" font-size="10.5" '
                 f'fill="{INK_3}" font-family="system-ui,sans-serif">lowest risk</text>')

    return (f'<svg viewBox="0 0 {W} {H}" width="100%" height="{H}" role="img" '
            f'aria-label="Score {score}, {band_label}">' + "".join(parts) + "</svg>")


def gauge_svg(score):
    """Where this applicant sits among applicants.

    Bands are drawn at equal width because they are population quintiles: each
    holds 20% of the book. The marker is placed by interpolating the score
    within its own band, so position reads as percentile. Each band carries its
    observed default rate, so colour never carries meaning alone.
    """
    W, H = 760, 112
    pad_l, pad_r = 10, 10
    track_y, track_h = 56, 24
    inner = W - pad_l - pad_r
    seg = inner / len(BANDS)

    # Percentile position: which band, then how far through it.
    pos = pad_l + percentile_of(score) * len(BANDS) * seg

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
        f'<text x="{pos:.1f}" y="{track_y - 26}" text-anchor="middle" font-size="13" '
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
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">Credit Risk Scorecard</div>'
    '<div class="hero-sub">Score a loan applicant, and see exactly which factors '
    'drove the decision.</div>'
    '<span class="badge">Model <b>LightGBM</b></span>'
    '<span class="badge">AUC <b>0.690</b></span>'
    '<span class="badge">KS <b>0.283</b></span>'
    '<span class="badge">Trained on <b>163,987 loans</b></span>'
    '<span class="badge">Calibrated: predicts <b>18.25%</b> vs actual <b>18.3%</b></span>'
    '<span class="badge">Excludes <b>int_rate</b> (leakage)</span>'
    '</div>',
    unsafe_allow_html=True,
)

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
    c1, c2 = st.columns([1, 1.25])
    with c1:
        st.markdown(
            f'<div class="card" style="padding:.6rem .8rem 1rem .8rem">'
            f'{dial_svg(score, prob, band_color, band_label)}</div>',
            unsafe_allow_html=True,
        )
    with c2:
        vs = prob * 100 / BASE_RATE
        st.markdown(
            f'<div class="card card-tinted" style="background:linear-gradient(135deg,'
            f'{band_color} 0%, {darken(band_color, 0.82)} 100%); color:{ink_on(band_color)}">'
            f'<div class="card-label">Default probability</div>'
            f'<div class="risk-hero">{prob:.1%}</div>'
            f'<div class="big-sub">Portfolio average is {BASE_RATE}%, so this applicant '
            f'carries <b>{vs:.1f}x</b> the average risk.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card"><div class="card-label" style="color:{INK_3}">'
            f'Portfolio position</div>'
            f'<div style="font-size:1.35rem;font-weight:700;color:{INK};margin-bottom:.3rem">'
            f'{band_label} of applicants</div>'
            f'<div class="note">Loans scoring in this band defaulted '
            f'<b style="color:{INK}">{band_rate}%</b> of the time in the held-out test set. '
            f'That is an observed outcome, not a prediction.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Where this applicant sits")
    st.markdown(f'<div class="card">{gauge_svg(score)}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Each band holds 20% of the held-out test set, so they are drawn '
        'at equal width and position reads as percentile. The figure under each band is '
        'the default rate <i>actually observed</i> in it.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="insight"><b>Why the range is 462&ndash;603, not 300&ndash;850.</b> '
        'The model reaches AUC 0.69 on 14 coarse application fields, with no credit '
        'bureau score and no payment history. A narrow probability range maps to a '
        'narrow score range. Rescaling to fill 300&ndash;850 would look more familiar but '
        'would imply discriminating power the model does not have.</div>',
        unsafe_allow_html=True,
    )

with tab_why:
    st.markdown("### What moved this score")
    st.markdown(f'<div class="card">{contributions_svg(contrib, values, labels)}</div>',
                unsafe_allow_html=True)
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
                f'<div class="reason-row"><div class="rank">{i}</div>'
                f'<div class="reason-name">{labels.get(feat, feat)}</div>'
                f'<div class="reason-val">{raw}</div></div>'
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
