# %% [markdown]
# # Phase 4 — Exploratory Data Analysis
#
# Three questions this notebook has to answer before any modelling starts:
#
# 1. Which variables actually separate good loans from bad ones?
# 2. Did the Phase 3 engineered features earn their place, or are they noise?
# 3. Is the `int_rate` leakage diagnosis from Phase 2 confirmed by a second,
#    independent measure?
#
# The main tool is **Weight of Evidence / Information Value**, which is the
# standard feature-screening method in credit scorecard work. It is used here
# rather than plain correlation because it handles categorical variables
# natively, captures non-monotonic relationships, and is what a credit risk
# team would expect to see.

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
FIGS = ROOT / "outputs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 60)
sns.set_theme(style="whitegrid", palette="deep")

df = pd.read_csv(ROOT / "data" / "lending_club_clean.csv")
TARGET = "bad_loan"
BASE_RATE = df[TARGET].mean()

print(f"{df.shape[0]:,} rows x {df.shape[1]} columns | base default rate {BASE_RATE:.2%}")

# %% [markdown]
# ## 4.1 Default rate by category
#
# Raw rates alone are misleading on thin categories, so every rate is shown
# with a **95% Wilson confidence interval**. A category with 30 loans can post
# a dramatic-looking default rate that is statistically indistinguishable from
# the base rate — the interval makes that obvious at a glance.

# %%
def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval — behaves correctly on small / extreme samples."""
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def rate_table(col):
    g = df.groupby(col, observed=True)[TARGET].agg(["size", "sum", "mean"])
    g.columns = ["n", "bad", "default_rate"]
    ci = [wilson_ci(r.bad, r.n) for r in g.itertuples()]
    g["ci_low"], g["ci_high"] = [c[0] for c in ci], [c[1] for c in ci]
    g["vs_base"] = g.default_rate / BASE_RATE
    # "Significant" = the interval excludes the overall base rate.
    g["significant"] = (g.ci_high < BASE_RATE) | (g.ci_low > BASE_RATE)
    return g.sort_values("default_rate", ascending=False)


for col in ["term", "home_ownership", "verification_status", "region", "purpose"]:
    print(f"===== {col} =====")
    t = rate_table(col).copy()
    for c in ["default_rate", "ci_low", "ci_high"]:
        t[c] = (t[c] * 100).round(2)
    t["vs_base"] = t.vs_base.round(2)
    print(t.to_string())
    print()

# %% [markdown]
# The standout results:
#
# - **`term`** is the strongest single categorical. 60-month loans default at
#   roughly twice the rate of 36-month loans. This is partly self-selection —
#   borrowers who need the longer term are those who cannot afford the shorter
#   one — which is exactly why `payment_to_income` was engineered in Phase 3.
# - **`purpose`** spans a wide range: `SMALL_BUSINESS` is far and away the
#   riskiest category, while `CAR`, `WEDDING` and `CREDIT_CARD` sit well below
#   base. Small business lending being riskiest than consumer lending is a
#   well-known pattern, and it showing up here is a good sanity check that the
#   data behaves like reality.
# - **`verification_status`** is the counter-intuitive one, and it gets its own
#   investigation in §4.5.
# - Some `purpose` levels have wide intervals and are **not** significant —
#   which is the point of computing them.

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

t = rate_table("purpose")
axes[0].barh(t.index, t.default_rate * 100, color="#4C72B0")
axes[0].errorbar(t.default_rate * 100, range(len(t)),
                 xerr=[(t.default_rate - t.ci_low) * 100, (t.ci_high - t.default_rate) * 100],
                 fmt="none", ecolor="#333", capsize=3, linewidth=1)
axes[0].axvline(BASE_RATE * 100, color="#C44E52", linestyle="--", label=f"base {BASE_RATE:.1%}")
axes[0].set_xlabel("Default rate (%)")
axes[0].set_title("Default rate by loan purpose (95% CI)")
axes[0].legend()

t2 = rate_table("term")
axes[1].bar(t2.index, t2.default_rate * 100, color="#4C72B0", width=0.5)
axes[1].errorbar(range(len(t2)), t2.default_rate * 100,
                 yerr=[(t2.default_rate - t2.ci_low) * 100, (t2.ci_high - t2.default_rate) * 100],
                 fmt="none", ecolor="#333", capsize=4)
axes[1].axhline(BASE_RATE * 100, color="#C44E52", linestyle="--", label=f"base {BASE_RATE:.1%}")
axes[1].set_ylabel("Default rate (%)")
axes[1].set_title("Default rate by term")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGS / "03_default_by_category.png", dpi=140)
plt.show()

# %% [markdown]
# ## 4.2 Weight of Evidence and Information Value
#
# For each bin of a variable:
#
# $$WoE = \ln\left(\frac{\%\ good}{\%\ bad}\right) \qquad
#   IV = \sum (\%good - \%bad) \times WoE$$
#
# The conventional reading of total IV in credit risk:
#
# | IV | Interpretation |
# |---|---|
# | < 0.02 | Not predictive — drop |
# | 0.02 – 0.1 | Weak |
# | 0.1 – 0.3 | Medium |
# | 0.3 – 0.5 | Strong |
# | **> 0.5** | **Suspiciously strong — check for leakage** |
#
# That last row is the one to watch, and it is why IV is a useful second
# opinion on `int_rate` rather than just a ranking tool.

# %%
def woe_iv(series, target, bins=10):
    """WoE/IV for one variable. Numerics are quantile-binned; NaN is its own bin."""
    if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 12:
        binned = series.astype("object").fillna("__MISSING__")
    else:
        binned = pd.qcut(series, bins, duplicates="drop").astype("object")
        binned = binned.fillna("__MISSING__")

    g = pd.DataFrame({"bin": binned, "y": target}).groupby("bin", observed=True).y.agg(["size", "sum"])
    g.columns = ["n", "bad"]
    g["good"] = g.n - g.bad

    # Laplace smoothing so an all-good or all-bad bin doesn't produce infinity.
    g["pct_bad"] = (g.bad + 0.5) / (g.bad.sum() + 0.5 * len(g))
    g["pct_good"] = (g.good + 0.5) / (g.good.sum() + 0.5 * len(g))
    g["woe"] = np.log(g.pct_good / g.pct_bad)
    g["iv"] = (g.pct_good - g.pct_bad) * g.woe
    g["default_rate"] = g.bad / g.n
    return g, g.iv.sum()


def iv_label(iv):
    if iv > 0.5:
        return "SUSPICIOUS - check leakage"
    if iv > 0.3:
        return "strong"
    if iv > 0.1:
        return "medium"
    if iv > 0.02:
        return "weak"
    return "not predictive"


FEATURES = [c for c in df.columns if c not in [TARGET, "addr_state"]]
iv_rows = []
woe_tables = {}
for col in FEATURES:
    tbl, iv = woe_iv(df[col], df[TARGET])
    woe_tables[col] = tbl
    iv_rows.append({"feature": col, "iv": round(iv, 4), "strength": iv_label(iv)})

iv_df = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
iv_df

# %% [markdown]
# ### Reading the IV table
#
# **`int_rate` tops the table at IV ≈ 0.38 — 2.6x the next strongest feature.**
#
# But note carefully what this does *not* say. It is classed "strong", and it
# does **not** cross the 0.5 threshold that the rule of thumb reserves for
# leakage. If I were relying on that heuristic alone, `int_rate` would sail
# through as an excellent feature.
#
# I am reporting that honestly rather than quietly rounding it up to fit the
# Phase 2 conclusion, because the lesson is the useful part: **the rule of
# thumb does not catch this.** What catches it is the mechanism — Lending Club
# set the rate from their own risk grade at origination — and the perfectly
# monotonic decile pattern that such a mechanism produces. Domain knowledge
# identified the leak; IV only corroborates that the column is unusually
# dominant. A threshold is a prompt to investigate, never a verdict.
#
# Below `int_rate`, the ordering is a realistic credit-risk picture: loan size
# relative to income and loan term lead, then revolving utilisation and DTI.
# Geography contributes essentially nothing (`region` IV = 0.0008), which
# retrospectively justifies collapsing 50 states rather than one-hot encoding
# them — those 50 columns would have bought nothing.

# %%
plot_df = iv_df[iv_df.iv > 0.005].sort_values("iv")
# Highlight the leaky column — flagged by domain reasoning, not by the IV bands.
colors = ["#C44E52" if f == "int_rate" else "#4C72B0" for f in plot_df.feature]

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(plot_df.feature, plot_df.iv, color=colors)
for x, lab in [(0.02, "weak"), (0.1, "medium"), (0.3, "strong"), (0.5, "suspicious")]:
    ax.axvline(x, color="#999", linestyle=":", linewidth=1)
    ax.text(x, len(plot_df) - 0.4, lab, fontsize=7, color="#666", ha="center")
ax.set_xlabel("Information Value")
ax.set_title("Feature predictive power\n(red = excluded as leakage, on domain grounds — note it sits below 'suspicious')")
plt.tight_layout()
plt.savefig(FIGS / "03_information_value.png", dpi=140)
plt.show()

# %% [markdown]
# ### WoE detail for the strongest legitimate feature

# %%
best = iv_df[iv_df.feature != "int_rate"].iloc[0].feature
print(f"WoE breakdown — {best}\n")
t = woe_tables[best].copy()
t["default_rate"] = (t.default_rate * 100).round(2)
print(t[["n", "bad", "default_rate", "woe", "iv"]].round(4).to_string())

# %% [markdown]
# Negative WoE means a bin is worse than average, positive means better. A
# scorecard is in effect a weighted sum of these — which is why a WoE table
# like this is directly explainable to a declined applicant, and why credit
# teams keep using logistic regression long after gradient boosting existed.

# %% [markdown]
# ## 4.3 Did the engineered features earn their place?
#
# The honest test: compare each engineered feature's IV against the raw columns
# it was built from. If the derived feature does not beat its own inputs, it
# added complexity for nothing and should be dropped.

# %%
comparisons = {
    "loan_to_income": ["loan_amnt", "annual_inc"],
    "payment_to_income": ["loan_amnt", "annual_inc", "term_months"],
    "acct_open_rate": ["total_acc", "longest_credit_length"],
    "has_delinquency": ["delinq_2yrs"],
    "log_annual_inc": ["annual_inc"],
    "emp_length_missing": ["emp_length"],
    "over_limit": ["revol_util"],
}
ivs = dict(zip(iv_df.feature, iv_df.iv))

print(f"{'engineered':<22}{'IV':>8}   {'best input':<24}{'input IV':>9}   verdict")
print("-" * 86)
for child, parents in comparisons.items():
    best_parent = max(parents, key=lambda p: ivs.get(p, 0))
    child_iv, parent_iv = ivs[child], ivs[best_parent]
    if child_iv > parent_iv:
        verdict = f"KEEP  (+{child_iv - parent_iv:.3f})"
    elif child_iv > 0.02:
        verdict = "keep - adds a different view"
    else:
        verdict = "weak on its own"
    print(f"{child:<22}{child_iv:>8.4f}   {best_parent:<24}{parent_iv:>9.4f}   {verdict}")

# %% [markdown]
# This is a mixed scorecard, and reporting it honestly is more useful than
# claiming all nine features were inspired.
#
# **The clear win is `loan_to_income`** (IV 0.145). It beats every column it
# was built from — nearly 2.3x `annual_inc` alone — and is the strongest
# non-leaky feature in the entire dataset. That is what good feature
# engineering looks like: no new information, rearranged into the form the
# question is actually about.
#
# **`payment_to_income` disappoints** at 0.058. It does not beat `term_months`.
# In hindsight the reason is clear — by dividing by the term I folded the term
# *into* the ratio, and in doing so blurred a variable that is powerful on its
# own. It survives only as a different view of affordability, and Phase 5's
# feature-selection step is where it has to prove that or be dropped.
#
# **Three features are weak: `acct_open_rate`, `has_delinquency`,
# `over_limit`.** Each scores below 0.02. `over_limit` is unsurprising in
# retrospect — it flags 289 rows out of 163,987, so however sharp the effect,
# it cannot move a portfolio-level statistic. I keep them for the tree model,
# which can use narrow features inside interactions, but I am not going to
# pretend they are carrying the model.
#
# **`log_annual_inc` scoring identically to `annual_inc` is expected, not a
# failure.** IV is computed on quantile bins and a log transform is monotonic,
# so it cannot move the bins. The transform exists to fix the *scale* the
# logistic coefficient is fitted on — a benefit invisible to IV by
# construction. Phase 5 is where it pays off.

# %% [markdown]
# ### The `emp_length_missing` paradox — significance vs. predictive power
#
# Worth pausing on, because it looks like a contradiction. Phase 2 found this
# flag overwhelmingly significant (Fisher p < 0.001, 1.46x default lift). Here
# it scores IV 0.009 — "not predictive".
#
# Both are correct, because they answer different questions:
#
# - **The Fisher test asks:** is the effect real, or chance? With 5,804 rows
#   behind it, the answer is emphatically real.
# - **IV asks:** how much does this move risk *across the whole portfolio*?
#   The flag covers 3.5% of applicants, so even a genuine 1.46x lift on that
#   slice barely shifts a population-weighted measure.
#
# A large effect on a small group is exactly this: unarguably real, modest in
# aggregate. The practical conclusion is to **keep the flag** — it is honest,
# costless, and materially sharpens the model's judgement on the specific
# applicants it fires for — while not expecting it near the top of a global
# importance chart in Phase 7. Confusing "statistically significant" with
# "important" is one of the most common analytical errors, and the two
# measures disagreeing here is a good illustration of why both get computed.

# %% [markdown]
# ## 4.4 Correlation structure
#
# Checking for multicollinearity before fitting a linear model — correlated
# inputs make logistic regression coefficients unstable and un-interpretable,
# which defeats the whole point of using it here.

# %%
num_cols = df.select_dtypes("number").columns.drop(TARGET)
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, square=True, cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_title("Correlation between numeric features")
plt.tight_layout()
plt.savefig(FIGS / "03_correlation.png", dpi=140)
plt.show()

# %%
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
strong = pairs[pairs.abs() > 0.7].sort_values(key=abs, ascending=False)
print("Pairs correlated above |0.7|:\n")
print(strong.round(3).to_string() if len(strong) else "  none")

# %% [markdown]
# The strong pairs are the ones deliberately created in Phase 3 — a derived
# feature is correlated with its own parent by construction
# (`log_annual_inc`/`annual_inc`, `has_delinquency`/`delinq_2yrs`,
# `loan_to_income`/`payment_to_income`).
#
# This is a real constraint on the linear model, not a cosmetic one. Phase 5
# handles it by giving logistic regression a curated subset that keeps one
# variable from each correlated pair, while LightGBM gets everything — trees
# are untroubled by collinearity, they just split on whichever is more useful.

# %% [markdown]
# ## 4.5 The `verification_status` puzzle
#
# The rate table showed something that looks backwards: **verified** applicants
# default *more* than unverified ones. Taken at face value that would suggest
# income verification makes borrowers riskier, which is nonsense.
#
# This is worth chasing, because "the data says something implausible" is
# usually the data telling you about a process, not about borrowers.

# %%
print(rate_table("verification_status")[["n", "default_rate", "vs_base"]].round(4).to_string())
print()
print("Median loan size and income by verification status:")
print(df.groupby("verification_status").agg(
    median_loan=("loan_amnt", "median"),
    median_income=("annual_inc", "median"),
    median_ltv=("loan_to_income", "median"),
    pct_60_month=("term_months", lambda s: (s == 60).mean()),
).round(3).to_string())

# %% [markdown]
# There it is. Verified applicants take **substantially larger loans**, borrow
# a higher multiple of their income, and take the 60-month term far more often.
#
# The causality runs the other way round from how the raw rate reads: Lending
# Club *chose to verify* the applications that were already riskier — bigger
# loans get more scrutiny. `verification_status` is therefore partly a marker
# of the lender's own suspicion, not a property of the borrower.
#
# It is a mild cousin of the `int_rate` problem: another column carrying a
# trace of the lender's internal process. Unlike `int_rate` it is weak enough
# to keep, but it is exactly the kind of variable to flag before someone reads
# a SHAP plot in Phase 7 and concludes that verification causes default.

# %% [markdown]
# ## 4.6 How the strongest features separate the two classes

# %%
top = [f for f in iv_df.feature if f not in ("int_rate",)][:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, col in zip(axes.ravel(), top):
    if not pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 12:
        t = rate_table(col)
        ax.bar(t.index.astype(str), t.default_rate * 100, color="#4C72B0")
        ax.axhline(BASE_RATE * 100, color="#C44E52", linestyle="--")
        ax.set_ylabel("default rate (%)")
        ax.tick_params(axis="x", rotation=30)
    else:
        binned = pd.qcut(df[col], 10, duplicates="drop")
        t = df.groupby(binned, observed=True)[TARGET].mean() * 100
        ax.plot(range(len(t)), t.values, marker="o", color="#4C72B0")
        ax.axhline(BASE_RATE * 100, color="#C44E52", linestyle="--")
        ax.set_ylabel("default rate (%)")
        ax.set_xlabel("decile (low to high)")
    ax.set_title(col, fontsize=11)
fig.suptitle("Default rate across the strongest non-leaky features", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "03_top_features.png", dpi=140)
plt.show()

# %% [markdown]
# ## Phase 4 summary
#
# **`int_rate` dominates, but IV alone would not have caught it.** At 0.38 it
# is 2.6x the next feature yet below the 0.5 "suspicious" threshold. The
# leakage case rests on the mechanism and the monotonic decile pattern, not on
# a heuristic — worth knowing which piece of evidence is actually load-bearing.
#
# **The engineered features are a mixed result, honestly reported.**
# `loan_to_income` is the best non-leaky feature in the dataset.
# `payment_to_income` underperforms because dividing by term blurred a strong
# variable. Three features are weak in aggregate and are kept only for the
# tree model.
#
# **Significance and importance are different things.** `emp_length_missing` is
# highly significant *and* low-IV, because it affects 3.5% of applicants
# sharply rather than everyone mildly. Both measures are right.
#
# **Found a second, subtler process artefact.** `verification_status` looks
# backwards until you see that verified applicants borrow more — the variable
# partly encodes the lender's own suspicion. Weak enough to keep, important
# enough to flag before anyone interprets it causally.
#
# **Multicollinearity is real but understood.** It sits almost entirely between
# derived features and their parents, so Phase 5 gives logistic regression a
# curated subset and LightGBM the full set.
#
# Risk drivers, in order: loan-to-income, loan term, revolving utilisation,
# DTI. Geography contributes essentially nothing — which retrospectively
# justifies the Phase 3 decision to collapse 50 states into 4 regions.
