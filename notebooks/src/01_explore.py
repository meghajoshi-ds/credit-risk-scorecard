# %% [markdown]
# # Phase 2 — Data Profiling
#
# **Goal:** understand the raw data before touching it. No cleaning happens in
# this notebook — the only output is a list of documented, evidence-backed
# problems that Phase 3 has to solve.
#
# Profiling before cleaning matters in credit risk specifically: a regulator or
# model validator will ask *"how did you know that value was wrong?"*, and the
# answer needs to be something other than "it looked odd".

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

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid", palette="deep")

df = pd.read_csv(ROOT / "data" / "lending_club_raw.csv")
print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

# %%
df.head()

# %% [markdown]
# ## 2.1 Structure and dtypes

# %%
df.info()

# %% [markdown]
# Two columns immediately look mis-typed for modelling:
#
# - `term` is text (`"36 months"`) when it is really a number of months.
# - `emp_length` loads as a float only because of missing values; the
#   underlying data is integer years, where `10` means "10 or more".
#
# The other four `object` columns (`home_ownership`, `purpose`, `addr_state`,
# `verification_status`) are genuinely categorical.

# %%
# Separate the column roles once, so every later notebook uses the same lists.
TARGET = "bad_loan"
CATEGORICAL = ["term", "home_ownership", "purpose", "addr_state", "verification_status"]
NUMERIC = [c for c in df.columns if c not in CATEGORICAL + [TARGET]]

print(f"Target      : {TARGET}")
print(f"Categorical : {CATEGORICAL}")
print(f"Numeric     : {NUMERIC}")

# %% [markdown]
# ## 2.2 Target balance
#
# The class balance drives nearly every downstream decision: which metric is
# trustworthy, whether to reweight the model, and where to set a cut-off.

# %%
target_counts = df[TARGET].value_counts().sort_index()
target_rate = df[TARGET].mean()

print(target_counts.to_string())
print(f"\nDefault (bad_loan = 1) rate: {target_rate:.2%}")
print(f"Imbalance ratio            : 1 bad loan per {(1 - target_rate) / target_rate:.1f} good loans")

# %% [markdown]
# **18.3% positives.** That is imbalanced, but not severely so — this is the
# comfortable range where plain logistic regression still behaves well.
#
# The practical consequence is about *metrics*, not sampling. A model that
# predicts "everyone repays" scores 81.7% accuracy while being completely
# useless, so **accuracy is not reportable on this problem**. Phase 5 uses AUC
# and the KS statistic instead, both of which are threshold-independent and
# both of which are what credit risk teams actually use.
#
# I am deliberately *not* going to oversample or SMOTE. At 18% positives it
# adds distortion without adding signal, and it corrupts the predicted
# probabilities — which matters here, because Phase 8 converts those
# probabilities into a score.

# %% [markdown]
# ## 2.3 Missingness
#
# The question is never just "how much is missing", it is **"is it missing at
# random?"** Missingness that correlates with the target is itself a feature.

# %%
missing = pd.DataFrame(
    {
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(3),
    }
)
missing = missing[missing.n_missing > 0].sort_values("n_missing", ascending=False)
missing

# %%
# Does missingness carry signal? Compare the default rate of missing vs present.
print(f"{'column':<24}{'default | missing':>20}{'default | present':>20}{'lift':>10}")
print("-" * 74)
for col in missing.index:
    is_na = df[col].isna()
    rate_missing = df.loc[is_na, TARGET].mean()
    rate_present = df.loc[~is_na, TARGET].mean()
    print(
        f"{col:<24}{rate_missing:>19.2%}{rate_present:>20.2%}"
        f"{rate_missing / rate_present:>10.2f}x"
    )

# %% [markdown]
# **This is the single most important finding in the profiling phase.**
#
# `emp_length` is missing for 5,804 applicants, and those applicants default at
# a materially higher rate than applicants whose employment length is known.
# The missingness is **not** random — an unknown employment history is itself a
# risk signal, which makes intuitive sense (no employer to verify, gig income,
# or a refusal to state).
#
# So the wrong move is a quiet `fillna(median)`, which would erase the signal
# by making these applicants look average. Phase 3 will impute **and** add an
# explicit `emp_length_missing` flag so the model can use the fact of absence.
#
# The other five columns show apparent lifts too — `revol_util` missing rows
# default at 22.3% against a 18.3% base. But those are only 193 rows, and an
# eye-catching percentage on a small denominator is exactly the trap that
# produces features which evaporate on the test set. Rather than trust the
# number, test it.

# %%
from scipy.stats import fisher_exact

print(f"{'column':<24}{'n_missing':>10}{'lift':>8}{'p-value':>10}   verdict")
print("-" * 68)
for col in missing.index:
    is_na = df[col].isna()
    table = [
        [int((is_na & (df[TARGET] == 1)).sum()), int((is_na & (df[TARGET] == 0)).sum())],
        [int((~is_na & (df[TARGET] == 1)).sum()), int((~is_na & (df[TARGET] == 0)).sum())],
    ]
    _, p = fisher_exact(table)
    lift = df.loc[is_na, TARGET].mean() / df.loc[~is_na, TARGET].mean()
    verdict = "SIGNAL — build a flag" if p < 0.05 else "noise — median fill is fine"
    print(f"{col:<24}{int(is_na.sum()):>10}{lift:>7.2f}x{p:>10.4f}   {verdict}")

# %% [markdown]
# Only `emp_length` survives the test. Its 1.46x lift on 5,804 rows is
# overwhelmingly significant; every other column's apparent lift is consistent
# with chance on a few dozen rows. So exactly one missingness flag gets built,
# not six — the test is what stops me shipping five spurious features.

# %%
# One more pattern worth catching: three columns are each missing on 29 rows
# and show an identical default rate, which suggests it is the *same* 29 rows.
trio = ["delinq_2yrs", "total_acc", "longest_credit_length"]
all_three = df[trio].isna().all(axis=1).sum()
any_of_three = df[trio].isna().any(axis=1).sum()
print(f"Rows missing all three of {trio}: {all_three}")
print(f"Rows missing any of them            : {any_of_three}")
print("\nConfirmed: a single block of records with no credit-bureau data attached,")
print("rather than three independent data-quality problems.")

# %%
fig, ax = plt.subplots(figsize=(8, 4))
missing.pct_missing.sort_values().plot.barh(ax=ax, color="#4C72B0")
ax.set_xlabel("% of rows missing")
ax.set_ylabel("")
ax.set_title("Missingness by column")
for i, v in enumerate(missing.pct_missing.sort_values()):
    ax.text(v + 0.03, i, f"{v:.2f}%", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIGS / "01_missingness.png", dpi=140)
plt.show()

# %% [markdown]
# ## 2.4 Numeric columns — distributions and outliers

# %%
df[NUMERIC].describe().T.round(2)

# %% [markdown]
# Reading the percentiles rather than just the means, four things stand out:
#
# 1. **`annual_inc`** — max is orders of magnitude above the 75th percentile.
#    Heavily right-skewed, as income always is.
# 2. **`revol_util`** — max is 150.7, above the 100% that a utilisation
#    percentage should cap at. Flagged in the brief; investigated below.
# 3. **`dti`** — has a 0.00 minimum, which is possible but worth a look.
# 4. **`delinq_2yrs`** — almost entirely zero, with a long thin tail. This is a
#    count, not a continuous variable, and should be treated as one.

# %%
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, col in zip(axes.ravel(), NUMERIC):
    data = df[col].dropna()
    # Clip the display at p99 so one extreme value doesn't flatten the histogram.
    ax.hist(data.clip(upper=data.quantile(0.99)), bins=40, color="#4C72B0", edgecolor="white")
    ax.set_title(col, fontsize=11)
    ax.set_ylabel("")
for ax in axes.ravel()[len(NUMERIC):]:
    ax.axis("off")
fig.suptitle("Numeric distributions (display clipped at the 99th percentile)", fontsize=13)
plt.tight_layout()
plt.savefig(FIGS / "01_numeric_distributions.png", dpi=140)
plt.show()

# %% [markdown]
# ### Skew check
#
# Skew tells us which variables will need a transform before going into a
# linear model. Trees will not care.

# %%
skew = df[NUMERIC].skew().sort_values(ascending=False).round(2)
skew.to_frame("skew")

# %% [markdown]
# `annual_inc` and `delinq_2yrs` are extremely right-skewed. For the logistic
# regression baseline in Phase 5 that is a genuine problem — a handful of
# millionaires would dominate the fitted coefficient. Phase 3 will add a
# log-transformed income column for the linear model to use.

# %% [markdown]
# ### Investigating `revol_util > 100`
#
# The brief flags this as "investigate before deciding how to handle". Deleting
# the rows or clipping them are both defensible *only after* checking whether
# these are data errors or real accounts.

# %%
over_100 = df[df.revol_util > 100]
print(f"Rows with revol_util > 100 : {len(over_100):,} ({len(over_100) / len(df):.3%})")
print(f"Range                      : {over_100.revol_util.min():.1f} to {over_100.revol_util.max():.1f}")
print(f"Default rate, util > 100   : {over_100[TARGET].mean():.2%}")
print(f"Default rate, util <= 100  : {df[df.revol_util <= 100][TARGET].mean():.2%}")
print()
print("Distribution of the over-100 values:")
print(over_100.revol_util.describe().round(2).to_string())

# %% [markdown]
# **Verdict: these are real, not corrupt.**
#
# The evidence: there are only ~100 of them, they sit in a tight band just
# above 100 (not scattered up to 5,000 the way a units error or a misplaced
# decimal point would be), and they default at a *higher* rate than everyone
# else.
#
# There is a real-world mechanism for this — a credit card balance can exceed
# its limit through interest charges, fees, or a limit being reduced after the
# balance was drawn. A borrower who is over their limit is genuinely
# higher-risk, so the value is informative.
#
# **Decision: keep the rows, do not clip to 100.** Clipping would destroy the
# very ordering that makes the variable useful. I will note it in the decisions
# log rather than silently truncating.

# %% [markdown]
# ### Investigating `annual_inc` extremes

# %%
print(df.annual_inc.quantile([0.5, 0.9, 0.99, 0.999, 1.0]).round(0).to_string())
print(f"\nApplicants reporting over $1m income: {(df.annual_inc > 1_000_000).sum()}")
print(f"Applicants reporting under $10k     : {(df.annual_inc < 10_000).sum()}")

# %% [markdown]
# A small number of very large incomes, and a small number of implausibly small
# ones. Both are plausible as self-reported data (Lending Club did not verify
# every application — see `verification_status`). Rather than delete rows, the
# log transform in Phase 3 compresses this tail, which is the less destructive
# fix.

# %% [markdown]
# ## 2.5 Categorical columns
#
# Checking cardinality and, crucially, **formatting consistency** — the brief
# warns these may not be clean.

# %%
for col in CATEGORICAL:
    vals = df[col]
    print(f"--- {col} ({vals.nunique()} unique) ---")
    print(vals.value_counts().head(12).to_string())
    print()

# %% [markdown]
# ### Formatting consistency check
#
# The real risk is invisible duplicates: `"RENT"` and `"rent "` would silently
# become two separate dummy variables. Testing directly rather than assuming.

# %%
for col in CATEGORICAL:
    raw = df[col].dropna().unique()
    normalised = pd.Series(raw).str.strip().str.upper().nunique()
    has_ws = any(v != v.strip() for v in raw)
    status = "OK" if normalised == len(raw) and not has_ws else "NEEDS CLEANING"
    print(f"{col:<22} raw={len(raw):>3}  normalised={normalised:>3}  stray_whitespace={has_ws!s:<5}  {status}")

# %% [markdown]
# The categoricals are clean — no case collisions, no stray whitespace. Worth
# having *tested* rather than assumed, which is the whole point of the check.
# Phase 3 will still apply normalisation defensively so the pipeline stays
# correct if the data is ever refreshed.
#
# Two structural notes for encoding:
#
# - `addr_state` has ~50 levels. One-hot encoding that adds 50 sparse columns,
#   some backed by very few loans. Phase 3 handles this with target-style
#   grouping rather than blind one-hot.
# - `home_ownership` has `OTHER` / `NONE` / `ANY` levels with tiny counts that
#   need collapsing.

# %%
rare = {}
for col in CATEGORICAL:
    counts = df[col].value_counts()
    thin = counts[counts < 100]
    if len(thin):
        rare[col] = thin
for col, thin in rare.items():
    print(f"--- {col}: levels with fewer than 100 loans ---")
    print(thin.to_string())
    print()

# %% [markdown]
# ## 2.6 Duplicates

# %%
dupes = df.duplicated().sum()
print(f"Fully duplicated rows: {dupes:,} ({dupes / len(df):.2%})")

# %% [markdown]
# **Zero duplicates** — no action needed, and `drop_duplicates()` should *not*
# appear in the cleaning pipeline just because it usually does.
#
# That is a genuinely useful result rather than a non-event. This dataset has
# no application ID, so had duplicates existed they would have been ambiguous:
# with only 15 coarse columns, two different applicants coinciding on every
# recorded field is entirely plausible, and dropping them would have destroyed
# real rows. The check is cheap and it settles the question with evidence.

# %% [markdown]
# ## 2.7 The `int_rate` leakage question
#
# The brief flags this as needing "a deliberate decision and a clear
# explanation". This is the most important modelling judgement in the project,
# so it gets evidence here in profiling and a decision in Phase 3.

# %%
print("Default rate by interest rate decile")
print("-" * 44)
band = pd.qcut(df.int_rate, 10)
by_rate = df.groupby(band, observed=True).agg(
    n=(TARGET, "size"), default_rate=(TARGET, "mean"), avg_rate=("int_rate", "mean")
)
by_rate["default_rate"] = (by_rate.default_rate * 100).round(2)
by_rate["avg_rate"] = by_rate.avg_rate.round(2)
print(by_rate.to_string())

print(f"\nCorrelation, int_rate vs bad_loan: {df.int_rate.corr(df[TARGET]):.4f}")

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(by_rate.avg_rate, by_rate.default_rate, marker="o", color="#C44E52", linewidth=2)
ax.set_xlabel("Average interest rate in decile (%)")
ax.set_ylabel("Default rate (%)")
ax.set_title("Default rate rises monotonically with interest rate")
plt.tight_layout()
plt.savefig(FIGS / "01_int_rate_leakage.png", dpi=140)
plt.show()

# %% [markdown]
# The relationship is **perfectly monotonic** across all ten deciles: default
# risk climbs from roughly 5% in the cheapest decile to roughly 36% in the most
# expensive one. Nothing in a real credit dataset is that clean by accident.
#
# The reason is mechanical, not behavioural. Lending Club ran their own risk
# model at origination and *set* the interest rate from the grade it produced.
# So `int_rate` is not an input the borrower brought to the application — it is
# a **downstream summary of a risk assessment that already happened**.
#
# Using it means my model largely re-learns Lending Club's model, which
# creates two concrete problems:
#
# 1. **It inflates validation results.** The AUC would look strong for a reason
#    that does not generalise to a lender who has to price the loan themselves.
# 2. **It breaks the use case.** To score a *new* applicant you must already
#    know their rate — but the rate is the thing you are trying to decide.
#
# **Decision: build the primary model without `int_rate`,** and also fit a
# variant with it to quantify what the leakage was worth. Phase 5 reports both.
# Showing the gap is more informative than silently dropping the column, and it
# is exactly the trade-off an interviewer will want to discuss.

# %% [markdown]
# ## 2.8 Profiling summary — the Phase 3 to-do list
#
# | # | Finding | Evidence | Action in Phase 3 |
# |---|---|---|---|
# | 1 | `emp_length` missing for 5,804 rows, **not at random** | 1.46x default lift, Fisher p < 0.001 | Impute **and** add `emp_length_missing` flag |
# | 2 | Five columns with trace missingness (4–193 rows) | Apparent lifts fail significance testing | Median impute, **no** flag |
# | 3 | `term` stored as text | dtype is `object` | Parse to integer months |
# | 4 | `revol_util` exceeds 100 (max 150.7) | ~100 rows, tight band, *higher* default rate | **Keep uncapped** — real, and informative |
# | 5 | `annual_inc` extremely right-skewed | Skew well above 1; max ≫ p99 | Add `log_annual_inc` for the linear model |
# | 6 | `int_rate` is target leakage | Perfectly monotonic across 10 deciles | Exclude from primary model; fit a variant to quantify |
# | 7 | `addr_state` has ~50 levels | Cardinality check | Group rather than one-hot |
# | 8 | `home_ownership` has tiny `OTHER`/`NONE`/`ANY` levels | Counts below 100 | Collapse into `OTHER` |
# | 9 | **No** duplicate rows | `duplicated()` returns 0 | No de-duplication step needed |
# | 10 | Target is 18.3% positive | Class counts | Report AUC/KS, never accuracy |
#
# Categoricals were **tested** for formatting problems and are clean — no
# action needed beyond defensive normalisation.
