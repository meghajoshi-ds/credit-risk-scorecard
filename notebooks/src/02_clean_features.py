# %% [markdown]
# # Phase 3 — Cleaning & Feature Engineering
#
# Every change in this notebook traces back to a numbered finding from the
# Phase 2 profiling table. Nothing is cleaned "because that's what you do" —
# if I cannot point at the evidence, the step does not happen.
#
# **One structural decision up front.** This notebook performs only
# *row-wise deterministic* transforms — ones whose result for a given row does
# not depend on any other row. Type fixes, ratios, flags, category mapping.
#
# Anything that has to *learn* a value from the data — the median used to fill
# missing values, the categories seen when encoding — is deliberately **not**
# done here. Those are fitted inside the scikit-learn `Pipeline` in Phase 5, on
# the training fold only.
#
# The reason is leakage. If I compute a median across all 163,987 rows and use
# it to fill the test set, then test-set information has reached the model
# before evaluation, and the reported AUC is mildly optimistic. The effect is
# small for a median, but the *habit* is what matters: the moment the same
# pattern is applied to a target-encoded variable it stops being small. So the
# clean CSV this notebook writes deliberately still contains `NaN`s.

# %%
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 60)

raw = pd.read_csv(ROOT / "data" / "lending_club_raw.csv")
df = raw.copy()
print(f"Raw: {raw.shape[0]:,} rows x {raw.shape[1]} columns")

# %% [markdown]
# ## 3.1 Finding #3 — `term` is stored as text
#
# Parsing with a regex rather than a hard-coded `{"36 months": 36}` lookup, so
# that a refreshed file containing a 48-month product does not silently produce
# `NaN`. I then assert the result, because a parse that quietly fails is worse
# than one that crashes.

# %%
df["term_months"] = df["term"].str.extract(r"(\d+)").astype(int)

assert df.term_months.notna().all(), "term parsing produced nulls"
assert set(df.term_months.unique()) <= {36, 60}, "unexpected term value"
print(df.groupby("term_months").size().to_string())
print("\nterm -> term_months parsed and validated")

# %% [markdown]
# ## 3.2 Finding #7/#8 — categorical cleaning
#
# Phase 2 *tested* these columns and found them already consistent. I still
# normalise defensively: the cost is one line, and it makes the pipeline
# correct if the data is ever refreshed from a messier export.

# %%
CATS = ["home_ownership", "purpose", "addr_state", "verification_status"]
for col in CATS:
    df[col] = df[col].str.strip().str.upper()
print("Normalised (strip + uppercase) — no values changed in this file, by design.")

# %% [markdown]
# ### Collapsing rare levels in `home_ownership`
#
# `NONE` (30 loans) and `ANY` (1 loan) cannot support a stable coefficient. A
# single loan becomes a dummy variable that the model can fit perfectly, which
# is pure overfitting. They are semantically close to `OTHER`, so they merge.

# %%
before = df.home_ownership.value_counts()
df["home_ownership"] = df.home_ownership.replace({"NONE": "OTHER", "ANY": "OTHER"})
after = df.home_ownership.value_counts()

print("before -> after")
print(pd.concat([before, after], axis=1, keys=["before", "after"]).fillna(0).astype(int).to_string())

# %% [markdown]
# ### Finding #7 — `addr_state` has 50 levels
#
# One-hot encoding 50 states adds 50 mostly-sparse columns, four of which are
# backed by fewer than 15 loans (ME has 3). Those columns cannot generalise.
#
# The tempting fix is **target encoding** — replace each state with its
# observed default rate. I am not doing that, and the reason is worth stating:
# target encoding leaks the label into the feature, and on levels with 3 rows
# it essentially memorises those rows' outcomes.
#
# Instead I map states to the four **US Census regions**. This mapping is
# external domain knowledge, fixed in advance, and completely independent of
# the target — so it cannot leak by construction, and it collapses 50 sparse
# columns into 4 dense ones. I keep the raw state too, so Phase 4 can still
# check whether region actually captures the geography.

# %%
CENSUS_REGION = {
    "NORTHEAST": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "MIDWEST": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "SOUTH": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "DC", "WV", "AL", "KY",
              "MS", "TN", "AR", "LA", "OK", "TX"],
    "WEST": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
}
state_to_region = {s: r for r, states in CENSUS_REGION.items() for s in states}

df["region"] = df.addr_state.map(state_to_region)

unmapped = df.loc[df.region.isna(), "addr_state"].unique()
assert len(unmapped) == 0, f"states missing from the region map: {unmapped}"
print(df.region.value_counts().to_string())
print("\nAll 50 states mapped — no unmapped values.")

# %% [markdown]
# ## 3.3 Finding #1 — the `emp_length` missingness flag
#
# The headline result from profiling: `emp_length` is missing for 5,804
# applicants who default at 1.46x the rate of applicants whose employment
# length is known (Fisher exact p < 0.001).
#
# The flag has to be created **before** imputation, because imputation is what
# destroys the information. It is row-wise and deterministic, so it belongs
# here rather than in the pipeline.

# %%
df["emp_length_missing"] = df.emp_length.isna().astype(int)

print(f"Flagged {df.emp_length_missing.sum():,} applicants with unknown employment length")
print(f"Default rate, flag = 1 : {df.loc[df.emp_length_missing == 1, 'bad_loan'].mean():.2%}")
print(f"Default rate, flag = 0 : {df.loc[df.emp_length_missing == 0, 'bad_loan'].mean():.2%}")

# %% [markdown]
# Note what would have happened *without* this flag: after median imputation
# these 5,804 applicants would carry `emp_length = 6` and look indistinguishable
# from an average borrower, and their 26% default rate would become invisible
# to the model. One line of code recovers it.

# %% [markdown]
# ## 3.4 Feature engineering
#
# Five new features. Every one uses only information available **at the moment
# of application** — nothing derived from the loan's outcome, and nothing
# derived from `int_rate` (see §3.6).

# %% [markdown]
# ### Feature 1 — `loan_to_income`
#
# The single most standard affordability ratio in consumer lending. A $30k loan
# means something entirely different to someone earning $40k than to someone
# earning $400k, and neither `loan_amnt` nor `annual_inc` alone captures that
# interaction. A linear model cannot construct a ratio on its own, so handing
# it one is real work rather than decoration.

# %%
df["loan_to_income"] = df.loan_amnt / df.annual_inc

print(df.loan_to_income.describe().round(3).to_string())

# %% [markdown]
# ### Feature 2 — `payment_to_income`
#
# `loan_to_income` ignores that a 60-month loan spreads the same principal over
# nearly twice as long. This approximates the monthly principal payment as a
# share of monthly income.
#
# It is deliberately an *approximation that excludes interest*, because the
# only way to include interest is via `int_rate` — the leaky column. I would
# rather have a slightly cruder feature that is honest at scoring time than a
# precise one I cannot compute for a new applicant.

# %%
df["payment_to_income"] = (df.loan_amnt / df.term_months) / (df.annual_inc / 12)

print(df.payment_to_income.describe().round(4).to_string())

# %% [markdown]
# ### Feature 3 — `log_annual_inc` (finding #5)
#
# Income skew is **35.5** — extreme. For logistic regression that is a real
# problem: the model fits a coefficient on a raw dollar scale where a single
# $7.1m applicant sits 70x beyond the median, dominating the fit.
#
# The log transform compresses that tail. Trees are invariant to monotonic
# transforms and will ignore it, which is fine — the column exists for the
# linear model's benefit.

# %%
df["log_annual_inc"] = np.log10(df.annual_inc)

print(f"skew, raw annual_inc : {df.annual_inc.skew():.2f}")
print(f"skew, log_annual_inc : {df.log_annual_inc.skew():.2f}")

# %% [markdown]
# From 35.5 to near-symmetric — exactly what the transform is for.

# %% [markdown]
# ### Feature 4 — `acct_open_rate`
#
# Credit-history *velocity*: accounts opened per year of credit history. Two
# applicants can both have 20 accounts, but one built them over 25 years and
# the other over 3 — the second is a much more aggressive credit seeker.
#
# `longest_credit_length` is 0 for 11 applicants, so the denominator gets a
# `+1` rather than producing infinities. Guarding division is unglamorous and
# it is where dataset bugs actually come from.

# %%
df["acct_open_rate"] = df.total_acc / (df.longest_credit_length + 1)

assert np.isfinite(df.acct_open_rate.dropna()).all(), "non-finite values in acct_open_rate"
print(df.acct_open_rate.describe().round(3).to_string())
print("\nNo infinities — denominator guard worked.")

# %% [markdown]
# ### Feature 5 — `has_delinquency`
#
# `delinq_2yrs` is skewed at 5.96 and is overwhelmingly zero — it is a count
# with a long thin tail, not a continuous variable. The practical question in
# underwriting is usually binary: *has this applicant missed payments at all?*
#
# Keeping both lets the model use the robust binary version and the raw count.

# %%
df["has_delinquency"] = (df.delinq_2yrs > 0).astype("float")
df.loc[df.delinq_2yrs.isna(), "has_delinquency"] = np.nan  # don't invent a value

print(df.has_delinquency.value_counts(dropna=False).to_string())
print(f"\nDefault rate, has_delinquency = 1 : {df.loc[df.has_delinquency == 1, 'bad_loan'].mean():.2%}")
print(f"Default rate, has_delinquency = 0 : {df.loc[df.has_delinquency == 0, 'bad_loan'].mean():.2%}")

# %% [markdown]
# Note the second line: where `delinq_2yrs` is missing, `has_delinquency` stays
# missing. A naive `(df.delinq_2yrs > 0)` would have turned 29 unknowns into a
# confident "no delinquencies", which is fabricating a favourable fact about a
# borrower.

# %% [markdown]
# ## 3.5 Finding #4 — `revol_util` above 100%
#
# Profiling established these are real: 289 rows, tightly clustered between
# 100.1 and 150.7, defaulting at 25.6% against an 18.3% base rate. A data-entry
# error would scatter; this does not.
#
# **Decision: keep them uncapped.** Clipping to 100 would flatten the most
# distressed borrowers onto the same value as merely maxed-out ones, deleting
# the signal that makes the variable useful. I add a flag instead, so the
# model can treat "over limit" as a distinct state rather than just a slightly
# larger number.

# %%
df["over_limit"] = (df.revol_util > 100).astype("float")
df.loc[df.revol_util.isna(), "over_limit"] = np.nan

print(f"over_limit = 1 : {int(df.over_limit.sum()):,} loans, "
      f"default rate {df.loc[df.over_limit == 1, 'bad_loan'].mean():.2%}")
print(f"over_limit = 0 : {int((df.over_limit == 0).sum()):,} loans, "
      f"default rate {df.loc[df.over_limit == 0, 'bad_loan'].mean():.2%}")

# %% [markdown]
# ## 3.6 Finding #6 — the `int_rate` decision
#
# Profiling showed default rate climbing monotonically across all ten interest
# rate deciles, from ~6% to ~33%. That is because Lending Club set the rate
# from their own internal risk grade — so `int_rate` is a *summary of a risk
# assessment that already happened*, not an applicant characteristic.
#
# I keep the column in the clean file but **tag it**, so the modelling notebook
# has to make an explicit choice rather than sweeping it up with `df.drop`.
# Phase 5 fits the model both ways and reports the gap.

# %%
LEAKY = ["int_rate"]
ENGINEERED = ["term_months", "region", "emp_length_missing", "loan_to_income",
              "payment_to_income", "log_annual_inc", "acct_open_rate",
              "has_delinquency", "over_limit"]

print(f"Leaky (excluded from the primary model) : {LEAKY}")
print(f"Engineered in this notebook             : {len(ENGINEERED)} features")
for f in ENGINEERED:
    print(f"  - {f}")

# %% [markdown]
# ## 3.7 Imputation — computed here, applied in Phase 5
#
# As set out at the top, I am **not** filling the `NaN`s in the saved file.
# What I do here is decide and document the *strategy* per column, and record
# the values the training fold would produce — so the choice is visible and
# reviewable rather than buried in a pipeline argument.

# %%
IMPUTATION_PLAN = {
    "emp_length": ("median", "Missingness is signal — but it is already captured "
                             "by emp_length_missing, so the fill value itself is neutral."),
    "revol_util": ("median", "193 rows; apparent lift failed significance testing."),
    "annual_inc": ("median", "4 rows; negligible."),
    "delinq_2yrs": ("median", "29 rows, all the same records — no bureau data attached."),
    "total_acc": ("median", "Same 29 records."),
    "longest_credit_length": ("median", "Same 29 records."),
}

plan = pd.DataFrame(
    [(c, s, f"{df[c].median():.2f}", int(df[c].isna().sum()), why)
     for c, (s, why) in IMPUTATION_PLAN.items()],
    columns=["column", "strategy", "median_value", "n_missing", "rationale"],
)
plan

# %% [markdown]
# Median rather than mean throughout, because Phase 2 showed these
# distributions are skewed and the mean would be pulled by the same outliers
# the log transform exists to control.
#
# The derived columns `has_delinquency` and `over_limit` inherit their parents'
# missingness and are median-filled alongside them in the pipeline.

# %% [markdown]
# ## 3.8 Validation before saving
#
# A cleaning notebook that does not check its own output is just hoping. These
# assertions are the contract the clean file has to satisfy.
#
# The subtle one is missingness in derived features. `loan_to_income` divides
# by `annual_inc`, which is missing for 4 rows — so the ratio is missing for
# those 4 rows too. That is correct and unavoidable: the right standard is not
# "derived features have no nulls", it is **"a derived feature is missing
# exactly where its inputs are missing"** — inheriting missingness is fine,
# inventing it is a bug.

# %%
checks = []

checks.append(("row count unchanged", len(df) == len(raw)))
checks.append(("no rows silently dropped", df.index.equals(raw.index)))
checks.append(("target untouched", df.bad_loan.equals(raw.bad_loan)))
checks.append(("target still binary", set(df.bad_loan.unique()) == {0, 1}))
checks.append(("term_months is integer", df.term_months.dtype.kind == "i"))

# A derived feature may inherit missingness from its inputs, but it must never
# invent missingness of its own. Checking each against its true parent columns.
DERIVED_PARENTS = {
    "loan_to_income": ["loan_amnt", "annual_inc"],
    "payment_to_income": ["loan_amnt", "annual_inc", "term_months"],
    "log_annual_inc": ["annual_inc"],
    "acct_open_rate": ["total_acc", "longest_credit_length"],
    "has_delinquency": ["delinq_2yrs"],
    "over_limit": ["revol_util"],
}
for child, parents in DERIVED_PARENTS.items():
    inherited = df[parents].isna().any(axis=1)
    checks.append((f"{child} missing only where its inputs are",
                   df[child].isna().equals(inherited)))
checks.append(("no infinities anywhere",
               np.isinf(df.select_dtypes("number").fillna(0)).sum().sum() == 0))
checks.append(("region fully mapped", df.region.notna().all()))
checks.append(("emp_length flag matches source", df.emp_length_missing.sum() == raw.emp_length.isna().sum()))
checks.append(("missingness preserved, not filled",
               df[list(IMPUTATION_PLAN)].isna().sum().sum() == raw[list(IMPUTATION_PLAN)].isna().sum().sum()))

for name, passed in checks:
    print(f"[{'PASS' if passed else 'FAIL'}] {name}")

assert all(p for _, p in checks), "validation failed — not saving"
print(f"\nAll {len(checks)} checks passed.")

# %%
out_path = ROOT / "data" / "lending_club_clean.csv"
df.to_csv(out_path, index=False)

print(f"Saved -> {out_path.relative_to(ROOT)}")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns "
      f"({raw.shape[1]} raw + {df.shape[1] - raw.shape[1]} engineered)")
df.head()

# %% [markdown]
# ## Phase 3 summary
#
# | Finding | Decision | Why not the obvious alternative |
# |---|---|---|
# | #1 `emp_length` missing, non-random | Impute **+ flag** | A plain `fillna` erases a 1.46x risk signal |
# | #2 Trace missingness | Median, no flags | Apparent lifts failed significance testing |
# | #3 `term` is text | Regex parse + assert | A dict lookup breaks silently on new products |
# | #4 `revol_util` > 100 | **Keep uncapped** + flag | Clipping destroys the ordering that carries the signal |
# | #5 Income skew 35.5 | Add `log_annual_inc` | Dropping outliers deletes real applicants |
# | #6 `int_rate` leakage | Tag, exclude, quantify in Phase 5 | Silent drop hides the most interesting trade-off |
# | #7 50 states | Census regions | Target encoding would leak the label |
# | #8 Rare `home_ownership` | Collapse to `OTHER` | A 1-loan dummy is memorisation |
# | #9 Duplicates | None found, no action | `drop_duplicates()` by reflex is not a decision |
#
# **Five engineered features** carry genuine domain reasoning: `loan_to_income`
# and `payment_to_income` (affordability), `acct_open_rate` (credit-seeking
# velocity), `log_annual_inc` (linear-model conditioning), `has_delinquency`
# (robust binary from a sparse count).
#
# Imputation and encoding are deferred to the Phase 5 pipeline so they are
# fitted on training data only.
