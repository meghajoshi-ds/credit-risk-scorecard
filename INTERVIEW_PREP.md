# Interview Preparation — Credit Risk Scorecard

Everything you need to discuss this project confidently. All figures here are
taken from the executed notebooks, so they match what an interviewer would see
if they opened the repo.

**How to use this:** read §1–§3 until you can say them without looking. Skim
§4 and §5. Know §6 exists so nothing catches you out. §7 is what you say when
asked "what next?"

---

## 1. The 30-second pitch

> "I built a credit default model on 164,000 Lending Club loans. The point
> wasn't maximum accuracy — it was making defensible decisions and being able
> to explain any individual decline, which is what a lender is legally
> required to do.
>
> The most interesting finding was that the strongest predictor in the data,
> interest rate, was actually leakage — Lending Club sets the rate using their
> own risk model at origination, so it's a summary of a risk assessment that's
> already happened rather than something the applicant brings. I excluded it,
> and measured what that cost: 2.4 AUC points.
>
> Final model is a LightGBM at 0.69 AUC, with SHAP explanations that convert
> into plain-language decline reasons, and a 300–850 scorecard validated as
> monotonic across all ten population deciles."

**Why this works:** it leads with judgment, not metrics. It shows you know the
regulatory context. And it volunteers a limitation before being asked.

---

## 2. Numbers to know cold

### The data
| | |
|---|---|
| Rows | 163,987 loans |
| Columns | 15 raw → 24 after feature engineering |
| Default rate | **18.3%** (30,016 bad / 133,971 good) |
| Imbalance | 1 bad loan per 4.5 good |

### The results
| Model | AUC | Gini | KS | Brier |
|---|---|---|---|---|
| Logistic regression | 0.680 | 0.361 | 0.265 | 0.140 |
| **LightGBM (final)** | **0.690** | **0.381** | **0.283** | **0.139** |
| LogReg *with* `int_rate` | 0.704 | 0.408 | 0.299 | 0.137 |

Split: **60/20/20** train/validation/test. Early stopping watches the
validation fold, so the test fold was touched exactly once.

- **5-fold CV:** 0.677 ± 0.004 → the result is stable, not a lucky split
- **`int_rate` was worth 2.4 AUC points** — the cost of excluding leakage
- **LightGBM beat logistic regression by only ~1 AUC point**

### The scorecard
- Range produced: roughly **430–575** (on a 300–850 scale)
- Default rate across 10 population deciles: **40.0% → 5.2%** (~8x gradient)
- Monotonic at every band — asserted in code, not assumed

### Three findings worth quoting exactly
| Finding | Numbers |
|---|---|
| `emp_length` missingness is a risk signal | 5,804 rows missing; **26.3%** default vs **18.0%** = 1.46x lift, Fisher p < 0.001 |
| `revol_util` > 100% is real, not an error | 289 rows, range 100.1–150.7, default **25.6%** vs 18.3% base |
| `verification_status` reads backwards | Verified default **20.1%** vs unverified **15.2%** — because verified applicants borrow more (median **$14,000 vs $8,900**, and 29.6% vs 5.1% take 60-month terms) |

### Top features by Information Value
| Feature | IV |
|---|---|
| `int_rate` | 0.382 ← *excluded as leakage* |
| `loan_to_income` | 0.145 ← **best legitimate feature** |
| `term` | 0.129 |
| `revol_util` | 0.079 |
| `dti` | 0.074 |
| `region` | 0.001 ← geography predicts almost nothing |

---

## 3. The six stories

These are the substance of the interview. Each is structured so you can tell
it in under a minute.

### Story 1 — The `int_rate` leakage decision ⭐ *your best one*

**Setup:** Interest rate was by far the strongest predictor. Default rate rose
perfectly monotonically across all ten interest-rate deciles, from about 5% to
36%.

**The insight:** Nothing in real credit data is that clean by accident. Lending
Club runs its own risk model at origination and *sets the rate from the grade
it produces*. So `int_rate` isn't an applicant characteristic — it's a
downstream summary of a risk assessment that already happened.

**What I did:** Excluded it from the primary model, then fitted a variant *with*
it to quantify the cost — 2.4 AUC points (0.680 → 0.704). Reported both.

**Why exclusion was right, in two sentences:**
1. You can't deploy it — scoring a new applicant needs their rate, but the rate
   is set *from* the risk assessment. The input doesn't exist at decision time.
2. It answers the wrong question. "How well can I assess risk from application
   data?" is 0.680, not 0.704.

**The kicker — use this one.** Information Value is the standard credit-risk
tool for spotting leakage; anything above 0.5 is "suspicious". `int_rate`
scored **0.38 — it sailed through.** The heuristic did not catch it. What
caught it was understanding how the data was generated.

> "A threshold is a prompt to investigate, not a verdict."

---

### Story 2 — Missingness as signal ⭐ *your best data-prep story*

**Setup:** `emp_length` was missing for 5,804 applicants.

**The insight:** Those applicants default at 26.3% against an 18.0% base — a
1.46x lift. The missingness isn't random; not stating your employment is itself
a risk signal (no employer to verify, gig income, or a refusal to answer).

**What I did:** Created an `emp_length_missing` flag **before** imputing, so the
information survives. A plain `fillna(median)` would have made those applicants
look average and erased a real signal.

**The part that shows discipline:** Five other columns showed tempting lifts
too — `revol_util`'s missing rows default at 22.3%. But that's only 193 rows.
I ran **Fisher exact tests on all six columns**. Only `emp_length` was
significant (p < 0.001); the rest were consistent with chance.

> "So I built one flag instead of six. The test is what stopped me shipping
> five features that would have evaporated on the test set."

---

### Story 3 — Investigating rather than reflexively cleaning

**Setup:** `revol_util` (credit utilisation) had values above 100%, max 150.7.
An obvious "data error" to clip or drop.

**What I checked:** Only 289 rows, clustered tightly between 100.1 and 150.7 —
*not* scattered up to 5,000 the way a units error or misplaced decimal would
be. And they default at 25.6% vs 18.3%.

**The mechanism:** A credit card balance genuinely can exceed its limit —
through interest, fees, or a limit reduction after the balance was drawn.
Someone over their limit is genuinely higher-risk.

**Decision:** Kept them uncapped, and added an `over_limit` flag.

> "Clipping to 100 would have flattened the most distressed borrowers onto the
> same value as merely maxed-out ones — deleting exactly the ordering that
> makes the variable useful."

---

### Story 4 — Recommending the *simpler* model

**Setup:** LightGBM beat logistic regression by about 1 AUC point (0.680 →
0.690).

**The insight:** That's a small gain from a much more complex model. What it
tells you is that the signal in 14 coarse application fields is close to
linear — there aren't rich interactions left to find.

**The recommendation:** One AUC point doesn't obviously justify losing
coefficient-level explainability in a regulated setting. This is exactly why
banks still run logistic regression scorecards.

> "The gap was small enough that the explainable model is easy to defend.
> Reporting that the complex model barely won is more useful than only
> reporting the winner."

---

### Story 5 — Significance is not importance

**Setup:** `emp_length_missing` is highly significant (Fisher p < 0.001) but
has an Information Value of just 0.009 — classed "not predictive".

**Why both are right:**
- The Fisher test asks *"is this effect real?"* → emphatically yes, on 5,804 rows.
- IV asks *"how much does this move the whole portfolio?"* → not much, because
  it fires on only 3.5% of applicants.

**A large effect on a small group looks exactly like this.**

> "I kept the flag because it materially sharpens the model's judgement on the
> applicants it fires for — but I didn't expect it near the top of a global
> importance chart, and it wasn't."

Confusing "statistically significant" with "important" is one of the most
common analytical errors. Being able to articulate the difference is a strong
signal.

---

### Story 6 — Testing a technique and rejecting it

**Setup:** Predicted probabilities need to be trustworthy, because Phase 8
converts them into a credit score. The textbook move at 18.3% positives is
`class_weight="balanced"` or SMOTE.

**What I did:** Neither — and then ran a **control experiment** to prove the
decision rather than assert it. I fitted an otherwise identical LightGBM *with*
`class_weight="balanced"` purely to measure the damage:

| Model | AUC | Brier | Mean predicted | Actual |
|---|---|---|---|---|
| `class_weight="balanced"` | 0.6898 | 0.218 | **45.8%** | 18.3% |
| Unweighted (chosen) | 0.6903 | **0.139** | **18.25%** | 18.3% |
| Unweighted + isotonic | 0.6901 | 0.139 | 18.27% | 18.3% |

**The finding:** Reweighting made the model believe bad loans were two and a
half times as common as they are — Brier 57% worse — while AUC differed by
0.0005. It bought nothing and wrecked the probability scale.

**The part that shows honesty:** I then tested post-hoc isotonic calibration on
the unweighted model, expecting an improvement. It **added nothing** —
identical Brier to four decimal places, fractionally worse KS. So I didn't keep
it.

> "Calibration turned out to be a modelling decision, not a post-processing
> step. Once the class weights were gone there was no error left to correct, so
> the extra component would have been complexity for its own sake. I'd rather
> report that the technique wasn't needed than ship it to look sophisticated."

**Why this lands:** most candidates either don't check calibration at all, or
apply `CalibratedClassifierCV` because a tutorial said to. Running the control,
getting a null result, and *acting* on the null result is what model validation
work actually looks like.

---

## 4. Concepts you must be able to explain

If you can't explain the metric, don't quote the number.

| Concept | One-sentence explanation |
|---|---|
| **AUC** | Probability that a randomly chosen bad loan is ranked riskier than a randomly chosen good one. 0.5 = coin flip. |
| **Gini** | Just `2 × AUC − 1`. Same information, the scale credit teams prefer. |
| **KS statistic** | The largest gap between the cumulative distributions of good and bad borrowers — "how cleanly does the score separate the two populations?" The headline number on a scorecard. |
| **Brier score** | Measures whether the predicted *probabilities* are accurate, not just correctly ordered. Lower is better. |
| **Why not accuracy?** | At 18.3% positives, predicting "everyone repays" scores 81.7% accuracy while being useless. |
| **Information Value** | Measures how much a variable separates good from bad, summed across its bins. Handles categoricals natively, unlike correlation. |
| **Weight of Evidence** | `ln(%good / %bad)` for each bin. Negative = worse than average. A scorecard is essentially a weighted sum of these. |
| **SHAP** | Decomposes a single prediction into per-feature contributions that sum exactly to the output — which is what makes it usable for legally required decline reasons. |
| **Data leakage** | Using information that wouldn't exist at decision time, or that encodes the answer. Inflates validation scores and breaks in production. |
| **VIF** | Variance Inflation Factor — how much a coefficient's variance is inflated by correlation with other inputs. Above ~5 is a warning. |
| **Class imbalance handling** | I used **neither** resampling nor class weights. At 18.3% positives the imbalance is mild, and both interventions cost more than they pay — see the control experiment in Story 6. The imbalance is handled at the decision threshold instead. |
| **Calibration** | Whether "20% risk" actually means 20 in 100 default. Mine are well calibrated: mean predicted 18.25% vs actual 18.3%. |
| **Monotonicity** | A higher score must mean lower risk at *every* band. Non-negotiable for a scorecard. |
| **PDO** | Points to Double the Odds — the standard probability→score conversion. I used PDO=20, 600 = 50:1 odds. |

### Two more technical points you should own

**Why imputation happens in the pipeline, not the cleaning notebook.**
If you compute a median across all 164,000 rows and use it to fill the test
set, test-set information has reached the model before evaluation. The effect
is small for a median, but the habit matters — the same pattern with target
encoding is a serious leak. So the clean CSV deliberately still contains
`NaN`s, and imputation is fitted inside the scikit-learn pipeline on the
training fold only. It also lets LightGBM use its native missing-value
handling, which learns a direction per split and beats median filling.

**Why the two models got different feature sets.**
Logistic regression got a curated, decorrelated set — collinear inputs make
coefficients unstable, which defeats the whole point of using an explainable
model. LightGBM got everything, because trees are indifferent to collinearity.

---

## 5. Likely questions, with answers

**"Walk me through your project."**
→ Use the §1 pitch, then offer: *"Happy to go deeper on the data preparation or
the leakage decision — whichever is more useful."*

**"Your AUC is only 0.69. Isn't that weak?"**
→ *"It is modest, and it's a data ceiling rather than a modelling failure —
14 coarse application fields, no bureau score, no payment history, no
origination date. With a bureau score this would be meaningfully higher. I'd
rather report 0.69 honestly than 0.70 by including a leaky variable."*

**"Why not just use interest rate? It's the best predictor."**
→ Story 1. Emphasise that it's unusable at decision time.

**"How did you handle missing values?"**
→ Story 2. The Fisher testing is the part that impresses.

**"Why logistic regression at all, if you had LightGBM?"**
→ Story 4. Explainability in a regulated setting, and the gap was only 1 point.

**"How would you explain a decline to a customer?"**
→ *"SHAP gives per-applicant contributions that sum to the prediction, so I map
the top risk-increasing factors to plain language. For my highest-risk test
case it produced: loan term requested, size of loan relative to income,
existing debt relative to income, and credit utilisation."*

**"What would you do with more time?"**
→ §7. Lead with out-of-time validation and reject inference.

**"What's the weakest part of this project?"**
→ *"No out-of-time validation. Credit models should be tested on later loans
than they're trained on, because conditions shift — but the data has no
origination date, so I could only split randomly, which flatters the result."*

**"How did you pick your cut-off?"**
→ *"I didn't pick one — I produced the trade-off table, because it's a policy
decision, not a statistical one. Rejecting the worst-scoring 20% takes the
approved book's default rate from 18.3% to 14.3%, but turns away 3,957 good
customers. The right answer depends on the margin on a good loan versus loss
given default."*

**"Did you check for overfitting?"**
→ *"5-fold CV gave 0.677 ± 0.004 and the held-out test score sits inside that
spread. LightGBM used early stopping — it stopped at 191 trees of a possible
600. I also set `min_child_samples=100` so no leaf fits a handful of loans."*

---

## 6. Limitations — say these before you're asked

Volunteering these reads as model risk awareness. Getting caught out doesn't.

1. **No out-of-time validation** *(the biggest)* — no origination date in the
   data, so only a random split was possible.
2. **Selection bias** — the data contains only *accepted* loans. The model
   learned who defaults among applicants already judged creditworthy, not among
   everyone who applied. The industry fix is **reject inference**; Lending Club
   publishes a rejected-applications file that would partly address it.
3. **Modest discrimination** — AUC 0.69, limited by the inputs.
4. **`verification_status` is partly a process variable** — it encodes the
   lender's suspicion, not just borrower risk. Don't interpret it causally.

**Two limitations that used to be on this list are now fixed** — worth
mentioning that way, because "I found it and fixed it" beats either silence or
a standing caveat:
- Probabilities were inflated by class weighting. Class weights were dropped;
  Brier fell from 0.218 to 0.139.
- Early stopping used the test set. There is now a dedicated validation fold.

**Also worth admitting freely:** one of my engineered features underperformed.
`payment_to_income` scored IV 0.058, below `term_months` at 0.129 — dividing by
the term folded a strong variable into the ratio and blurred it. Saying this
unprompted is more credible than claiming nine out of nine features worked.

---

## 7. Improvements — what you'd do next

Ordered by what a credit risk interviewer would most want to hear.

### Tier 1 — the ones that matter most

**1. Out-of-time validation.** Train on older loans, test on newer. The single
biggest gap. Requires sourcing origination dates, which the full Lending Club
dataset has even though this extract doesn't.

**2. Reject inference.** Bring in the rejected-applications file to address the
selection bias. Standard techniques: parcelling, fuzzy augmentation, or
reweighting. This is a genuinely specialist topic — knowing it exists puts you
ahead of most candidates.

**3. A real points-based scorecard.** Bin every variable, fit logistic
regression on the **WoE-transformed** bins, and convert coefficients into a
points table:
```
Employment length  < 1 yr  →  -12 points
                   1-3 yrs →   -4 points
                   10+ yrs →  +15 points
```
That's the format banks deploy and regulators review. The WoE values are
already computed in notebook 03, so this is closer than it sounds.

**4. Fair lending / disparate impact analysis.** Under ECOA/Reg B this is a
legal requirement, not an optional extra. Test approval rates across
protected-attribute proxies — `addr_state` is a geographic proxy that warrants
this before any real deployment. *Finding* a problem is better than not looking.

### Tier 2 — technical quality

**5. Eliminate training/serving skew.** The feature engineering currently
exists twice — in notebook 02 and again in `app.py`. Extracting it to a shared
`src/features.py` that both import is the correct fix.

**6. A profit curve.** Add assumptions (margin on a good loan, loss given
default ~50–70%) to turn the cut-off table into an actual recommended
threshold. Converts a statistical output into a business recommendation.

### Deliberately *not* worth doing

**Hyperparameter tuning, stacking, more model families.** You'd gain maybe
0.005 AUC. The ceiling is the data, not the algorithm. Saying this out loud is
itself a good signal — it shows you can tell the difference between effort and
progress.

---

## 8. Things not to say

| Don't say | Because |
|---|---|
| "I got 81% accuracy" | Accuracy is meaningless at 18.3% positives — and quoting it suggests you don't know that |
| "The model is 69% accurate" | AUC is not accuracy. They are different things. |
| "I used SMOTE to fix the imbalance" | You didn't — you used neither SMOTE nor class weights, and you have a control experiment showing why |
| "Interest rate was my best feature" | It was excluded. Leading with it undoes your best story. |
| "The data was clean" | You found six distinct problems and tested for a seventh |
| "I'd try a neural network next" | Signals you think the algorithm is the constraint. It isn't. |

---

## 9. Quick reference — where things live

| | |
|---|---|
| Project write-up | `README.md` |
| Profiling | `notebooks/01_explore.ipynb` |
| **Cleaning & features** | `notebooks/02_clean_features.ipynb` ← your strongest section |
| EDA, WoE/IV | `notebooks/03_eda.ipynb` |
| Models, SHAP, scorecard | `notebooks/04_modelling.ipynb` |
| Readable, no-setup versions | `outputs/html/*.html` |
| All 13 charts | `outputs/figures/` |
| Interactive demo | `streamlit run app.py` |

**Final advice:** read `02_clean_features.ipynb` end to end at least once. You
need to be able to explain why utilisation above 100% was kept *in your own
words* — the notebook has the reasoning, but it has to come from you, not be
read off a screen.
