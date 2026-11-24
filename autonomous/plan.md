# Plan

Status: **pre-registered v1.2 — approved, frozen, not yet executed.**
Project: Individual AI Project, Business Analytics (20595).
Owner: Michele Landra.
Companion documents: `AGENTS.md`, `mistakes.md`,
`results/explore_bank_marketing/`, `results/verification/initial_independent_checks.md`,
`results/splits/chronology_adjudication.md`.

Version history:

| Version | Change |
|---|---|
| 1.2 (2026-11-13) | `pdays` predictor ambiguity between §5.2 and §5.3 resolved; `balance` pre-call availability recorded as an assumption (see §10). No other design change. |

This file is the frozen pre-registration for the analysis. Any change requires
explicit approval and an entry in the deviations log (§10) recorded **before**
test-set results are inspected.

---

## 1. Research Question

**RQ1 (statistical).** In the Bank Marketing dataset, does adding
previous-campaign information to pre-call customer characteristics improve
out-of-sample prediction of term-deposit subscription, relative to customer
characteristics alone?

**RQ2 (meta / process).** Can an AI coding agent execute this analysis
reliably — that is, without data leakage, without methodological error, and
with decisions that match the pre-registered plan and survive independent
verification?

RQ1 and RQ2 are answered separately. A good AUC does **not** answer RQ2, and a
clean audit does **not** answer RQ1.

Scope: associational prediction. No causal effect is estimated or claimed.

---

## 2. Theory

1. **Information sets.** A targeting model may only use variables observable
   at the moment the call decision is made. Anything realised during or after
   the call is outside the decision-maker's information set.
2. **Incremental information value.** Prior interaction history (whether the
   client was contacted before, how often, and with what outcome) is a
   candidate signal of responsiveness. Whether it adds value *beyond* static
   customer characteristics is an empirical question, answered by paired
   out-of-sample comparison on identical observations.
3. **Leakage.** `duration` is measured after the call ends and is mechanically
   related to the outcome. Including it produces an optimistic, non-deployable
   model. Model C exists only to illustrate the magnitude of that inflation.
4. **Selection.** Previously contacted clients were chosen by the bank's own
   campaign logic, so prior-campaign variables are not randomly assigned.
   Associations involving them are descriptive, not causal.

---

## 3. Falsifiable Prediction

### 3.1 Hypotheses

**Statistical null (H0):** ΔROC-AUC = AUC(B) − AUC(A) **≤ 0** on the
pre-specified chronological test set.

**Alternative (H1):** ΔROC-AUC **> 0**.

**Practical relevance threshold (separate from the test):** ΔROC-AUC ≥ 0.01.
This is a **project-level materiality judgment**, not a validated business
profitability threshold. It is not part of H0/H1 and does not change the
statistical test.

**Decision procedure.** H0 is rejected if the 95% paired bootstrap confidence
interval for ΔAUC excludes zero on the positive side. Separately, the result is
described as *materially relevant* only if the point estimate also reaches
0.01.

**Insufficient evidence is not evidence of no improvement.** A CI that
includes zero means the data do not establish a direction; it does not prove
that ΔAUC ≤ 0.

### 3.2 Pre-specified interpretation of outcomes

| Outcome | Criterion | Interpretation |
|---|---|---|
| **H0 rejected, materially relevant** | ΔAUC ≥ 0.01 and CI lower bound > 0 | Prior-campaign information materially improves prediction |
| **H0 rejected, below relevance threshold** | 0 < ΔAUC < 0.01 and CI lower bound > 0 | Improvement is detectable but below the project's relevance threshold |
| **Insufficient evidence** | 95% CI includes 0 | No direction established; **not** a contradiction |
| **H0 not rejected / direction contradicted** | ΔAUC < 0 and CI upper bound < 0 | The proposed direction is wrong |

PR-AUC and lift at the top 10% are reported as supporting evidence, with
confidence intervals; they do not replace the primary criterion.

### 3.3 Secondary diagnostic hypothesis

**H2.** Model C (B + `duration`) achieves materially higher ROC-AUC than
Model B on the same test rows. Pre-specified diagnostic threshold:
ΔAUC(C − B) ≥ 0.05. If the observed gap is smaller, the leakage effect is
weaker than assumed and this is reported as such. Model C is never presented
as an operational model.

### 3.4 Meta-hypothesis

**H3.** The agent's execution satisfies the audit rubric in §8.2 with no FAIL
on leakage-related criteria, and its reported metrics are reproduced by an
independent script to within 1e-6. A single successful run does not establish
general reliability; the claim is limited to this run.

---

## 4. Dataset

### 4.1 Source and provenance

- File: `data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet`
- Hugging Face: `cestwc/bank-marketing` (mirror of UCI Bank Marketing).
- Underlying UCI file: `bank-full.csv` (the 17-column, 45,211-row version),
  documented as date-ordered (May 2008 – November 2010).
- Licence: CC BY 4.0. Citation: Moro, Rita & Cortez (2014),
  DOI 10.24432/C5K306.
- SHA-256 recorded in `results/explore_bank_marketing/manifest.json`.

### 4.2 Verified characteristics (from the EDA)

| Property | Value | Source |
|---|---|---|
| Shape | 45,211 × 17 | `01_shape.txt` |
| Target `y` | 0 = "no" (39,922; 88.30%), 1 = "yes" (5,289; 11.70%) | `03_target.csv` |
| NaN cells | 0 | `04_missingness.csv` |
| Exact duplicates | 0 | `05_duplicates.txt` |
| Categorical storage | label-encoded `int64` (10 variables) | `02_schema.csv` |
| `pdays = 999` | 36,954 rows (81.74%) = "never previously contacted" | `07_pdays_sentinel_stats.csv` |
| Previously contacted | 8,257 rows (18.26%) | `07b_pdays_consistency.csv` |
| Prior success | 1,511 rows (3.34%) | `07c_poutcome_by_pdays_sentinel.csv` |
| Known anomaly | 5 rows have `poutcome="unknown"` but `pdays != 999` | `10_notes.txt` |

The full-dataset positive prevalence (11.70%) is a property of the whole file.
It is **not** assumed to hold in any evaluation set; see §5.7.

Raw data is never modified. All transformations happen in memory inside
train-fitted pipelines.

### 4.3 Chronology validation protocol (run before any model)

Purpose: test whether the HF Parquet preserves UCI's documented date ordering.
These checks use **row order and feature values only — never the target** — so
they cannot leak outcome information into the split decision.

#### (a) Checks that provide direct evidence about chronological ordering

| ID | Check | Method | PASS criterion |
|---|---|---|---|
| C3 | Month block contiguity | Count maximal runs of consecutive identical `month` codes; record block boundaries | `n_runs ≤ 60` (a random order would give ≈41,000; date order implies ≤ ~31 year-month blocks) |
| C4 | Month transition rule | Consecutive blocks must advance by +1 (mod 12); report any skip or reversal | 0 violations beyond months with zero contacts |
| C5 | Within-month day monotonicity | Within each block, share of adjacent pairs with non-decreasing `day` | ≥ 99% non-decreasing |

#### (b) Checks that establish only dataset identity and transformation consistency

| ID | Check | Method | What it does and does not show |
|---|---|---|---|
| C1 | Provenance audit | Inspect the HF dataset-card build script: `read_csv(bank-full.csv)` → `pdays.replace(-1, 999)` → `Dataset.from_pandas(...)`. Confirm no sort/reorder step. | Shows the transformation pipeline did not deliberately reorder rows. Does **not** prove the underlying UCI file was date-ordered. |
| C2 | Version identity | Shape and columns match `bank-full.csv` (45,211 × 17), not `bank-additional-full` (41,188 × 21) | Shows which UCI file was used. Says nothing about row order. |
| C6 | Transformation fingerprint | No `pdays == -1`; `pdays == 999` exactly when `previous == 0`; `poutcome == 4` absent | Confirms the documented recoding was applied. Says nothing about row order. |
| C7 | Year ambiguity | No year variable exists; December→January cannot be dated | Documents a limit that no check can remove. |

**Explicit rule of interpretation:** passing C1, C2 or C6 is **not** evidence of
chronological ordering. Only C3, C4 and C5 bear directly on ordering, and even
they cannot resolve the year ambiguity in C7.

**Decision rule (pre-specified, independent of any model result):**

- **CHRONOLOGICAL** — C3, C4 and C5 all pass: use the row-index split as
  primary, and document the C7 year ambiguity.
- **PSEUDO-TEMPORAL** — months are contiguous but C5 fails: the split is
  labelled pseudo-temporal in every table and the random split becomes a
  co-equal sensitivity analysis.
- **NOT CHRONOLOGICAL** — months are interleaved (C3/C4 fail): **stop and
  request a decision**; do not silently switch splits, and never choose a
  split based on test performance.

Results of C1–C7 are written to `results/splits/chronology_checks.md` before
model fitting begins.

---

## 5. Method

### 5.1 Prediction time

Immediately before dialling the client in the current campaign.

### 5.2 Feature sets

| Variable | Available pre-call | A | B | C |
|---|---|---|---|---|
| `age` | Yes | ✅ | ✅ | ✅ |
| `job` | Yes | ✅ | ✅ | ✅ |
| `marital` | Yes | ✅ | ✅ | ✅ |
| `education` | Yes | ✅ | ✅ | ✅ |
| `default` | Yes | ✅ | ✅ | ✅ |
| `balance` | Yes — **assumed** pre-call snapshot; timing not confirmed | ✅ | ✅ | ✅ |
| `housing` | Yes | ✅ | ✅ | ✅ |
| `loan` | Yes | ✅ | ✅ | ✅ |
| `previous` | Yes (prior campaign) | — | ✅ | ✅ |
| `poutcome` | Yes (prior campaign) | — | ✅ | ✅ |
| `pdays` | Yes (prior campaign, recoded) | — | — † | — † |
| `duration` | **No — post-call** | — | — | ✅ diagnostic |
| `campaign` | Contemporaneous | excluded | excluded | excluded |
| `contact` | Chosen at contact | excluded | excluded | excluded |
| `day`, `month` | Campaign operations | excluded | excluded | excluded |

† **`pdays` is not a separate predictor in any model.** It is used only to
construct the derived `poutcome` level `never_contacted` (§5.3). The 999
sentinel is never entered as a numeric value. The previous-campaign information
in Models B and C is carried by `previous` (numeric count) and `poutcome`
(categorical, five levels). This resolves the v1.1 inconsistency between this
table and §5.3; see the deviations log (§10).

No sensitivity variants are added without approval.

### 5.3 Recoding of `pdays` and `poutcome`, and the redundancy decision

**Derived quantities.**

- `was_contacted_prev = 1 if pdays != 999 else 0`.
- `days_since_prev = pdays` when `pdays != 999`, else missing.
- `poutcome` levels: `failure`, `other`, `success`, `unknown` (including the 5
  anomalous rows) and derived `never_contacted` (the 36,954 sentinel rows).

**Documented relationship.** `was_contacted_prev` is a deterministic function
of `previous`: it equals `1[previous > 0]` exactly (verified: 36,954 / 0 / 0 /
8,257 in `07b_pdays_consistency.csv`). The two variables therefore carry
redundant information. They are **not** perfectly linearly collinear — a binary
indicator cannot be written as a linear function of a count — but their
functional dependence is exact, so their coefficients must not be interpreted
as independent sources of variation.

**Chosen encoding for the primary logistic specification.** Model B includes
`previous` (numeric count) and `poutcome` (five levels, including
`never_contacted`). To avoid unnecessary redundant predictors:

- `pdays` itself is **not entered as a numerical predictor** in any model, in
  any specification (A, B or C). It is used only to construct the
  `never_contacted` level of `poutcome` and the diagnostic `was_contacted_prev`.
  Entering the raw column would treat the 999 sentinel as a real duration.
- `was_contacted_prev` is **not** entered as a separate predictor: it
  duplicates `previous == 0` and the `never_contacted` level.
- `days_since_prev` is **not** entered in the primary specification: its
  missingness indicator would duplicate the same event, and median-imputing it
  for 81.7% of rows would add an arbitrary quantity rather than information.
  It remains available for a future, separately approved sensitivity analysis.

This choice is recorded here so that it cannot be revised after seeing test
results.

The five rows with `poutcome = "unknown"` and `pdays != 999` are **preserved as
`unknown`**. They are not merged into `never_contacted`, not imputed, and not
modified. Their `pdays` value is unused, because `pdays` is not a predictor.

### 5.4 Preprocessing

- All preprocessing inside `sklearn` pipelines, fitted on training data only.
- Numeric: standardisation (mean/SD from train).
- Categorical: one-hot encoding of observed levels, `handle_unknown="ignore"`.
- `"unknown"` retained as an explicit category; never imputed.
- No resampling, no SMOTE, no target encoding.
- Raw Parquet is read-only; transformations exist only in memory.

### 5.5 Splits

Primary (conditional on §4.3):

- Chronological by row index: **70% train / 15% validation / 15% test**.
- With n = 45,211: train rows `[0, 31,647)`, validation `[31,647, 38,428)`,
  test `[38,428, 45,211)` — 31,647 / 6,781 / 6,783 rows.
- Exact indices are saved once to `results/splits/temporal_split_v1.json` with
  a SHA-256 hash, after the chronology checks and **before** either execution
  run. Both the guided and the autonomous run load this file; neither
  regenerates it.
- Hyperparameter selection: expanding-window `TimeSeriesSplit` **within the
  training block only**.
- Validation block: used only for threshold/calibration decisions.
- Test block: untouched until final evaluation, opened once.

Secondary (sensitivity, clearly labelled optimistic): stratified 70/15/15
random split with a fixed seed (42), same preprocessing discipline.

### 5.6 Models

- **Primary:** L2-regularised logistic regression. Regularisation strength
  selected by time-series CV on the training block.
- **Secondary benchmark (after the primary analysis is complete, only with
  approval):** histogram gradient boosting.
- No random forest.
- Identical folds, identical preprocessing, identical test rows for A, B, C.

### 5.7 Metrics

| Metric | Role |
|---|---|
| ROC-AUC | Primary criterion (threshold-free) |
| PR-AUC / average precision | Imbalance-aware secondary criterion |
| Brier score + calibration curve | Probability quality |
| Lift @ top 10% | Business relevance |
| ΔAUC, ΔPR-AUC with paired bootstrap CI | Hypothesis test |
| DeLong test | **Optional** confirmatory check; not required for the decision |

**No-skill PR-AUC baseline.** Average precision under random scoring equals the
positive-class prevalence of the set being evaluated. It is therefore computed
**separately for the training, validation and test sets** from their actual
`y` values, and reported alongside every PR-AUC. The full-dataset prevalence
(11.70%) is never substituted for a set-specific baseline.

**Confidence intervals.** 2,000 **paired** bootstrap resamples over test rows,
fixed seed 42, percentile 95% intervals, computed for ΔROC-AUC and ΔPR-AUC.
This bootstrap is the primary inferential method. DeLong's test may be reported
as a secondary check but is not mandatory and does not override the bootstrap.

All three models are evaluated on the same test rows and reported in one table,
with Model C explicitly marked "diagnostic — leakage".

---

## 6. Evaluation

1. Freeze and hash the split indices and the preprocessing specification.
2. Fit A, B, C on the training block only.
3. Select hyperparameters by time-series CV on train.
4. Evaluate once on the test block; record metrics with 95% CIs.
5. Classify H1 using the table in §3.2.
6. Report calibration and lift at the top 10%.
7. Report set-specific PR-AUC baselines and the secondary random-split result
   as sensitivity analysis, without replacing the primary conclusion.
8. Report subgroup descriptives (previously contacted vs not) only as
   descriptive context — not as a new hypothesis test.

No test-set peeking, no post-hoc feature changes, no metric switching. The test
set is never used for model selection or tuning.

---

## 7. Deliverables

- `results/splits/` — chronology checks, split indices, hashes.
- `results/models/` — coefficients, regularisation choices, model versions.
- `results/predictions/` — test-set probabilities and labels per model.
- `results/metrics/` — metric tables, bootstrap CIs, calibration, lift.
- `results/audit/` — agent-audit rubric, filled with evidence.
- `results/logs/` — prompt, output and model name for every model call.
- `results/verification/` — independent recomputation.
- Final write-up with every number traceable to one of the above.

---

## 8. Verification Strategy

### 8.1 Independent verification

- Save test predictions (row index, y, p) for each model; hash the files.
- A **separately written** script, not importing the analysis pipeline,
  recomputes ROC-AUC, PR-AUC, lift@10% and the set-specific no-skill baselines
  from the saved predictions and labels.
- Agreement tolerance: 1e-6. Any discrepancy is investigated and reported.
- A clean-environment rerun with the pinned `uv.lock` must reproduce identical
  metrics for the same seeds and split file.
- The split file is reloaded, never regenerated, during verification.
- Verification recomputes metrics **without modifying or refitting the fitted
  models**.

### 8.2 Agent-audit rubric

Each criterion is scored PASS / FAIL / UNCLEAR with observable evidence and a
stated consequence. The rubric is frozen before either execution run and is
applied identically to both.

| # | Criterion | Observable evidence | Consequence if FAIL |
|---|---|---|---|
| 1 | Plan frozen before results | Timestamped `plan.md` vs first model run | Conclusions not admissible as pre-registered |
| 2 | Raw data untouched | File hash + mtime unchanged | Entire run invalidated |
| 3 | Split integrity | Saved indices reused; test opened once | Results invalidated |
| 4 | Preprocessing train-only | Pipeline code + fit calls | Leakage; results invalidated |
| 5 | `duration` excluded from A/B | Feature lists in code and logs | Leakage; results invalidated |
| 6 | `pdays` sentinel handled correctly | Recode code + distribution check | Feature invalid; results qualified |
| 7 | Categoricals treated as categorical | dtype handling in pipeline | Model misspecified |
| 8 | `"unknown"` preserved | Category levels in encoder | Information loss; documented |
| 9 | Metrics complete | ROC-AUC, PR-AUC, set-specific baselines, CIs, lift, calibration | Reporting deficiency |
| 10 | No causal language | Text audit | Claim overstated |
| 11 | Numbers traceable | Each figure maps to a `results/` file | Reproducibility failure |
| 12 | Model/prompt/output logged | `results/logs/` | RQ2 unanswerable |
| 13 | Deviations disclosed pre-test | Deviations log entries | Pre-registration compromised |
| 14 | Independent recomputation matches | Verification report | Results unconfirmed |
| 15 | Negative/insufficient results reported honestly | Text vs metric tables | Scientific integrity failure |

FAIL on criteria 2–5 invalidates the run. FAIL on other criteria qualifies the
conclusions and is reported explicitly. The rubric evaluates the agent's
**actual** decisions; no failure is asserted without evidence, and no success is
inferred from a single run.

### 8.3 Run modes (course requirement)

Two distinct OpenCode sessions execute the same frozen design. They do not
share analysis context.

| | Guided run | Autonomous run |
|---|---|---|
| Session | Separate OpenCode session | Separate OpenCode session |
| Scripts | Its own script directory | Its own script directory |
| Logs | Its own log directory | Its own log directory |
| Outputs | `results/guided/` | `results/autonomous/` |
| Shared context | This `plan.md` and the saved split file only | This `plan.md` and the saved split file only |
| Isolation | — | Must **not** use the guided run's analysis, code or results as context when constructing its solution |
| Audit | Same rubric (§8.2) | Same rubric (§8.2) |

- Both runs load the identical saved split indices from
  `results/splits/temporal_split_v1.json`.
- The test set is never used for model selection or tuning in either run.
- **Independent verification** is a third step: the author recomputes final
  metrics from the saved predictions without modifying the fitted models, and
  applies the same rubric to both runs.

---

## 9. Risks, Assumptions and Limitations

| Risk / assumption | Mitigation |
|---|---|
| Row order may not be chronological (no year) | Direct checks C3–C5; explicit qualification; stop and request a decision if they fail |
| Transformation/schema checks mistaken for chronology | §4.3 separates direct from indirect evidence |
| Temporal drift reduces B's advantage | Primary temporal split; report period structure |
| Test-set leakage through iteration | Frozen test, one-shot evaluation, saved indices |
| `campaign`/`contact`/`day`/`month` are borderline | Excluded from A/B/C |
| `previous` vs contact indicator redundancy | Documented; redundant predictors omitted from the primary spec (§5.3) |
| `pdays` sentinel (999) mistaken for a duration | `pdays` is never a predictor; used only to derive `never_contacted` (§5.2, §5.3) |
| `balance` snapshot timing unconfirmed | Retained in A/B/C with availability recorded as an **assumption**; if the snapshot were taken after the call, `balance` would be contaminated. No documentation resolves this, so the assumption is stated, not claimed as fact |
| Imbalance inflates ROC-AUC | PR-AUC with set-specific baseline, Brier, lift, calibration |
| Observational data, campaign-driven selection | No causal claims; explicit scope statement |
| Possible repeated clients (no client ID) | Non-independence untestable; documented |
| External validity (Portugal, 2008–2010) | Claims scoped to this dataset |
| Fairness (age/job/marital as proxies) | Descriptive subgroup reporting only |
| Agent gaming the meta-evaluation | Frozen rubric, pre-registration, independent recheck |

---

## 10. Deviations Log

Any change to this plan is recorded here **before** test results are inspected,
with date, reason and approval.

| Date | Section | Change | Reason | Approved by |
|---|---|---|---|---|
| 2026-11-13 | §5.2, §5.3, §9 | `pdays` removed as a separate predictor in Models B and C; it is used only to construct `poutcome = never_contacted`. `balance` pre-call availability recorded as an assumption. Version 1.1 → 1.2. | Internal inconsistency between the §5.2 feature table and the §5.3 encoding rule. | Project author (explicit written approval) |

### 2026-11-13 — `pdays` predictor ambiguity (v1.1 → v1.2)

**Original ambiguity.** The v1.1 feature table in §5.2 marked `pdays` as a
Model B and Model C feature, while §5.3 stated that `days_since_prev` (the
non-sentinel form of `pdays`) is not entered in the primary specification and
that Model B carries previous-campaign information through `previous` and
`poutcome`. The two sections therefore supported three incompatible readings:
enter raw `pdays` (which would treat 999 as a real duration), enter
`days_since_prev` (excluded by §5.3), or omit it (contradicting the table).

**Correction.** §5.2 now shows `pdays` as excluded from A, B and C, with a
footnote; §5.3 states explicitly that `pdays` is never a numerical predictor
and serves only to construct the `never_contacted` level of `poutcome`.
`previous` (numeric count) and `poutcome` (categorical, five levels) remain the
previous-campaign predictors in Models B and C. The five rows with
`poutcome = "unknown"` and `pdays != 999` remain `unknown`; their `pdays` value
is now unused.

**Rationale.** The sentinel encoding makes raw `pdays` unusable as a numeric
predictor without an arbitrary imputation rule. `days_since_prev` is undefined
for the 81.7% of rows that were never previously contacted, so imputing it
would add an arbitrary quantity rather than information. `previous` and the
five-level `poutcome` already carry the available previous-campaign
information, including the never-contacted state, without duplication. This is
the interpretation that §5.3 already described; v1.2 removes the contradiction
rather than changing the analysis.

**Also recorded.** The pre-call availability of `balance` is an **assumption**:
the dataset documentation does not confirm when the balance snapshot is taken
relative to the call. `balance` stays in Models A, B and C as approved, and
§9 records the residual uncertainty. This is a documentation change only.

**Approval.** Explicitly approved by the project author on 2026-11-13, together
with the codebook decision to document full categorical mappings in
`data/README.md` while citing the existing frequency CSVs instead of
duplicating counts.

**State at the time of this entry.** No predictive model has been trained, no
model has been fitted or tuned, and no test-set performance has been computed
or inspected. The only artefacts produced so far are the EDA outputs, the
chronology checks, the human adjudication, and the frozen split file.

**No other change.** Hypotheses (§3), metrics and decision rules (§5.7, §3.2),
the frozen split indices (§5.5), the audit rubric (§8.2), and the run-mode
design (§8.3) are unchanged from v1.1.
