# Guided run — training and validation stage

Script: `src/my_analysis/guided_train_validate.py`
Command: `uv run python -m my_analysis.guided_train_validate`
Generated: 2026-11-13T19:30:11+00:00
Plan: `plan.md` v1.2 (design)
Split file records plan v1.1 (version in force when the split was created; boundaries unchanged)

**Scope.** Train + validation only. The test block (6,783 rows) was **not**
sliced, scored or inspected. No model was refit on train+validation. Model C is
a **diagnostic — leakage** model and is not deployable.

## Row counts

| Block | Rows | Used |
|---|---|---|
| train | 31,647 | yes — fitting and CV |
| validation | 6,781 | yes — metrics |
| test | 6,783 | **no** |

No rows were filtered, merged or dropped (total 45,211).

## Feature sets

| Model | Role | Numeric | Categorical |
|---|---|---|---|
| A | customer characteristics only | age, balance | job, marital, education, default, housing, loan |
| B | A + previous-campaign | age, balance, previous | job, marital, education, default, housing, loan, poutcome_derived |
| C | diagnostic — leakage | age, balance, previous, duration | job, marital, education, default, housing, loan, poutcome_derived |

`pdays` is not a predictor in any model. `poutcome` is recoded to five levels,
with `never_contacted` = `pdays == 999` (36,954 rows) and the five
anomalous rows kept as `unknown`.

## Preprocessing

Fitted on the training block only (and inside each CV fold):
`StandardScaler` on numeric columns; `OneHotEncoder(handle_unknown="ignore")`
on categorical columns; `remainder="drop"`. No imputer (0 NaN), no resampling,
no target encoding. Spec hash: `73887c4654f695cd103475530400af074419b8d2b2b3c44291bd4ffe4ee8c7b9`.

## Hyperparameter selection

`GridSearchCV`, expanding-window `TimeSeriesSplit(n_splits=5)` on the
training block only, scored by mean CV ROC-AUC. `C ∈ [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]`.

| Model | C | mean CV ROC-AUC | SD | rank |
|---|---|---|---|---|
| A | 0.001 | 0.561896 | 0.025067 | 4 |
| A | 0.003 | 0.564002 | 0.023365 | 2 |
| A | 0.01 | 0.564748 | 0.021766 | 1 |
| A | 0.03 | 0.563197 | 0.020221 | 3 |
| A | 0.1 | 0.560820 | 0.018926 | 5 |
| A | 0.3 | 0.558515 | 0.017465 | 6 |
| A | 1 | 0.558062 | 0.017587 | 7 |
| A | 3 | 0.557391 | 0.017928 | 8 |
| A | 10 | 0.556791 | 0.018249 | 9 |
| B | 0.001 | 0.565022 | 0.025245 | 4 |
| B | 0.003 | 0.566844 | 0.023885 | 2 |
| B | 0.01 | 0.567769 | 0.023136 | 1 |
| B | 0.03 | 0.566289 | 0.021940 | 3 |
| B | 0.1 | 0.563626 | 0.020834 | 5 |
| B | 0.3 | 0.558279 | 0.022639 | 6 |
| B | 1 | 0.556409 | 0.022856 | 7 |
| B | 3 | 0.555060 | 0.023217 | 8 |
| B | 10 | 0.554938 | 0.023229 | 9 |
| C | 0.001 | 0.907256 | 0.075867 | 1 |
| C | 0.003 | 0.907003 | 0.075245 | 2 |
| C | 0.01 | 0.906258 | 0.074906 | 3 |
| C | 0.03 | 0.905351 | 0.074337 | 4 |
| C | 0.1 | 0.904288 | 0.073406 | 5 |
| C | 0.3 | 0.902941 | 0.072996 | 6 |
| C | 1 | 0.901227 | 0.073315 | 7 |
| C | 3 | 0.900934 | 0.072913 | 8 |
| C | 10 | 0.900373 | 0.073301 | 9 |

Selected: A: C=0.01 (CV AUC 0.564748), B: C=0.01 (CV AUC 0.567769), C: C=0.001 (CV AUC 0.907256)

## Validation metrics

No-skill average precision is the **validation-set prevalence**
(0.115027), not the full-file prevalence.

| Model | Role | ROC-AUC | Avg precision | AP no-skill | Brier | Lift@10% |
|---|---|---|---|---|---|---|
| A | customer characteristics only | 0.640340 | 0.208401 | 0.115027 | 0.103874 | 2.3209 |
| B | A + previous-campaign information | 0.659535 | 0.228795 | 0.115027 | 0.103818 | 2.4747 |
| C | diagnostic - leakage (B + duration); NOT deployable | 0.868365 | 0.472761 | 0.115027 | 0.090059 | 4.6417 |

### Lift at the top 10%

| Model | k | k/n | TP in top k | Precision@k | Recall@k | Expected TP by chance | Lift |
|---|---|---|---|---|---|---|---|
| A | 678 | 0.099985 | 181 | 0.266962 | 0.232051 | 77.99 | 2.3209 |
| B | 678 | 0.099985 | 193 | 0.284661 | 0.247436 | 77.99 | 2.4747 |
| C | 678 | 0.099985 | 362 | 0.533923 | 0.464103 | 77.99 | 4.6417 |

## Base rates, calibration and interpretation limits

**Observed base-rate shift (empirical, from this run).** The training block has
prevalence 0.058141 (1,840/31,647) while the
validation block has 0.115027 (780/6,781).
The validation period is roughly 1.98x more positive
than the training period. This is a property of the chronological split, not of
the modelling code, and it is why the no-skill AP reference is the
validation prevalence.

**Consequence for Brier and calibration.** Every model's mean predicted
probability on validation is below the validation prevalence:

| Model | mean predicted | validation prevalence | ratio |
|---|---|---|---|
| A | 0.054843 | 0.115027 | 0.4768 |
| B | 0.052808 | 0.115027 | 0.4591 |
| C | 0.057152 | 0.115027 | 0.4969 |

The models are therefore **under-calibrated in level** on validation: they rank
reasonably but systematically under-predict the positive rate. The Brier scores
above should be read with this in mind; they are not evidence of a well
calibrated probability model. Per-decile detail is in
`results/guided/calibration_deciles.csv`. No recalibration, prior correction or
threshold tuning was applied — none is pre-registered for this stage.

**Consequence for the CV estimate.** The mean CV ROC-AUC within the training
block (0.556–0.568 for A and B) is much lower than the validation ROC-AUC
(0.640 and 0.660). The cross-validation folds are all *within* the training
period, where the positive rate is low and comparatively flat; the validation
block is a later period with a higher positive rate. The two numbers are not
estimates of the same quantity and should not be compared as if one validated
the other.

**Scope.** These are descriptive held-out performance estimates for the guided
run. No test-set number exists yet, no final conclusion is drawn, and nothing
here supports a causal reading. Model C is a leakage diagnostic: its higher
scores measure how much of the outcome is encoded in call duration, not a
deployable gain.

## Files

| File | Contents |
|---|---|
| `results/guided/validation_metrics.csv` | one row per model, all validation metrics |
| `results/guided/validation_predictions.csv` | `row_index`, `y`, `p_A`, `p_B`, `p_C` for the 6,781 validation rows |
| `results/guided/hyperparameter_selection.csv` | full CV grid per model |
| `results/guided/calibration_deciles.csv` | decile calibration per model |
| `results/guided/model_coefficients.csv` | fitted coefficients on the standardised/one-hot scale |
| `results/guided/run_manifest.json` | hashes, versions, config, fold sizes, artifact hashes |
| `results/guided/run_warnings.txt` | captured Python warnings (convergence etc.) |

Every number above comes from one of these files.

## Implementation notes and warnings

- **`penalty="l2"` FutureWarning (scikit-learn 1.9.1).** `plan.md` §5.6 specifies
  an L2-penalised logistic regression. In scikit-learn 1.9 the `penalty`
  argument is deprecated in favour of `l1_ratio`; the code keeps
  `penalty="l2"` so the specification is stated explicitly, and the resulting
  `FutureWarning` is captured verbatim in `results/guided/run_warnings.txt`.
  The fitted model is unchanged (L2, `lbfgs`). No convergence warnings were
  raised.
- **`C` selected by CV ROC-AUC, not by validation performance.** Selection used
  the training block only; validation was scored once, after selection.
- **`pdays`** is read only to derive `never_contacted`; it is never passed to a
  scaler, encoder or estimator.
- **Test block.** `results/guided/run_manifest.json` records
  `test_evaluated: false` and `split.test_rows_used: 0`.
