# Final test — independent verification (human-executed)

**Status:** Completed by the human project author.

**Attribution and scope of this document.** The verification recorded below was
performed **independently by the human project author**, directly in Windows
PowerShell. It was **not** performed by the AI agent. This document only records
the evidence supplied by the human author; the agent did not rerun the analyses,
did not refit any model, and did not modify any prediction, metric, bootstrap,
training, validation or test output. Where the saved artifacts are cited below,
they are cited as the source files that the human author independently checked,
not as a substitute for the human's calculations.

**Date documented:** 2026-11-14.

---

## 1. Artifacts covered by the verification

The verification covers the final test outputs of both independent executions.

### Guided execution

- `results/guided/test/test_predictions.csv`
- `results/guided/test/test_metrics.csv`
- `results/guided/test/hypothesis_tests_test.json`
- `results/guided/test/run_manifest_test.json`

(Guided workspace: `C:\Users\alber\OneDrive\Documenti\my-analysis`.)

### Autonomous execution

- `results/autonomous/test/test_predictions.csv`
- `results/autonomous/test/test_metrics.csv`
- `results/autonomous/test/hypothesis_tests_test.json`
- `results/autonomous/test/run_manifest_test.json`

(Autonomous workspace: `C:\Users\alber\OneDrive\Documenti\my-analysis-autonomous`.)

---

## 2. Row alignment and positive counts

The human author verified that both test prediction files contain exactly
**6,783 rows**, aligned to the frozen test indices `[38428, 45211)` (i.e. rows
`38428` through `45210` inclusive), with **2,669 positive outcomes** (`y = 1`).

This is consistent with the saved metric files, which record `n_test = 6783`
(guided) / `n = 6783` (autonomous), `n_positive = 2669` (guided) /
`positives = 2669` (autonomous), and prevalence `0.39348370927318294` in both.

---

## 3. Independent recomputation of point metrics

The human author independently recomputed, from the saved test prediction CSVs:

- ROC-AUC,
- average precision (PR-AUC),
- Brier score,
- lift at the top 10%.

For **Models A, B and C in both executions**, every recalculated metric matched
the corresponding reported value within **1e-6**.

The reported values in the saved artifacts (the values the independent
recomputation matched) are:

### Guided execution (`results/guided/test/test_metrics.csv`)

| Model | ROC-AUC | Average precision | Brier | Lift@10% |
|---|---|---|---|---|
| A | 0.5556035709881709 | 0.437868541560834 | 0.34269133584152267 | 1.1919846682825095 |
| B | 0.5858003804279422 | 0.4670448372049348 | 0.34242745574504724 | 1.3644101234428725 |
| C | 0.767392793580775 | 0.6193567633789513 | 0.3380019092016863 | 1.735499689983654 |

### Autonomous execution (`results/autonomous/test/test_metrics.csv`)

| Model | ROC-AUC | Average precision | Brier | Lift@10% |
|---|---|---|---|---|
| A | 0.5556035709881709 | 0.437868541560834 | 0.34269133584152267 | 1.1919846682825095 |
| B | 0.5858003804279422 | 0.4670448372049348 | 0.34242745574504724 | 1.3644101234428725 |
| C | 0.7673445251690624 | 0.6238923290083126 | 0.3313793426264262 | 1.7579899667437013 |

Models A and B have identical reported metrics across the two executions,
consistent with their identical predictions (Section 4). Model C differs because
the two executions selected different `C` values for Model C (guided
`C = 0.001`; autonomous `C = 0.01`), so Model C is not expected to be identical.

---

## 4. Cross-execution identity of Models A and B

The human author verified that the **guided and autonomous Models A and B
produced identical test predictions**, with a **maximum absolute difference of
exactly zero**.

This is a prediction-level check on the saved `p_A` / `p_model_a` and `p_B` /
`p_model_b` columns. It does not extend to Model C, whose two executions used
different selected regularisation strengths (see Section 3).

---

## 5. Label integrity

The human author verified that the test labels stored in the prediction files
matched the corresponding labels in the original Parquet dataset for the frozen
test rows.

This confirms row alignment between the saved predictions and the source data;
it is a label-consistency check, not a re-estimation of the models.

---

## 6. Independent paired bootstrap (primary B − A comparison)

The human author independently recomputed **2,000 paired bootstrap resamples**
for the primary **B − A** comparison, using
`numpy.random.default_rng(42)` and percentile 95% confidence intervals.

- **All 2,000 resamples were valid** (none skipped).
- The independent 95% percentile confidence intervals were:

| Quantity | Independent 95% CI |
|---|---|
| Δ ROC-AUC (B − A) | `[0.026085, 0.034267]` |
| Δ average precision (B − A) | `[0.024286, 0.034239]` |

Both the guided and the autonomous reported confidence intervals matched the
independent recomputation **within 1e-6**.

The reported values in the saved artifacts are identical across the two
executions for this comparison:

- `results/guided/test/hypothesis_tests_test.json` and
  `results/autonomous/test/hypothesis_tests_test.json` both report for `B-A`:
  - Δ ROC-AUC 95% CI: `[0.02608477195825771, 0.03426655663048662]`
  - Δ average precision 95% CI: `[0.024286032131931503, 0.03423888328606934]`
  - `n_valid_resamples = 2000`
  - `resamples_skipped_single_class = 0`

These reported intervals round to the independent intervals above and agree
within 1e-6.

---

## 7. Verification scope and limitations

**What this verification establishes.** The human author checked, independently
and from the saved files:

- row alignment and the number of test rows and positive outcomes;
- the test labels against the original Parquet dataset;
- ROC-AUC, average precision, Brier score and lift@10% for Models A, B and C in
  both executions;
- the primary B − A paired bootstrap (2,000 resamples, seed 42) and its 95%
  percentile confidence intervals;
- the cross-execution identity of Models A and B test predictions.

**What this verification does not establish.** This is a check of saved
predictions, metrics, bootstrap inference, row alignment and labels. It does
**not** independently establish that every model implementation, feature
definition, preprocessing step, hyperparameter-selection decision or
methodological choice was correct. It also does not independently recompute the
secondary C − B comparison; the bootstrap check recorded here concerns the
primary B − A comparison.

These results should therefore be read as an independent confirmation of the
reported test outputs, not as a full methodological audit.

---

## 8. Summary of verified facts

| # | Verified fact | Result |
|---|---|---|
| 1 | Test rows in both prediction files | 6,783 rows, indices `[38428, 45211)`, 2,669 positives |
| 2 | Independent recomputation of ROC-AUC, AP, Brier, lift@10% | Performed from saved CSVs |
| 3 | Agreement with reported metrics (A, B, C; both executions) | Within 1e-6 |
| 4 | Guided vs autonomous Models A and B predictions | Identical, max abs diff = 0 |
| 5 | Test labels vs original Parquet | Match |
| 6 | Independent paired bootstrap, B − A | 2,000 resamples, `numpy.random.default_rng(42)` |
| 7 | Independent 95% CI, Δ ROC-AUC | `[0.026085, 0.034267]` |
| 8 | Independent 95% CI, Δ average precision | `[0.024286, 0.034239]` |
| 9 | Reported CIs vs independent CIs | Within 1e-6 (both executions) |
| 10 | Valid bootstrap resamples | 2,000 of 2,000 |
