# Corrected guided random-split sensitivity evaluation — report

Script: `src/my_analysis/guided_random_split_sensitivity_evaluate.py`
Generated (UTC): 2026-11-14T11:25:58+00:00
Plan reference: `plan.md` §5.5 (secondary sensitivity split)

**Status: POST-TEST SECONDARY SENSITIVITY ANALYSIS.** This evaluation uses the
human-verified corrected random split. It is **not** a new independent
confirmation, **not** a replacement for the primary chronological test, and
**not** a basis for changing the primary conclusion. The random split is
**optimistic** because it mixes temporal information between training and
evaluation.

## 1. Verified corrected split

| Item | Value |
|---|---|
| Split artifact | `results/sensitivity_corrected_v1/random_split_v1.json` |
| Canonical fingerprint | `6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0` |
| Sizes | train 31,647 / validation 6,781 / test 6,783 |
| Positive counts | train 3,702 / validation 793 / test 794 |
| Test prevalence | 0.117057 |
| Complete coverage | True |
| Pairwise disjoint | True |
| Random-test vs chronological-test overlap | 1,022 |
| Split-only status verified | True |

The fingerprint, sizes and class counts were verified before any model was
constructed or fitted. The index arrays were loaded directly from the split
JSON; no split was regenerated.

## 2. Frozen features, preprocessing and hyperparameters

Feature sets, codebook relabelling, `poutcome_derived` construction and the
`StandardScaler` + `OneHotEncoder(handle_unknown="ignore")` preprocessing are
unchanged from the guided primary analysis. Preprocessing is fitted on the
corrected random training block only.

Frozen guided hyperparameters (no re-selection, no CV):

| Model | Role | C | penalty | solver | max_iter | random_state |
|---|---|---|---|---|---|---|
| A | customer characteristics only | 0.01 | l2 | lbfgs | 1000 | 42 |
| B | A + previous-campaign information | 0.01 | l2 | lbfgs | 1000 | 42 |
| C | **diagnostic — leakage; NOT deployable** | 0.001 | l2 | lbfgs | 1000 | 42 |

The random validation block (6,781 rows) was verified but
**never loaded into a fitting object** and was not used for hyperparameter
selection, calibration, model selection or refitting.

## 3. Corrected random-test metrics

Average-precision no-skill baseline is the **corrected random-test prevalence**
(0.117057), not the chronological-test prevalence and not
the full-dataset prevalence.

| Model | Role | ROC-AUC | Average precision | AP no-skill | Brier | Lift@10% | C |
|---|---|---|---|---|---|---|---|
| A | customer characteristics only | 0.648781 | 0.197967 | 0.117057 | 0.100110 | 2.1546 | 0.01 |
| B | A + previous-campaign information | 0.703351 | 0.323390 | 0.117057 | 0.092144 | 3.2004 | 0.01 |
| C | diagnostic - leakage (B + duration); NOT deployable | 0.871345 | 0.465148 | 0.117057 | 0.082550 | 4.5360 | 0.001 |

Model C is a **non-deployable duration-leakage diagnostic**. Its higher scores
measure how much of the outcome is encoded in call `duration`, not a deployable
gain.

## 4. Calibration summaries

| Model | Bins | Mean predicted | Observed rate | Weighted mean abs gap | Max abs gap |
|---|---|---|---|---|---|
| A | 10 | 0.118559 | 0.117057 | 0.010369 | 0.026117 |
| B | 10 | 0.118883 | 0.117057 | 0.011595 | 0.037667 |
| C | 10 | 0.116994 | 0.117057 | 0.051115 | 0.130147 |

The full ten-bin tables are in `calibration.csv`.

## 5. Paired bootstrap (2,000 resamples, seed 42)

| Comparison | Metric | Point estimate | 95% percentile CI | Valid resamples |
|---|---|---|---|---|
| B − A | ΔROC-AUC | 0.054570 | [0.041805, 0.066600] | 2,000 |
| B − A | ΔAP | 0.125422 | [0.098713, 0.154288] | 2,000 |
| C − B | ΔROC-AUC | 0.167994 | [0.147389, 0.188793] | 2,000 |
| C − B | ΔAP | 0.141759 | [0.101859, 0.182669] | 2,000 |

These intervals are **descriptive sensitivity intervals**, not a new
pre-registered hypothesis test. They do not override the primary chronological
bootstrap result.

## 6. Descriptive comparison with the primary chronological test

**Descriptive only.** The two evaluations use different test rows and different
prevalences; the differences below are not a test of a pre-registered
hypothesis and must not be read as a causal or confirmatory result.

| Model | Chrono ROC-AUC | Random ROC-AUC | Δ (random − chrono) | Chrono AP | Random AP | Δ AP | Chrono Brier | Random Brier | Δ Brier |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.555604 | 0.648781 | +0.093178 | 0.437869 | 0.197967 | -0.239901 | 0.342691 | 0.100110 | -0.242582 |
| B | 0.585800 | 0.703351 | +0.117551 | 0.467045 | 0.323390 | -0.143655 | 0.342427 | 0.092144 | -0.250283 |
| C | 0.767393 | 0.871345 | +0.103952 | 0.619357 | 0.465148 | -0.154208 | 0.338002 | 0.082550 | -0.255452 |

Primary chronological B − A (for reference, from
`results/guided/test/hypothesis_tests_test.json`): ΔROC-AUC
0.030197
[0.026085,
0.034267];
ΔAP 0.029176
[0.024286,
0.034239].

The random split generally produces higher apparent performance than the
chronological split. That is expected: random splitting lets the model learn
from rows interleaved in time with the evaluation rows, whereas the
chronological split asks for performance on a later, unseen period.

## 7. Warnings and limitations

- **Optimistic estimate.** Random splitting mixes temporal information; the
  random-split metrics are not an estimate of future-period performance.
- **Not independent.** The random test rows overlap all chronological blocks,
  including the primary chronological test block.
- **Different prevalence.** Random-test prevalence is
  0.117057 versus 0.393484 chronological; AP and Brier are
  prevalence-sensitive.
- **Model C is not deployable.** It uses `duration`, which is unknown before a
  call ends.
- **Secondary.** This report must not replace the primary chronological test.

## 8. Artifacts

- `metrics.csv`
- `predictions.csv`
- `calibration.csv`
- `bootstrap.json`
- `run_manifest_evaluation.json`
- `console_output_evaluation.txt`
- `execution_log_evaluation.md`

The human-verified split-only artifacts were not modified.
