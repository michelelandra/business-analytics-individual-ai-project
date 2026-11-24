# Guided random-split sensitivity analysis — report

Script: `src/my_analysis/guided_random_split_sensitivity.py`
Generated (UTC): 2026-11-14T09:56:04+00:00
Plan reference: `plan.md` §5.5 (secondary sensitivity split)

**Status: secondary post-test sensitivity analysis.** This is not a new
independent confirmation, not a replacement for the primary chronological
evaluation, and not a basis for changing the primary conclusion. The random
split is **optimistic** because it mixes temporal information between training
and evaluation.

## 1. Split

| Item | Value |
|---|---|
| Method | stratified random 70/15/15, `train_test_split`, seed 42 |
| Dataset SHA-256 | `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f` |
| Split SHA-256 | `e43308ca7a0485fdf114be43372be3c0cc07017897ed766895de47e02f6eca3b` |
| Sizes | train 31,647 / validation 6,781 / test 6,783 |
| Test positives | 794 / 6,783 |
| Test prevalence | 0.117057 |
| Complete coverage | True |
| Pairwise disjoint | True |
| Reproducible from seed | True |

Full indices and generation details are in `results/sensitivity/random_split_v1.json`;
a human-readable summary is in `results/sensitivity/random_split_v1.md`.

**Stratification disclosure.** The target labels are used **only** to assign
split membership. They are not used for feature construction, preprocessing,
hyperparameter selection, calibration or model selection.

## 2. Overlap with the primary chronological blocks

| Random block | Chronological train [0, 31,647) | Chronological validation [31,647, 38,428) | Chronological test [38,428, 45,211) |
|---|---|---|---|
| random_train | 22,121 | 4,800 | 4,726 |
| random_validation | 4,758 | 974 | 1,049 |
| random_test | 4,768 | 1,007 | 1,008 |

Every random block draws rows from all chronological periods. This overlap is
the mechanism that makes the random-split estimate optimistic: rows that are
temporally adjacent can appear in both the random training and random test
blocks.

## 3. Frozen features, preprocessing and hyperparameters

Feature sets, codebook relabelling, `poutcome_derived` construction and the
`StandardScaler` + `OneHotEncoder(handle_unknown="ignore")` preprocessing are
unchanged from the guided primary analysis. Preprocessing is fitted on the
random training block only.

Frozen guided hyperparameters (no re-selection, no CV):

| Model | Role | C | penalty | solver | max_iter | random_state |
|---|---|---|---|---|---|---|
| A | customer characteristics only | 0.01 | l2 | lbfgs | 1000 | 42 |
| B | A + previous-campaign information | 0.01 | l2 | lbfgs | 1000 | 42 |
| C | **diagnostic — leakage; NOT deployable** | 0.001 | l2 | lbfgs | 1000 | 42 |

The random validation block (6,781 rows) was recorded but **not used** for
hyperparameter selection, calibration fitting, model selection or refitting.

## 4. Random-test metrics

Average-precision no-skill baseline is the **random-test prevalence**
(0.117057), not the chronological-test prevalence and not the
full-dataset prevalence.

| Model | Role | ROC-AUC | Average precision | AP no-skill | Brier | Lift@10% | C |
|---|---|---|---|---|---|---|---|
| A | customer characteristics only | 0.679998 | 0.223176 | 0.117057 | 0.098600 | 2.3814 | 0.01 |
| B | A + previous-campaign information | 0.728580 | 0.366773 | 0.117057 | 0.089903 | 3.4650 | 0.01 |
| C | diagnostic - leakage (B + duration); NOT deployable | 0.867877 | 0.449270 | 0.117057 | 0.083370 | 4.2966 | 0.001 |

Model C is a **non-deployable leakage diagnostic**. Its higher scores measure
how much of the outcome is encoded in call `duration`, not a deployable gain.

## 5. Paired bootstrap (2,000 resamples, seed 42)

| Comparison | Metric | Point estimate | 95% percentile CI | Valid resamples |
|---|---|---|---|---|
| B − A | ΔROC-AUC | 0.048582 | [0.036106, 0.061362] | 2,000 |
| B − A | ΔAP | 0.143597 | [0.114035, 0.172903] | 2,000 |
| C − B | ΔROC-AUC | 0.139297 | [0.119442, 0.160083] | 2,000 |
| C − B | ΔAP | 0.082497 | [0.041390, 0.125256] | 2,000 |

These intervals are **descriptive sensitivity intervals**, not a new
pre-registered hypothesis test. They do not override the primary chronological
bootstrap result.

## 6. Descriptive comparison with the chronological result

**Descriptive only.** The two evaluations use different test rows and different
prevalences; the differences below are not a test of a pre-registered
hypothesis and must not be read as a causal or confirmatory result.

| Model | Chrono ROC-AUC | Random ROC-AUC | Δ (random − chrono) | Chrono AP | Random AP | Δ AP | Chrono Brier | Random Brier | Δ Brier |
|---|---|---|---|---|---|---|---|---|---|
| A | 0.555604 | 0.679998 | +0.124394 | 0.437869 | 0.223176 | -0.214692 | 0.342691 | 0.098600 | -0.244091 |
| B | 0.585800 | 0.728580 | +0.142780 | 0.467045 | 0.366773 | -0.100272 | 0.342427 | 0.089903 | -0.252525 |
| C | 0.767393 | 0.867877 | +0.100484 | 0.619357 | 0.449270 | -0.170087 | 0.338002 | 0.083370 | -0.254632 |

Primary chronological B − A (for reference, from
`results/guided/test/hypothesis_tests_test.json`): ΔROC-AUC
0.030197
[0.026085,
0.034267];
ΔAP 0.029176
[0.024286,
0.034239].

ROC-AUC is higher on the random split for all three models. That is expected:
random splitting lets the model learn from rows interleaved in time with the
evaluation rows, whereas the chronological split asks for performance on a
later, unseen period. AP and Brier are lower on the random split, but those
metrics are prevalence-sensitive and the random-test prevalence is far lower
(0.117057 vs 0.393484), so that difference is not a like-for-like performance
comparison.

## 7. Prevalence sensitivity of AP and Brier

| Block | Prevalence | Positives / rows |
|---|---|---|
| Random train | 0.116978 | 3,702 / 31,647 |
| Random validation | 0.116944 | 793 / 6,781 |
| Random test | 0.117057 | 794 / 6,783 |
| Chronological test | 0.393484 | 2,669 / 6,783 |

The random test prevalence (0.117057) is much lower than the
chronological test prevalence (0.393484). **Average precision and
Brier score are prevalence-sensitive.** AP has a no-skill baseline equal to the
evaluated set's prevalence, and Brier score depends on the base rate. Therefore
random-test AP and Brier are not directly comparable to chronological-test AP
and Brier, and the comparison table above is descriptive only.

## 8. Methodological decisions made after seeing primary test results

All of the following were fixed by the approved sensitivity protocol and
implemented without using the observed primary test results to choose a
feature, hyperparameter, split seed or model specification:

1. Random split treated as the pre-registered secondary sensitivity analysis
   (`plan.md` §5.5), not as a replacement for the primary analysis.
2. Two-step stratified `train_test_split` with seed 42 and exact sizes
   31,647 / 6,781 / 6,783.
3. Indices sorted ascending after generation; membership unchanged.
4. Original guided feature sets and preprocessing reused verbatim.
5. Original guided `C` values frozen (A 0.01, B 0.01, C 0.001); no re-selection
   or cross-validation.
6. Random validation block recorded but not used for fitting, tuning,
   calibration or model selection.
7. Same metrics, lift definition and 2,000-resample paired bootstrap (seed 42)
   as the primary evaluation.
8. Model C retained only as a labelled non-deployable leakage diagnostic.
9. No recalibration, threshold tuning or post-hoc feature change.
10. Comparison with the chronological result reported descriptively only.

## 9. Warnings and limitations

- **Optimistic estimate.** Random splitting mixes temporal information; the
  random-split metrics are not an estimate of future-period performance.
- **Not independent.** The random test rows overlap all chronological blocks,
  including the primary chronological test block.
- **Different prevalence.** Random-test prevalence is
  0.117057 versus 0.393484 chronological; AP and Brier
  are prevalence-sensitive.
- **Stratification uses labels.** Labels assign split membership only; this is
  disclosed and does not affect feature construction or tuning.
- **Frozen C.** The random split was not used to re-select regularisation; the
  primary selected values were reused.
- **Model C is not deployable.** `duration` is post-contact information.
- **Single split seed.** One random split was generated and evaluated; no
  repeated random seeds were used, so split-to-split variability is not
  characterised.
- **Deterministic rerun disclosed.** A console-hash bookkeeping defect caused a
  deterministic rerun with an identical split hash and identical metrics; see
  `execution_log.md`.
- **No causal claims.** Nothing here supports a causal reading.

## 10. Files created

All files are under `results/sensitivity/`:

| File | Contents |
|---|---|
| `random_split_v1.json` | full indices, class counts, generation details, hash, overlap |
| `random_split_v1.md` | human-readable split summary |
| `metrics.csv` | one row per model, random-test metrics |
| `predictions.csv` | `row_index`, `y`, `p_A`, `p_B`, `p_C` for the 6,783 random-test rows |
| `calibration.csv` | decile calibration per model |
| `bootstrap.json` | 2,000-resample paired bootstrap results |
| `report.md` | this report |
| `run_manifest_sensitivity.json` | hashes, config, preservation check, environment |
| `console_output.txt` | captured console output |
| `prompt_log.md` | prompt and model-call log |
| `execution_log.md` | execution record |

Every number above comes from one of these files, produced by
`src/my_analysis/guided_random_split_sensitivity.py`.
