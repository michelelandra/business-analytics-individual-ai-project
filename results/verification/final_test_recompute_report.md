# Independent recomputation — frozen chronological final-test metrics

**Status: PASS.** Generated 2026-11-16T16:39:08+00:00 by
`results/verification/verify_final_test_recompute.py`.

> **Model C is a NON-DEPLOYABLE duration-leakage diagnostic.** It is reported
> only to quantify leakage inflation and is not an operational model.

**Scope.** Metric-level recomputation from the saved chronological final-test
predictions. No model was fitted, no prediction was generated, and no analysis
was rerun. `data/raw/` was not opened. This is an OpenCode-generated,
independently implemented computational check; it **supplements but does not
replace** the human PowerShell verification in
`results/verification/final_test_independent_verification.md`.

## Environment

| Item | Value |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.1 |
| Platform | Windows-11-10.0.26200-SP0 |
| Bootstrap | 2000 paired resamples, `default_rng(42)`, `integers(0, n, n)`, percentile 95% |
| Tolerance | 1e-06 |

## Recomputed point metrics (test n = 6,783; positives = 2,669; prevalence = 0.393483709273)

| Model | ROC-AUC | Average precision | Brier | Lift@10% | TP in top 10% |
|---|---:|---:|---:|---:|---:|
| A | 0.555603570988 | 0.437868541561 | 0.342691335842 | 1.191984668283 | 318/678 |
| B | 0.585800380428 | 0.467044837205 | 0.342427455745 | 1.364410123443 | 364/678 |
| C | 0.767392793581 | 0.619356763379 | 0.338001909202 | 1.735499689984 | 463/678 |

## Checks

### Integrity and metadata

| hash | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `hash_hypothesis_tests_test.json` | 8e2788be011ec63c0eaf94a2caa787ad6269c9bc1f9eaf65905d62ca04f840b5 | 8e2788be011ec63c0eaf94a2caa787ad6269c9bc1f9eaf65905d62ca04f840b5 | — | PASS |
| `hash_test_calibration.csv` | 506cf277679993e22dad1cc4302573abf3b45b7e2898480df67b2aa24a0965a7 | 506cf277679993e22dad1cc4302573abf3b45b7e2898480df67b2aa24a0965a7 | — | PASS |
| `hash_test_metrics.csv` | 48e7fa1dcf2c510abd106fced42b1af251426ea902d808f767cfbde5e0a8f0fa | 48e7fa1dcf2c510abd106fced42b1af251426ea902d808f767cfbde5e0a8f0fa | — | PASS |
| `hash_test_predictions.csv` | 2dc5a653e567ebd5821233ee9b72ce984d9d3e2790671b7620d298cbc74b4139 | 2dc5a653e567ebd5821233ee9b72ce984d9d3e2790671b7620d298cbc74b4139 | — | PASS |
| `hash_analysis_script` | 2ac3a3eac188254343bc0f63fd284a29de348a7dd078a19be8d01868e666ba36 | 2ac3a3eac188254343bc0f63fd284a29de348a7dd078a19be8d01868e666ba36 | — | PASS |

| split | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `split_canonical_hash` | d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c | d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c | — | PASS |
| `split_matches_manifest` | d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c | d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c | — | PASS |
| `split_sizes` | {'train': 31647, 'validation': 6781, 'test': 6783} | {'train': 31647, 'validation': 6781, 'test': 6783} | — | PASS |

| dataset | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `dataset_metadata_consistency` | 0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f | 0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f | — | PASS |

### Prediction structure

| row | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `n_rows` | True | True | — | PASS |
| `n_positive` | True | True | — | PASS |

| index | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `row_index_contiguous` | True | True | — | PASS |
| `row_alignment_split` | True | True | — | PASS |

| prevalence | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `prevalence` | 0.393483709273 | 0.393483709273 | 0.000e+00 | PASS |

| nulls | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `no_nulls` | False | False | — | PASS |

| model C | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `model_c_role` | True | True | — | PASS |
| `model_c_json_note` | True | True | — | PASS |

### Model metrics

| A | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `A_roc_auc` | 0.555603570988 | 0.555603570988 | 0.000e+00 | PASS |
| `A_roc_auc_mannwhitney` | 0.555603570988 | 0.555603570988 | 0.000e+00 | PASS |
| `A_pr_auc` | 0.437868541561 | 0.437868541561 | 0.000e+00 | PASS |
| `A_brier` | 0.342691335842 | 0.342691335842 | 0.000e+00 | PASS |
| `A_brier_sklearn` | 0.342691335842 | 0.342691335842 | 0.000e+00 | PASS |
| `A_mean_predicted` | 0.0688427181796 | 0.0688427181796 | 0.000e+00 | PASS |
| `A_lift` | 1.19198466828 | 1.19198466828 | 0.000e+00 | PASS |
| `A_lift_k` | 678 | 678 | 0.000e+00 | PASS |
| `A_lift_tp` | 318 | 318 | 0.000e+00 | PASS |
| `A_lift_precision` | 0.469026548673 | 0.469026548673 | 0.000e+00 | PASS |
| `A_lift_recall` | 0.119145747471 | 0.119145747471 | 0.000e+00 | PASS |
| `A_lift_expected` | 266.781954887 | 266.781954887 | 0.000e+00 | PASS |

| B | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `B_roc_auc` | 0.585800380428 | 0.585800380428 | 0.000e+00 | PASS |
| `B_roc_auc_mannwhitney` | 0.585800380428 | 0.585800380428 | 0.000e+00 | PASS |
| `B_pr_auc` | 0.467044837205 | 0.467044837205 | 0.000e+00 | PASS |
| `B_brier` | 0.342427455745 | 0.342427455745 | 0.000e+00 | PASS |
| `B_brier_sklearn` | 0.342427455745 | 0.342427455745 | 0.000e+00 | PASS |
| `B_mean_predicted` | 0.0676601805258 | 0.0676601805258 | 0.000e+00 | PASS |
| `B_lift` | 1.36441012344 | 1.36441012344 | 0.000e+00 | PASS |
| `B_lift_k` | 678 | 678 | 0.000e+00 | PASS |
| `B_lift_tp` | 364 | 364 | 0.000e+00 | PASS |
| `B_lift_precision` | 0.536873156342 | 0.536873156342 | 0.000e+00 | PASS |
| `B_lift_recall` | 0.136380666916 | 0.136380666916 | 0.000e+00 | PASS |
| `B_lift_expected` | 266.781954887 | 266.781954887 | 0.000e+00 | PASS |

| C | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `C_roc_auc` | 0.767392793581 | 0.767392793581 | 0.000e+00 | PASS |
| `C_roc_auc_mannwhitney` | 0.767392793581 | 0.767392793581 | 1.110e-16 | PASS |
| `C_pr_auc` | 0.619356763379 | 0.619356763379 | 0.000e+00 | PASS |
| `C_brier` | 0.338001909202 | 0.338001909202 | 0.000e+00 | PASS |
| `C_brier_sklearn` | 0.338001909202 | 0.338001909202 | 0.000e+00 | PASS |
| `C_mean_predicted` | 0.062979606932 | 0.062979606932 | 0.000e+00 | PASS |
| `C_lift` | 1.73549968998 | 1.73549968998 | 0.000e+00 | PASS |
| `C_lift_k` | 678 | 678 | 0.000e+00 | PASS |
| `C_lift_tp` | 463 | 463 | 0.000e+00 | PASS |
| `C_lift_precision` | 0.682890855457 | 0.682890855457 | 0.000e+00 | PASS |
| `C_lift_recall` | 0.17347321094 | 0.17347321094 | 0.000e+00 | PASS |
| `C_lift_expected` | 266.781954887 | 266.781954887 | 0.000e+00 | PASS |

### Paired bootstrap: B − A

| B-A | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `B-A_delta_roc_auc_point` | 0.0301968094398 | 0.0301968094398 | 0.000e+00 | PASS |
| `B-A_delta_roc_auc_ci_lo` | 0.0260847719583 | 0.0260847719583 | 0.000e+00 | PASS |
| `B-A_delta_roc_auc_ci_hi` | 0.0342665566305 | 0.0342665566305 | 0.000e+00 | PASS |
| `B-A_delta_roc_auc_n_valid` | 2000 | 2000 | 0.000e+00 | PASS |
| `B-A_delta_pr_auc_point` | 0.0291762956441 | 0.0291762956441 | 0.000e+00 | PASS |
| `B-A_delta_pr_auc_ci_lo` | 0.0242860321319 | 0.0242860321319 | 0.000e+00 | PASS |
| `B-A_delta_pr_auc_ci_hi` | 0.0342388832861 | 0.0342388832861 | 0.000e+00 | PASS |
| `B-A_delta_pr_auc_n_valid` | 2000 | 2000 | 0.000e+00 | PASS |

### Paired bootstrap: C − B (leakage diagnostic)

| C-B | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `C-B_delta_roc_auc_point` | 0.181592413153 | 0.181592413153 | 0.000e+00 | PASS |
| `C-B_delta_roc_auc_ci_lo` | 0.1654750378 | 0.1654750378 | 0.000e+00 | PASS |
| `C-B_delta_roc_auc_ci_hi` | 0.197654539939 | 0.197654539939 | 0.000e+00 | PASS |
| `C-B_delta_roc_auc_n_valid` | 2000 | 2000 | 0.000e+00 | PASS |
| `C-B_delta_pr_auc_point` | 0.152311926174 | 0.152311926174 | 0.000e+00 | PASS |
| `C-B_delta_pr_auc_ci_lo` | 0.130024239293 | 0.130024239293 | 0.000e+00 | PASS |
| `C-B_delta_pr_auc_ci_hi` | 0.173742875904 | 0.173742875904 | 0.000e+00 | PASS |
| `C-B_delta_pr_auc_n_valid` | 2000 | 2000 | 0.000e+00 | PASS |

| bootstrap | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `boot_skipped` | 0 | 0 | 0.000e+00 | PASS |
| `boot_n` | 2000 | 2000 | 0.000e+00 | PASS |
| `boot_seed` | 42 | 42 | 0.000e+00 | PASS |

### Frozen classifications

| H1 | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `H1_classification` | H0 rejected, materially relevant | H0 rejected, materially relevant | — | PASS |

| H2 | reported | recomputed | abs diff | result |
|---|---:|---:|---:|---|
| `H2_delta` | 0.181592413153 | 0.181592413153 | 0.000e+00 | PASS |
| `H2_material` | True | True | — | PASS |

## Integrity

- Protected files hashed before: 112
- Protected files hashed after: 112
- Added by this run: none
- Removed: none
- Changed: none

Dataset and split metadata were verified from saved manifests and the saved
split file only. The canonical split hash
`d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c` was recomputed from the saved index arrays without
opening the raw dataset, and the dataset SHA-256 was checked for consistency
between the test manifest and the split file.

## Limitations

- This verifies **metric arithmetic, bootstrap inference and saved-file
  integrity**. It does **not** verify feature engineering, categorical handling,
  the derived previous-campaign variable, preprocessing, hyperparameter
  selection, or model training.
- Labels are taken from the saved prediction file. That they match the raw
  Parquet was verified separately by the human author; it is not re-established
  here.
- Agreement is expected to be near-exact because the same saved probabilities
  and the same documented bootstrap convention are used. The check therefore
  tests reproducibility of the reported numbers, not the correctness of the
  modelling pipeline.
- AP is computed with scikit-learn, whose tie handling differs from naive
  hand implementations; this is deliberate and matches the frozen artifacts.

## Result

**PASS** — 77 of 77 checks passed.
No discrepancy exceeded tolerance; all protected artefacts are unchanged.
