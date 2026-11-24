# Model export validation (post-test usability artifact)

**Status: PASS - Models A and B exported.** Generated
2026-11-16T17:28:06+00:00 by
`src/my_analysis/export_frozen_guided_pipelines.py`.

## What this is, and what it is not

The original Guided run did **not** serialise its fitted models. This phase
reconstructs Models A and B from the frozen specification and exports them as
complete scikit-learn pipelines, so a reviewer can obtain a prediction without
retraining.

These are **post-test reconstructed artifacts, not the objects that produced
the original results**. They were built after the experiment ended. No original
metric, prediction, manifest or script was changed, and the frozen results
remain the only evidence for the reported findings.

## Procedure

1. Dataset SHA-256 and the frozen chronological split were verified against the
   recorded values (`verify_inputs`).
2. Features were decoded and prepared with the original Guided implementation
   (`prepare_frame`), and pipelines were built with the original factory
   (`build_pipeline`). Both are imported from
   `src/my_analysis/guided_test_evaluate.py`; no preprocessing or estimator
   specification was rewritten.
3. Models A and B were fitted on the original **31,647 training rows only**
   (rows `[0, 31,647)`). The validation block was not used for fitting.
4. Predictions on the original 6,781 validation rows were compared with
   `results/guided/validation_predictions.csv`.
5. Only after both comparisons passed at tolerance 1e-12 were the pipelines
   written to `models/A.joblib` and `models/B.joblib`.

## Numerical checks

| Model | Frozen C | Validation rows | Max abs difference | Tolerance | Result |
|---|---:|---:|---:|---:|---|
| A | 0.01 | 6781 | 0.000e+00 | 1e-12 | PASS |
| B | 0.01 | 6781 | 0.000e+00 | 1e-12 | PASS |

Saved predictions file SHA-256:
`f4197e99aa802eac3306b909dcdb587c08baa646b8a73427581d4e1864e57610`.

## Files

| File | SHA-256 | Bytes |
|---|---|---|
| `models/A.joblib` | `cab4df0760ea73ed27b759d23d13f3a027d27ebe750278a4a2521007eaa056d6` | 4849 |
| `models/B.joblib` | `403b16b6febe9f3b2bd627934d3afce136ddc3e94efe13c90a0046e9e1fd40a3` | 5153 |

## What was NOT done

- No chronological test label was read; the test block was not touched.
- No final-test metric was computed and the frozen test evaluation was not rerun.
- Model C was not trained (non-deployable leakage diagnostic).
- No hyperparameter search, calibration, feature selection or model redesign.
- No existing file, prediction, metric or manifest was modified.

## Limitations

- These pipelines are reconstructions. They reproduce the saved validation
  predictions to the stated tolerance, but they are not the original in-memory
  objects, and no test-block comparison was performed.
- Loading them requires a compatible scikit-learn version (recorded in
  `models/export_manifest.json`: scikit-learn 1.9.1); a different
  version may warn or change behaviour.
- The exported models inherit the project's calibration defect: on the
  chronological test block the mean predicted probability is about 0.068 while
  the observed subscription rate is 0.393. They are usable for ranking, not for
  probability statements about an individual customer.
- The export contains no inference interface and no input-schema helper. The
  pipelines still expect the dataset's integer codebook plus the
  `poutcome_derived` rule, so scoring a new customer requires a later phase.
