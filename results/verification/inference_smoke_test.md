# Inference smoke test (post-test usability artifact)

**Status: PASS.** Executed 2026-11-16 on the project machine, after Phase A
(model export) and Phase B (inference interface). Nothing below is
reconstructed or estimated: every command and output was copied from an actual
run.

This test checks that the new inference script works. It says nothing about
model quality, and it is **not** a new evaluation of Models A or B.

- Script under test: `src/my_analysis/predict.py`
- Models used: `models/A.joblib`, `models/B.joblib`
- Input data: `models/example_customer.csv` (two synthetic customers, invented
  for this test; not taken from any real record)
- Environment: Python 3.13.5, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.1,
  joblib 1.6.0, Windows 11

---

## 1. Models load and produce finite scores in [0, 1]

Command (run from the repository root):

```powershell
.venv\Scripts\python.exe -m my_analysis.predict --input models\example_customer.csv
```

Actual output:

```text
Frozen Guided Models A and B - inference (post-test usability artifact)
Input file: C:\Users\alber\OneDrive\Documenti\my-analysis\models\example_customer.csv
Rows scored: 2
Model A: customer characteristics only; numeric ['age', 'balance'], categorical ['job', 'marital', 'education', 'default', 'housing', 'loan']
Model B: A + previous-campaign information; numeric ['age', 'balance', 'previous'], categorical ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome_derived']
Frozen C: A=0.01, B=0.01 (fixed; nothing was fitted, tuned or recalibrated)
Row labels come from column 'customer_id'.

 row  customer              p_A         p_B      A %      B %
-------------------------------------------------------------
   2  SYNTHETIC-001  0.053667  0.054235    5.37%    5.42%
   3  SYNTHETIC-002  0.082571  0.089312    8.26%    8.93%

Calibration warning
  These scores are not trustworthy individual probabilities. On the chronological
  test block the mean predicted probability is about 6.8% while about 39.3% of
  customers actually subscribed (results/guided/test/test_metrics.csv).
  Models A and B are useful for RANKING customers relative to each other, not
  for statements such as 'this customer has a 5% chance'. A low score does not
  mean the customer will not subscribe.

Model C was not loaded: it is a non-deployable leakage diagnostic.
Nothing was fitted, and no file was written or modified by this script.
```

Exit code: `0`.

| Check | A | B |
|---|---|---|
| Score is finite | yes | yes |
| Score within [0, 1] | yes (0.053667, 0.082571) | yes (0.054235, 0.089312) |
| Two rows scored, one per customer | yes | yes |

The script itself refuses to print a score if any value is non-finite or outside
`[0, 1]`; the guard did not trigger.

## 2. The same input produces the same output on a repeated run

Two consecutive runs, stdout redirected to files and hashed:

```powershell
.venv\Scripts\python.exe -m my_analysis.predict --input models\example_customer.csv > run1.txt 2>&1
.venv\Scripts\python.exe -m my_analysis.predict --input models\example_customer.csv > run2.txt 2>&1
Get-FileHash run1.txt, run2.txt -Algorithm SHA256
```

Actual result:

```text
exit1=0  exit2=0
sha256 run1: baa13ab7edec6c197b1047afd1a411c163d86db8f3f051ee04d5829b2774c2e1
sha256 run2: baa13ab7edec6c197b1047afd1a411c163d86db8f3f051ee04d5829b2774c2e1
identical: True
```

The full output, including the printed table, is byte-identical between runs.

## 3. Invalid input produces a clear error

Each case used a temporary CSV outside the repository (in the system temp
folder), derived from the example file with one defect introduced.

| Case | Exit code | Message |
|---|---:|---|
| Unknown categorical code (`job = 99`) | 2 | `row 2: 'job' code 99 is unknown. Valid codes: 0=admin., 1=blue-collar, ...` |
| Missing required column (`pdays` dropped) | 2 | `missing required column(s): pdays (see models/input_schema.md for the full schema)` |
| Blank cell (`age` empty) | 2 | `row 2: 'age' is missing (blank)` |
| Non-numeric value (`age = thirty`) | 2 | `row 2: 'age' must be a number, got 'thirty'` |

Raw output of the first case:

```text
ERROR: the input could not be scored.
  - row 2: 'job' code 99 is unknown. Valid codes: 0=admin., 1=blue-collar, 2=entrepreneur, 3=housemaid, 4=management, 5=retired, 6=self-employed, 7=services, 8=student, 9=technician, 10=unemployed, 11=unknown

Nothing was written. See models/input_schema.md for the required schema.
```

In every case no score was printed and no file was created.

## 4. Columns that must not be used are ignored, not consumed

A CSV carrying `duration` and `y` alongside the valid features scores
successfully and prints:

```text
Note: the outcome column is present but ignored: scoring never uses the true result
Note: 'duration' is present but ignored: it is unavailable before a call and is not a feature of Models A or B
   2  SYN-X     0.045883  0.046292    4.59%    4.63%
```

Exit code `0`. The scores are produced without those columns entering the model.

## 5. No fitting occurs

- Grepping `src/my_analysis/predict.py` for `.fit(`, `fit_transform`,
  `partial_fit`, `GridSearch`, `calibrat*` returns only prose strings in the
  docstring and the printed "nothing was fitted" line. There is no estimator
  training call anywhere in the script.
- The script loads models with `joblib.load` only. It imports `CODEBOOK`,
  `MODELS` and `prepare_frame` from
  `src/my_analysis/guided_test_evaluate.py`; it never imports `build_pipeline`,
  `load_train_validation` or `load_test_block`, so it cannot construct a new
  estimator or read labels.
- Model file hashes after the test runs still equal the values recorded in
  `models/export_manifest.json`:

| File | SHA-256 | Matches manifest |
|---|---|---|
| `models/A.joblib` | `cab4df0760ea73ed27b759d23d13f3a027d27ebe750278a4a2521007eaa056d6` | yes |
| `models/B.joblib` | `403b16b6febe9f3b2bd627934d3afce136ddc3e94efe13c90a0046e9e1fd40a3` | yes |

File modification timestamps are unchanged from the export (18:28:07), i.e.
`joblib.load` did not rewrite them.

The hash-guard that rejects an altered model file was not exercised
destructively: corrupting a model to test it would violate the protected-file
rule. The check exists in `load_models()` and is therefore untested end-to-end.

## 6. No historical prediction or evaluation artifact was modified

Every artifact recorded in the two run manifests still matches its recorded
SHA-256, checked after the inference runs:

| Artifact | Hash matches manifest |
|---|---|
| `results/guided/validation_predictions.csv` | yes (`f4197e99…e57610`) |
| `results/guided/validation_metrics.csv` | yes (`93507180…`) |
| `results/guided/calibration_deciles.csv` | yes (`77596fce…`) |
| `results/guided/model_coefficients.csv` | yes (`e3b99deb…`) |
| `results/guided/hyperparameter_selection.csv` | yes (`79760759…`) |
| `results/guided/test/test_predictions.csv` | yes (`2dc5a653…`) |
| `results/guided/test/test_metrics.csv` | yes (`48e7fa1d…`) |
| `results/guided/test/test_calibration.csv` | yes (`506cf277…`) |
| `results/guided/test/hypothesis_tests_test.json` | yes (`8e2788be…`) |

The script never opens a file for writing, so this is expected rather than
fortunate.

---

## Limitations

1. These models are post-test reconstructions, not the objects that produced
   the original results (`models/export_manifest.json`,
   `results/verification/model_export_validation.md`).
2. The smoke test uses two synthetic rows. It demonstrates that the plumbing
   works; it is not evidence of predictive accuracy, and it must not be cited
   as a test-set result.
3. Scores are poorly calibrated (mean ≈ 0.068 versus an observed rate ≈ 0.393
   on the chronological test block). Ranking only; see §7 of
   `models/input_schema.md`.
4. `duration` is deliberately unavailable, so scores from this interface will
   not reproduce any figure that used it. Models A and B never used it.
5. The models are tied to scikit-learn 1.9.1; the script refuses to score if the
   saved pipelines were pickled by an incompatible version.
