# Guided random-split sensitivity analysis — execution log

Append-only execution record. This file belongs to the guided workspace only.

## 2026-11-14T09:56:04+00:00 — sensitivity execution (final attempt)

### Scope

Secondary post-test sensitivity analysis using a stratified random split. The
primary chronological evaluation and every existing artifact were preserved.
The autonomous workspace was not accessed.

### Approvals in force

The authorising prompt is recorded verbatim in
`results/sensitivity/prompt_log.md`. It authorises the guided sensitivity
analysis only and requires the analysis to stop before the autonomous run.

### Files read

- `data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet` (verified by SHA-256)
- `results/splits/temporal_split_v1.json` (verified by SHA-256)
- `results/guided/test/test_metrics.csv` (descriptive comparison only)
- `results/guided/test/hypothesis_tests_test.json` (descriptive comparison only)

### Command

```
$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m my_analysis.guided_random_split_sensitivity
```

### Split

- Method: `train_test_split`, stratified on `y`, seed 42.
- Sizes: train 31,647 / validation 6,781 / test 6,783.
- Class counts: train 27,945/3,702;
  validation 5,988/793;
  test 5,989/794 (negative/positive).
- Test prevalence: 0.117057.
- Split SHA-256: `e43308ca7a0485fdf114be43372be3c0cc07017897ed766895de47e02f6eca3b`.
- Complete coverage: True;
  pairwise disjoint: True;
  reproducible from seed: True.

### Model calls (frozen settings)

| Model | Estimator | C | penalty | solver | max_iter | random_state | n train | n test |
|---|---|---|---|---|---|---|---|---|
| A | LogisticRegression | 0.01 | l2 | lbfgs | 1000 | 42 | 31,647 | 6,783 |
| B | LogisticRegression | 0.01 | l2 | lbfgs | 1000 | 42 | 31,647 | 6,783 |
| C | LogisticRegression | 0.001 | l2 | lbfgs | 1000 | 42 | 31,647 | 6,783 |

Models were fitted on the random training block only. The random validation
block was recorded but not used for fitting, tuning, calibration, model
selection or refitting.

### Results

| Model | ROC-AUC | Average precision | AP no-skill | Brier | Lift@10% |
|---|---|---|---|---|---|
| A | 0.679998 | 0.223176 | 0.117057 | 0.098600 | 2.3814 |
| B | 0.728580 | 0.366773 | 0.117057 | 0.089903 | 3.4650 |
| C | 0.867877 | 0.449270 | 0.117057 | 0.083370 | 4.2966 |

B − A ΔROC-AUC: point 0.048582, 95% CI
[0.036106,
0.061362].
B − A ΔAP: point 0.143597, 95% CI
[0.114035,
0.172903].
Bootstrap: 2,000 paired resamples, seed 42, skipped
single-class resamples 0.

### Warnings

- model A: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
- model B: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
- model C: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.

### Preservation

Baseline files hashed before execution: 69. The run
compared every tracked file after execution; `run_manifest_sensitivity.json`
records the result. Any unexpected modification would have aborted the run.

### Execution attempts (disclosed)

- Attempt 1 (2026-11-14T09:53:35Z) aborted with an `AttributeError` while
  assembling the split document, before any model was fitted and before any
  metric was computed. No analysis artifact was produced; only the console log
  existed and was overwritten.
- Attempt 2 (2026-11-14T09:53:55Z) completed the full analysis.
- Attempt 3 (2026-11-14T09:56:02Z) was a deterministic rerun after fixing a
  console-output hash-ordering defect in the script. The split hash and every
  metric were identical to Attempt 2. The delivered artifacts are from
  Attempt 3. No protocol, feature, hyperparameter, split seed or model
  specification was changed, and no result was selected between attempts.
