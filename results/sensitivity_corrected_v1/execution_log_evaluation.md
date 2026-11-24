# Corrected guided random-split sensitivity evaluation — execution log

## 2026-11-14T11:26:00+00:00 — corrected evaluation

### Scope and authorization

PHASE 4 of the controlled correction, explicitly authorized by the human
project author after independent verification that the corrected Guided index
sets match both the authorized procedure and the existing Autonomous split.

This is a **post-test secondary sensitivity analysis**. It does not replace the
primary chronological test. No subgroup analysis was started.

### Inputs verified before model construction

- Corrected split: `results/sensitivity_corrected_v1/random_split_v1.json`
- Canonical fingerprint: `6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`
  (expected `6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`)
- Sizes: train 31,647 / validation
  6,781 / test
  6,783
- Positive counts: train
  3,702 / validation
  793 / test
  794
- Complete coverage: True
- Pairwise disjoint: True
- Random-test vs chronological-test overlap:
  1,022
- Split-only status verified: True
- Split-producing script hash verified:
  True
- Dataset hash verified: True

### Command

```
$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m my_analysis.guided_random_split_sensitivity_evaluate
```

### Fit and evaluation

- Fitted on the corrected random training block only.
- Validation block used for fitting, tuning or calibration: **no**.
- Evaluated on the corrected random test block.
- No split was regenerated; the index arrays were loaded directly.

| Model | ROC-AUC | AP | Brier | Lift@10% |
|---|---|---|---|---|
| A | 0.648781 | 0.197967 | 0.100110 | 2.1546 |
| B | 0.703351 | 0.323390 | 0.092144 | 3.2004 |
| C | 0.871345 | 0.465148 | 0.082550 | 4.5360 |

### Paired bootstrap

- Resamples: 2000, seed 42
- Skipped single-class resamples: 0
- B−A ΔROC-AUC: 0.054570
  [0.041805, 0.066600]
- B−A ΔAP: 0.125422
  [0.098713, 0.154288]
- C−B ΔROC-AUC: 0.167994
  [0.147389, 0.188793]
- C−B ΔAP: 0.141759
  [0.101859, 0.182669]

### Warnings

- Model A: ["FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None."]
- Model B: ["FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None."]
- Model C: ["FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None."]

### Integrity

- Tracked files hashed before: 108
- Tracked files hashed after: 108
- Added: []
- Removed: []
- Changed: []
- Unexpected modification: False

The baseline includes the human-verified split-only artifacts, the original
erroneous run, the error archive, all primary artifacts, the frozen plan and
the dataset (verified separately by SHA-256). Only this evaluation's own eight
output files and the `evaluation_attempts/` subtree are excluded.

### Attempts and retries

Prior evaluation outputs archived before this attempt:
{'archived': False, 'attempt': None, 'files': [], 'path': None}

### Status

- Evaluation succeeded: see manifest.
- Model C: NON-DEPLOYABLE duration-leakage diagnostic.
- Subgroup analysis: **not started**.
- Final report: **not written**.
