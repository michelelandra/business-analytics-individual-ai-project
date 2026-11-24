# Guided preflight report — reproduction of validation predictions

Generated (UTC): 2026-11-14T09:29:39+00:00
Script: `src/my_analysis/guided_test_evaluate.py` (SHA-256 `2ac3a3eac188254343bc0f63fd284a29de348a7dd078a19be8d01868e666ba36`)
Plan: `plan.md` v1.2 (frozen)
Dataset SHA-256: `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f`
Split SHA-256: `d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c`

**Scope.** Preparation and reproduction only. The test block was not
opened. No test metric, test prediction or test outcome was computed.

## Reproduction of the saved validation predictions

Tolerance: max absolute difference <= 1e-12.

| Model | Frozen C | n train | n validation | max abs diff | rows > tol | Result |
|---|---|---|---|---|---|---|
| A | 0.01 | 31,647 | 6,781 | 0.000e+00 | 0 | PASS |
| B | 0.01 | 31,647 | 6,781 | 0.000e+00 | 0 | PASS |
| C | 0.001 | 31,647 | 6,781 | 0.000e+00 | 0 | PASS |

**Overall: PASS.**

The models were re-fitted from scratch on the original training block
with the frozen selected `C` values and the original preprocessing,
solver and seed. They were then scored on the frozen validation block
and compared with `results/guided/validation_predictions.csv`.

## What the preflight read (and did not read)

- Parquet row groups: 46 of 1,000 rows.
- Feature row groups read: [0, 38] (39,000 rows).
- Label row groups read: [0, 31] (32,000 rows; 31,647 used for training).
- Highest row index read from the label column: 31,999 (test block starts at 38,428).

**Technical limitation, stated accurately.** Validation features require
row group 38 (rows 38,000–38,999), which physically contains 572 test
*feature* rows. Parquet cannot read a partial row group, so those rows
are loaded and immediately sliced away; they are never inspected or used.
No test *label* is read: the `y` column is read only from row groups
0–31 (rows 0–31,999), entirely before the test block.

## Static audit (no new search, features or settings)

- No hyperparameter search: True
- Forbidden names found: []
- Feature sets match the frozen lists: True
- Selected C matches the frozen values: True
- Selected C: {'A': 0.01, 'B': 0.01, 'C': 0.001}
- `max_iter` = 1000, seed = 42

## Environment

- python: 3.13.5
- scikit_learn: 1.9.1
- numpy: 2.5.4
- pandas: 3.1.1
- platform: Windows-11-10.0.26200-SP0

## Gated test mode

`--mode test` exists but was **not executed**. It requires an explicit
authorisation token and re-runs this preflight check (fail-closed)
before opening the test block once.

## Warnings captured during the refits

- model A: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
- model B: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
- model C: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
