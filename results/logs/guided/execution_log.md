# Guided run — execution log (final-evaluation preparation)

Append-only execution record for the guided final-evaluation stage. The
original training/validation run log is `results/guided/guided_run_log.md`
and is not modified.

## 2026-11-14 — preflight preparation and reproduction

### Scope

Preparation and reproduction only. The test block was **not** opened. No test
metric, test prediction or test outcome was computed. The gated test mode was
implemented but **not executed**.

### Approvals in force

G1 train-block-only fitting; G2 random-split sensitivity deferred until after
primary test evaluation + independent verification; G3 subgroup descriptives
deferred to the same point; G4 no recalibration/threshold/tuning; G5 separate
test artifacts permitted, originals preserved; G6 no deviations, `plan.md`
v1.2 preserved; G7 logging begins now with the earlier gap disclosed; G8
crash → preserve evidence and stop; G9 disclosure approved; G10 both test
evaluations before cross-run comparison.

### Files read

- `plan.md` (v1.2, frozen)
- `src/my_analysis/guided_train_validate.py` (original, for exact settings)
- `results/guided/validation_predictions.csv`
- `results/guided/run_manifest.json`
- `results/splits/temporal_split_v1.json`
- Parquet metadata (row-group layout only) and feature columns.

### Actions and commands

1. Inspected Parquet metadata: 46 row groups × 1,000 rows (last 211).
2. Created `src/my_analysis/guided_test_evaluate.py` with two modes:
   - `--mode preflight` (default, safe);
   - `--mode test` (gated by `--confirm-test-evaluation`, fail-closed, not run).
3. Ran:
   `uv run python -m my_analysis.guided_test_evaluate --mode preflight`
   Console captured to `results/guided/preflight/preflight_console.txt`.

### Results

| Model | Frozen C | Max absolute prediction difference | Rows > 1e-12 | Result |
|---|---|---|---|---|
| A | 0.01 | 0.000e+00 | 0 | PASS |
| B | 0.01 | 0.000e+00 | 0 | PASS |
| C | 0.001 | 0.000e+00 | 0 | PASS |

All three models reproduced the saved validation predictions exactly. The
reproduction is bit-identical, not merely within tolerance.

### Data-isolation accounting

- Label column read only from Parquet row groups 0–31 (rows 0–31,999); highest
  label row index read = 31,999, before the test block start (38,428).
- No test label was read. `preflight_checks.json` records
  `"test_labels_read": false`.
- Validation features require row group 38 (rows 38,000–38,999), which
  physically contains 572 test **feature** rows. Parquet cannot read a partial
  row group, so those rows are loaded and immediately sliced away. They are
  never inspected or used. This limitation is stated in the report rather than
  claiming complete isolation.

### Static audit

- No `GridSearchCV`, `TimeSeriesSplit` or `C_GRID` used: PASS.
- Feature sets equal the frozen lists: PASS.
- Selected `C` = {A: 0.01, B: 0.01, C: 0.001}: PASS.
- `max_iter` = 1000, seed = 42, `penalty="l2"`, solver `lbfgs`: PASS.

### Errors and warnings

- No errors.
- The expected scikit-learn 1.9 `FutureWarning` for `penalty="l2"` appeared on
  each fit (3 lines). It is the same known, documented warning as in the
  original guided run; no code or setting was changed in response.

### Test-mode confirmation

`results/guided/test/` does **not** exist. `--mode test` was not invoked. The
gate requires an explicit authorisation token and re-runs the preflight check
before opening the test block.

### Preservation

No existing guided or autonomous script, result, log, manifest, `plan.md`,
`AGENTS.md`, `mistakes.md`, `data/README.md` or split file was modified. The
autonomous workspace was not touched.
