# Autonomous run — execution log (final-evaluation preparation)

Append-only execution record for the autonomous final-evaluation stage. The
original training/validation log is `results/autonomous/autonomous_run_log.md`
and is not modified.

## 2026-11-14 — preflight preparation and reproduction

### Scope

Preparation and reproduction only. The test block was **not** opened. No test
metric, test prediction or test outcome was computed. The gated test mode was
implemented but **not executed**. All work was confined to the autonomous
workspace; the guided workspace was not accessed.

### Approvals in force

G1 train-block-only fitting; G2 random-split sensitivity deferred until after
primary test evaluation + independent verification; G3 subgroup descriptives
deferred to the same point; G4 no recalibration/threshold/tuning; G5 separate
test artifacts permitted, originals preserved; G6 no deviations, `plan.md`
v1.2 preserved; G7 logging begins now with the earlier gap disclosed; G8
crash → preserve evidence and stop; G9 disclosure approved; G10 both test
evaluations before cross-run comparison.

### Files read

- `AGENTS.md`, `plan.md` (v1.2, frozen), `data/README.md`
- `src/my_analysis/autonomous_train_val.py` (original, for exact settings)
- `results/autonomous/models/model_settings.json`
- `results/autonomous/predictions/validation_predictions_{a,b,c}.csv`
- `results/autonomous/run_manifest.json`
- `results/autonomous/logs/console_output.txt`
- `results/autonomous/autonomous_run_log.md`
- `results/splits/temporal_split_v1.json`
- Parquet metadata and the required feature columns (no test labels).

### Actions and commands

1. Inspected Parquet metadata: 46 row groups × 1,000 rows (last 211).
2. Captured a SHA-256/mtime baseline of all 60 pre-existing files in the
   autonomous workspace (excluding `data/raw/` and `.venv/`) before writing
   anything.
3. Created `src/my_analysis/autonomous_test_evaluate.py` with two modes:
   - `--mode preflight` (default, safe);
   - `--mode test` (gated by `--confirm-test-evaluation`, fail-closed, not run).
4. Ran (the project package build via `uv run` fails because `pyproject.toml`
   declares `readme = "README.md"` and no README exists; the locked `.venv` is
   used directly instead, as in the original autonomous run):

   ```
   $env:PYTHONPATH="src"
   .venv\Scripts\python.exe -m my_analysis.autonomous_test_evaluate --mode preflight
   ```

   Console captured to `results/autonomous/preflight/preflight_console.txt`.
   Exit code 0.

### Results

Source: `results/autonomous/preflight/preflight_checks.json`.

| Model | Frozen C | Max absolute prediction difference | Rows > 1e-12 | Result |
|---|---|---|---|---|
| A | 0.01 | 0.000e+00 | 0 | PASS |
| B | 0.01 | 0.000e+00 | 0 | PASS |
| C | 0.01 | 0.000e+00 | 0 | PASS |

All three models reproduced the saved validation predictions exactly. The
reproduction is bit-identical, not merely within tolerance. Frozen source of
truth: `results/autonomous/models/model_settings.json` (SHA-256
`89a39d0d84266f036edccb3ed41fb1f96579b665aecd3a032e1c13b14df2a86d`), which
records `C=0.01` for A/B/C, `max_iter=5000`, `solver=lbfgs`, `penalty=l2`.

### Data-isolation accounting

Source: `results/autonomous/preflight/preflight_checks.json` →
`row_group_policy`.

- Label column read only from Parquet row groups 0–31 (rows 0–31,999); highest
  label row index read = 31,999, before the test block start (38,428). The 353
  validation labels in rows 31,647–31,999 were loaded and discarded by the
  training slice; no test label was read.
- Validation targets were taken from the saved prediction artifacts, not from
  the Parquet label column.
- No test label was read. `preflight_checks.json` records
  `"test_labels_read": false`.
- Validation **features** require row group 38 (rows 38,000–38,999), which
  physically contains 572 test **feature** rows (rows 38,428–38,999). Parquet
  cannot read a partial row group, so those rows were loaded and immediately
  sliced away. `test_feature_rows_used` = 0. This limitation is stated in the
  report rather than claiming complete isolation.

### Static audit

Source: `preflight_checks.json` → `static_audit`.

- No `GridSearchCV`, `TimeSeriesSplit` or `C_GRID` used: PASS.
- Feature sets equal the frozen lists: PASS.
- Selected `C` = {A: 0.01, B: 0.01, C: 0.01}: PASS.
- `max_iter` = 5000, seed = 42: PASS.
- The script additionally cross-checks each constant against
  `model_settings.json` before fitting and aborts on any mismatch.

### Errors and warnings

- No errors. Exit code 0.
- The expected scikit-learn 1.9 `FutureWarning` for `penalty="l2"` appeared on
  each of the three final fits (3 captured entries). It is the same known,
  documented warning as in the original autonomous run; no code or setting was
  changed in response.

### Test-mode confirmation

`results/autonomous/test/` does **not** exist. `--mode test` was not invoked,
including its refusal path. The gate requires an explicit authorisation token
and re-runs the preflight check before opening the test block.

### Preservation

Source: `results/autonomous/preflight/preservation_check.json`.

- All 60 pre-existing files under the workspace (excluding `data/raw/` and
  `.venv/`) have unchanged SHA-256 and mtime: 0 changed, 0 missing.
- `plan.md` SHA-256 `103ea4de7128f680aba8d39279e59f3f00fc7c0c1d2e29929b63743d910532dd`
  (unchanged, matches the guided workspace copy seen earlier).
- `AGENTS.md`, `mistakes.md`, `data/README.md`, the raw dataset and the frozen
  split file are unchanged.
- The guided workspace was not accessed or modified.
