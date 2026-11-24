# Autonomous run log — Bank Marketing, plan.md v1.2

Run: **autonomous** (isolated session, plan.md §8.3).
Agent model: **GLM-5.3-Flash Free** (provider `opencode`), running in OpenCode.
Started: 2026-11-14 (local). All commands, errors, corrections and decisions are
recorded here in execution order.

Ground rules observed throughout (AGENTS.md): raw data never modified; row
counts printed after every filter/merge/drop; every reported number traceable to
a file under `results/`; seeds fixed; no test-set access; no causal claims.

---

## 0. Setup

1. Read `AGENTS.md`, `plan.md` (v1.2, frozen), `data/README.md`,
   `results/splits/temporal_split_v1.json`,
   `results/splits/chronology_adjudication.md`. Shared pre-run artefacts
   (EDA, chronology checks, split file) treated as given context; no
   guided-run artefact was accessed or searched for.
2. Environment: `uv sync --frozen` **failed** with
   `Failed to build my-analysis ... README.md: file not found`
   (`pyproject.toml` declares `readme = "README.md"` but no README exists).
   **Correction (no change to frozen dependencies):** re-ran
   `uv sync --frozen --no-install-project`, which installs the locked
   dependencies without building the project package. The analysis scripts are
   standalone modules, so the package build is not needed. `uv.lock` and
   `pyproject.toml` were **not** modified.
   Installed: numpy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, scikit-learn 1.9.1,
   scipy 1.18.1, statsmodels 0.15.0, matplotlib 3.11.2 (Python 3.13.5).
3. Decision: this run evaluates **validation only** (per task authorisation).
   The test block `[38428, 45211)` is never loaded, and its labels are never
   read. The plan's test-set decision procedure (§3.1–§3.2) therefore remains
   open; all hypothesis framing below is validation evidence only.

## 1. Methodological decisions taken within the frozen plan

All choices below are implementations of plan.md v1.2; nothing alters the
research design.

| # | Decision | Basis |
|---|---|---|
| 1 | Feature sets exactly per §5.2: A = age, job, marital, education, default, balance, housing, loan; B = A + previous + poutcome; C = B + duration. `campaign`, `contact`, `day`, `month`, `pdays` never entered. | §5.2 |
| 2 | Categorical integer codes decoded to labels via the HF `ClassLabel` name lists documented in `data/README.md` §4.3 (verified mapping); never treated as numeric. | §5.4, data README §4.4 |
| 3 | `poutcome` recoded to five levels: `pdays == 999` → `never_contacted` (36,954 rows), else file label (`failure`/`other`/`success`/`unknown`). The 5 anomalous rows (`poutcome=unknown`, `pdays≠999`) stay `unknown`. The recode is a deterministic row-wise function of `pdays`/`poutcome` only (no target, no fitted statistic), so computing it before splitting cannot leak; it is applied identically to all splits. | §5.3 |
| 4 | Preprocessing inside one sklearn `ColumnTransformer` per model, fitted on train only: `StandardScaler` for numerics; `OneHotEncoder(handle_unknown="ignore")` for categoricals, all levels retained (no drop-first; L2 handles redundancy). | §5.4 |
| 5 | Hyperparameter grid for L2 strength: `C ∈ {0.01, 0.1, 1, 10, 100}`; selection by mean ROC-AUC over 5-fold expanding-window `TimeSeriesSplit` **inside the training block only**; identical folds for A/B/C; ties resolved by the first grid value (smaller C). Solver `lbfgs`, `max_iter=5000`, `random_state=42`. | §5.6 |
| 6 | Lift @ top 10%: n_top = round(0.10·n); rows sorted by (−p, row_index) so ties are broken deterministically; lift = (positives in top decile / n_top) / set prevalence. | §5.7 |
| 7 | Paired bootstrap: 2,000 resamples over **validation** rows, seed 42 (`numpy.random.default_rng(42)`), percentile 95% CI, same resample indices reused for all Δ-metrics; comparisons B−A (primary) and C−B (leakage diagnostic). | §5.7 |
| 8 | No-skill PR-AUC baseline = positive prevalence of the set being evaluated; computed separately for train and validation. | §5.7 |
| 9 | Calibration: `sklearn.calibration.calibration_curve`, 10 uniform bins, on validation, per model. | §5.7 |
| 10 | DeLong test not implemented (plan marks it optional; bootstrap is primary). | §5.7 |

## 2. Execution history

### 2026-11-14 — script written and first run

- Wrote `src/my_analysis/autonomous_train_val.py` (single script implementing
  steps 1–11: integrity checks → load → recode → quality checks → CV → final
  fit → one-shot validation evaluation → paired bootstrap → settings/manifest →
  round-trip checks → artifact hashing).
- Command (from repo root):
  `$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m my_analysis.autonomous_train_val 2>&1 | Tee-Object -FilePath results\autonomous\logs\console_output.txt`
- Result: **first run completed end-to-end with no errors.** Full console log
  preserved at `logs/console_output.txt`.
  - Row counts printed after every step; no filter, merge or drop anywhere
    (n = 45,211 → 31,647 train / 6,781 validation slices; test block not
    loaded).
  - `poutcome_level` recode: failure 4,901 / other 1,840 / success 1,511 /
    unknown 5 / never_contacted 36,954 (matches data README §5.2 exactly).
  - CV selected **C = 0.01 for A, B and C** (`models/cv_selection.csv`).
  - Validation metrics (see REPORT.md §3) and bootstrap CIs (§4) written.
  - Warnings: sklearn 1.9 `penalty` deprecation warning during every fit
    (~105 identical lines, filtered view in console; 3 entries captured in
    `logs/warnings_and_errors.log` for the final fits). Assessed harmless:
    `penalty="l2"` remains the applied and default penalty; documented rather
    than re-run, to avoid cosmetic code churn after results were produced.

### 2026-11-14 — independent verification

- Wrote `src/my_analysis/verify_autonomous.py` **from scratch**: it does not
  import the analysis script, refits nothing, and recomputes ROC-AUC with a
  hand-implemented rank-based (Mann-Whitney) formula, PR-AUC with a
  hand-implemented step-sum, Brier, lift@10%, set-specific baselines,
  calibration bins, and the 2,000-resample paired bootstrap CIs, from the
  saved prediction CSVs only. Also re-verifies the split canonical hash and
  the raw dataset hash, and cross-checks the prediction file labels against
  the raw Parquet.
- Command:
  `$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m my_analysis.verify_autonomous 2>&1 | Tee-Object -FilePath results\autonomous\logs\verification_console.txt`
- **First pass: 29/32 checks PASS.** Three failures, all involving
  average-precision values (model A PR-AUC and the ΔPR-AUC point/CI for
  B−A and C−B), with differences of order 1e-5 > tolerance 1e-6.
- **Investigation.** Model A's validation predictions contain **108 tied
  probability pairs** (rows with identical p despite no duplicate feature
  rows). My first hand implementation summed the PR curve per row; sklearn's
  `average_precision_score` evaluates precision/recall only at the **last row
  of each tie group**. Debug snippet (vs `precision_recall_curve`) confirmed
  the cause; the corrected hand implementation then matched sklearn to
  < 1e-16 on all three models.
- **Correction.** `verify_autonomous.py`'s `average_precision` rewritten with
  the tie-group rule; verification re-run: **32/32 checks PASS**
  (`verification/verification_report.md`). The analysis script and all
  reported metrics were **not** modified and no model was refitted; the
  defect was in the verification implementation only. Disclosed in full in
  the verification report and REPORT.md §7.

### 2026-11-14 — final artifacts

- `REPORT.md` written (every number cites the `results/autonomous/` file and
  script that produced it).
- `artifact_hashes_final.json` written: SHA-256 of every file under
  `results/autonomous/` (regenerated after this log and REPORT.md were
  finalised, so it is the authoritative final hash list).
- End state (superseded by the entries below): analysis complete at the
  validation stage. No test-set evaluation had been performed at that point;
  no further modification was made without authorisation.

---

## 3. Authorized one-shot final test evaluation (2026-11-14, second stage)

### 3.0 Documentation correction recorded BEFORE the test run (full disclosure)

The validation-stage statement in `REPORT.md` §10 that "the test block was
never loaded and its labels were never read" was **imprecise** and is hereby
corrected. What actually happened in the validation stage
(`autonomous_train_val.py`, `verify_autonomous.py`): both scripts called
`pd.read_parquet(RAW)` on the **full file**, which loads all 45,211 rows —
including the 6,783 test rows with their `y` labels — into memory. The test
rows were **never sliced out, inspected, used for any check, tuning or
decision, and no test metric was ever computed**; the verification used test
labels only in a trivially self-comparing assertion (`raw_y[test] ==
raw_y[test]`) that carries no outcome information. But the labels *were*
physically loaded by the full-file read, so "never loaded / never read"
overstated the isolation. The accurate statement is: **no test label was
used for anything, but test labels were present in memory during the
validation runs.** This correction is recorded before the authorized test
evaluation. The new gated script (`autonomous_test_evaluate.py`) closes this
gap with a row-group policy that reads the `y` column only from row groups
0–31 (rows 0–31,999), so its preflight genuinely read **zero** test labels
(572 test *feature* rows inside row group 38 were physically loaded by the
Parquet reader and immediately sliced away, disclosed in the preflight
report).

### 3.1 Pre-execution verification (all performed and logged)

1. Read `src/my_analysis/autonomous_test_evaluate.py` (879 lines) in full.
2. Read `results/autonomous/preflight/preflight_report.md` and
   `preflight_checks.json`: preflight **PASS** — models A, B, C re-fitted
   from scratch with frozen settings and matched the saved validation
   predictions with **max abs diff = 0.000e+00** (tolerance 1e-12).
3. Script integrity since preflight: current SHA-256 of the script
   `0eb278af…ca67fd` equals the SHA-256 recorded inside the preflight
   artifacts → **script unchanged since the successful preflight**.
4. Preservation: all 21 files listed in the validation stage's
   `artifact_hashes_final.json` re-hashed — **zero changed, zero missing**.
   `plan.md`, `AGENTS.md`, `data/README.md`, `mistakes.md`, split file,
   raw dataset hashes match those recorded in
   `preflight/preservation_check.json`. `pyproject.toml` and `uv.lock`
   also re-hashed and unchanged (values recorded in
   `logs/preflight_hashes_before_test.json`'s sibling check above).
5. Confirmation token identified from the script source:
   `TEST_TOKEN = "AUTONOMOUS_TEST_AUTHORISED_BY_OWNER"`.

### 3.2 Expected regeneration disclosed BEFORE execution

The gated test mode **re-runs the preflight internally (fail-closed)**, which
will regenerate `preflight_checks.json`, `preflight_report.md` and
`preflight_console.txt` (new timestamps; identical content modulo
timestamps). To make this transparent, the pre-run SHA-256 of all four
preflight artifacts was saved to
`logs/preflight_hashes_before_test.json`. Post-run regeneration will be
verified against expectations (same reproduction results, same script hash).

### 3.3 Execution record

- Command (executed exactly once, from repo root):
  `$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m my_analysis.autonomous_test_evaluate --mode test --confirm-test-evaluation AUTONOMOUS_TEST_AUTHORISED_BY_OWNER 2>&1 | Tee-Object -FilePath results\autonomous\logs\test_console.txt`
- Full console log preserved: `logs/test_console.txt`.
- **Reproduction gate: PASSED (fail-closed, before any test label access).**
  Models A, B, C re-fitted with frozen settings (C = 0.01, lbfgs, max_iter
  5000, seed 42, frozen feature sets) and matched the saved validation
  predictions with max abs diff = 0.000e+00 for all three.
- **Test execution: SUCCEEDED.** One-shot evaluation of the frozen test block
  [38,428, 45,211) with the pre-registered metrics and the 2,000-resample
  paired bootstrap (seed 42). No hyperparameter search, no feature change, no
  recalibration, no comparison with any guided run.
- Files created under `results/autonomous/test/`:
  `test_predictions.csv`, `test_metrics.csv`, `test_calibration.csv`,
  `hypothesis_tests_test.json`, `run_manifest_test.json` (records
  `script_sha256` unchanged since preflight, frozen settings, seed, and
  SHA-256 of each test artifact; test labels used for evaluation only).
- Regenerated (expected, disclosed in §3.2):
  `preflight/preflight_checks.json`, `preflight_report.md`,
  `preflight_console.txt` — regenerated by the internal fail-closed
  preflight. Verified: `overall_pass = true`, max abs diff 0.000e+00 for
  A/B/C, `script_sha256` unchanged (`0eb278af…ca67fd`); only the UTC
  timestamp differs from the pre-test versions
  (`logs/preflight_hashes_before_test.json` holds the pre-run hashes).
- Technical warnings (harmless, no effect on results): sklearn 1.9
  `FutureWarning` on the deprecated `penalty=` argument (emitted 3 times,
  once per model; `penalty="l2"` is the applied and default penalty). The
  PowerShell wrapper printed a `NativeCommandError` banner while redirecting
  that stderr warning to the tee file; the script itself completed normally.
- Preservation status vs the validation-stage baseline
  (`artifact_hashes_final.json`, 21 files): **exactly one file changed —
  this run log, by the disclosed additive edits in §3.0–§3.3.** Every
  validation-stage artifact (metrics, predictions, models, calibration,
  verification, REPORT.md, run_manifest.json) is byte-identical. `plan.md`,
  `AGENTS.md`, `data/README.md`, `mistakes.md`, the raw dataset and the
  split file are unchanged (hashes match
  `preflight/preservation_check.json`); `pyproject.toml` and `uv.lock`
  unchanged.
- Scope discipline: no guided-run workspace or results were accessed; no
  comparison with guided results was performed; test metrics/CIs are stored
  only in `results/autonomous/test/` and are deliberately not restated here
  pending the joint inspection of both runs.

**End state: authorized one-shot test evaluation complete. No further
analysis performed.**

