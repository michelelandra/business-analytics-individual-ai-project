# Guided-run decision log

**Run mode:** `guided` (the first, human-supervised execution of the pre-registered design).
This log covers the training + validation stage only. A future `results/autonomous/`
directory does not exist and nothing here feeds it.

---

## 1. Identity

| Item | Value |
|---|---|
| Agent | OpenCode coding agent |
| Model | `opencode/big-pickle` ("Big Pickle") |
| Session role | Guided execution, stage 1 of 2 (train + validation) |
| Human supervisor | Project author (`Michele Landra`, per `pyproject.toml`) |
| Date of run | 2026-11-13 |
| Plan followed | `plan.md` v1.2 (frozen) |

## 2. Instructions received (verbatim summary)

The human issued a single instruction message with 12 numbered requirements.
The operative content:

1. Use the exact frozen train and validation indices; do not use or evaluate the test set.
2. Implement Models A (customer characteristics), B (A + previous campaign),
   C (B + `duration`, non-deployable leakage diagnostic).
3. Use the exact feature sets and encoding rules from `plan.md` v1.2: categoricals
   as categories, no `pdays`/999 sentinel as a numeric predictor, derive
   `never_contacted` exactly, preserve the five anomalies as `unknown`.
4. Fit all preprocessing on training data only.
5. Follow the pre-registered hyperparameter procedure using training data only.
6. Compute and save ROC-AUC, average precision, Brier, lift@10%, with the no-skill
   AP reference set to validation prevalence.
7. Save predictions and results in `results/guided/` with script references.
8. Reproducibility: seeds, versions, settings, artifacts.
9. Begin this decision log.
10. No autonomous outputs; strict separation from future `results/autonomous/`.
11. Do not modify `plan.md`, `AGENTS.md`, `data/README.md`, `mistakes.md`, raw data, split files.
12. No refit on train+validation, no test labels, no test metrics, no final conclusions, no causal claims.

Authorization was limited to "the scripts and result artifacts necessary for this
guided training/validation stage".

## 3. Files read before implementing

- `AGENTS.md` (6 operating rules)
- `plan.md` v1.2 (§§4.2, 4.3, 5.1–5.7, 8.2, 9, 10)
- `data/README.md` (codebook §4.3/§4.4, sentinel §5.1, anomalies §5.3, leakage §9)
- `results/splits/temporal_split_v1.json` (indices, hash, verification block)
- `results/splits/chronology_adjudication.md`
- `results/explore_bank_marketing/` (schema, missingness, sentinel stats, notes)

## 4. Implementation decisions

Every choice below is either (a) dictated by `plan.md` v1.2, or (b) an
implementation detail the plan leaves open. Category (b) items are marked
**[choice]** and are the only things a reviewer could reasonably want changed.

### 4.1 Feature matrices

| Model | Numeric | Categorical | n input cols |
|---|---|---|---|
| A | `age`, `balance` | `job`, `marital`, `education`, `default`, `housing`, `loan` | 8 |
| B | A + `previous` | A + `poutcome_derived` | 10 |
| C | B + `duration` | B | 11 |

- `pdays` is **never** a predictor. It is read solely to evaluate `pdays == 999`.
- No `was_contacted_prev` or `days_since_prev` column is constructed or used.
- `campaign`, `contact`, `day`, `month`, `y` are never in any matrix.
- **[choice]** The raw integer categorical codes are mapped to their documented
  string labels via the codebook in `data/README.md` §4.3 *before* one-hot
  encoding. This is a 1:1 relabel that uses no target information. It makes the
  encoder's `get_feature_names_out()` readable (`cat__job_technician` rather than
  `cat__job_9`) and cannot change the fitted model, because a one-hot encoding of
  an arbitrary bijection of the levels spans the same column space.
  Verified: no code failed to map (the script aborts on an unmapped code).

### 4.2 Derived `poutcome`

```
poutcome_derived = "never_contacted"  if pdays == 999
                 = poutcome label      otherwise
```

- Result: `never_contacted` = 36,954; `unknown` = 5 (the anomalies).
- The script **asserts** both counts and aborts otherwise. Both matched.
- The five anomalous rows therefore stay in the `unknown` level, as required.

### 4.3 Preprocessing

`ColumnTransformer` inside a `Pipeline`:
- `StandardScaler()` on numeric columns.
- `OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=float64)` on
  categorical columns.
- `remainder="drop"`.

Because the whole transformer is a pipeline step, it is refitted inside every
cross-validation fold on that fold's training portion only, and once on the full
training block for the final fit. It never sees validation rows during fitting.

No imputer (0 NaN cells), no resampling, no SMOTE, no target encoding, no
feature selection.

**[choice]** `OneHotEncoder` keeps **all** levels (no `drop="first"` baseline),
so for a binary variable such as `housing` both `housing_no` and `housing_yes`
enter the design matrix. They are collinear with the intercept, and the L2
penalty splits the effect symmetrically (e.g. `+0.14` / `−0.14` in
`model_coefficients.csv`). This does not change predictions, rankings or any
metric — only the individual coefficient values, which should therefore be read
as relative contributions on the standardised scale, not as effects relative to
a reference level. Dropping a baseline was rejected because it would make the
coefficient table depend on an arbitrary reference choice.

### 4.4 Hyperparameters

- `LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, class_weight=None, random_state=42)`.
- `GridSearchCV` over `C ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}`.
- `cv = TimeSeriesSplit(n_splits=5)` (expanding window), the **same splitter
  object** reused for A, B and C so all three see identical folds.
- Scoring: `roc_auc` (mean over folds); `refit=True` on the full training block.
- Selection used training data only. Validation was scored exactly once per
  model, after selection.

**[choice]** `TimeSeriesSplit` was chosen over `StratifiedKFold` because the
training block is chronologically ordered and the plan's evaluation is temporal;
random K-fold would leak future training rows into earlier folds.

### 4.5 Metrics

Computed on the 6,781 validation rows only:
- ROC-AUC, average precision, Brier score.
- No-skill AP reference = **validation prevalence** (0.115027), per the plan.
- Lift@10%: `k = floor(0.10 × 6781) = 678`; ranking by a **stable** descending
  sort of predicted probability (ties broken by original row order); lift =
  `precision@k / prevalence`.
- Calibration deciles via `pd.qcut` on predicted probability, `duplicates="drop"`.

**[choice]** The lift tie-breaking rule (stable sort) is not specified in the
plan; it is recorded in `run_manifest.json → lift_definition`.

## 5. Warnings, errors and corrections

| # | Event | Resolution |
|---|---|---|
| 1 | `scikit-learn` was not installed; `plan.md` §5.4/§5.6 require `sklearn` pipelines | Ran `uv add scikit-learn` → `scikit-learn==1.9.1`, plus `joblib`, `threadpoolctl`, `cloudpickle`. This modified `pyproject.toml` and `uv.lock`, the only changes outside `results/guided/` and the new script. Disclosed because it is a dependency change, not a data or design change. |
| 2 | **First run crashed**: `KeyError: 'lift'` at manifest construction. Cause: the lift dict was spread into the metrics row with a `lift_` prefix, producing `lift_lift`, while the manifest read `row["lift"]`. Most artifacts had already been written; `run_manifest.json` and `validation_report.md` were missing. | Fixed by writing the lift keys explicitly (`lift_k`, `lift_precision_at_k`, …, `lift`) and updating the report's lift table to match. Re-ran from scratch. No partial artifact was kept. |
| 3 | `FutureWarning: 'penalty' was deprecated in version 1.8` on every fit | **Not** silenced. `plan.md` §5.6 specifies an L2 penalty, so the code states `penalty="l2"` explicitly rather than relying on a default. The warning is captured verbatim in `run_warnings.txt` and explained in the report. The fitted model is unchanged. |
| 4 | No convergence warnings | Checked `run_warnings.txt`: only the 3 `penalty` warnings, one per model. |
| 5 | PowerShell displayed mojibake (`�?"`) for em-dashes when reading the UTF-8 report | Cosmetic console codepage artefact, not a file defect. Verified the file is valid UTF-8 and renders correctly. |
| 6 | `results/splits/temporal_split_v1.json` records `plan_version: "1.1"` (it was created under v1.1) | Not an error. The manifest now records both `plan_version: "1.2"` (design executed) and `split_plan_version: "1.1"` with an explanatory note. The split boundaries are unchanged and were not regenerated. |

## 6. Integrity checks performed at runtime

The script aborts (exit 1) on any of these:

- Dataset SHA-256 ≠ `0cd36af1…bf45f`.
- Recomputing the canonical hash of `indices` ≠ the recorded `split_hash_sha256`,
  or that hash ≠ `d4284248…fe5c`.
- Block sizes ≠ 31,647 / 6,781 / 6,783.
- Any overlap between train / validation / test index sets.
- Block order not strictly `train.max() < validation.min() < test.min()`.
- Any categorical code that fails to map to a label.
- `never_contacted` ≠ 36,954 or `unknown` ≠ 5.

All checks passed. Row accounting printed at runtime (AGENTS.md rule 2):

```
[rows] total=45,211; train=31,647; validation=6,781; test=6,783 (NOT used)
[rows] no filtering, merging or dropping applied (train+validation+test = 45,211)
```

**Filters applied: none. Rows dropped: 0.** The test block was sliced only to
assert its size and disjointness — its labels were never read into `y_train` or
`y_val`, never scored, and no test prediction exists.

## 7. Commands executed

```powershell
# from C:\Users\alber\OneDrive\Documenti\my-analysis
uv add scikit-learn
uv run python -m my_analysis.guided_train_validate
```

Two full runs were performed: the first failed at artifact-manifest time
(error #2 above), the second succeeded. The third and fourth runs were executed
after cosmetic report changes and produced **byte-identical** metrics, confirming
determinism.

## 8. Human approval required

The following were implemented under the standing authorization in the
instruction message and did **not** need a separate approval, but are recorded
because they touch something outside `results/guided/`:

- `uv add scikit-learn` → `pyproject.toml`, `uv.lock`.
- Creation of `src/my_analysis/guided_train_validate.py`.

**Explicitly not done, and requiring approval before any future step:**

- Refitting on train + validation.
- Reading test labels or computing any test metric.
- Any recalibration, prior correction, or decision-threshold tuning.
- Any change to `plan.md`, `AGENTS.md`, `data/README.md`, `mistakes.md`,
  `data/raw/`, or the split files.
- Any `results/autonomous/` output.

## 9. Artifacts produced

All under `results/guided/`, all hashed in `run_manifest.json`:

| File | Contents |
|---|---|
| `run_manifest.json` | config, hashes, versions, fold sizes, selected `C`, artifact hashes, `test_evaluated: false` |
| `validation_metrics.csv` | one row per model: ROC-AUC, AP, AP no-skill, Brier, lift block, selected `C`, CV AUC |
| `validation_predictions.csv` | 6,781 rows: `row_index`, `y`, `p_A`, `p_B`, `p_C` |
| `hyperparameter_selection.csv` | full 27-row `C` grid (9 per model) with mean/SD CV ROC-AUC and rank |
| `calibration_deciles.csv` | 10 deciles × 3 models: n, mean predicted, observed rate |
| `model_coefficients.csv` | fitted coefficients and intercept on the standardised / one-hot scale |
| `run_warnings.txt` | 3 captured `FutureWarning` lines |
| `validation_report.md` | human-readable summary, including base-rate shift and calibration caveats |
| `guided_run_log.md` | this file |

## 10. Environment

| Component | Version |
|---|---|
| Python | 3.13.5 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.1 |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| Platform | Windows-11-10.0.26200-SP0 |
| Seed | 42 |
