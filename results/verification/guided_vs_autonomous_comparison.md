# Guided vs autonomous run — comparative audit (training/validation stage)

Audit date: 2026-11-14.
Auditor: OpenCode agent, authorised comparative audit (no refit, no rerun, no test evaluation).
Plan audited: `plan.md` **v1.2 (frozen)**. The guided copy and the autonomous copy are
byte-identical: SHA-256 `103EA4DE…532DD` for both files.
Runs audited:

| Run | Workspace | Agent model | Main script | Independent verification |
|---|---|---|---|---|
| Guided | `C:\Users\alber\OneDrive\Documenti\my-analysis` | `opencode/big-pickle` | `src/my_analysis/guided_train_validate.py` | human-run one-liner, recorded in `results/verification/guided_validation_independent_check.md` |
| Autonomous | `C:\Users\alber\OneDrive\Documenti\my-analysis-autonomous` | `GLM-5.3-Flash Free (opencode)` | `src/my_analysis/autonomous_train_val.py` | `src/my_analysis/verify_autonomous.py` |

**Method of this audit.** Read-only. I read the frozen plan, both analysis scripts, both
verification scripts, and every result/log/manifest file listed in §9. I ran no analysis,
fitted no model, and inspected no test outcome. The only executed code was a read-only
pandas comparison of the two saved validation prediction files (§4.1); it read existing
artefacts and wrote nothing.

**Scope limit.** Both runs stopped at validation. Neither executed the pre-registered
test-set decision procedure (`plan.md` §3.1–§3.2). Every number below is validation
evidence only.

---

## 1. Headline findings

1. **Models A and B are numerically identical across the two runs.** Not merely equal at
   the reported precision: the saved validation predictions are bit-identical
   (`max |p_guided − p_autonomous| = 0.000e+00` over all 6,781 rows for both A and B). The
   fitted coefficients are also identical (e.g. `num__age = -0.05842904778480015` in both).
2. **Model C differs only because the two runs selected a different `C`.** Guided selected
   `C = 0.001`; autonomous selected `C = 0.01`. All 6,781 Model C predictions differ
   (`max |Δp| = 0.2505`), consistent with a different penalty strength.
3. **The two hyperparameter grids differ** (9 values vs 5 values) and the overlap is only
   `{0.01, 0.1, 1.0, 10.0}`. On every overlapping point the two runs' CV scores agree to
   the last displayed digit (§3.2), proving the fold construction, preprocessing and seed
   are equivalent.
4. **`plan.md` §5.6 does not specify grid values or a grid range.** It only requires
   "regularisation strength selected by time-series CV on the training block". Therefore
   neither grid is a plan violation. The grid difference is an **implementation choice**
   left open by the design — and it is the sole reason Model C is not directly comparable
   between runs.
5. **The autonomous run's test-isolation claims are stronger than its code.** The
   autonomous script loads the full Parquet file including `y` and computes
   `df["y"].unique()` over all 45,211 rows; its verification script reads the entire `y`
   column and indexes the test labels. No evidence indicates test labels were used for
   fitting, tuning or evaluation, but the statements `"test_labels_read": false`,
   "the block is NOT loaded", and "its labels were never read" are factually wrong.
   The guided run's wording is more precise and accurate.
6. **The autonomous run's `test_untouched` verification check is a tautology** —
   `np.array_equal(raw_y[test], raw_y[test])` — and can never fail. This is a
   verification-design defect, not a data leak.
7. **The autonomous encoding documentation is inconsistent with its code, but the model
   is not misspecified.** It creates decoded `*_label` columns but passes the original
   integer code columns for the six base categoricals. The guided run passes decoded
   strings. Because one-hot encoding of a bijection spans the same column space, the
   coefficients are numerically identical; this is a documentation error, not a
   predictive-modelling error.
8. **Both runs preserved the frozen design** in feature sets, split, recoding, train-only
   preprocessing, diagnostic-only Model C, and no causal claims. The material gaps are
   shared process gaps (§7), not design violations.

---

## 2. Feature sets and preprocessing

### 2.1 Feature sets (identical in both runs)

| Model | Numeric | Categorical |
|---|---|---|
| A | `age`, `balance` | `job`, `marital`, `education`, `default`, `housing`, `loan` |
| B | A + `previous` | A + `poutcome_level` (five levels) |
| C (diagnostic) | B + `duration` | B |

Both scripts exclude `campaign`, `contact`, `day`, `month` everywhere and never pass
`pdays` to a scaler, encoder or estimator. Both derive `never_contacted` from
`pdays == 999` and preserve the five `poutcome = unknown, pdays != 999` rows as `unknown`.
Both verified the same recode counts: `never_contacted = 36,954`, `unknown = 5`.
Source: `src/my_analysis/guided_train_validate.py` lines 100–141 and
`src/my_analysis/autonomous_train_val.py` lines 66–101; console logs in both runs.

The five anomalous rows all fall in the **test block** (`train+validation = 0`), so the
preserve-as-`unknown` rule has no effect at the validation stage. Both runs report this
correctly.

### 2.2 Preprocessing (functionally identical)

| Element | Guided | Autonomous |
|---|---|---|
| Numeric | `StandardScaler` | `StandardScaler` |
| Categorical | `OneHotEncoder(handle_unknown="ignore")`, all levels | `OneHotEncoder(handle_unknown="ignore")`, all levels |
| `remainder` | `"drop"` | `"drop"` |
| Fitted on | training block only, and inside each CV fold | training block only, and inside each CV fold |
| Imputation / resampling / target encoding | none | none |

Both apply their deterministic recode to the full dataframe before slicing; neither fits
any statistic on validation rows. This is not leakage because the recode is a row-wise
function of `pdays`/`poutcome` with no target or fitted statistic.

**Differences that are implementation choices, not design deviations:**

- `max_iter`: guided 1000, autonomous 5000. No convergence warnings in either run.
- Categorical representation: guided passes decoded strings; autonomous passes integer
  codes (see §6).
- Calibration binning: guided `pd.qcut` (equal-count deciles); autonomous
  `calibration_curve(strategy="uniform")` (equal-width bins). `plan.md` §5.7 says only
  "calibration curve" and does not specify a binning rule, so the two calibration tables
  are not directly comparable.

---

## 3. Hyperparameter search grids and selected `C`

### 3.1 The grids differ

| Run | Grid | Points | Selected `C` |
|---|---|---|---|
| Guided | `{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0}` | 9 | A = 0.01, B = 0.01, **C = 0.001** |
| Autonomous | `{0.01, 0.1, 1.0, 10.0, 100.0}` | 5 | A = 0.01, B = 0.01, **C = 0.01** |

Both use expanding-window `TimeSeriesSplit(n_splits=5)` on the training block, mean
ROC-AUC as the selection score, identical folds for A/B/C, and `refit=True`.
Sources: `results/guided/hyperparameter_selection.csv` (27 rows) and
`results/autonomous/models/cv_selection.csv` (15 rows).

### 3.2 Overlapping grid points reproduce exactly

For every shared value of `C`, the mean CV ROC-AUC is identical to the last displayed
digit:

| Model | `C` | Guided mean CV | Autonomous mean CV |
|---|---|---|---|
| A | 0.01 | 0.5647482833554879 | 0.5647482833554879 |
| A | 0.1 | 0.5608203388326298 | 0.5608203388326298 |
| B | 0.01 | 0.567769439442778 | 0.567769439442778 |
| B | 1.0 | 0.5564089206017694 | 0.5564089206017694 |
| C | 0.01 | 0.9062575571188642 | 0.9062575571188642 |
| C | 0.1 | 0.9042877208460803 | 0.9042877208460803 |

This is strong evidence that both runs used the same folds, the same preprocessing
semantics and the same seed, and that the A/B result is implementation-independent.

### 3.3 Does the plan specify grid values?

No. `plan.md` §5.6 reads: "Primary: L2-regularised logistic regression. Regularisation
strength selected by time-series CV on the training block." It specifies neither the
candidate values nor a range, nor a tie-breaking rule. The autonomous run's
first-grid-value tie-break and the guided run's `GridSearchCV` default tie-break are both
consistent with the text. **Neither grid is a plan violation.** Calling either run
non-compliant on this point would be incorrect.

### 3.4 Why Model C is not directly comparable

- Both runs' Model C CV curves **decrease monotonically as `C` increases** over their own
  grids. In the guided grid, `C = 0.001` (0.9072556) is the best and is the **lower
  boundary**; in the autonomous grid, `C = 0.01` (0.9062576) is the best and is also the
  **lower boundary**. Neither run found an interior optimum for Model C.
- The autonomous grid simply does not contain any value below `0.01`, so it **cannot**
  select the guided value `0.001`. The guided grid does contain `0.01`, and its CV table
  shows `0.01` beating `0.1` — so the two runs agree about every point they share.
- The guided `0.001` vs `0.01` CV gap is `0.000998` with fold SD ≈ `0.0759` (about
  0.013 SD, or 0.03 standard errors of the 5-fold mean). It is not a meaningful
  separation. The same is true of the autonomous `0.01` vs `0.1` gap (`0.001970`).
- Consequently the Model C validation difference (§4.2) is a **direct artefact of
  regularisation strength**, not evidence that one implementation is better. A fair
  cross-run comparison of Model C would require a common grid, or comparing both runs at
  the shared point `C = 0.01` — which would require refitting the guided Model C and is
  therefore out of scope for this audit.

**This is the main unresolved comparability issue.** It is a design omission in
`plan.md` (no grid specified), not an agent error. See `mistakes.md` entry H-1.

---

## 4. Validation performance of Models A, B and C

Validation set: 6,781 rows, 780 positives, prevalence `0.11502728211178292`; top-decile
`k = 678`. Source: `results/guided/validation_metrics.csv` and
`results/autonomous/metrics/validation_metrics.csv`.

### 4.1 Models A and B: identical reported metrics

| Model | Metric | Guided | Autonomous | Identical |
|---|---|---|---|---|
| A | ROC-AUC | 0.6403399647067369 | 0.6403399647067369 | yes |
| A | PR-AUC | 0.20840129342966013 | 0.20840129342966013 | yes |
| A | Brier | 0.10387433447017923 | 0.10387433447017923 | yes |
| A | Lift@10% | 2.3208550790409195 | 2.3208550790409195 | yes |
| B | ROC-AUC | 0.6595350561231248 | 0.6595350561231248 | yes |
| B | PR-AUC | 0.228794684830157 | 0.228794684830157 | yes |
| B | Brier | 0.10381795657395584 | 0.10381795657395584 | yes |
| B | Lift@10% | 2.4747239240602075 | 2.4747239240602075 | yes |

Read-only comparison of the saved prediction files confirms this at the prediction level:
for A and B, `max |p_guided − p_autonomous| = 0.000e+00` and `n_diff > 0 = 0` over all
6,781 rows. The top-decile precision and `n_top_decile` also match
(A 0.2669616519174041, B 0.28466076696165193, `k = 678`).

### 4.2 Model C: differs because `C` differs

| Metric | Guided (`C = 0.001`) | Autonomous (`C = 0.01`) | Δ (autonomous − guided) |
|---|---|---|---|
| ROC-AUC | 0.8683653152 | 0.8810467486 | +0.0126814334 |
| PR-AUC | 0.4727607524 | 0.4886233732 | +0.0158626208 |
| Brier | 0.0900593759 | 0.0871141505 | −0.0029452254 |
| Lift@10% | 4.6417101581 | 4.7699341956 | +0.1282240375 |

Model C predictions differ on all 6,781 rows (`max |Δp| = 0.2505`). Because the only
material difference between the runs for Model C is the selected `C` (§3.4), these
differences should be read as a regularisation effect, **not** as a difference in
implementation quality. Model C remains a leakage diagnostic in both runs and is not
deployable.

### 4.3 Uncertainty quantification

- **Autonomous:** computed 2,000 paired bootstrap resamples over validation rows
  (seed 42, percentile 95% CI), reusing identical resample indices across models.
  B − A ΔROC-AUC = +0.0191950914, CI [+0.0134325017, +0.0247870681];
  C − B ΔROC-AUC = +0.2215116925, CI [+0.1982860298, +0.2448157938].
  Source: `results/autonomous/metrics/hypothesis_tests_validation.json`.
- **Guided:** computed **no** bootstrap CIs at validation. Its only uncertainty summary is
  the fold SD of the CV scores. `plan.md` §5.7 defines the bootstrap over **test** rows,
  so the guided omission is defensible at the validation stage — but it does leave the
  guided validation report with less uncertainty evidence than the autonomous one.
- Both runs correctly label the validation bootstrap as provisional: the pre-registered
  decision applies to the test block, which neither run evaluated.

### 4.4 Calibration

Both runs report the same substantive finding: train prevalence 5.81% (1,840/31,647) vs
validation 11.50% (780/6,781), so all models under-predict the validation base rate.
The numeric calibration tables differ in construction (equal-count deciles vs
equal-width bins, §2.2) and should not be compared bin by bin. Both runs explicitly
decline to recalibrate, reweight or re-tune in response, which is consistent with the
frozen plan.

---

## 5. Reproducibility and verification procedures

### 5.1 What each run did

| Element | Guided | Autonomous |
|---|---|---|
| Seed | 42 (`random_state`, bootstrap N/A) | 42 (`random_state`, `default_rng(42)`) |
| Input hashes checked in-script | raw dataset + split canonical hash | raw dataset + split canonical hash |
| Manifest with artifact hashes | yes (`results/guided/run_manifest.json`) | yes (`run_manifest.json`, `artifact_hashes_final.json`) |
| Repeat runs | 4 runs; runs 3–4 byte-identical metrics | 1 analysis run |
| Independent metric recomputation | human-run one-liner, 4 metrics × 3 models, recorded in `results/verification/guided_validation_independent_check.md` | separately written script, 32/32 checks PASS |
| Bootstrap CI verification | N/A (none computed) | independently re-implemented and verified |
| Calibration verification | no | yes (max diff ≤ 1.1e-16) |
| Clean-environment rerun from pinned `uv.lock` | not reported | not reported |

### 5.2 Assessment

- The autonomous run has the stronger verification *artefact*: a separately written
  script that does not import the pipeline, recomputes ROC-AUC with a rank-based formula,
  re-implements AP, recomputes Brier/lift/baselines/bootstrap/calibration, and checks the
  split and dataset hashes. It reproduces all reported metrics to `< 1e-16`.
- The guided verification is a human-executed one-liner that recomputes only ROC-AUC, AP,
  Brier and lift from the saved predictions, to 6 decimal places. It does not recompute
  calibration and there is no bootstrap to recompute. It is a genuine independent check
  but narrower.
- **Neither run performed the `plan.md` §8.1 clean-environment rerun** from the pinned
  `uv.lock` that must "reproduce identical metrics for the same seeds and split file".
  The guided run's four repeats were in the same environment; the autonomous run was a
  single execution plus a metric recomputation. This is a **shared execution gap**.
- **Neither verification checks that test labels were not read.** Both only check that no
  test metric was computed. This gap allowed the autonomous run's tautological
  `test_untouched` check to pass (§6.2). This is a **shared verification-design gap**.

### 5.3 Determinism evidence

The strongest determinism evidence is cross-run, not within-run: the overlapping CV
points agree exactly (§3.2) and the A/B predictions agree exactly (§4.1), despite
different agents, different code and different `max_iter`. This is much stronger than a
same-environment repeat and should be treated as such.

---

## 6. Test isolation audit

### 6.1 What the code actually does

**Autonomous analysis script** (`src/my_analysis/autonomous_train_val.py`):

- Line 303: `df = pd.read_parquet(RAW)` — reads **all 45,211 rows and all 17 columns**,
  including `y`.
- Line 309: `if not set(df["y"].unique()) <= {0, 1}` — computes the unique values of the
  **entire** `y` column, including test labels.
- Line 321: `df = decode_categoricals(df)` — applies the deterministic recode to all rows,
  including test rows.
- Lines 325–327: `poutcome_level` counts over the **full file**, including test rows.
- Lines 339–344: `anomalies_all` counts over the **full file**, including test rows
  (`poutcome`/`pdays`, not `y`).
- Lines 329–332: only then slices `train_df`/`val_df` and builds `y_train`/`y_val`.

**Autonomous verification script** (`src/my_analysis/verify_autonomous.py`):

- Line 150: `raw_y = pd.read_parquet(RAW, columns=["y"])["y"].to_numpy(dtype=int)` — reads
  the **entire** `y` column.
- Line 151: `np.array_equal(raw_y[val_idx], y)` — uses validation labels only.
- Lines 243–245:
  `np.array_equal(raw_y[split["indices"]["test"]], raw_y[split["indices"]["test"]])` —
  indexes the test labels and compares them to themselves. Always true.

**Guided analysis script** (`src/my_analysis/guided_train_validate.py`):

- Line 241: `y_all = pd.read_parquet(RAW, columns=["y"])["y"].to_numpy(...)` — reads the
  entire `y` column.
- Lines 242–244: immediately slices `y_train = y_all[idx_train]` and `y_val = y_all[idx_val]`,
  then `del y_all`. Test labels are read into memory but never indexed.
- Lines 246–266: applies the deterministic recode to all rows and counts
  `never_contacted`/`unknown` over the full file, including test rows.

### 6.2 Reading vs using — the correct distinction

| Question | Guided | Autonomous |
|---|---|---|
| Were test labels read into memory? | **Yes** (`y_all`, entire column) | **Yes** (`df["y"]`, `.unique()`; `raw_y` in verification) |
| Were test labels used for fitting? | No | No |
| Were test labels used for tuning / `C` selection? | No | No |
| Were test labels used for evaluation / metrics? | No | No |
| Were test rows predicted? | No | No |
| Were test features processed by the deterministic recode? | Yes | Yes |
| Were test features used in full-file diagnostic counts? | Yes (`never_contacted`/`unknown`) | Yes (`poutcome_level`, `anomalies_all`) |

The honest summary is: **both runs read the test block; neither used it for any modelling
decision or metric.** The recode is deterministic and target-free, so applying it to test
rows before slicing does not leak information into fitting.

### 6.3 Contradictions between code and statements

| Source | Statement | Accurate? |
|---|---|---|
| Autonomous `run_manifest.json` | `"test_labels_read": false` | **False** — the analysis script reads all `y` values and the verification script reads the full `y` column |
| Autonomous `logs/console_output.txt` | "the block is NOT loaded and its labels are NEVER read" | **False** — the block is loaded (all 17 columns), and labels are read |
| Autonomous `REPORT.md` §Scope | "was never loaded and its labels were never read" | **False** — same reason |
| Autonomous `REPORT.md` §10 | "No script in this run sliced rows `[38,428, 45,211)` … or read their labels for any purpose" | **False at the "read" level**; true at the "used for a metric/model" level |
| Autonomous `REPORT.md` §10 | "the split file's test index list is used only for size/hash verification" | **Incomplete** — `verify_autonomous.py` also indexes `raw_y[test]` |
| Guided `run_manifest.json` | `"test_labels_inspected": false` | **Imprecise** — the full `y` column is read; "inspected" is defensible but "read" would be false |
| Guided `guided_run_log.md` | "its labels were never read into `y_train` or `y_val`" | **Accurate** — precise wording |
| Guided `validation_report.md` | test block "was not sliced, scored or inspected" | **Accurate in substance** — it was read in full but not sliced, scored or examined |

**Severity.** This is a **reporting/integrity-claim defect**, not a leakage defect. There
is no evidence of test-label use in fitting, tuning or evaluation in either run. The
autonomous run's claims are simply stronger than its code supports; the guided run's
wording is careful and mostly correct.

### 6.4 The tautological check

`verify_autonomous.py` lines 243–245 define `test_untouched` as
`np.array_equal(raw_y[test], raw_y[test])`, which is `True` for any array. Combined with a
hard-coded size check, the verification reports
"test block size 6,783; NOT evaluated … PASS" unconditionally. It cannot detect a
violation. This should be replaced by a check that fails if test rows appear in any
prediction file, any metric input, or any fitted pipeline. It is a **verification-design
defect**, not evidence of a leak. See `mistakes.md` entry A-2.

---

## 7. Errors encountered and how each agent corrected them

### 7.1 Guided run

| # | Error | Correction | Outcome |
|---|---|---|---|
| G-1 | `scikit-learn` not installed; plan requires sklearn pipelines | `uv add scikit-learn` → 1.9.1; this modified `pyproject.toml` and `uv.lock` (disclosed) | resolved |
| G-2 | First run crashed with `KeyError: 'lift'` during manifest construction (`lift_lift` vs `lift`) | rewrote lift keys explicitly, updated report table, re-ran from scratch; no partial artefacts kept | resolved |
| G-3 | sklearn 1.9 `FutureWarning` for `penalty="l2"` on every fit | deliberately not silenced; `penalty="l2"` kept to match the plan; warning captured in `run_warnings.txt` and explained | accepted, documented |
| G-4 | PowerShell mojibake when displaying em-dashes | cosmetic console codepage artefact; file verified valid UTF-8 | no action needed |
| G-5 | split file records `plan_version: "1.1"` while the design executed is v1.2 | recorded both versions in the manifest with an explanatory note; boundaries unchanged | documented |

### 7.2 Autonomous run

| # | Error | Correction | Outcome |
|---|---|---|---|
| A-1 | `uv sync --frozen` failed: `README.md` missing, referenced by `pyproject.toml` | `uv sync --frozen --no-install-project`; no change to `pyproject.toml`/`uv.lock` | resolved |
| A-2 | First verification pass 29/32: PR-AUC and ΔPR-AUC differed by ~9.6e-6 (> 1e-6 tolerance) due to tie handling (108 tied pairs in Model A) | rewrote the verification script's AP with sklearn's tie-group rule; re-ran → 32/32 PASS; analysis metrics unchanged, no refit | resolved, analysis unaffected |
| A-3 | sklearn 1.9 `FutureWarning` for `penalty="l2"` | documented in `logs/warnings_and_errors.log`; left as-is to avoid post-result churn | accepted, documented |
| A-4 | Overstated test-isolation claims (§6.3) | not corrected by the autonomous run | **open** |
| A-5 | Tautological `test_untouched` check (§6.4) | not corrected by the autonomous run | **open** |
| A-6 | Documentation says categoricals were decoded; code passes integer codes (§8) | not corrected by the autonomous run | **open (documentation)** |

Both runs handled the sklearn `penalty` deprecation the same way and both disclosed it.
Both self-corrected their genuine code errors successfully (G-2, A-2). Neither
self-correction altered any reported analysis metric.

---

## 8. Categorical encoding audit

### 8.1 What was passed to `OneHotEncoder`

- **Guided:** `CODEBOOK` maps integer codes to strings on the full dataframe
  (lines 246–248); `MODELS` then selects the string columns; the encoder sees
  `admin.`, `blue-collar`, … Output names are `cat__job_admin.`, `cat__job_blue-collar`, …
- **Autonomous:** `decode_categoricals` creates `job_label`, `marital_label`, etc.
  (line 146), and `poutcome_level` is a decoded string column. **But** `MODELS["A"/"B"/"C"]`
  (lines 85–101) lists the **original** columns `job`, `marital`, `education`, `default`,
  `housing`, `loan`, so the encoder receives **integer codes** for those six. Output names
  are `cat__job_0`, `cat__job_1`, … Only `poutcome_level` is decoded.

The `*_label` columns are therefore created and then unused. This is why the coefficient
tables differ in feature naming but agree numerically.

### 8.2 Is this a modelling error?

No. `OneHotEncoder` treats integer codes as **categories**, not as numeric magnitudes; it
does not standardise them or assume order. A one-hot encoding of an arbitrary bijection of
the levels spans the same column space, so the fitted model is identical. The evidence is
direct:

| Feature | Guided coefficient | Autonomous coefficient |
|---|---|---|
| `num__age` | -0.05842904778480015 | -0.05842904778480015 |
| job level 0 | `cat__job_admin.` 0.01568307706382793 | `cat__job_0` 0.01568307706382793 |
| job level 1 | `cat__job_blue-collar` -0.050447518071248205 | `cat__job_1` -0.050447518071248205 |
| job level 2 | `cat__job_entrepreneur` -1.744347590005682e-05 | `cat__job_2` -1.744347590005682e-05 |

Sources: `results/guided/model_coefficients.csv`,
`results/autonomous/models/coefficients_a.csv`.

### 8.3 What is actually wrong

The **documentation** is wrong, not the model. Autonomous `REPORT.md` §2 states:
"Categorical integer codes decoded via the verified HF `ClassLabel` name lists
(data README §4.3); never used as numerics." The "never used as numerics" half is
defensible (the encoder treats them categorically), but the "decoded" half is false for
the six base categoricals. The run log §1 row 2 makes the same claim. `model_settings.json`
records `"categorical": "OneHotEncoder(handle_unknown='ignore'), levels learned from train
only, no drop-first"` without clarifying that codes were passed.

**Rubric consequence.** `plan.md` §8.2 criterion 7 ("Categoricals treated as categorical |
dtype handling in pipeline") is a **PASS** on behaviour — the encoder treated them as
categorical. The evidence trail is misleading, which is a documentation/integrity issue,
not a model-misspecification issue. See `mistakes.md` entry A-3.

---

## 9. Compliance with the frozen research design

| Plan requirement | Guided | Autonomous | Verdict |
|---|---|---|---|
| §5.2 feature sets A/B/C | exact | exact | PASS both |
| §5.2 `duration` only in C | yes | yes | PASS both |
| §5.2/§5.3 `pdays` never a predictor | yes | yes | PASS both |
| §5.3 five-level `poutcome`, `never_contacted` from 999 | yes (36,954 / 5) | yes (36,954 / 5) | PASS both |
| §5.3 anomalies preserved as `unknown` | yes | yes | PASS both |
| §5.4 preprocessing train-only, no resampling | yes | yes | PASS both |
| §5.4 `unknown` retained as a category | yes | yes | PASS both |
| §5.5 frozen split reused, not regenerated | yes | yes | PASS both |
| §5.5 CV within training block only | yes | yes | PASS both |
| §5.6 L2 logistic, CV-selected strength | yes | yes | PASS both |
| §5.6 exact grid values | not specified | not specified | **N/A — no violation** |
| §5.7 ROC-AUC, PR-AUC, Brier, lift, set-specific baseline | yes | yes | PASS both |
| §5.7 bootstrap CI | not computed at validation | computed at validation | autonomous more complete; guided omission defensible (test-scoped) |
| §5.7 calibration | yes (qcut) | yes (uniform) | PASS both; not comparable bin-to-bin |
| §8.1 separately written verification | human one-liner | independent script | PASS both, different depth |
| §8.1 clean-environment rerun | not reported | not reported | **shared gap** |
| §8.2 #2 raw data untouched | verified by hash | verified by hash | PASS both |
| §8.2 #3 split integrity | PASS | PASS | PASS both |
| §8.2 #4 preprocessing train-only | PASS | PASS | PASS both |
| §8.2 #5 `duration` excluded from A/B | PASS | PASS | PASS both |
| §8.2 #6 `pdays` sentinel | PASS | PASS | PASS both |
| §8.2 #7 categoricals treated as categorical | PASS | PASS (behaviour); docs inconsistent | PASS both |
| §8.2 #8 `unknown` preserved | PASS | PASS | PASS both |
| §8.2 #9 metrics complete | no validation CI | yes | autonomous more complete |
| §8.2 #10 no causal language | PASS | PASS | PASS both |
| §8.2 #11 numbers traceable | PASS | PASS | PASS both |
| §8.2 #12 model/prompt/output logged in `results/logs/` | model name only; no `results/logs/` | model name only; no `results/logs/` | **shared gap** |
| §8.2 #13 deviations disclosed pre-test | no deviations | no deviations | PASS both |
| §8.2 #14 independent recomputation matches | PASS (6 dp) | PASS (< 1e-16) | PASS both |
| §8.2 #15 honest negative/insufficient reporting | PASS | PASS | PASS both |
| §3.1–§3.2 test decision | not executed | not executed | correctly out of scope |
| No causal claims | PASS | PASS | PASS both |

**Overall.** Both runs comply with the frozen design on every substantive modelling
requirement. The differences are implementation choices the plan left open (grid, binning,
`max_iter`, encoding representation, validation CI) and shared process gaps (clean-env
rerun, prompt logging). Neither run executed the test procedure, which is correct for
this stage.

---

## 10. Unresolved issues requiring a decision before final test evaluation

These are ordered by importance. None of them blocks the audit itself; all require a
human decision before the test block is opened.

1. **Freeze a single `C` grid for the test stage (H-1).** The two runs selected different
   Model C strengths because the plan specifies no grid. Before the final evaluation,
   decide either (a) a common grid for all models, or (b) to report the test-stage Model C
   result at the shared point `C = 0.01` for both runs (which requires a guided refit and
   therefore explicit approval). Without this, Model C test results will not be
   comparable across runs. **This is the single most important open decision.**
2. **Decide whether validation-stage bootstrap CIs are required (H-2).** The autonomous
   run computed them; the guided run did not. The plan's CI clause is test-scoped, so this
   is a reporting-asymmetry decision, not a compliance failure.
3. **Decide the test-stage verification standard (H-3).** Neither run's verification can
   detect test-label reading. Before the test run, specify a negative control that fails
   if test rows enter any prediction, metric input or fitted pipeline, and require the
   clean-environment rerun from `uv.lock` that §8.1 already mandates.
4. **Correct the autonomous test-isolation wording before it is carried into the final
   report (A-4).** `"test_labels_read": false`, "NOT loaded", and "labels were never read"
   should be replaced with an accurate statement: the file is loaded in full, test labels
   are read into memory and aggregated, but are not used for fitting, tuning or
   evaluation.
5. **Fix or remove the tautological `test_untouched` check (A-5).** It currently cannot
   fail and should not be cited as evidence.
6. **Correct the autonomous categorical-decoding documentation (A-6).** State that integer
   codes were passed to `OneHotEncoder` for the six base categoricals and that `*_label`
   columns were unused; or change the code to pass the label columns. The fitted model is
   unaffected either way.
7. **Decide the prompt-logging gap (H-4).** Neither run archived prompts under
   `results/logs/` as `plan.md` §8.2 #12 and `AGENTS.md` rule 5 require. Decide whether
   the final stage needs this, given RQ2.
8. **Decide the calibration-binning rule for the test stage (H-5).** `qcut` vs `uniform`
   produces non-comparable calibration tables; the plan does not specify one.

**What must NOT change without a pre-test deviations entry:** the frozen split, the
feature sets, the hypotheses and decision rules, and `plan.md` itself. This audit did not
modify `plan.md` and did not decide any new grid.

---

## 11. Files read and files changed

### Read (no modification)

Guided workspace:
`plan.md`; `src/my_analysis/guided_train_validate.py`;
`results/guided/{validation_metrics.csv, hyperparameter_selection.csv,
model_coefficients.csv, validation_predictions.csv, validation_report.md,
guided_run_log.md, run_manifest.json, run_warnings.txt, calibration_deciles.csv}`;
`results/verification/guided_validation_independent_check.md`.

Autonomous workspace:
`src/my_analysis/autonomous_train_val.py`; `src/my_analysis/verify_autonomous.py`;
`results/autonomous/{REPORT.md, autonomous_run_log.md, run_manifest.json,
artifact_hashes_final.json}`;
`results/autonomous/metrics/{validation_metrics.csv, hypothesis_tests_validation.json,
validation_calibration_a.csv, validation_calibration_b.csv, validation_calibration_c.csv}`;
`results/autonomous/models/{cv_selection.csv, model_settings.json, coefficients_a.csv,
coefficients_b.csv, coefficients_c.csv}`;
`results/autonomous/predictions/validation_predictions_{a,b,c}.csv`;
`results/autonomous/verification/{verification_report.md, verification_checks.json}`;
`results/autonomous/logs/{console_output.txt, verification_console.txt,
warnings_and_errors.log}`.

### Created by this audit

- `results/verification/guided_vs_autonomous_comparison.md` (this file).
- `mistakes.md` — appended evidence-based entries (no existing row removed).

### Explicitly not changed

`plan.md` (both copies), `AGENTS.md`, `data/raw/`, `data/README.md`, the split files, all
existing guided and autonomous result files, both scripts, and the autonomous workspace
as a whole. No model was refitted, no analysis was rerun, no test outcome was inspected.

---

## 12. One-paragraph conclusion

The guided and autonomous executions are far more similar than their headline Model C
difference suggests. Models A and B are bit-identical, and every overlapping CV point
agrees exactly, which demonstrates that the two independently written pipelines implement
the same design. Model C diverges only because `plan.md` §5.6 leaves the regularisation
grid unspecified: the guided run selected `C = 0.001` from a 9-point grid, the autonomous
run selected `C = 0.01` from a 5-point grid, and both selections sit on their grid's lower
boundary with no meaningful CV separation from the neighbouring value. Neither choice is
a plan violation, but the difference makes Model C validation results non-comparable as
executed. The genuine defects are reporting and verification defects in the autonomous
run — overstated test-isolation claims, a tautological `test_untouched` check, and
categorical-decoding documentation that does not match the code — plus shared process gaps
(no clean-environment rerun, no prompt log). None of these is a leakage or
model-misspecification error. The most important decision before the test block is opened
is to freeze a common `C` grid (or an explicit shared comparison point) so that Model C
test results are comparable across runs.
