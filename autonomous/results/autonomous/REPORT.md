# Autonomous run — analysis report (validation only)

Run: autonomous execution of plan.md **v1.2 (frozen)** for the Bank Marketing
dataset. Agent: GLM-5.3-Flash Free (opencode) in OpenCode. Date: 2026-11-14.
Main script: `src/my_analysis/autonomous_train_val.py`; independent
verification: `src/my_analysis/verify_autonomous.py`.

**Scope.** This run was authorised to execute the training/validation part of
the pre-registered design. The frozen test block `[38,428, 45,211)` was never
loaded and its labels were never read; the pre-registered test-set decision
procedure (plan §3.1–§3.2) is **not executed here** and remains open. Every
statement below is therefore **validation evidence only**. No causal claim is
made anywhere: the design is associational prediction (plan §1, §2).

---

## 1. Integrity of inputs

| Check | Result | Source |
|---|---|---|
| Raw dataset SHA-256 | `0cd36af1…bf45f` — matches pre-registered value | `autonomous_train_val.py` console log |
| Split-file canonical hash | `d4284248…bfe5c` — matches pre-registered value; recomputed from the stored indices, file only **loaded**, never regenerated | same |
| Split sizes | train 31,647 / validation 6,781 / test 6,783 — as frozen | `results/splits/temporal_split_v1.json` |
| Rows filtered / merged / dropped | **0** (45,211 throughout; transformations are row-wise relabelings only) | console log `[rows]` entries |
| `pdays == 999` ⇔ `previous == 0` in train and validation | True | console log `[check]` |
| 5 anomalous rows (`poutcome=unknown`, `pdays≠999`) | all 5 lie in the test block (0 in train+validation); preserved as `unknown` by the same deterministic rule — never merged, imputed or dropped | console log |
| `duration` in Model C only; `pdays`, `campaign`, `contact`, `day`, `month` in no model | verified by assertion in code | console log `[check]` |

## 2. Models actually fitted (within the frozen plan)

- **A** — customer characteristics only: `age`, `balance` (standardised) +
  one-hot `job`, `marital`, `education`, `default`, `housing`, `loan`
  (27 coefficients).
- **B** — A + previous-campaign information: `previous` (standardised) +
  five-level `poutcome` (`never_contacted` derived from `pdays == 999`;
  32 coefficients).
- **C** — B + `duration` (**DIAGNOSTIC — LEAKAGE, never deployable**; 33
  coefficients).

- Preprocessing inside one sklearn `ColumnTransformer` per model, **fitted on
  the training block only**; `OneHotEncoder(handle_unknown="ignore")`, all
  observed train levels retained, no drop-first; `StandardScaler` for numerics.
- Categorical integer codes decoded via the verified HF `ClassLabel` name
  lists (data README §4.3); never used as numerics.
- L2 logistic regression, `lbfgs`, `max_iter=5000`, `random_state=42`.
  Selected strength: **C = 0.01 for all three models** (best mean ROC-AUC over
  5-fold expanding-window `TimeSeriesSplit` **inside the training block
  only**; identical folds for A/B/C; grid {0.01, 0.1, 1, 10, 100}).
  Source: `results/autonomous/models/cv_selection.csv`, `models/model_settings.json`.
- Seeds: 42 everywhere (bootstrap RNG `numpy.random.default_rng(42)`;
  `LogisticRegression(random_state=42)`); `TimeSeriesSplit` is deterministic.

## 3. Validation metrics (all three models, identical rows)

Source: `results/autonomous/metrics/validation_metrics.csv` (reproduced
independently to < 1e-16 by `verification/verification_report.md`).

| Model | ROC-AUC | PR-AUC | PR-AUC no-skill baseline (= prevalence) | Brier | Lift @ top 10% |
|---|---|---|---|---|---|
| A | 0.640340 | 0.208401 | 0.115027 | 0.103874 | 2.3209 |
| B | 0.659535 | 0.228795 | 0.115027 | 0.103818 | 2.4747 |
| **C (diagnostic — leakage)** | 0.881047 | 0.488623 | 0.115027 | 0.087114 | 4.7699 |

n = 6,781 validation rows, 780 positives (prevalence 0.115027). Baselines are
**set-specific** (validation prevalence), not the whole-file 11.70%.

## 4. Hypothesis evidence (validation rows only — NOT the pre-registered test decision)

Source: `results/autonomous/metrics/hypothesis_tests_validation.json`
(2,000 paired bootstrap resamples over validation rows, seed 42, percentile
95% CIs; identical resamples reused across models and metrics).

| Comparison | ΔROC-AUC (point) | 95% CI | ΔPR-AUC (point) | 95% CI |
|---|---|---|---|---|
| B − A (primary, H1 direction) | **+0.019195** | [+0.013433, +0.024787] | +0.020393 | [+0.014212, +0.026577] |
| C − B (leakage diagnostic, H2) | **+0.221512** | [+0.198286, +0.244816] | +0.259829 | [+0.216862, +0.302504] |

- **H1 direction (validation):** ΔROC-AUC > 0 with a CI that excludes zero on
  the positive side. If the plan's decision table were applied to these
  validation numbers, the outcome would be *"H0 rejected, materially
  relevant"* (ΔAUC 0.0192 ≥ 0.01 materiality threshold and CI lower bound
  > 0). **This classification is provisional by construction**: the
  pre-registered decision is defined on the chronological **test** set, which
  this run did not evaluate.
- **H2 (leakage diagnostic):** ΔAUC(C − B) = 0.2215, far above the
  pre-specified 0.05 diagnostic threshold. The `duration` leakage effect is
  **larger** than the plan assumed, not weaker. Model C is not an operational
  model and is never presented as one.

## 5. Calibration and probability quality

Source: `results/autonomous/metrics/validation_calibration_{a,b,c}.csv`.

- **All models under-predict on validation.** Models A and B concentrate
  predictions in [0, 0.1): bin 0 (n = 6,702/6,703) has mean predicted ≈ 0.052–
  0.054 against observed positive rate ≈ 0.111–0.112. Model C spreads more
  but still under-predicts in most bins (e.g. bin 0: predicted 0.028 vs
  observed 0.064; top bin predicted 0.961 vs observed 0.600, n = 15).
- Mechanism: **temporal prevalence drift**. The training block has 1,840/31,647
  = 5.81% positives; the validation block has 780/6,781 = 11.50%. A model
  calibrated to the earlier period necessarily under-predicts the later one.
  Ranking metrics (ROC-AUC, PR-AUC, lift) are unaffected by this shift;
  Brier scores and calibration are not directly comparable across periods.
  This drift is a dataset property, discovered at validation time; it changes
  no methodological choice (no re-tuning, no reweighting, no re-fitting was
  done in response).

## 6. Answer to the research question (validation evidence only)

On the frozen chronological validation block, adding previous-campaign
information (`previous` + five-level `poutcome`) to pre-call customer
characteristics improves out-of-sample prediction of term-deposit subscription:
ΔROC-AUC = +0.0192 (95% paired-bootstrap CI [+0.0134, +0.0248]) and
ΔPR-AUC = +0.0204 (CI [+0.0142, +0.0266]), with lift@10% rising from 2.32 to
2.47. The interval excludes zero on the positive side and the point estimate
exceeds the project's 0.01 materiality threshold. Whether this constitutes the
pre-registered rejection of H0 depends on the **test-set** evaluation, which is
deliberately outside this run. I am not sure the same magnitude will hold on
the test block, because the train→validation prevalence drift (5.8% → 11.5%)
shows the relationship is being evaluated under distribution shift.

## 7. Errors encountered and corrections

1. **Environment build failure.** `uv sync --frozen` failed
   (`README.md` missing, referenced by `pyproject.toml`). Corrected without
   touching frozen dependencies: `uv sync --frozen --no-install-project`
   (scripts are standalone modules; the project package is not needed).
2. **sklearn 1.9 deprecation warning** (`penalty` is deprecated in favour of
   `l1_ratio`). Harmless: `penalty="l2"` is still the applied penalty and
   `l2` is also the sklearn default. Recorded in
   `logs/warnings_and_errors.log`; left as-is rather than re-running with
   cosmetic-only changes, to avoid any post-result code churn.
3. **Verification discrepancy (real, investigated, fixed — plan §8.1).**
   First verification pass: 29/32 checks passed; PR-AUC recomputation and
   ΔPR-AUC bootstrap values differed from the reported ones by ~1e-5, above
   the 1e-6 tolerance. Root cause: **tie handling in average precision** —
   model A's validation predictions contain 108 tied probability pairs; a
   per-row step-sum differs from sklearn's definition, which evaluates the PR
   curve only at the last row of each tie group. The verification script's
   hand-implemented AP was corrected to the standard definition; afterwards
   every metric reproduces to < 1e-16. **The reported analysis metrics were
   never changed and no model was refitted**; the discrepancy was in the
   verification script, not the analysis.

## 8. Unresolved limitations (carried, per plan §9)

- The **test-set decision procedure is not yet executed**; all conclusions
  above are validation-scoped.
- Row order is chronological only up to the **year ambiguity** (no year
  variable; automated verdict was `INCONCLUSIVE`, human adjudication accepted
  the ordering).
- **`balance` snapshot timing is an assumption**, not documented; if
  post-call, it would contaminate A, B and C alike.
- **Prevalence drift** between train (5.81%) and validation (11.50%) degrades
  absolute calibration; coefficients and thresholds are period-specific.
- Observational data; prior-campaign variables are **selection-biased** by the
  bank's own campaign logic — descriptive associations only.
- No client identifier → possibly non-independent observations (untestable).
- Hyperparameters were selected on ROC-AUC alone (plan §5.6); the strong
  regularization choice (C = 0.01) compresses predicted probabilities, which
  contributes to the compressed calibration of A/B.
- The bootstrap CI is over validation rows; it quantifies sampling uncertainty
  within this period, not across periods.

## 9. Exact scripts and artifacts created

Scripts (this run only; nothing existing was modified):
- `src/my_analysis/autonomous_train_val.py` — pipeline, fitting, CV,
  evaluation, bootstrap, artifacts.
- `src/my_analysis/verify_autonomous.py` — independent recomputation
  (does not import the analysis script; refits nothing).

Artifacts under `results/autonomous/` (SHA-256 of every file in
`artifact_hashes_final.json` and `run_manifest.json`):

| Path | Content |
|---|---|
| `REPORT.md` | this report |
| `autonomous_run_log.md` | full execution log (commands, errors, decisions) |
| `metrics/validation_metrics.csv` | §3 table |
| `metrics/hypothesis_tests_validation.json` | §4 bootstrap statistics |
| `metrics/validation_calibration_{a,b,c}.csv` | §5 calibration tables |
| `predictions/validation_predictions_{a,b,c}.csv` | row_index, y, p per model (validation rows only) |
| `models/model_settings.json` | seeds, grids, chosen C, feature lists, versions |
| `models/cv_selection.csv` | full CV grid results |
| `models/coefficients_{a,b,c}.csv` | fitted coefficients |
| `logs/console_output.txt` | full console log of the analysis run |
| `logs/verification_console.txt` | full console log of verification |
| `logs/warnings_and_errors.log` | warnings raised during final fits |
| `verification/verification_report.md` | 32/32 checks PASS |
| `verification/verification_checks.json` | machine-readable check results |
| `run_manifest.json` | run identity, agent model, artifact hashes at run time |
| `artifact_hashes_final.json` | final SHA-256 of every artifact |

## 10. Test-set confirmation

**The test set was not evaluated.** No script in this run sliced rows
`[38,428, 45,211)`, computed any metric on them, or read their labels for any
purpose (the split file's test index list is used only for size/hash
verification of the split itself). The prediction files contain validation
rows only. Test-set evaluation will not be attempted without authorisation.
