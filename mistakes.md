# Mistakes

| Date | Agent Action | Mistake | How I Detected It | Correction | Lesson Learned |
| ---- | ------------ | ------- | ----------------- | ---------- | -------------- |
| 2026-11-14 | **[Human design omission] H-1** — `plan.md` §5.6 pre-registers CV selection but no grid values or range | The guided run used 9 values `{0.001…10}` and the autonomous run 5 values `{0.01…100}`. Model C therefore selected `C=0.001` (guided) vs `C=0.01` (autonomous), so the Model C validation results are not directly comparable. | Comparative audit of `results/guided/hyperparameter_selection.csv` and `results/autonomous/models/cv_selection.csv`. | Not corrected: the plan is frozen and this audit must not amend it. Decision required before the test run (common grid or a shared comparison point). | Pre-register exact grids when cross-run comparability matters. |
| 2026-11-14 | **[Human design omission] H-2** — `plan.md` §5.7 defines bootstrap CIs over test rows only and says nothing about validation-stage CIs | Guided computed no validation CIs; autonomous computed them. This is a reporting asymmetry (rubric #9), not a compliance failure. | Compared `results/guided/validation_metrics.csv` (no CI columns) with `results/autonomous/metrics/hypothesis_tests_validation.json`. | Not corrected. Decision required whether validation CIs are mandatory. | Specify metric requirements stage by stage, not only for the final test. |
| 2026-11-14 | **[Human design omission] H-3** — `plan.md` §8.1 requires metric recomputation but no negative control that test labels were not read | Both verification procedures can pass while test labels are read into memory; the autonomous `test_untouched` check was a tautology. | Read `verify_autonomous.py` lines 243–245 and `results/verification/guided_validation_independent_check.md`. | Not corrected. Specify a check that fails if test rows enter any prediction, metric input or fitted pipeline. | Verification must include checks that can fail. |
| 2026-11-14 | **[Human/process omission] H-4** — `plan.md` §8.2 #12 and `AGENTS.md` rule 5 require model/prompt/output logging under `results/logs/` | Neither run created `results/logs/` or archived prompts; only the agent model name was recorded (in the manifest/settings). | Directory audit of both workspaces: `results/logs/` does not exist in either. | Not corrected. Decision required whether RQ2 needs prompt logs. | Log prompts at call time rather than reconstructing them later. |
| 2026-11-14 | **[Human design omission] H-5** — `plan.md` §5.7 says "calibration curve" without a binning rule | Guided used `pd.qcut` (equal-count deciles); autonomous used `calibration_curve(strategy="uniform")` (equal-width bins). The calibration tables are not comparable bin-to-bin. | Compared `results/guided/calibration_deciles.csv` with autonomous `results/autonomous/metrics/validation_calibration_{a,b,c}.csv`. | Not corrected. Decide a binning rule for the test stage. | Specify binning when calibration tables will be compared. |
| 2026-11-14 | **[Human/process omission] H-6** — `plan.md` §8.1 requires a clean-environment rerun from the pinned `uv.lock` that reproduces identical metrics | Neither run reported such a rerun: the guided run repeated in the same environment; the autonomous run ran once plus a metric recomputation. | Read both run logs and manifests. | Not corrected. Execute the clean-environment rerun before the test run. | Schedule the clean-env rerun as an explicit deliverable. |
| 2026-11-14 | **[Agent mistake — autonomous] A-1** — claimed the test block was "never loaded" and its labels "never read" (`run_manifest.json` `"test_labels_read": false`; console "NOT loaded … NEVER read"; `REPORT.md` "never loaded … never read") | Contradicted by the code: `pd.read_parquet(RAW)` loads all 45,211 rows × 17 columns, `df["y"].unique()` aggregates the full label column, and `verify_autonomous.py` reads the entire `y` column and indexes `raw_y[test]`. | Code audit of `autonomous_train_val.py` lines 303–310 and `verify_autonomous.py` lines 150, 243–245. | Not corrected in the autonomous workspace (audit must not modify it). Corrected wording recorded in `results/verification/guided_vs_autonomous_comparison.md` §6.3. | Distinguish *reading* labels from *using* them for fitting/tuning/evaluation; never claim more than the code does. |
| 2026-11-14 | **[Agent mistake — autonomous] A-2** — `test_untouched = np.array_equal(raw_y[test], raw_y[test])` in `verify_autonomous.py` | Tautology: always `True`, so the check can never fail, yet it was reported as PASS evidence that the test block was untouched. | Read `verify_autonomous.py` lines 243–245. | Not corrected; flagged for replacement with a failing negative control. | A verification check that cannot fail is not a check. |
| 2026-11-14 | **[Agent mistake — autonomous] A-3** — documentation claims categorical codes were decoded and "never used as numerics" | `decode_categoricals` creates `*_label` columns, but `MODELS` passes the original integer code columns to `OneHotEncoder` for `job`, `marital`, `education`, `default`, `housing`, `loan`; only `poutcome_level` is decoded. Evidence: autonomous coefficients are `cat__job_0` vs guided `cat__job_admin.`, with numerically identical values. | Compared `results/autonomous/models/coefficients_a.csv` with `results/guided/model_coefficients.csv`; read `autonomous_train_val.py` lines 85–101 and 135–157. | Not corrected; the documentation should state that integer codes were passed. The fitted model is unaffected (identical coefficients prove one-hot equivalence). | Documentation must name the columns actually passed to the transformer. |
| 2026-11-14 | **[Agent mistake — guided, minor] A-4** — `run_manifest.json` records `"test_labels_inspected": false` while `guided_train_validate.py` reads the entire `y` column | Imprecise field: test labels are read into memory (then deleted) although they are never indexed or examined. | Read `guided_train_validate.py` lines 241–244. | Not corrected; the log wording elsewhere ("never read into `y_train` or `y_val`") is accurate. | Use precise terms: "read", "indexed", "used for fitting/tuning/evaluation". |
| 2026-11-13 | **[Successful self-correction — guided] S-1** — first run crashed with `KeyError: 'lift'` during manifest construction | Coding error: the lift dict was spread with a `lift_` prefix, producing `lift_lift`, while the manifest read `row["lift"]`. Most artefacts had been written; `run_manifest.json` and `validation_report.md` were missing. | Crash output; `results/guided/guided_run_log.md` §5 #2. | Rewrote the lift keys explicitly, updated the report table, re-ran from scratch; no partial artefact kept. | Validate the artifact schema before writing files. |
| 2026-11-14 | **[Successful self-correction — autonomous] S-2** — verification first pass 29/32; PR-AUC and ΔPR-AUC differed by ~9.6e-6 (> 1e-6 tolerance) | Hand-implemented AP summed per row; sklearn evaluates precision/recall only at the last row of each tie group (Model A had 108 tied probability pairs). | Verification run and comparison against `precision_recall_curve`; `results/autonomous/verification/verification_report.md`. | Corrected the AP tie handling in `verify_autonomous.py`; re-ran → 32/32 PASS. Analysis metrics unchanged, no model refitted. | Match sklearn's tie semantics in independent implementations. |
| 2026-11-14 | **[Successful self-correction — autonomous] S-3** — `uv sync --frozen` failed with `README.md: file not found` | The project build could not run because `pyproject.toml` declares a README that does not exist. | Error message; `autonomous_run_log.md` §0.2. | Re-ran `uv sync --frozen --no-install-project`; did not modify `pyproject.toml` or `uv.lock`. | Prefer non-mutating environment fixes. |
| 2026-11-13 | **[Successful self-correction — guided] S-4** — `scikit-learn` was absent when the pipeline was implemented | Required dependency not provisioned at implementation time. | Import failure / environment check; `guided_run_log.md` §5 #1. | `uv add scikit-learn`; disclosed that `pyproject.toml` and `uv.lock` changed. | Check environment prerequisites before writing pipeline code; disclose lockfile changes. |
| 2026-11-14 | **[Accepted, documented — both] S-5** — sklearn 1.9 emits `FutureWarning` for `penalty="l2"` on every fit | Not a defect: `plan.md` §5.6 requires an L2 penalty, so stating `penalty="l2"` explicitly is correct; the warning is cosmetic (L2 is also the default). | `results/guided/run_warnings.txt` and `results/autonomous/logs/warnings_and_errors.log`. | Documented, not silenced; no model or metric changed. | Prefer an explicit specification over warning suppression. |
| 2026-11-14 | **[Agent mistake — guided sensitivity] A-5** — implemented the random split train-first | The authorized procedure was test-first: hold out 6,783 test rows first, then 6,781 validation rows from the remainder. The code used `test_size=13564` first and `test_size=6783` second, producing different membership despite correct sizes and internal reproducibility. The read-only audit's proposed correction snippet also reversed sklearn's `(train, test)` return order; the human caught it before execution. | Independent human regeneration of the authorized procedure; guided matched FALSE for all three blocks; guided versus autonomous shared rows were 26,036/31,647 train, 563/6,781 validation and 3,381/6,783 test; guided random-test/chronological-test overlap was 1,008 versus 1,022 for the authorized reference. | Archived the entire erroneous run and script under `results/sensitivity_erroneous_v1/` with a SHA-256 inventory; wrote `results/audit/guided_random_split_protocol_discrepancy.md`; created `guided_random_split_sensitivity_corrected.py` and generated the corrected split only. No models fitted; model evaluation awaits human verification of the exact index sets. | Implement the exact authorized call order, verify sklearn return-value semantics, and compare against an independently implemented reference; internal reproducibility is not conformance. |

---

## Status update — 2026-11-16 (documentation correction phase)

Append-only. This section adds later status; it does not alter or remove any row
above. The rows keep the meaning they had when written.

**A-5 — status closed, not rewritten.** The A-5 row ends: "No models fitted;
model evaluation awaits human verification of the exact index sets." That was
accurate on 2026-11-14, when the row was written. It is no longer the current
state.

- The human project author independently regenerated the authorised test-first
  split and verified that all three corrected Guided index sets match the
  authorised procedure and the Autonomous split.
- PHASE 4 was then authorised, and the corrected evaluation was executed on
  2026-11-14T11:26:00Z by
  `src/my_analysis/guided_random_split_sensitivity_evaluate.py`, which loads the
  existing index arrays and does not regenerate the split. Artifacts:
  `results/sensitivity_corrected_v1/report_evaluation.md`,
  `run_manifest_evaluation.json`, `metrics.csv`, `bootstrap.json`,
  `execution_log_evaluation.md`.
- Split fingerprint
  `6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`;
  B−A random ΔROC-AUC 0.054570, 95% CI [0.041805, 0.066600].
- This remains a post-test secondary sensitivity result. It is optimistic by
  construction and does not replace the primary chronological test.
- The A-5 lesson is unchanged, and it is why the error was caught: internal
  reproducibility is not conformance.

**H-4 — superseded, not deleted.** H-4 records that "`results/logs/` does not
exist in either" workspace and that no prompts were archived. That was accurate
for the directory audit performed on 2026-11-14.

- A guided log directory now exists: `results/logs/guided/prompt_log.md` and
  `results/logs/guided/execution_log.md`, created at the final-evaluation
  preparation stage on 2026-11-14.
- They were therefore created **late, not contemporaneously**. The deficiency
  H-4 describes is unchanged for the earlier stages: no verbatim prompt log
  exists for the original training/validation run (2026-11-13) or for the
  guided-vs-autonomous comparative audit, and none has been reconstructed.
- This update does not invent or backfill any prompt.
- H-4's statement about the autonomous workspace records the 2026-11-14 audit
  and has not been re-checked here.

**Files touched by this documentation phase:** `mistakes.md` (this section),
`results/audit/guided_random_split_protocol_discrepancy.md` (dated status note),
`results/sensitivity/README.md` (new), `results/verification/provenance_update.md`
(new). No script, plan, dataset, prediction, metric, bootstrap or manifest was
changed; SHA-256 checks before and after are recorded in the phase report.

---

## Reflection — 2026-11-16 (calibration risk)

**H-7 (calibration) — calibration was treated as a reporting line, not as a
risk.** This entry is recorded as a dated section rather than a new table row so
that no earlier row is rewritten.

**Why this deserved more attention earlier.** The validation report already
showed the warning signs before the test block was opened: prevalence rose from
0.058141 in training to 0.115027 in validation, and the mean predicted
probabilities were roughly half the validation rate (0.054843 for A, 0.052808
for B, against 0.115027;
`results/guided/validation_report.md` §"Consequence for Brier and calibration").
That was a direct warning that the probability scale would not transfer to a
later period. The gap was documented, but it was not escalated: the only related
`mistakes.md` entry was H-5, about calibration *binning rules*, which did not
treat the level gap itself as a defect. There was no stop-and-review, no
explicit "these probabilities are not usable" decision, and no acceptance
criterion that could fail. The test block then repeated the pattern at larger
magnitude: mean predicted 0.0688 (A) and 0.0677 (B) against an observed
prevalence of 0.3935 (`results/guided/test/test_metrics.csv`).

Documentation is not mitigation. A known, measured defect should have been
recorded as a defect with an owner and a decision attached, at the moment it was
first observed, not summarised in a results table and carried forward.

**Why the test results must not be used to retune, recalibrate or reselect.**
The chronological test block has now been evaluated once. Fitting a calibration
map on it, choosing a decision threshold from it, or revisiting the
regularisation choice because of it would turn an out-of-sample estimate into an
in-sample one, invalidate the reported confidence intervals, and make the headline
number impossible to verify independently. The same applies to using the test
block to decide that Model B "needs" recalibration and then reporting a
recalibrated Brier: that number would no longer be comparable with the frozen
result. The correct handling is to report the defect, keep the frozen artifacts
frozen, and if a calibrated model is genuinely required, specify the procedure in
advance and fit it without the test block. The artifacts do not establish whether
the level gap is driven by base-rate shift, regularisation or another mechanism;
using the test set to find out would not be a fair test of that question either.

**What should be planned in advance in future projects.**

1. A calibration acceptance criterion agreed before the test — for example a
   maximum absolute decile gap, or a calibration slope/intercept target — rather
   than treating calibration as a descriptive appendix to ROC-AUC.
2. A block-to-block drift check (prevalence, and the distribution of key
   predictors) with a pre-registered reporting rule and a named decision-maker.
3. Either a pre-registered recalibration step that uses training/validation only,
   or an explicit up-front statement that the probabilities will not be
   interpreted, so the limitation cannot surface for the first time in the final
   table.
4. A constant-prevalence hindsight Brier reference, labelled as a diagnostic, so
   that the size of any level error is visible immediately instead of being
   inferred.
5. A rule that once the test block is evaluated, nothing — selection, threshold,
   calibration or feature choice — may use it.
6. A "not deployable as a probability model" declaration in the results table
   whenever calibration fails, so a ranking gain cannot be read as deployment
   readiness.

**Lesson.** Discrimination and calibration are separate claims. A pipeline can
pass every ranking check and still produce probabilities that are unusable for
decisions; the earlier that is written down as a defect, the cheaper it is to
handle and the less likely it is to be mistaken for a result.

**Files touched by this documentation phase:**
`results/guided/test/calibration_interpretation.md` (new) and `mistakes.md` (this
section only). No metric, prediction, script, plan, manifest or raw file was
changed; SHA-256 checks before and after are recorded in the phase report.
