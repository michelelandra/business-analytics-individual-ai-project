# Autonomous execution — run notes

This directory holds the **Autonomous** execution of the same pre-registered
Bank Marketing study as the Guided run in the repository root. It is preserved as
a second, independent execution so that the primary result can be compared
against a run that was not directed step by step.

**Run identity.** Agent: `GLM-5.3-Flash Free` (`opencode`), separate OpenCode
session. Date: 2026-11-14. Frozen design: `plan.md` v1.2. Training/validation
artifacts and the independent verification script were produced by that
autonomous session. The final-evaluation script was added later, in a
separately authorised session (see §7, Provenance).

**At a glance**

| Item | Value | Source |
|---|---|---|
| Models fitted | A, B, C (C is a leakage diagnostic, never deployable) | `results/autonomous/models/model_settings.json` |
| Rows: train / validation / test | 31,647 / 6,781 / 6,783 | `results/autonomous/run_manifest.json` |
| Selected `C` | A = 0.01, B = 0.01, C = 0.01 | `results/autonomous/models/model_settings.json` |
| Validation ROC-AUC (A / B / C) | 0.6403 / 0.6595 / 0.8810 | `results/autonomous/metrics/validation_metrics.csv` |
| Test ROC-AUC (A / B / C) | 0.5556 / 0.5858 / 0.7673 | `results/autonomous/test/test_metrics.csv` |
| B − A ΔROC-AUC on test | 0.0302, 95% CI [0.0261, 0.0343] | `results/autonomous/test/hypothesis_tests_test.json` |

---

## 1. Autonomous run overview

The course required two executions of one frozen design: a **Guided** run in
which the human supervisor issued numbered instructions at each stage, and an
**Autonomous** run in which an agent executed the same design in a separate
workspace without using the Guided run's analysis, code or results as context
(`plan.md` §8.3).

The purpose of this second run is **not** to produce a second set of headline
numbers. It answers a different question: *would an implementation that never
saw the first implementation reproduce the result?* That is a test of the plan's
specificity and of the result's dependence on one agent's choices, not a
replication with new data. The two runs share one dataset, one frozen split and
one pre-registration.

Both executions remain in this repository in full. The Guided run is the
**primary** analysis and is the basis of the report in the repository root; the
Autonomous run is a **cross-check**. Where they disagree, the comparison audit in
[`../results/verification/guided_vs_autonomous_comparison.md`](../results/verification/guided_vs_autonomous_comparison.md)
records why, and no number is averaged, merged or substituted.

---

## 2. Experimental design

### 2.1 Shared with the Guided run

| Element | Value |
|---|---|
| Research question | Does adding previous-campaign information (Model B) improve prediction of term-deposit subscription over customer characteristics alone (Model A), on a chronological holdout? |
| Dataset | UCI Bank Marketing (`bank-full`), Hugging Face mirror; 45,211 rows, 17 columns; CC BY 4.0 |
| Frozen split | Chronological: train `[0, 31,647)`, validation `[31,647, 38,428)`, test `[38,428, 45,211)` |
| Selection metric | ROC-AUC |
| Primary comparison | B − A ΔROC-AUC, paired bootstrap, 2,000 resamples, seed 42, percentile 95% CI |
| Materiality threshold | ΔROC-AUC ≥ 0.01 (`plan.md` §3.2) |

The frozen split file `results/splits/temporal_split_v1.json` in this directory is
**byte-identical** to the Guided run's copy. The dataset is likewise identical
(SHA-256 `0cd36af1…bf45f`). The Autonomous run loaded the saved indices; it did
not regenerate the split.

### 2.2 Models

| Model | Features | Role |
|---|---|---|
| **A** | `age`, `balance`, `job`, `marital`, `education`, `default`, `housing`, `loan` | Baseline: customer characteristics only |
| **B** | A + `previous` and `poutcome_level` | Model under test (H1) |
| **C** | B + `duration` | **Leakage diagnostic. Never deployable.** |

**On Model C.** `duration` is the length of the marketing call and is only known
after the call. Model C is fitted and reported solely to quantify how much
apparent performance a duration-contaminated model gains, and is labelled
`DIAGNOSTIC - LEAKAGE, never deployable` in its own results file. No result from
Model C should be read as an achievable performance level, and this run makes no
operational recommendation based on it. On test, C reaches ROC-AUC 0.7673 against
B's 0.5858, a C − B gap of 0.1815 (95% CI 0.1666–0.1964) — consistent with the
pre-registered reading that this feature carries outcome information unavailable
at prediction time.

### 2.3 Evaluation approach

Logistic regression (L2, `lbfgs`, `max_iter=5000`) inside a pipeline with
`StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown="ignore")`
for categoricals, all fitted on the training block only. Regularisation strength
is chosen by mean ROC-AUC over 5-fold expanding-window `TimeSeriesSplit` on the
training block. The test block was evaluated **once**, as a gated stage, after a
preflight reproduction passed at tolerance `1e-12`.

Reported metrics: ROC-AUC, PR-AUC with the set-specific no-skill baseline,
Brier score, lift at the top decile, and calibration by decile. No causal claim
is made anywhere in this run; the design is associational prediction.

---

## 3. Code and directory structure

```
autonomous/
├── README.md                       this file
├── plan.md                         frozen pre-registration (v1.2), identical to the root copy
├── AGENTS.md                       operating rules (frozen), identical to the root copy
├── mistakes.md                     UNPOPULATED HISTORICAL TEMPLATE — see §6
├── pyproject.toml / uv.lock        declared and pinned dependencies
├── data/
│   ├── README.md                   dataset documentation, provenance, licence, codebook
│   └── raw/                        the Parquet file used by this run
├── src/my_analysis/
│   ├── autonomous_train_val.py     original training/validation run
│   ├── verify_autonomous.py        independent recomputation (32 checks)
│   ├── autonomous_test_evaluate.py gated preflight + one-shot test evaluation
│   ├── autonomous_random_split_sensitivity.py  post-test sensitivity analysis
│   ├── create_temporal_split.py    split construction (shared with the Guided run)
│   ├── explore_bank_marketing.py   exploratory data checks
│   └── validate_chronology.py      chronology checks C1–C7
└── results/
    ├── autonomous/
    │   ├── REPORT.md               validation-stage analysis report
    │   ├── autonomous_run_log.md   original run log
    │   ├── run_manifest.json       settings and artifact hashes
    │   ├── metrics/                validation metrics, calibration, CIs
    │   ├── models/                 coefficients, CV selection, model settings
    │   ├── predictions/            saved validation predictions (A, B, C)
    │   ├── preflight/              preflight reproduction checks
    │   ├── test/                   one-shot test metrics, predictions, manifest
    │   ├── verification/           independent verification report + checks
    │   └── logs/                   console output, warnings, preflight hashes
    ├── splits/                     frozen split indices
    ├── explore_bank_marketing/     EDA outputs
    ├── sensitivity/                post-test random-split sensitivity
    └── logs/autonomous/            prompt log and execution log
```

### Where to look

| You want | Open |
|---|---|
| The frozen design | `plan.md` |
| The code that produced the validation results | `src/my_analysis/autonomous_train_val.py` |
| The validation report | `results/autonomous/REPORT.md` |
| Validation numbers | `results/autonomous/metrics/validation_metrics.csv` |
| Test numbers | `results/autonomous/test/test_metrics.csv` |
| Test predictions | `results/autonomous/test/test_predictions.csv` |
| Independent recomputation | `results/autonomous/verification/verification_report.md` |
| Machine-checkable verification | `results/autonomous/verification/verification_checks.json` |
| Exact model settings actually used | `results/autonomous/models/model_settings.json` |
| Original instructions and prompts | `results/logs/autonomous/prompt_log.md`, `results/logs/autonomous/execution_log.md` |
| Cross-run comparison | [`../results/verification/guided_vs_autonomous_comparison.md`](../results/verification/guided_vs_autonomous_comparison.md) |

`results/autonomous/artifact_hashes_final.json` records hashes of the artifacts as
produced. Note that `run_manifest.json` records a **canonical** hash of the split
contents (`d4284248…`), recomputed from the stored indices; it is not the
byte-level hash of the JSON file.

---

## 4. Guided versus Autonomous

The authoritative comparison is
[`../results/verification/guided_vs_autonomous_comparison.md`](../results/verification/guided_vs_autonomous_comparison.md)
(written after both runs, by a read-only audit). The summary below is a reading
aid; if it and that document ever disagree, that document governs.

### 4.1 Verified identical

- **Pre-registration**: `plan.md` v1.2, byte-identical in both directories.
- **Dataset**: same file, SHA-256 `0cd36af1…bf45f`.
- **Split**: same saved indices, same row counts, byte-identical split file.
- **Feature sets** for A, B and C, including the `pdays`/`poutcome` recode and the
  five preserved `unknown` anomalies.
- **Preprocessing**: scaler and one-hot encoder fitted on the training block only,
  no drop-first, no resampling.
- **Search machinery**: 5-fold expanding-window `TimeSeriesSplit`, mean ROC-AUC,
  identical folds for A/B/C, `refit=True`, seed 42.
- **Overlapping grid points**: at every `C` value present in both grids, the mean
  CV ROC-AUC agrees to the last displayed digit.
- **Models A and B on validation**: identical coefficients and identical
  predictions.
- **Models A and B on test**: measurements are bit-identical
  (`max |p_guided − p_autonomous| = 0.000e+00` over 6,783 rows).

### 4.2 Verified differences

| Aspect | Guided | Autonomous |
|---|---|---|
| `C` grid | 9 values, `{0.001 … 10.0}` | 5 values, `{0.01 … 100.0}` |
| Selected `C` | A = 0.01, B = 0.01, **C = 0.001** | A = 0.01, B = 0.01, **C = 0.01** |
| Validation CIs | not computed | computed |
| Random-split sensitivity | original run implemented train-first instead of the authorised test-first order, and is archived as invalid (`../results/sensitivity_erroneous_v1/`) | its split matched the authorised procedure for all three blocks |

### 4.3 Results side by side

| | Guided | Autonomous | Same? |
|---|---:|---:|---|
| Validation ROC-AUC A | 0.6403 | 0.6403 | yes |
| Validation ROC-AUC B | 0.6595 | 0.6595 | yes |
| Validation ROC-AUC C | 0.8684 | 0.8810 | **no** |
| Test ROC-AUC A | 0.5556035710 | 0.5556035710 | yes |
| Test ROC-AUC B | 0.5858003804 | 0.5858003804 | yes |
| Test ROC-AUC C | 0.7673927936 | 0.7673445252 | **no** (Δ = −4.8e-05) |
| B − A ΔROC-AUC | 0.0302 | 0.0302 | yes |
| B − A 95% CI | [0.0261, 0.0343] | [0.0261, 0.0343] | yes |

**How to read this.** The A/B result is implementation-independent: two runs that
never saw each other's code produced the same test predictions to the last
floating-point digit. Only Model C differs, and it differs for one documented
reason — the two grids do not contain the same values, so the two runs selected
different regularisation strengths. The guided grid reaches `0.001`, the
autonomous grid stops at `0.01`, and both CV curves decrease monotonically as `C`
rises, so each run selected its **lower boundary**. Neither found an interior
optimum. The guided `0.001` vs `0.01` gap is about 0.013 fold standard deviations —
not a meaningful separation.

This grid difference is a **design omission in `plan.md`**, which specifies that
`C` is selected by time-series CV but does not specify grid values or a range
(`mistakes.md` entry H-1). It is therefore not an agent error and not evidence
that one implementation is better. A like-for-like comparison of Model C would
require a common grid, which would mean refitting both runs — deliberately not
done.

---

## 5. Reproducibility

### 5.1 Declared environment

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; both are
byte-identical to the Guided run's copies. Python is pinned by the repository's
`.python-version` (3.13). The environment recorded during the autonomous
preflight was:

| Component | Version |
|---|---|
| Python | 3.13.5 |
| scikit-learn | 1.9.1 |
| NumPy | 2.5.4 |
| pandas | 3.1.1 |
| Platform | Windows-11 |

Source: `results/autonomous/preflight/preflight_checks.json`.

### 5.2 Recorded setup command

The original autonomous run recorded that `uv sync --frozen` failed because
`pyproject.toml` references `README.md` and no such file existed in that
workspace, and that the working form was:

```powershell
uv sync --frozen --no-install-project
```

Source: `../results/verification/guided_vs_autonomous_comparison.md` §7.2, entry
A-1. The missing file that caused that failure is this README; the command above
is recorded as having worked and is quoted here as history, not re-tested.

### 5.3 Inspecting existing results without recomputing

Every reported number is already stored. Prefer reading it:

- metrics: `results/autonomous/metrics/validation_metrics.csv`,
  `results/autonomous/test/test_metrics.csv`
- intervals: `results/autonomous/metrics/hypothesis_tests_validation.json`,
  `results/autonomous/test/hypothesis_tests_test.json`
- calibration: `results/autonomous/metrics/validation_calibration_*.csv`,
  `results/autonomous/test/test_calibration.csv`
- model settings: `results/autonomous/models/model_settings.json`
- verification outcomes: `results/autonomous/verification/verification_checks.json`
  (32 named checks, all PASS)

One read-only command is documented in the repository and is safe because it only
prints — quoted from
[`../results/verification/initial_independent_checks.md`](../results/verification/initial_independent_checks.md),
to be run from this directory:

```powershell
uv run python -c "import pandas as pd; from pathlib import Path; d=pd.read_parquet(next(Path('data/raw').glob('*.parquet'))); print('Shape:', d.shape); print('Target:', d['y'].value_counts().sort_index().to_dict()); print('pdays=999:', int(d['pdays'].eq(999).sum()))"
```

### 5.4 Scripts that must not be rerun

The following scripts are the **record**, not a pipeline. Each writes to fixed
filenames under `results/`; running one again would replace saved evidence with
newly computed values and break the correspondence between this directory and the
preserved artifacts.

| Script | Writes into | Why not to rerun |
|---|---|---|
| `src/my_analysis/autonomous_train_val.py` | `results/autonomous/` | would overwrite the original validation run |
| `src/my_analysis/autonomous_test_evaluate.py` (`test` mode) | `results/autonomous/test/` | one-shot frozen test evaluation; the test block has been opened once |
| `src/my_analysis/verify_autonomous.py` | `results/autonomous/verification/` | would overwrite the saved verification report and checks |
| `src/my_analysis/autonomous_random_split_sensitivity.py` | `results/sensitivity/` | post-test sensitivity analysis, already recorded |

There is **no** full clean-environment retraining reproduction of this run in the
repository, and none is claimed. Reproducibility here means: the saved predictions
support the saved metrics, the settings are recorded, the environment is pinned,
and an independent script reproduced the metrics from the saved predictions.

---

## 6. Errors and limitations

### 6.1 `mistakes.md` in this directory is an empty template

> **`autonomous/mistakes.md` is an unpopulated historical template. It must not be
> read as evidence that no errors occurred in this run.**

The file contains a header row and an empty table. It was carried over as part of
the original workspace and has deliberately **not** been filled in afterwards,
because writing entries now would fabricate a contemporaneous record that never
existed. The run's real error record is elsewhere, in documents written at the
time.

### 6.2 Where the error analysis actually is

| Document | What it records |
|---|---|
| `../results/verification/guided_vs_autonomous_comparison.md` §7.2 | The autonomous run's errors, each with correction and outcome |
| `../results/audit/guided_random_split_protocol_discrepancy.md` | Root-cause audit of the Guided split error, and the evidence that the Autonomous split matched the authorised procedure |
| `results/autonomous/logs/warnings_and_errors.log` | Warnings actually emitted during the run |
| `results/autonomous/REPORT.md` §7 | Errors as understood at the time of the validation report |
| `../mistakes.md` | The project-wide ledger (Guided-centric; H/A/S entries) |

The autonomous run's six documented items, from §7.2 of the comparison audit.
The item labels below are that audit's own numbering and are **not** the same
numbering as the H/A/S entries in [`../mistakes.md`](../mistakes.md):

| Item | Error | Outcome |
|---|---|---|
| A-1 | `uv sync --frozen` failed because `README.md` was missing | Resolved with `--no-install-project`; no dependency change |
| A-2 | First verification pass was 29/32: PR-AUC differed by ~9.6e-6 because of tie handling (108 tied pairs in Model A) | Verification script rewritten with the tie-group rule; 32/32 PASS; **no analysis metric changed, no refit** |
| A-3 | scikit-learn 1.9 `FutureWarning` for `penalty="l2"` | Documented, left as-is to avoid post-result churn |
| A-4 | Overstated test-isolation claims | **Open** |
| A-5 | A tautological "test untouched" check | **Open** |
| A-6 | Documentation said categorical codes were decoded; the code passes integer codes | **Open (documentation)** |

Items A-4 to A-6 were not corrected by the autonomous run. They are recorded here
rather than repaired, because repairing them after the fact would erase the
evidence of what the run actually claimed.

### 6.3 The isolation claim, stated accurately

This matters for how the run should be cited.

The autonomous `run_manifest.json`, `logs/console_output.txt`,
`autonomous_train_val.py` docstring and `REPORT.md` all state, in some form, that
the test block "was never loaded and its labels were never read". **That wording is
not supported by the code.** Both runs load the full dataset and read the entire
`y` column into memory; the verification script indexes the raw label array at test
indices.

What *is* supported:

- test labels were **not** used for fitting, tuning, `C` selection or any modelling
  decision in either run;
- test rows were **not** predicted during training/validation;
- the deterministic, target-free recode was applied to the full file, including
  test rows;
- the test block was scored exactly once, at the gated stage, after the preflight
  reproduction passed at `1e-12`.

The honest summary, from §6.2 of the comparison audit, is: **both runs read the test
block; neither used it for any modelling decision or metric.** The defect is a
reporting/integrity-claim defect, not a leakage defect.

### 6.4 Limitations carried

- **Model C is not comparable across runs** and is not deployable (§4.2–§4.3).
- **Calibration is unresolved.** On test, mean predicted probability is roughly
  half the observed prevalence. This is documented and measured, not corrected;
  recalibrating after seeing the test block would invalidate the design.
- **No second independent holdout.** The post-test random split is a sensitivity
  analysis, optimistic by construction, and overlaps the chronological test block.
- **Prompt logs for the original run are missing.** `results/logs/autonomous/`
  records prompts only from the final-evaluation preparation stage onwards and
  states plainly that the earlier prompts were **not** archived and were **not**
  reconstructed. No verbatim prompt log exists for the original training/validation
  run or for the original verification run.
- **No clean-environment retraining reproduction** is demonstrated.
- **Association, not causation.** Observational data, campaign-driven contact
  selection; no causal claim is made.

---

## 7. Provenance

Not everything in this directory was produced by the autonomous run on
2026-11-14. The distinction matters, so it is recorded file by file.

### 7.1 Original autonomous evidence (2026-11-14, `GLM-5.3-Flash Free`)

| Artifact | Notes |
|---|---|
| `src/my_analysis/autonomous_train_val.py` | Original training/validation script |
| `src/my_analysis/verify_autonomous.py` | Independent recomputation, written separately from the pipeline |
| `src/my_analysis/create_temporal_split.py`, `explore_bank_marketing.py`, `validate_chronology.py` | Shared analysis stages, identical in both runs |
| `results/autonomous/REPORT.md` | Validation-stage report written at the time |
| `results/autonomous/autonomous_run_log.md` | Original run log |
| `results/autonomous/run_manifest.json` | Settings and artifact hashes, `generated_utc` 2026-11-14T08:35:24Z |
| `results/autonomous/metrics/`, `models/`, `predictions/` | Validation metrics, coefficients, saved predictions |
| `results/autonomous/verification/` | 32-check verification report |
| `results/autonomous/logs/console_output.txt`, `warnings_and_errors.log` | Console and warning capture |

### 7.2 Added later, under separate authorisation

| Artifact | Added when | Notes |
|---|---|---|
| `src/my_analysis/autonomous_test_evaluate.py` | Final-evaluation preparation stage, 2026-11-14, session `opencode/big-pickle` | Written as a **new** file; the original training script and its artifacts were left untouched. Two modes: preflight, then a gated test mode |
| `results/autonomous/preflight/` | Same stage | Reproduction at tolerance `1e-12`; test mode implemented but not executed at that point |
| `results/autonomous/test/` | Same stage, after authorisation | One-shot test evaluation, `run_manifest_test.json` `generated_utc` 2026-11-14T09:34:58Z |
| `src/my_analysis/autonomous_random_split_sensitivity.py` | After the primary test evaluation | Post-test sensitivity analysis; a new script, not a revision |
| `results/sensitivity/` | After the primary test evaluation | Random-split sensitivity outputs |
| `results/logs/autonomous/prompt_log.md`, `execution_log.md` | From the final-evaluation preparation stage onwards | Verbatim from that point; earlier gap declared, not reconstructed |
| `results/autonomous/artifact_hashes_final.json` | Final-evaluation stage | Hash inventory of the artifacts as produced |

### 7.3 Written by later audits, about — not by — this run

| Artifact | Author | Status |
|---|---|---|
| [`../results/verification/guided_vs_autonomous_comparison.md`](../results/verification/guided_vs_autonomous_comparison.md) | Read-only audit session | Compares the two runs; the source of §4 above |
| [`../results/audit/guided_random_split_protocol_discrepancy.md`](../results/audit/guided_random_split_protocol_discrepancy.md) | Read-only audit session | Guided split error; records that the Autonomous split was correct |
| [`../results/verification/provenance_update.md`](../results/verification/provenance_update.md) | Documentation-correction session | Who produced what across the whole project |
| [`../results/verification/final_test_recompute_report.md`](../results/verification/final_test_recompute_report.md) | Automated recomputation, 77 checks | Recomputes the **Guided** frozen test metrics; supplements the human check |

Nothing in 7.3 altered any autonomous result, prediction or model.

### 7.4 Trust boundary

Numbers from this run should be cited from `results/autonomous/…`, and any
characterisation of how the two runs relate should be cited from
`../results/verification/guided_vs_autonomous_comparison.md`. This README is a
guide to those files, not a substitute for them, and it introduces no new evidence.

---

## Related documents

- Repository overview and the Guided analysis: [`../README.md`](../README.md)
- Frozen pre-registration: [`plan.md`](plan.md)
- Operating rules for agents: [`AGENTS.md`](AGENTS.md)
- Project-wide error ledger: [`../mistakes.md`](../mistakes.md)
- Cross-run comparison: [`../results/verification/guided_vs_autonomous_comparison.md`](../results/verification/guided_vs_autonomous_comparison.md)
- Dataset documentation: [`data/README.md`](data/README.md)
- Interactive dashboard (not a required deliverable): [`../dashboard/project_dashboard.html`](../dashboard/project_dashboard.html)
