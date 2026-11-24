# Individual AI Project - Bank Term-Deposit Prediction

Business Analytics (20595). Single-author project by Michele Landra.

This README is a navigation guide to the completed project: what was asked, what
was done, what the verified results are, and where every claim can be checked.
Every number quoted below is copied from a file under `results/`; the source file
is named next to it.

---

## 1. Project overview

The project studies whether **previous-campaign information** improves
out-of-sample prediction of bank term-deposit subscription beyond **customer
characteristics alone**. The research question is deliberately comparative: the
target is the *difference* between two nested models, not the accuracy of either
model in isolation.

Two further questions are part of the design. The first is a leakage diagnostic:
how much apparent performance is added by `duration`, the call length, which is
only known after the call. The second is a process question: can an AI coding
agent execute a pre-registered analysis whose results can be independently
verified?

The design was frozen before the test block was opened (`plan.md`, v1.2). The
primary result is the **chronological** test evaluation. Secondary analyses are
reported as sensitivity checks and do not replace it.

## 2. Data and methodology

**Dataset.** The Bank Marketing dataset (UCI, via a Hugging Face mirror), 45,211
rows, binary target `y` (term-deposit subscription). Dataset SHA-256
`0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f`, as recorded
in `results/guided/test/run_manifest_test.json`. The raw file was never modified.

**Split.** A frozen chronological row-index split, 70/15/15, created by
`src/my_analysis/create_temporal_split.py` and stored in
[`results/splits/temporal_split_v1.json`](results/splits/temporal_split_v1.json):
train 31,647 rows, validation 6,781, test 6,783. The canonical split hash is
`d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c`. The test
block was evaluated once, after a preflight that reproduced the validation
predictions exactly
([`results/guided/preflight/preflight_report.md`](results/guided/preflight/preflight_report.md)).

**Models.** All three are L2-penalised logistic regressions (`lbfgs`,
`max_iter=1000`, `random_state=42`), fitted on the training block only, with
`StandardScaler` for numeric columns and `OneHotEncoder(handle_unknown="ignore")`
for categoricals.

| Model | Features | Role |
|---|---|---|
| A | `age`, `balance`, six customer categoricals | Customer characteristics only |
| B | A + `previous`, `poutcome_derived` | Adds previous-campaign information |
| C | B + `duration` | **Non-deployable leakage diagnostic** |

`C` is never an operational model. Duration is recorded after the call, so a
model that uses it cannot support a pre-call decision. Model C exists only to
measure how much of the outcome is encoded in call timing.

Selected regularisation strengths were `C=0.01` (A), `0.01` (B) and `0.001` (C),
chosen by time-series cross-validation on the training block
([`results/guided/validation_report.md`](results/guided/validation_report.md);
`results/guided/test/run_manifest_test.json`). No recalibration, threshold tuning
or refitting was performed at the test stage.

## 3. Main findings

Verified chronological test results (6,783 rows, 2,669 positives, prevalence
0.39348370927318294) from
[`results/guided/test/test_metrics.csv`](results/guided/test/test_metrics.csv):

| Model | ROC-AUC | Average precision | Brier | Lift@10% |
|---|---:|---:|---:|---:|
| A | 0.555604 | 0.437869 | 0.342691 | 1.191985 |
| B | 0.585800 | 0.467045 | 0.342427 | 1.364410 |
| C (diagnostic) | 0.767393 | 0.619357 | 0.338002 | 1.735500 |

**Primary comparison (B minus A).** From
[`results/guided/test/hypothesis_tests_test.json`](results/guided/test/hypothesis_tests_test.json),
using a paired bootstrap over test rows (2,000 resamples, seed 42, percentile
95% intervals):

- Delta ROC-AUC = **0.03019680943977132**, 95% CI
  **[0.02608477195825771, 0.03426655663048662]**.
- Delta average precision = **0.02917629564410079**, 95% CI
  **[0.024286032131931503, 0.03423888328606934]**.
- Pre-registered classification: **H0 rejected, materially relevant** (the
  threshold is Delta AUC >= 0.01 with a CI lower bound above zero; `plan.md`
  section 3.2).

**Practical meaning.** Adding previous-campaign information improves the ranking
of customers by roughly three ROC-AUC points on a later time period. The
improvement is small but consistently positive: the whole confidence interval
lies above zero. This is a predictive association on held-out later data, not a
causal effect, and it does not establish that contacting a customer because of
this score would change their behaviour.

**The calibration limitation is severe and must travel with these numbers.**
Mean predicted probabilities are 0.068843 (A) and 0.067660 (B) against an
observed test prevalence of 0.393484 - roughly five to six times too low. The
Brier scores are therefore worse than a hindsight constant predictor that is
told the test prevalence (0.238654, an explicitly labelled diagnostic, not an
operational baseline). Discrimination and calibration are separate claims: A and
B rank customers usefully but do not output usable probabilities. See
[`results/guided/test/calibration_interpretation.md`](results/guided/test/calibration_interpretation.md)
and the decile table
[`results/guided/test/test_calibration.csv`](results/guided/test/test_calibration.csv).

Model B should be described as a **ranking prototype with a documented
probability-scale defect**, not as deployment-ready.

## 4. Secondary analyses

**Corrected random-split sensitivity (secondary, optimistic).** After the
primary test, a random 70/15/15 split was evaluated as a sensitivity check. This
split mixes time periods between training and evaluation, so it is optimistic by
construction and **does not replace the chronological result**. B minus A on the
random test block: Delta ROC-AUC 0.054570, 95% CI [0.041805, 0.066600]; Delta AP
0.125422, 95% CI [0.098713, 0.154288]
([`results/sensitivity_corrected_v1/report_evaluation.md`](results/sensitivity_corrected_v1/report_evaluation.md)).
The corrected split is stored in `results/sensitivity_corrected_v1/random_split_v1.json`
with canonical fingerprint
`6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`. The random
test block overlaps the chronological test block by 1,022 rows, which is one
reason the two results are not independent.

**Descriptive subgroup analysis.** Splitting the chronological test by whether a
customer had been contacted before (`pdays` sentinel):

| Group | n | Prevalence | B minus A Delta ROC-AUC |
|---|---:|---:|---:|
| Never contacted | 3,343 | 0.331439 | 0.000144 |
| Previously contacted | 3,440 | 0.453779 | 0.069125 |

Source:
[`results/verification/subgroup_descriptives_test.md`](results/verification/subgroup_descriptives_test.md).
The apparent benefit is concentrated in previously contacted customers. These
are **descriptive** numbers with no subgroup hypothesis test attached, and they
should not be read as a confirmed moderation effect.

**Chronology caveat.** The automated chronology validation returned
**INCONCLUSIVE** because one transition moved forward by a month
([`results/splits/chronology_checks.md`](results/splits/chronology_checks.md)).
A documented human adjudication retained the chronological ordering with the
limitation disclosed; it did not upgrade the automated verdict
([`results/splits/chronology_adjudication.md`](results/splits/chronology_adjudication.md)).
The split remains the frozen primary design, but the year of each record cannot
be recovered from the data, so strict time ordering is an assumption rather than
a proven fact.

## 5. AI agent workflow and reflection

The analysis was executed twice by an AI coding agent, in two workspaces.

- **Guided run** (`opencode/big-pickle`): the human supervisor issued explicit,
  numbered instructions at each stage. Log:
  [`results/guided/guided_run_log.md`](results/guided/guided_run_log.md).
- **Autonomous run** (`GLM-5.3-Flash Free`): an independent execution of the same
  frozen plan in a separate workspace, used as a cross-check. The comparison is
  recorded in
  [`results/verification/guided_vs_autonomous_comparison.md`](results/verification/guided_vs_autonomous_comparison.md).

The comparison found that Models A and B produced **bit-identical** test
predictions across the two runs, but that Model C is **not comparable** because
the hyperparameter grids differed (guided selected `C=0.001`, autonomous
`C=0.01`; `mistakes.md` entry H-1). The same audit also documented overstated
test-isolation claims and a documentation/implementation mismatch on categorical
decoding in the autonomous run.

**Errors are logged, not hidden.** [`mistakes.md`](mistakes.md) records human
design omissions (H-entries), agent mistakes (A-entries), successful
self-corrections (S-entries) and a calibration reflection (H-7). The root-cause
audit for the split error is
[`results/audit/guided_random_split_protocol_discrepancy.md`](results/audit/guided_random_split_protocol_discrepancy.md).

**The original Guided random-split evaluation is INVALID.** It implemented a
train-first split instead of the authorised test-first procedure, so its index
membership was wrong even though block sizes were correct. It is preserved
unchanged as evidence under
[`results/sensitivity_erroneous_v1/`](results/sensitivity_erroneous_v1/) (see
`ARCHIVE_NOTICE.md` and `SHA256SUMS.txt`). The directory
[`results/sensitivity/`](results/sensitivity/) is flagged by its own README as
invalid and must not be cited. The valid version is the **corrected** evaluation
in [`results/sensitivity_corrected_v1/`](results/sensitivity_corrected_v1/),
whose index sets were independently regenerated by the human author before any
model was fitted.

## 6. Independent verification

Two independent checks were performed, by different actors.

**Human PowerShell verification.**
[`results/verification/final_test_independent_verification.md`](results/verification/final_test_independent_verification.md)
records checks executed directly by the human author: row alignment, labels
against the source Parquet, ROC-AUC, average precision, Brier and lift for A, B
and C, and a re-implemented B minus A paired bootstrap. Agreement was within
1e-6. That document also states its own limits.

**OpenCode-generated reproducible script.** A separate, independently written
script
([`results/verification/verify_final_test_recompute.py`](results/verification/verify_final_test_recompute.py))
recomputes the frozen metrics from the saved predictions without importing the
original metric code. Its saved output reports **77 of 77 checks passed**
([`results/verification/final_test_recompute_report.md`](results/verification/final_test_recompute_report.md);
machine-readable results in
[`final_test_recompute_checks.json`](results/verification/final_test_recompute_checks.json)).
It covers all three models' metrics, B minus A and C minus B point estimates and
bootstrap intervals, the frozen classification labels, split/metadata
consistency, and SHA-256 integrity before and after.

**What was NOT verified.**

- Feature engineering, categorical decoding, the derived previous-campaign
  variable, preprocessing and model training were not independently
  re-derived; the checks are metric-level.
- Labels are taken from the saved prediction file. Their match to the raw
  Parquet was checked by the human author, not re-established by the script.
- No clean-environment rerun from the pinned `uv.lock` was performed
  (`mistakes.md` H-6). Determinism evidence comes from repeated runs in the same
  environment.
- The verification establishes reproducibility of the reported numbers, not the
  correctness of the modelling pipeline.

## 7. Provenance and limitations

**Provenance.** The primary Guided and Autonomous analyses were produced by
OpenCode coding agents. The corrected sensitivity evaluation script was drafted
with ChatGPT assistance (reported by the human author; no repository file records
that assistance), then reviewed, authorised and executed manually by the human.
The human also ran the PowerShell checks. Later audits and documentation
corrections were OpenCode-generated and read-only with respect to results. The
full account is in
[`results/verification/provenance_update.md`](results/verification/provenance_update.md).

**Limitations.**

- **Base-rate shift.** Prevalence rises from 0.058141 (train) to 0.115027
  (validation) to 0.393484 (test). Any probability-level claim transfers poorly
  across blocks, and the cause of the shift is not established.
- **Calibration.** As described in section 3, the probabilities are far below
  the observed rate. Brier scores should not be quoted without that caveat.
- **Temporal ambiguity.** The chronology check was INCONCLUSIVE, one forward
  month transition was observed, and record years cannot be recovered. Ordering
  rests on a documented human adjudication.
- **Generalisation.** Results come from one chronological split of one dataset
  and one time period. They do not establish performance on future campaigns or
  other markets, and no causal claim is supported.
- **Model C.** Never deployable; its scores measure leakage, not usefulness.

## 8. Repository structure

The repository holds two independent executions of the same frozen design, the
interactive dashboard, the standalone error analysis and all saved evidence.
Guided code and results remain at the repository root; the independent
Autonomous execution is in [`autonomous/`](autonomous/).

| Path | Contents |
|---|---|
| [`plan.md`](plan.md) | Frozen pre-registration: hypotheses, metrics, decision rules |
| [`AGENTS.md`](AGENTS.md) | Operating rules for AI agents in this repository |
| [`mistakes.md`](mistakes.md) | Historical error, correction and reflection register (append-only) |
| [`error_analysis.md`](error_analysis.md) | Standalone Error Analysis deliverable: errors, locations, methods |
| [`src/my_analysis/`](src/my_analysis/) | Guided analysis scripts (root execution) |
| [`src/my_analysis/create_temporal_split.py`](src/my_analysis/create_temporal_split.py) | Chronological split |
| [`src/my_analysis/validate_chronology.py`](src/my_analysis/validate_chronology.py) | Chronology checks |
| [`src/my_analysis/guided_train_validate.py`](src/my_analysis/guided_train_validate.py) | Training and validation |
| [`src/my_analysis/guided_test_evaluate.py`](src/my_analysis/guided_test_evaluate.py) | One-shot final test evaluation |
| [`src/my_analysis/guided_random_split_sensitivity_corrected.py`](src/my_analysis/guided_random_split_sensitivity_corrected.py) | Corrected random split (split only) |
| [`src/my_analysis/guided_random_split_sensitivity_evaluate.py`](src/my_analysis/guided_random_split_sensitivity_evaluate.py) | Corrected random-split evaluation |
| [`results/guided/validation_report.md`](results/guided/validation_report.md) | Validation metrics and base-rate discussion |
| [`results/guided/test/`](results/guided/test/) | Primary chronological test artifacts |
| [`results/guided/test/calibration_interpretation.md`](results/guided/test/calibration_interpretation.md) | Calibration interpretation |
| [`results/verification/`](results/verification/) | Human and scripted verification |
| [`results/verification/provenance_update.md`](results/verification/provenance_update.md) | Who produced what |
| [`results/verification/final_test_recompute_report.md`](results/verification/final_test_recompute_report.md) | 77/77 recomputation report |
| [`results/splits/`](results/splits/) | Split file, chronology evidence, adjudication |
| [`results/sensitivity_corrected_v1/`](results/sensitivity_corrected_v1/) | Valid corrected sensitivity analysis |
| [`results/sensitivity_erroneous_v1/`](results/sensitivity_erroneous_v1/) | Archived invalid run (evidence) |
| [`results/audit/`](results/audit/) | Root-cause audit of the split error |
| [`results/logs/guided/`](results/logs/guided/) | Prompt and execution logs (partial: see H-4) |
| [`results/explore_bank_marketing/`](results/explore_bank_marketing/) | Exploratory data checks |
| [`autonomous/`](autonomous/) | Independent Autonomous execution: code, results, evidence |
| [`autonomous/README.md`](autonomous/README.md) | Autonomous run notes, verification status and limitations |
| [`dashboard/project_dashboard.html`](dashboard/project_dashboard.html) | Interactive dashboard (open in a browser) |
| [`dashboard/build_dashboard.py`](dashboard/build_dashboard.py) | Regenerates the dashboard from saved artifacts |
| [`models/`](models/) | Post-test reconstructed Models A and B, schema and manifest |
| [`data/README.md`](data/README.md) | Dataset documentation |
| `reflection.md` | **Pending** — one-page personal reflection (not yet written) |

**The Autonomous execution.** The Autonomous run was produced independently of
the Guided run and is copied here under [`autonomous/`](autonomous/); its notes
are in [`autonomous/README.md`](autonomous/README.md). It is not required to
reproduce the primary result, but it is the cross-check the comparison audit
relies on. `autonomous/mistakes.md` is an **empty template** (explained in
`autonomous/README.md` §6.1); its emptiness is a documentation gap, not evidence
that the Autonomous run was error-free.

### 8.1 Guided versus Autonomous

- **Guided** (root): the primary execution, built under explicit step-by-step
  human instruction. Code in `src/my_analysis/`, results in `results/guided/`.
- **Autonomous** ([`autonomous/`](autonomous/)): an independent execution of the
  same frozen plan in a separate workspace, copied here for comparison. The
  differences and the missing common Model C grid are analysed in
  [`error_analysis.md`](error_analysis.md) §5.2 and
  [`results/verification/guided_vs_autonomous_comparison.md`](results/verification/guided_vs_autonomous_comparison.md).

### 8.2 Original frozen results versus post-test additions

- **Original frozen results**: the chronological split, validation, the one-shot
  test evaluation and the verification of those numbers.
- **Post-test usability additions**: `models/A.joblib`, `models/B.joblib`,
  `src/my_analysis/predict.py` and the dashboard were added after the experiment
  finished. They change no reported result; see §10.

### 8.3 Human-directed versus automated verification

- **Human-directed**: the ad-hoc PowerShell checks in
  [`results/verification/final_test_independent_verification.md`](results/verification/final_test_independent_verification.md).
- **Automated**: the independently written 77-check recompute script in
  [`results/verification/final_test_recompute_report.md`](results/verification/final_test_recompute_report.md).
  It supplements, and does not replace, the human check.

### 8.4 Primary versus secondary analysis

- **Primary**: the chronological test block (`results/guided/test/`). This is the
  result the project reports.
- **Secondary**: the corrected random-split sensitivity
  ([`results/sensitivity_corrected_v1/`](results/sensitivity_corrected_v1/),
  optimistic by construction), the descriptive subgroup table and the chronology
  adjudication. None replaces the primary result.

### 8.5 A note on recorded absolute paths

Historical logs, console captures and prompt records contain absolute Windows
paths such as `C:\Users\alber\...`. These are **provenance records** of where a
command actually ran; they are not current path requirements. All analysis
scripts resolve paths relative to the repository root
(`Path(__file__).resolve().parents[...]`), so the project runs from any clone
location.

## 9. Submission

The official Blackboard submission requires **both**:

1. a **GitHub repository URL**, and
2. a **ZIP of that repository**,

submitted by **6 December 2026, 23:59**.

The repository is organised around the four required deliverables:

| # | Deliverable | Location |
|---|---|---|
| 1 | Agent code and supporting analysis | [`src/my_analysis/`](src/my_analysis/), [`autonomous/`](autonomous/), [`results/`](results/) |
| 2 | README explaining the repository | [`README.md`](README.md) (this file) |
| 3 | Error analysis | [`error_analysis.md`](error_analysis.md) |
| 4 | One-page personal reflection | `reflection.md` — **pending, not yet written** |

Deliverable 4 is still outstanding. It must follow the course syllabus reflection
points and be written by the author; it is deliberately not fabricated here, and
no link is given until the file exists.

**What is not the submission ZIP.** `report_evidence_package.zip` is an older
evidence archive that predates the latest documentation edits. It is excluded
from version control by `.gitignore` and is **not** the repository ZIP; it must
not be uploaded as one.

### 9.1 Post-verification privacy metadata change

One change was made to the repository **after** the verification snapshots were
frozen, and it is disclosed here rather than hidden:

- [`pyproject.toml`](pyproject.toml) and
  [`autonomous/pyproject.toml`](autonomous/pyproject.toml) had the `email` field
  removed from their `authors` metadata before public GitHub publication, to
  avoid publishing a personal email address. The author name is unchanged, and
  no other line in either file was edited.
- Scientific code, dependencies, dependency versions, `uv.lock`, the dataset,
  the saved models and all evaluation results were **not** changed. `authors`
  metadata is build metadata only: it plays no part in dependency resolution,
  installation or any reported number.
- [`results/verification/final_test_recompute_hashes.json`](results/verification/final_test_recompute_hashes.json)
  and
  [`autonomous/results/sensitivity/run_manifest_sensitivity.json`](autonomous/results/sensitivity/run_manifest_sensitivity.json)
  deliberately **retain the pre-change SHA-256 of `pyproject.toml`**
  (`99561f119a40ac1a2111c746fd95e24d9af129559e26526496721e532eb4e24a`). They are
  frozen evidence of the file state on their generation date and are still
  accurate as such.
- A strict re-run of the historical file-integrity assertions will therefore
  report `pyproject.toml` as modified relative to those two manifests. That
  expected discrepancy is the change documented in this subsection.
- The historical manifests must **not** be rewritten to make the hashes agree.
  Doing so would conceal a real post-verification edit, which is exactly the
  failure mode the append-only evidence trail exists to prevent.

If a written report is additionally requested, this README plus
`results/guided/test/calibration_interpretation.md` and
`results/verification/provenance_update.md` provide the narrative, numbers and
provenance needed to assemble one without re-running any analysis.

## 10. Setup and inference (added after the experiment)

**Read this first: this section and the files it describes were added after the
experiment had finished.** The original run never saved its fitted models and had
no inference path. Models A and B were later reconstructed from the frozen
specification and exported as files
([`models/export_manifest.json`](models/export_manifest.json),
[`results/verification/model_export_validation.md`](results/verification/model_export_validation.md)),
and [`src/my_analysis/predict.py`](src/my_analysis/predict.py) was added with
them. These are **post-test usability artifacts, not original experimental
deliverables**, and they change no reported result.

### Environment

Python 3.13 (`.python-version`); dependencies declared in `pyproject.toml` and
pinned by `uv.lock`. The recorded analysis environment is Python 3.13.5,
scikit-learn 1.9.1, pandas 3.1.1, NumPy 2.5.4
(`results/guided/run_manifest.json`, `versions`).

```powershell
uv sync
```

Re-running the training or evaluation scripts also needs the raw dataset at
`data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet` (SHA-256
`0cd36af1…bf45f`). The inference path below does **not** need it.

### Test the synthetic example

```powershell
.venv\Scripts\python.exe -m my_analysis.predict --input models\example_customer.csv
```

Expected table (two invented customers, not real records):

| customer | Model A | Model B |
|---|---:|---:|
| `SYNTHETIC-001` | 5.37% | 5.42% |
| `SYNTHETIC-002` | 8.26% | 8.93% |

### Score your own customers

Put the customers in a CSV with the 11 required columns (`age`, `balance`,
`previous`, `pdays`, `job`, `marital`, `education`, `default`, `housing`,
`loan`, `poutcome`) and pass it to the same command:

```powershell
.venv\Scripts\python.exe -m my_analysis.predict --input path\to\customers.csv
```

Categorical columns take the dataset's **integer codes**, not words; `pdays = 999`
means "never contacted"; `duration` is not required and is ignored. Invalid
codes or missing cells are reported instead of guessed. The full schema, the
codebook and the handling of unseen categories are in
[`models/input_schema.md`](models/input_schema.md). Scores are printed to the
terminal only: nothing is written to disk and no model is fitted.

### Saved models versus OpenCode

These are different things and should not be confused:

| Item | What it is |
|---|---|
| [`models/A.joblib`](models/A.joblib), [`models/B.joblib`](models/B.joblib) | The fitted scikit-learn pipelines (scaler + one-hot encoder + logistic regression), reconstructed post-test from the frozen settings. Standalone files; no AI service or network access is involved at prediction time. |
| `src/my_analysis/predict.py` | A small CLI that validates a CSV and calls `predict_proba` on those saved pipelines. |
| **OpenCode** | The AI coding agent used to build, run and audit the analysis. It is a development tool, **not** a model artifact, and it plays no part in scoring. |

### Caveat on the scores

The models are **not calibrated**: on the chronological test block the mean
predicted probability is about 6.8% against an observed subscription rate of
about 39.3% (`results/guided/test/test_metrics.csv`,
[`calibration_interpretation.md`](results/guided/test/calibration_interpretation.md)).
Use the scores to **rank** customers, never as a reliable probability for one
person. Model C is a non-deployable leakage diagnostic and is not exported.

### Cross-platform setup and reproducibility

**Prerequisites.** `uv` must be installed (it is the environment manager that
built and pins this project), and a Python 3.13 interpreter must be available —
`uv` can provision one if the system has none. `.venv` is machine-specific and
is not part of the distributable project; it is recreated, never copied.

**Recreate the environment** from the project root, with internet access:

```powershell
uv sync --frozen
```

`--frozen` installs exactly the versions pinned in `uv.lock` and leaves the
lockfile untouched, so a reviewer gets the same scikit-learn and pandas versions
that produced the recorded numbers instead of a silent upgrade.

**Cross-platform inference command** (works in PowerShell, `cmd`, macOS and
Linux shells):

```powershell
uv run python -m my_analysis.predict --input models/example_customer.csv
```

The `.venv\Scripts\python.exe -m …` form used earlier in this section is
**Windows-specific**: on macOS and Linux the interpreter lives at
`.venv/bin/python`, and that form also assumes the environment already exists.
`uv run` is the portable equivalent — it resolves the project environment itself
and needs no activation step.

**Network, dataset and offline use.** Installing dependencies may require
internet access. Once the environment is installed, scoring with the saved
pipelines needs neither the original dataset (`data/raw/`) nor a network
connection: the predictions come from local `.joblib` files only.

**Load model files only from trusted artifacts.** A `.joblib` file contains
pickled Python objects and is executed on load by design, so a model file is
code-adjacent: treat `models/*.joblib` like a script and load only the copies
shipped with this project after checking them against
[`models/export_manifest.json`](models/export_manifest.json). The saved
pipelines were serialised with scikit-learn 1.9.1; loading them under a
different version can warn or change behaviour, which is why
`src/my_analysis/predict.py` refuses to score when it detects an incompatible
pickle.

**Verified scope.** The analysis and inference commands were executed locally on
Windows 11 (Python 3.13.5, scikit-learn 1.9.1), including
`uv run --no-sync python -m my_analysis.predict --input models/example_customer.csv`,
which returns exactly the same two scores as the `.venv\Scripts\python.exe`
form. `uv sync --frozen` was **not** exercised here, because the environment
already exists, and a clean-environment run on macOS or Linux has **not** been
performed: those steps are documented and expected to work, but they are not
empirically verified in this project.

**OpenCode is not the model.** This distinction survives the setup change:
OpenCode is the AI coding agent used to build and audit the analysis, while
`models/A.joblib` and `models/B.joblib` are standalone scikit-learn pipelines
(scaler, one-hot encoder, logistic regression). Scoring involves no AI service,
no API call and no network request — only local arithmetic on the saved
coefficients.
