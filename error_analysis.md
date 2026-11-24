# Error Analysis

**Project:** Bank term-deposit prediction — a study of AI agent reliability.
**Period:** 13–16 November 2026.
**Scope:** the Guided execution, the Autonomous execution, the human design that
framed both, and the corrections recorded during them.

This retrospective report introduces no new experiment or statistic; every claim
is drawn from artifacts already in the repository. It is **not** a replacement for
the append-only ledger [`mistakes.md`](mistakes.md), treated here as primary
evidence and left unaltered.

One pre-registered design was run twice — under step-by-step human instruction
("Guided") and independently in a separate workspace ("Autonomous") — to test
whether a competent implementation that never saw the first would reproduce it.
Many of the most informative errors are therefore not coding mistakes but gaps in
the frozen design, or claims stronger than the code supported.

## 1. What counts as an error

| Category | Meaning | Count |
|---|---|---|
| **H** — human design omission | The frozen plan was under-specified, leaving a defensible choice open | 6 |
| **A** — agent mistake | An agent contradicted its instructions, its code, or the authorised procedure | 5 |
| **S** — self-correction / accepted warning | An error the agent found and fixed, or a warning deliberately not fixed | 5 |

The distinction matters: the most consequential defect (the random-split error)
was an **agent** error, while the second (the incomparable Model C grid) was a
**human design** omission no agent could have resolved. The ledger has 16 rows —
H-1…H-6, A-1…A-5, S-1…S-5 — while the calibration defect is a dated reflection
rather than a row, so no earlier row is rewritten ([`mistakes.md`](mistakes.md),
"Reflection — 2026-11-16").

## 2. How the errors were detected

| Instrument | Operated by | What it found |
|---|---|---|
| Frozen pre-registration and audit rubric | Human author | Design omissions H-1…H-6, visible only against a silent specification |
| Cross-run comparison and code inspection | Read-only agent audit | Different grids and binning; docs-versus-code mismatch; the tautological check |
| Independent recomputation (PowerShell) | **Human author** | Row alignment, labels, point metrics, paired bootstrap |
| Independent split regeneration | **Human author** | The random-split discrepancy (A-5) |
| Automated recompute script | Agent-written | 77 checks over frozen test artifacts, 0 failures at 1e-6 |

**Human versus automated.** Final-test verification was run by the human author
in ad-hoc PowerShell, not a stored script
([`provenance_update.md`](results/verification/provenance_update.md) §4). An
agent-written script produced 77 assertions and states that it "supplements but
does not replace" the human check
([`final_test_recompute_report.md`](results/verification/final_test_recompute_report.md));
it re-derives the same arithmetic, so agreement is expected by construction and
its value is repeatability, not independence. The human check added a separately
implemented reference procedure — which is how A-5 was caught.

## 3. Error inventory

Locations cite file paths plus section or function; line numbers are omitted
because the files were reorganised and citing them now would imply precision that
no longer holds.

| ID | Issue | Location | Detected by | Consequence | Status |
|---|---|---|---|---|---|
| **H-1** | No CV grid pre-registered | `plan.md` §5.6 | Cross-run CSVs | 9 vs 5 values; `C=0.001` vs `0.01`, not comparable | Not corrected |
| **H-2** | No validation-stage CIs | `plan.md` §5.7 | Validation metrics vs JSON | Guided none, Autonomous yes; asymmetry | Not corrected |
| **H-3** | No negative control on label reading | `plan.md` §8.1 | `verify_autonomous.py` | Both checks could pass while labels were read | Not corrected |
| **H-4** | Prompt logging required, no log directory | `plan.md` §8.2 #12; `AGENTS.md` | Directory audit | No contemporaneous prompt record | Partly superseded (§3.1) |
| **H-5** | No calibration binning rule | `plan.md` §5.7 | Calibration CSVs | Deciles vs equal-width; not comparable | Not corrected |
| **H-6** | Clean-environment rerun required | `plan.md` §8.1 | Run logs and manifests | No rerun in either workspace | Not corrected |
| **A-1** | Claimed test block "never loaded", labels "never read" | `autonomous/results/autonomous/run_manifest.json`; `autonomous_train_val.py` docstring | Code audit | Documentation contradicts code | Wording fixed in comparison §6.3 |
| **A-2** | Check compares an array with itself | `verify_autonomous.py`, `test_untouched` | Expression inspection | A "PASS" with no evidential weight | Flagged |
| **A-3** | Docs say codes decoded; encoder got integer codes | `autonomous_train_val.py`, `MODELS` | Coefficient comparison | Misleading trail; model unaffected | Not corrected |
| **A-4** | Manifest implies labels unread; they were read | `results/guided/run_manifest.json` | Guided loader | Imprecise claim; no leakage | Not corrected |
| **A-5** | Train-first split, not authorised test-first | `results/sensitivity/`; `results/sensitivity_erroneous_v1/` | Human regeneration and index comparison | Sensitivity run invalid | Corrected, re-run; kept |
| **S-1** | First Guided run crashed (`KeyError: 'lift'`) | `results/guided/guided_run_log.md` §5 | Crash output | Manifest and report missing | Fixed; re-ran |
| **S-2** | Hand-written AP off by ~9.6e-6 | `verify_autonomous.py` | 29/32; 108 tied pairs | Tolerance 1e-6 exceeded | Fixed; 32/32 |
| **S-3** | `uv sync --frozen` failed (missing README) | `autonomous_run_log.md` §0.2 | Error message | Environment not buildable as declared | `--no-install-project` |
| **S-4** | scikit-learn absent at implementation | `guided_run_log.md` §5 | Import failure | Pipeline could not run | `uv add`; lockfile change disclosed |
| **S-5** | `FutureWarning` for explicit L2 penalty | `run_warnings.txt`; autonomous warnings log | Warning capture | None; specification correct | Documented — not a defect |

### 3.1 Status updates, not rewrites

Two rows were later superseded without being edited. **A-5** closed "awaits human
verification"; a dated status section records the later authorised evaluation.
**H-4** said no log directory existed; one was later created — **late, not
contemporaneously** — so the deficiency stands for the earlier stages, and no
prompt was reconstructed ([`mistakes.md`](mistakes.md), "Status update —
2026-11-16"). Separately, [`autonomous/mistakes.md`](autonomous/mistakes.md) is
an unpopulated template: its emptiness is a documentation gap, not evidence of an
error-free run, and that run's errors appear in §7.2 of the comparison audit.

## 4. Case study: the A-5 split error

**Authorised procedure.** The sensitivity analysis required a stratified random
split at the chronological block sizes, with the test block separated **first**:
hold out 6,783 test rows from 45,211, then 6,781 validation rows from the
remaining 38,428, leaving 31,647 for training
([`guided_random_split_protocol_discrepancy.md`](results/audit/guided_random_split_protocol_discrepancy.md) §2).

**What the code did.** It separated training first — `test_size=13564`, then
`test_size=6783`. Block sizes were correct and the split reproducible from seed
42, but both facts were irrelevant: the membership differed.

**Why the checks passed.** Verification checked block sizes, coverage of 45,211
rows, disjointness, and reproducibility by re-calling the same function. That "is
a reproducibility check, not a conformance check… the same wrong function returns
the same wrong split" (§5): `reproducible_from_seed: true` was accurate and
worthless as evidence of correctness.

**How it was found.** The human author regenerated the authorised procedure and
compared index sets. Autonomous matched all three blocks; Guided matched none.
They shared only **26,036 of 31,647** training, **563 of 6,781** validation and
**3,381 of 6,783** test rows; the Guided random test block overlapped the
chronological test block in **1,008** rows versus **1,022** for the reference
(§4).

**A second error, in the repair.** The audit's proposed correction reversed
scikit-learn's return order, assigning the 38,428-row remainder to the test
variable. The human caught it before execution; the corrected script uses the
authorised order (§5).

**Resolution.** The corrected split was regenerated and checked against an
independent reference before any fitting: `all_checks_passed` and
`all_blocks_equal` are true, fingerprint `6975172c…a0`. A separate evaluation was
then run at 2026-11-14T11:26:00Z from the stored indices
([`conformance_check.json`](results/sensitivity_corrected_v1/conformance_check.json);
[`report_evaluation.md`](results/sensitivity_corrected_v1/report_evaluation.md)).
It reports a random-split B − A ΔROC-AUC of 0.054570 (95% CI 0.041805 to
0.066600) — a **post-test sensitivity result, optimistic by construction**, not a
replacement for the chronological test.

**Preservation.** The erroneous run, including the exact script, was archived
byte-for-byte with a SHA-256 inventory
([`ARCHIVE_NOTICE.md`](results/sensitivity_erroneous_v1/ARCHIVE_NOTICE.md);
[`SHA256SUMS.txt`](results/sensitivity_erroneous_v1/SHA256SUMS.txt)) and the
original directory kept with an invalidity banner
([`results/sensitivity/README.md`](results/sensitivity/README.md)). The invalid
numbers stay readable — and wrong — so the record of the error and its detection
survives.

## 5. Other issues affecting interpretation

**5.1 Test-block isolation claims.** Both runs load the full dataset and read the
whole label column; the Autonomous verifier also indexes raw labels at test
indices. "Never loaded" / "never read" claims in the Autonomous manifest, log,
docstring and report are false as written. The defensible claim is narrower:
labels were never used for fitting, tuning or any modelling decision, test rows
were never predicted during training, and the block was scored once at a gated
stage
([`guided_vs_autonomous_comparison.md`](results/verification/guided_vs_autonomous_comparison.md)
§6.2–§6.3). A reporting defect, not a leakage defect, but one that undermines
trust in everything else.

**5.2 Design differences and the missing common grid.** The grids differ, and the
Autonomous grid cannot reach Guided's `C=0.001`. Both CV curves decrease as `C`
rises, so each run selected its own lower boundary; neither found an interior
optimum, and the 0.001-versus-0.01 gap is about 0.013 fold standard deviations.
On overlapping points the runs agree to the last displayed digit (§3.2). Models A
and B are bit-identical on validation and test; Model C differs by 4.8e-05
ROC-AUC purely from regularisation strength
([`autonomous/README.md`](autonomous/README.md)). This is H-1, unresolved by
design.

**5.3 Incomplete prompt logging.** No prompt log survives for the original Guided
or Autonomous runs, or the original verification runs. Logging began at the
final-evaluation stage, and both logs declare the gap rather than reconstruct it
([`results/logs/guided/prompt_log.md`](results/logs/guided/prompt_log.md)). The
project cannot fully reconstruct how the agents were directed where most
decisions were made; reconstruction would have produced a document that looked
complete and was not.

**5.4 Calibration.** The most serious limitation. On validation, prevalence was
0.115027 against mean predicted probability of about 0.055 (A) and 0.053 (B); on
test, prevalence 0.3935 against 0.0688 and 0.0677. The warning was visible before
the test block was opened and was documented, but not escalated into a
stop-and-review, and no acceptance criterion could fail. The block was not used
to recalibrate ([`mistakes.md`](mistakes.md);
[`calibration_interpretation.md`](results/guided/test/calibration_interpretation.md)).
Ranking metrics remain usable; the probabilities do not represent risk.

**5.5 Chronology.** The data has no year column. Pre-registered checks returned
`INCONCLUSIVE` because one month transition skips forward. Human adjudication
accepted the ordering: no reversal across 30 blocks, day non-decreasing over
45,181 pairs, and the missing month consistent with zero contacts — a reasoned
judgement, not a proof
([`chronology_adjudication.md`](results/splits/chronology_adjudication.md)).

## 6. What was corrected, documented, or left open

| Status | Items |
|---|---|
| **Corrected** | S-1…S-4; A-5 (split, then separately re-evaluated) |
| **Documented, not corrected** | A-1…A-4; H-1…H-6 (plan frozen) |
| **Accepted as not a defect** | S-5 (correct explicit L2 specification; cosmetic warning) |
| **Open** | Calibration; incomparable Model C grid; missing original prompt logs; no clean-environment rerun; no second holdout |

Nothing here claims full test-label isolation, a reproduced clean environment, or
calibrated models. The primary evaluation remains the chronological test:
B − A ΔROC-AUC 0.030197, 95% CI 0.026085 to 0.034267, reproduced by the human
author within 1e-6
([`final_test_independent_verification.md`](results/verification/final_test_independent_verification.md))
and by a 77-check script over 112 unchanged files
([`final_test_recompute_checks.json`](results/verification/final_test_recompute_checks.json)).

## 7. Lessons

**Oversight mattered at one kind of task.** The agents' internal checks were
competent at arithmetic and weak at conformance. A-5 survived a passing suite
because the suite asked whether output was reproducible, not whether it was
authorised. The lesson is not "agents are unreliable" but the sharper one:
internal reproducibility is not conformance.

**Pre-registration constrains only as far as it constrains choices.** Six of the
sixteen rows are human design omissions. Where `plan.md` was silent — grids,
binning, validation intervals, a negative control — two competent implementations
diverged, and no agent diligence could have resolved the ambiguity. The
specification, not the agent, was limiting.

**Verification is only as good as its falsifiability.** A check that compares an
array with itself, or re-calls a function to confirm it repeats, yields
confidence without evidence. The checks that carried weight could have failed:
recomputation from independent formulas, index comparison against a separately
written reference, and label matching against the source data.

Preservation carries equal weight: the invalid run stays byte-identical and
labelled, letting this document describe a failure precisely rather than
summarise it.
