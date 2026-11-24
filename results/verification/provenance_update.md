# Provenance update — who produced what, and when

Recorded: 2026-11-16 by an OpenCode agent session, on an explicit
documentation-correction instruction from the human project author.

**Purpose.** The repository's provenance records were written at different times
by different agents and did not, in one place, distinguish (a) analysis produced
by an AI coding agent, (b) a script produced with assistance from a different AI
assistant and executed manually by the human, (c) checks executed by the human,
and (d) later read-only audits. This note consolidates those distinctions.

**Scope.** It adds no new evidence and changes no result. Every claim points to
a repository file, or is explicitly marked **author-reported** (no repository
artifact records it). It does not audit or describe the autonomous workspace
beyond what the repository already records.

---

## 1. Original Guided analysis (OpenCode)

| Item | Value | Evidence |
|---|---|---|
| Run mode | `guided` (human-supervised first execution) | `results/guided/guided_run_log.md` §1 |
| Agent | OpenCode coding agent | same |
| Model | `opencode/big-pickle` ("Big Pickle") | `results/guided/guided_run_log.md` §1; `results/logs/guided/prompt_log.md` |
| Human role | supervisor; issued numbered requirements | `results/guided/guided_run_log.md` §2 |
| Stages | train + validation (2026-11-13); preflight and gated one-shot test (2026-11-14) | `results/guided/guided_run_log.md`; `results/guided/preflight/preflight_report.md`; `results/guided/test/run_manifest_test.json` |

No ChatGPT or other assistant involvement is recorded for this stage in any
repository file.

## 2. Original Autonomous analysis (OpenCode)

| Item | Value | Evidence |
|---|---|---|
| Agent | OpenCode coding agent | `results/verification/guided_vs_autonomous_comparison.md` |
| Model | `GLM-5.3-Flash Free (opencode)` | same, runs table |

The autonomous workspace was not accessed while writing this note. The model
identifier is quoted from the comparative audit, which is the repository's
record of it.

## 3. Corrected Guided sensitivity evaluation script

- File: `src/my_analysis/guided_random_split_sensitivity_evaluate.py`
  (SHA-256 `eb536c4cabd0b6594552d382d4b987370b1fea0449c4a413482803f27906cada`,
  per `results/sensitivity_corrected_v1/run_manifest_evaluation.json`).
- **Author-reported:** the human project author states that this script was
  created with **ChatGPT assistance**, then reviewed, authorised and executed
  manually by the human. No repository file records the assistant, the prompt,
  or the date of that assistance; it is therefore recorded here as
  author-reported, not as repository evidence.
- Executed: 2026-11-14T11:26:00Z. Artifacts:
  `results/sensitivity_corrected_v1/` (`metrics.csv`, `predictions.csv`,
  `calibration.csv`, `bootstrap.json`, `report_evaluation.md`,
  `run_manifest_evaluation.json`, `console_output_evaluation.txt`,
  `execution_log_evaluation.md`).
- The script's docstring states "This script performs no LLM calls." That
  concerns **runtime behaviour** (no model or API call during execution) and is
  compatible with ChatGPT having assisted in drafting it. The two statements
  describe different things.

## 4. Human-executed verification (Windows PowerShell)

| Check | Executor | Evidence |
|---|---|---|
| Final test metrics, labels, row alignment, B−A bootstrap | Human, PowerShell | `results/verification/final_test_independent_verification.md` (documented 2026-11-14) |
| Subgroup descriptives | Human, PowerShell | `results/verification/subgroup_descriptives_test.md` |
| Corrected random-split index sets | Human (author-reported), independent regeneration | Summarised in `results/sensitivity_corrected_v1/report_evaluation.md` §1 and in the `authorization` field of `run_manifest_evaluation.json`; machine checks in `results/sensitivity_corrected_v1/conformance_check.json` |

The human checks were run as ad-hoc PowerShell, not from a stored script. No
re-runnable final-test verification script exists in the repository; what
survives is the written record of the checks and their results.

## 5. Read-only audits

| Audit | Executor | Evidence |
|---|---|---|
| Guided vs Autonomous comparative audit (2026-11-14) | OpenCode agent | `results/verification/guided_vs_autonomous_comparison.md` |
| Random-split protocol root-cause audit (2026-11-14) | OpenCode agent (`opencode/big-pickle`) | `results/audit/guided_random_split_protocol_discrepancy.md` |
| Final read-only pre-submission audit (2026-11-16) | OpenCode agent session | No repository artifact was produced; reported in the conversation |
| Documentation correction phase (2026-11-16) | OpenCode agent session | This file, `results/sensitivity/README.md`, and the dated status notes in `mistakes.md` and the protocol audit |

**Session model identifiers for the audit phases.** The 2026-11-16 sessions
report the configured model identifier `opencode/deepseek-v4.1-flash-free`.
**Author-reported:** earlier audit work was also run in a Gemini 3.5 Flash Lite
session. No repository artifact records either identifier, so both are listed as
session-reported rather than as documented evidence; the agents' own written
audits name no model.

## 6. How to read the project

| What you are reading | Who produced it |
|---|---|
| Model fitting, predictions, metrics, bootstrap, calibration tables | AI coding agent (OpenCode), under human authorisation, from the frozen `plan.md` |
| Corrected sensitivity script | ChatGPT-assisted drafting (author-reported); reviewed, authorised and executed by the human |
| Independent metric, label and bootstrap checks | Human, executed manually |
| Error audit, comparative audit, pre-submission audit | AI agents, read-only with respect to results |
| `plan.md`, `AGENTS.md` | Human |

No numerical result in this repository is attributed to ChatGPT drafting alone:
the `results/sensitivity_corrected_v1/` numbers were produced by the human
executing the corrected script. No other analysis artifact is attributed to
ChatGPT.

## 7. Limits of this note

- Prompt logs do not exist for the initial stages and have not been
  reconstructed; see `results/logs/guided/prompt_log.md` and `mistakes.md` H-4.
- Audit-session model identifiers are author-reported, not recorded by the
  agents at the time.
- This note does not modify `plan.md`, any script, the dataset, any prediction,
  metric, bootstrap result or run manifest.
