# Autonomous run — prompt log (final-evaluation preparation)

Purpose: accurate record of prompts and model calls for the autonomous
execution, started at the final-evaluation preparation stage (2026-11-14).
Created under `plan.md` §7 (`results/logs/`) and `AGENTS.md` rule 5.

## Historical gap (declared, not reconstructed)

No verbatim prompt log was archived for:

- the original autonomous training/validation run (2026-11-14, agent model
  `GLM-5.3-Flash Free` / `opencode`);
- the original independent verification run (`verify_autonomous.py`).

Those prompts are **not reconstructed here**. Reconstructing them from memory
would risk fabrication. The gap is disclosed as a rubric §8.2 #12 deficiency for
the earlier stage. From this file forward, prompts are recorded verbatim.

## 2026-11-14 — final-evaluation preparation authorisation (autonomous preflight)

- Agent/model: `opencode/big-pickle` ("Big Pickle"), OpenCode session.
- Session role: autonomous final-evaluation preparation (preflight only).
- Prompt (verbatim):

```
We are preparing the final evaluation of the autonomous Bank Marketing analysis.

This task is restricted to the AUTONOMOUS workspace. Maintain independence from the guided execution.

The original autonomous training/validation run is complete. Its scripts, predictions, selected hyperparameters, settings and results must remain unchanged.

CURRENT AUTHORIZATION: PREFLIGHT AND REPRODUCTION ONLY. NO TEST EVALUATION.

Please:

1. Read AGENTS.md, plan.md v1.2, data/README.md, the frozen split and the autonomous run's existing scripts, model settings, validation predictions and logs.

2. Create a NEW autonomous final-evaluation script with two explicitly separated execution modes:
   - preflight: permitted now;
   - test: implemented but NOT executed.

3. In preflight mode, refit Models A, B and C using the exact settings recorded by your original autonomous run:
   - Original selected hyperparameters.
   - Original feature sets and categorical handling.
   - Original preprocessing pipeline.
   - Original solver, max_iter and seed.
   - Original training indices only.

4. Do not run cross-validation, hyperparameter search or any new model selection.

5. Reproduce the original autonomous validation predictions and calculate the maximum absolute difference for each model.

6. Require max absolute difference <= 1e-12 for every model. If any model fails, stop and report the discrepancy. Do not access the test set.

7. During preflight, do not read test labels or calculate any test metric. Carefully account for Parquet row-group boundaries. Read training labels only from row groups that precede the test block. Validation targets may be read from the existing validation prediction artifacts where appropriate.

8. If accessing required validation features physically loads some test feature rows due to Parquet row-group boundaries, disclose exactly what was loaded and confirm those rows were not used. Do not claim that no test data was physically read if this is not true.

9. The future test mode must be explicitly gated, require a successful reproduction check and require a separate authorization token. Do not invoke it, including its refusal path.

10. Preserve all existing autonomous scripts, results, manifests and logs. Do not modify the guided workspace or access its contents.

11. Begin accurate prompt and execution logging for this preflight. Do not reconstruct or invent missing historical prompt logs.

12. Do not modify plan.md, AGENTS.md, mistakes.md, data/README.md, the raw dataset or the frozen split.

You are authorized to create new autonomous preflight scripts and report artifacts, and execute preflight mode only.

When finished, report:
- Original selected C for each autonomous model.
- Maximum absolute validation prediction difference for A, B and C.
- PASS/FAIL status against the 1e-12 tolerance.
- Exact files created.
- Any warnings, errors and data-isolation limitations.
- Confirmation that test mode was NOT executed.

Then STOP and await authorization for final test evaluation.
```
