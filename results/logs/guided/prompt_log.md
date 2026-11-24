# Guided run — prompt log

Purpose: accurate record of prompts and model calls for the guided execution,
started at the final-evaluation preparation stage (2026-11-14). This file is
created under `plan.md` §7 (`results/logs/`) and `AGENTS.md` rule 5.

## Historical gap (declared, not reconstructed)

No verbatim prompt log was archived for:

- the original guided training/validation run (2026-11-13);
- the guided-vs-autonomous comparative audit (2026-11-14).

Those prompts are **not reconstructed here**. Reconstructing them from memory
would risk fabrication. The gap is disclosed as a rubric §8.2 #12 deficiency
for the earlier stage. From this file forward, prompts are recorded verbatim.

## 2026-11-14 — final-evaluation preparation authorisation (guided preflight)

- Agent/model: `opencode/big-pickle` ("Big Pickle"), OpenCode session.
- Session role: guided final-evaluation preparation (preflight only).
- Prompt (verbatim):

```
I have reviewed the proposed final test-set evaluation protocol.

I approve the methodological decisions as follows:

G1: Use the original training block only. Do not refit on train + validation.

G2: Perform the pre-registered random-split sensitivity analysis only after the primary chronological test evaluation has been completed and independently verified. Keep it separate from the primary results.

G3: Perform the pre-registered subgroup descriptive analyses after the primary test evaluation, without changing the primary conclusions or model settings.

G4: No new calibration, probability threshold adjustment or post-validation tuning.

G5: New test-evaluation scripts and outputs may eventually be created in both workspaces. Preserve all original training/validation artifacts and keep the executions separate.

G6: No methodological deviation is currently approved. Preserve plan.md v1.2.

G7: Begin accurate prompt and execution logging now. Clearly disclose earlier logging gaps; do not fabricate historical prompts.

G8: If a test evaluation crashes, preserve technical evidence and stop. Do not inspect partial test metrics or restart the test evaluation without my explicit approval.

G9: Approve the proposed disclosure of premature test-label access, but do not claim the final test evaluation has already taken place. Distinguish reading test labels from using them for fitting, tuning or evaluation.

G10: Both guided and autonomous test evaluations must be completed before comparing their test results. Their settings must remain frozen.

IMPORTANT: THIS MESSAGE AUTHORIZES PREPARATION AND REPRODUCTION CHECKS ONLY. IT DOES NOT AUTHORIZE TEST EVALUATION.

CURRENT TASK — GUIDED PREFLIGHT

In the original my-analysis workspace:

1. Prepare a separate, reproducible guided final-evaluation script using the exact model specifications and selected hyperparameters from the existing guided run:
   - Model A: C=0.01
   - Model B: C=0.01
   - Model C: C=0.001

2. Preserve the original feature encoding, solver, preprocessing, training indices and model settings.

3. Implement a preflight-only execution mode that reconstructs the three models using the original training data and checks their regenerated validation predictions against the previously saved validation predictions.

4. Require a maximum absolute prediction difference <= 1e-12. Record the actual differences for A, B and C.

5. Do not calculate or inspect any test metrics, test predictions or test outcomes. Avoid loading test labels in the preflight phase. If the available Parquet-reading method cannot meet this restriction, stop and describe the technical limitation rather than claiming isolation.

6. Implement a separate, explicitly gated test-evaluation mode in the script, but DO NOT EXECUTE IT. Running preflight must not automatically trigger test evaluation.

7. Verify that the script uses no new hyperparameter search, new features or retrospective methodological changes.

8. Preserve all existing guided and autonomous scripts, results, logs and manifests.

9. Save the guided preflight report and current prompt/execution log in new, clearly identified files. Document the checks and their limitations accurately.

10. Do not modify the autonomous workspace at this stage. Its reproduction check will be prepared separately in its own OpenCode session.

11. Do not modify plan.md, AGENTS.md, mistakes.md, data/README.md or frozen split files.

When finished, report:
- Whether each guided model reproduced its saved validation predictions.
- The maximum absolute difference for each model.
- Any errors or unresolved issues.
- Exact new files created.
- Confirmation that the test-evaluation mode was NOT executed.

STOP and wait for my next authorization before evaluating any test data.
```
