# Guided random-split sensitivity analysis — prompt log

Purpose: record the authorising prompt and the model calls for the guided
random-split sensitivity analysis (`AGENTS.md` rule 5, `plan.md` §7).

## Agent / session

- Agent/model: `opencode/big-pickle` ("Big Pickle"), OpenCode session.
- Workspace: `C:\Users\alber\OneDrive\Documenti\my-analysis` (guided only).
- The autonomous workspace `my-analysis-autonomous` was **not** accessed.

## Authorising prompt (verbatim)

```
I approve the proposed RANDOM-SPLIT SENSITIVITY ANALYSIS protocol, with the methodological clarifications below.

You are authorized to IMPLEMENT AND EXECUTE the GUIDED sensitivity analysis only, in the original `my-analysis` workspace.

Follow the protocol you just proposed:

1. Generate one deterministic stratified random split across all 45,211 rows using `train_test_split`, seed 42, with exact sizes:
   - Training: 31,647
   - Validation: 6,781
   - Test: 6,783

2. Record the full split indices, class counts, split-generation details, dataset hash and a reproducible split hash. Verify complete coverage, disjointness and reproducibility.

3. Keep the original GUIDED feature sets and preprocessing exactly unchanged.

4. Freeze the original GUIDED hyperparameters:
   - Model A: C=0.01
   - Model B: C=0.01
   - Model C: C=0.001
   Preserve the original solver, regularization, max_iter and random_state settings.

5. Fit on the RANDOM TRAINING BLOCK ONLY. Do not use the random validation block for hyperparameter selection, calibration fitting, model selection or refitting.

6. Evaluate the random test block with the same metrics as the primary analysis: ROC-AUC, average precision, its prevalence baseline, Brier, lift@10%, calibration summary and 2,000 paired bootstrap comparisons with seed 42.

7. Include Model C only as a clearly labelled NON-DEPLOYABLE LEAKAGE DIAGNOSTIC.

8. Use a NEW script:
   `src/my_analysis/guided_random_split_sensitivity.py`

9. Save all results exclusively under:
   `results/sensitivity/`

   Include:
   - random_split_v1.json
   - random_split_v1.md
   - metrics.csv
   - predictions.csv
   - calibration.csv
   - bootstrap.json
   - report.md
   - run_manifest_sensitivity.json
   - console_output.txt
   - prompt_log.md
   - execution_log.md

10. Do not access or modify the autonomous workspace. The autonomous execution will be authorized separately.

11. Treat this as a SECONDARY POST-TEST SENSITIVITY ANALYSIS, not a new independent confirmation or replacement of the original chronological evaluation.

12. Explicitly disclose:
   - Overlap between the random blocks and primary chronological blocks.
   - That stratification uses the target labels solely to assign split membership.
   - Temporal information mixing caused by random splitting.
   - The different outcome prevalence of random vs chronological test.
   - That AP and Brier are prevalence-sensitive.
   - All post-test methodological decisions.

13. Do not use the previously observed primary test results to choose any sensitivity-analysis feature, hyperparameter, split seed or model specification.

14. Preserve every existing primary analysis artifact. Hash existing files before and after execution; report any unexpected modification.

15. If an integrity check fails, stop rather than silently changing the protocol.

Execute the guided sensitivity analysis once, verify its outputs and report:
- Execution status.
- Random split hash and class counts.
- Test metrics for Models A, B and C.
- B-minus-A bootstrap confidence interval.
- Comparison with the original chronological result, clearly marked descriptive only.
- Warnings, methodological limitations and files created.

Then STOP. Do not begin the autonomous sensitivity analysis.
```

## Model calls

No LLM calls are made by `src/my_analysis/guided_random_split_sensitivity.py`.
The machine-learning model calls are three frozen `LogisticRegression` fits
(A, B, C) on the random training block. Their exact configuration, training
size, test size and captured warnings are recorded in
`results/sensitivity/run_manifest_sensitivity.json` under `model_calls`, and in
`results/sensitivity/execution_log.md`.
