# Corrected guided random-split sensitivity — prompt log

## Agent / session

- Agent/model: `opencode/big-pickle` ("Big Pickle"), OpenCode session.
- Workspace: `C:\Users\alber\OneDrive\Documenti\my-analysis` (guided only).
- Autonomous workspace: not accessed.

## Authorizing prompt (verbatim)

```
I have reviewed your read-only root-cause audit and approve a CONTROLLED PARTIAL CORRECTION.

IMPORTANT: Your proposed correction snippet contained an additional error. sklearn's train_test_split returns (train/remainder, test/held-out), in that order. You reversed the return assignments in your audit.

The EXACT approved split implementation is:

```python
remaining_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=6783,
    stratify=y,
    random_state=42
)

train_idx, val_idx = train_test_split(
    remaining_idx,
    test_size=6781,
    stratify=y[remaining_idx],
    random_state=42
)
```

The final blocks must contain 31,647 training, 6,781 validation and 6,783 test rows.

YOU ARE AUTHORIZED TO PERFORM PHASES 0, 1 AND 2 ONLY.

PHASE 0 — PRESERVE THE ORIGINAL ERROR

1. Preserve the entire original `results/sensitivity/` directory, byte-for-byte, under:
   `results/sensitivity_erroneous_v1/`

2. Also preserve the exact original erroneous script:
   `src/my_analysis/guided_random_split_sensitivity.py`

3. Create a SHA-256 inventory verifying that every archived file matches its original.

4. Do not delete, overwrite or alter the original erroneous run.

5. Label the archived analysis clearly:
   "INVALID FOR THE AUTHORIZED SENSITIVITY ANALYSIS — PRESERVED AS AGENT ERROR EVIDENCE."

PHASE 1 — DOCUMENT THE ERROR

Create:
`results/audit/guided_random_split_protocol_discrepancy.md`

Document:
- The approved test-first procedure.
- The implemented train-first procedure.
- Exact source lines responsible.
- The independent human discovery and index-set comparisons.
- Why the agent's reproducibility checks failed to detect nonconformance.
- The additional reversed-assignment bug in your proposed correction snippet.
- All affected sensitivity outputs.
- Confirmation that the primary chronological analysis is unchanged.

Append a clearly labelled evidence-based entry to `mistakes.md`, without deleting or changing earlier entries.

PHASE 2 — CREATE THE CORRECTED SPLIT ONLY

1. Create a NEW script:
   `src/my_analysis/guided_random_split_sensitivity_corrected.py`

2. Save corrected outputs exclusively under:
   `results/sensitivity_corrected_v1/`

3. Implement an explicit `--mode split-only` that generates and verifies the corrected split WITHOUT constructing, fitting or evaluating any model.

4. Use the exact two-step train_test_split calls shown above, preserving their correct return-value order.

5. Check independently that:
   - Every index belongs to exactly one block.
   - All 45,211 observations are covered.
   - The block sizes are exact.
   - Class counts are 3,702 training positives, 793 validation positives and 794 test positives.
   - The random test overlaps the chronological test in exactly 1,022 rows.
   - A separately implemented reference procedure produces the same three index sets.

6. Save the corrected index arrays in `random_split_v1.json` with a canonical order-insensitive fingerprint, together with a split-only conformance report and execution log.

7. Preserve the original Guided features and hyperparameters in the corrected script for a possible later evaluation, but DO NOT execute any training or evaluation mode.

8. Do not copy or access any Autonomous files. The human project author will independently compare the corrected Guided index sets against the existing Autonomous index sets.

9. Do not modify any original primary-analysis scripts, predictions, metrics, manifests, data or frozen plan. Preserve and verify existing hashes.

10. Make all failures, corrections and execution attempts transparent in the logs.

CRITICAL STOP CONDITION:

After generating the corrected split, STOP.

Do NOT:
- Fit Models A, B or C.
- Generate any new model predictions.
- Calculate new performance metrics.
- Run bootstrap.
- Execute the full sensitivity analysis.
- Overwrite the erroneous sensitivity results.
- Start subgroup analysis.

In your final response report:
- Archive location and hash verification.
- Root-cause documentation path.
- Corrected split output paths.
- Exact split sizes and positive counts.
- Canonical index fingerprint.
- Results of the split-only conformance checks.
- Any errors, warnings or unexpected changes.

Then wait for explicit authorization before model fitting.
```

## Model calls

No model calls were made. `--mode split-only` constructs no estimator, fits no
model, generates no predictions, computes no metrics and runs no bootstrap.
The frozen feature sets and hyperparameters are preserved as constants only.
