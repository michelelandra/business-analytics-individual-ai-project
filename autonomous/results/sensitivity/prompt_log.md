# Prompt log — autonomous random-split sensitivity analysis (post-test)

Date: 2026-11-14 (local). Agent model: **GLM-5.3-Flash Free** (provider
`opencode`), running in OpenCode, in the isolated `my-analysis-autonomous`
workspace.

## Stage context

This stage was preceded by (all in this workspace):
1. The autonomous training/validation run (`src/my_analysis/autonomous_train_val.py`,
   `verify_autonomous.py`, artifacts under `results/autonomous/`).
2. The authorized one-shot gated test evaluation
   (`src/my_analysis/autonomous_test_evaluate.py`, artifacts under
   `results/autonomous/test/`).
3. An independent human verification of the primary analysis (performed by the
   project author outside this session).

## Prompt received for this stage (authorization)

> "We have completed the primary chronological analysis, the authorized
> one-shot final test, and the independent human verification.
>
> I now explicitly authorize the AUTONOMOUS POST-TEST RANDOM-SPLIT SENSITIVITY
> ANALYSIS in this isolated `my-analysis-autonomous` workspace only.
>
> This is a secondary robustness analysis. It must NOT replace or alter the
> frozen primary chronological results. [...]
>
> PROTOCOL: 1. Independently generate the random split ... two-step procedure
> (train_test_split test_size=6783, stratify=y, random_state=42; then
> test_size=6781, stratify=y_remaining, random_state=42) ... 2. Preserve exact
> original row indices; verify disjointness and full coverage. 3. Guided-run
> reference (31,647 / 3,702 positives; 6,781 / 793; 6,783 / 794; JSON SHA-256
> e43308ca7a0485fdf114be43372be3c0cc07017897ed766895de47e02f6eca3b) — treat as
> reference only; report your own split-file hash and a separately defined
> canonical fingerprint; do not force a match. 4. Preserve the original
> AUTONOMOUS feature engineering/preprocessing exactly. 5. Freeze C=0.01 for
> A/B/C, L2, lbfgs, max_iter=5000, random_state=42; no tuning, no new CV, no
> recalibration. 6. Fit on the RANDOM TRAINING BLOCK ONLY. 7. Evaluate on the
> RANDOM TEST BLOCK (ROC-AUC, AP + prevalence baseline, Brier, lift@10%,
> calibration). 8. 2,000 paired bootstrap resamples, seed 42, B−A and C−B.
> 9. Model C exclusively as non-deployable duration-leakage diagnostic.
> 10. New script src/my_analysis/autonomous_random_split_sensitivity.py.
> 11. Outputs under results/sensitivity/ (list of 10 files). 12. Preserve all
> existing artifacts; record and compare pre/post-execution hashes. 13.
> Disclosures (temporal mixing, overlap, not an independent confirmation,
> AP/Brier prevalence effects, frozen hyperparameters, post-test decisions,
> errors/reruns). 14. Preserve failure evidence; no concealment. 15. No
> access to the Guided workspace.
>
> AFTER EXECUTION report: success, split class counts and canonical
> fingerprint, ROC-AUC/AP/Brier/lift for A/B/C, B−A bootstrap CI, output
> paths, preservation checks, warnings/errors/reruns. Then STOP."

## Files read before implementation

- `AGENTS.md`, `plan.md` (v1.2 frozen) — read in the earlier stages of this
  same session and relied upon throughout; §5.5 (secondary random split) and
  §5.7 (metrics) govern this stage.
- `src/my_analysis/autonomous_train_val.py` — the original autonomous
  implementation whose feature engineering, categorical representation,
  preprocessing, metric definitions and bootstrap definitions are reused
  verbatim.
- `src/my_analysis/autonomous_test_evaluate.py` — read in the previous
  authorized stage; its frozen-settings constants and preservation pattern
  informed this script's structure.
- `src/my_analysis/verify_autonomous.py` — for the tie-handling and metric
  definitions previously validated.
- `results/autonomous/models/model_settings.json` — frozen settings source
  (verified: A/B/C all C=0.01, max_iter=5000, lbfgs, l2, seed 42).
- `results/autonomous/test/run_manifest_test.json`,
  `results/autonomous/preflight/preservation_check.json`,
  `results/autonomous/artifact_hashes_final.json` — preservation baselines.

No Guided-run file, workspace or result was accessed, searched for, copied or
compared.
