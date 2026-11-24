# Independent verification — autonomous run

Script: `src/my_analysis/verify_autonomous.py` (separately written; does not import `my_analysis.autonomous_train_val`; refits nothing).
Tolerance: 1e-06. Bootstrap: 2000 resamples, seed 42, re-implemented independently (rank-based AUC, hand-implemented AP).

## Checks

- prediction row_index == frozen validation indices (model A): **PASS**
- prediction row_index == frozen validation indices (model B): **PASS**
- prediction row_index == frozen validation indices (model C): **PASS**
- y in prediction files == raw parquet y at validation indices: **PASS**
- independent recomputation roc_auc (model A): **PASS**
- independent recomputation pr_auc (model A): **PASS**
- independent recomputation brier (model A): **PASS**
- independent recomputation lift_top10pct (model A): **PASS**
- independent recomputation prevalence_baseline (model A): **PASS**
- independent recomputation n_top_decile (model A): **PASS**
- independent recomputation roc_auc (model B): **PASS**
- independent recomputation pr_auc (model B): **PASS**
- independent recomputation brier (model B): **PASS**
- independent recomputation lift_top10pct (model B): **PASS**
- independent recomputation prevalence_baseline (model B): **PASS**
- independent recomputation n_top_decile (model B): **PASS**
- independent recomputation roc_auc (model C): **PASS**
- independent recomputation pr_auc (model C): **PASS**
- independent recomputation brier (model C): **PASS**
- independent recomputation lift_top10pct (model C): **PASS**
- independent recomputation prevalence_baseline (model C): **PASS**
- independent recomputation n_top_decile (model C): **PASS**
- independent bootstrap B-A dROC_AUC (point + CI): **PASS**
- independent bootstrap B-A dPR_AUC (point + CI): **PASS**
- independent bootstrap C-B dROC_AUC (point + CI): **PASS**
- independent bootstrap C-B dPR_AUC (point + CI): **PASS**
- independent calibration recomputation (model A): **PASS**
- independent calibration recomputation (model B): **PASS**
- independent calibration recomputation (model C): **PASS**
- split file canonical hash unchanged: **PASS**
- raw dataset hash unchanged: **PASS**
- test block size 6,783; NOT evaluated (no metric involving test rows computed anywhere in this verification): **PASS**

## Detail

    model A: rows=6,781, indices_match_frozen_split=True
    model B: rows=6,781, indices_match_frozen_split=True
    model C: rows=6,781, indices_match_frozen_split=True
    labels cross-checked against raw parquet: True
    model A: indep AUC=0.640339965 AP=0.208401293 Brier=0.103874334 lift=2.320855 baseline=0.115027282 n_top=678
    model B: indep AUC=0.659535056 AP=0.228794685 Brier=0.103817957 lift=2.474724 baseline=0.115027282 n_top=678
    model C: indep AUC=0.881046749 AP=0.488623373 Brier=0.087114151 lift=4.769934 baseline=0.115027282 n_top=678
    bootstrap B-A dROC_AUC: point=0.019195091 CI [0.013432502, 0.024787068]
    bootstrap B-A dPR_AUC: point=0.020393391 CI [0.014212005, 0.026576892]
    bootstrap C-B dROC_AUC: point=0.221511692 CI [0.198286030, 0.244815794]
    bootstrap C-B dPR_AUC: point=0.259828688 CI [0.216862328, 0.302503880]
    calibration A: max diffs pp=6.939e-18, pt=0.000e+00
    calibration B: max diffs pp=6.245e-17, pt=0.000e+00
    calibration C: max diffs pp=1.110e-16, pt=0.000e+00

## Discrepancy investigated during verification

The first version of this script computed average precision with
a per-row step-sum. Model A's validation predictions contain 108
pairs of rows with identical predicted probabilities (ties), and a
per-row sum differs from the reported sklearn
`average_precision_score` by ~9.6e-6 (> the 1e-6 tolerance).
Root cause: sklearn evaluates precision/recall only at the last
row of each tie group. The verification implementation was
corrected to the same (standard) definition; afterwards all PR-AUC
and bootstrap values reproduce exactly (diffs < 1e-16). The
reported analysis metrics were NOT changed and no model was
refitted.

**Overall: 32/32 checks PASS.**

