# Random-split sensitivity analysis — autonomous run (POST-TEST)

Generated (UTC): 2026-11-14T10:04:53+00:00
Script: `src/my_analysis/autonomous_random_split_sensitivity.py` (SHA-256 `e55c1711cfbfc5fce44d2844bd15beb4b2fd881d7f8af86741204c93ba382078`)
Plan: `plan.md` v1.2 (frozen), §5.5 secondary sensitivity analysis.

## Scope and status

This is a **secondary robustness analysis**, authorized after the
primary chronological analysis and the authorized one-shot test
evaluation were complete. It does **not** replace or alter the
frozen primary chronological results, and it is **not** an
independent confirmation of the primary test.

## Disclosures (required)

1. The random split uses observations from **all temporal periods**
   in every block; the chronological separation that defines the
   primary analysis is removed.
2. The random blocks overlap substantially with the frozen
   chronological blocks (see the split summary
   `random_split_v1.md`).
3. **This random split is not an independent confirmation of the
   primary test.**
4. Average precision and the Brier score are affected by the very
   different outcome prevalence regime: stratification fixes every
   block's prevalence at the whole-file 11.70% (train 3,702/31,647 = 0.1170), whereas the primary chronological blocks
   drift strongly (5.81% train vs 11.50% validation and higher in
   the test period). AP and Brier values here are therefore **not
   comparable** with the primary chronological values.
5. Hyperparameters were **frozen using the original analysis**:
   C = {'A': 0.01, 'B': 0.01, 'C': 0.01} (recorded in
   `results/autonomous/models/model_settings.json`); no feature
   change, no new CV, no recalibration.
6. **All additional implementation choices are post-test
   decisions**, including this script, the split, the report
   structure and the reference application of the decision rule.
7. No Guided-run file, workspace or result was accessed, copied or
   compared; the Guided-reported split hash was treated as
   reference only.

## Split

- Sizes: train 31,647 / validation 6,781 / test 6,783
- Positives: train 3,702 / validation 793 / test 794
- Verification: {'sizes_match_protocol': True, 'no_duplicate_indices': True, 'full_coverage_exactly_once': True, 'mutually_disjoint': True, 'all_indices_within_range': True}
- Split-file SHA-256 (as recorded inside the JSON; this is the hash of the
  file content **before** the self-referential hash field was appended):
  `12177a2662ccbe6ca66c726a0877e5501bcd71f6098a07a6237769e9d1c20513`
- Final on-disk `random_split_v1.json` SHA-256 (authoritative, recorded in
  `run_manifest_sensitivity.json`):
  `385e31cf8323b2a510cdfc7263b90587f75cea04a3409d19a2380e7509f73e03`
- Canonical index fingerprint: `6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`
  (order-insensitive; verified to reproduce deterministically on an
  independent in-memory re-run of the two-step split, recorded in
  `execution_log.md`).

## Frozen model settings (from the original autonomous analysis)

- L2 logistic regression, solver `lbfgs`, `max_iter=5000`, `random_state=42`
- Selected C: {'A': 0.01, 'B': 0.01, 'C': 0.01} — frozen, no new search
- Original autonomous feature engineering, categorical
  representation (integer code columns + derived `poutcome_level`)
  and preprocessing (StandardScaler / OneHotEncoder
  `handle_unknown='ignore'`), fitted on the random training block
  only.

## Test-block metrics (random test, evaluated once)

Model C is a **NON-DEPLOYABLE DURATION-LEAKAGE DIAGNOSTIC** in
every table.

| Model | ROC-AUC | AP (PR-AUC) | No-skill AP baseline | Brier | Lift @10% |
|---|---|---|---|---|---|
| A | 0.648781 | 0.197967 | 0.117057 | 0.100110 | 2.1546 |
| B | 0.703351 | 0.323390 | 0.117057 | 0.092144 | 3.2004 |
| C | 0.873980 | 0.489994 | 0.117057 | 0.077449 | 4.8006 |

Baseline = prevalence of the random test block itself.

## Paired bootstrap (2,000 resamples, seed 42)

| Comparison | ΔROC-AUC (point) | 95% CI | ΔAP (point) | 95% CI |
|---|---|---|---|---|
| B-A | 0.054570 | [0.042833, 0.067030] | 0.125422 | [0.098026, 0.151747] |
| C-B | 0.170629 | [0.151728, 0.190232] | 0.166604 | [0.132247, 0.202367] |

Identical resample indices reused across models and metrics, as in
the original autonomous analysis. The decision rule of plan.md
§3.2 applied to this random split (reference context only):
**n/a**
— this does not replace or alter the primary chronological
decision.

## Interpretation limits

- No causal claim is made anywhere (associational prediction only,
  plan §1–§2).
- Because stratification equalizes prevalence, the ROC-AUC values
  here are optimistic relative to a deployment setting with
  drifting prevalence; the comparison with the primary
  chronological result is a robustness contrast, not a
  replication.
- Warnings raised during fitting: 3 (see below).

### Warnings

- fit A: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
- fit B: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.
- fit C: FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10. To avoid this warning, leave 'penalty' set to its default value and use 'l1_ratio' or 'C' instead. Use l1_ratio=0 instead of penalty='l2', l1_ratio=1 instead of penalty='l1', l1_ratio set to a float between 0 and 1 instead of penalty='elasticnet', and C=np.inf instead of penalty=None.

