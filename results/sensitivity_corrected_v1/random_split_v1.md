# Corrected random split v1 — SPLIT-ONLY

**SPLIT-ONLY — NO MODEL RESULTS. Model fitting/evaluation requires separate authorization.**

Created (UTC): 2026-11-14T10:26:25+00:00
Script: `src/my_analysis/guided_random_split_sensitivity_corrected.py`
Dataset SHA-256: `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f`
Canonical index fingerprint: `6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`

**Role.** Secondary post-test sensitivity analysis. The split is
optimistic because random splitting mixes temporal information. It is
not an independent confirmation and not a replacement for the primary
chronological evaluation.

## Authorized procedure (test-first)

```python
remaining_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=6783, stratify=y, random_state=42
)
train_idx, val_idx = train_test_split(
    remaining_idx, test_size=6781, stratify=y[remaining_idx], random_state=42
)
```

`train_test_split` returns `(train, test)` in that order.

## Sizes and class counts

| Block | Rows | Negative | Positive | Prevalence |
|---|---|---|---|---|
| train | 31,647 | 27,945 | 3,702 | 0.116978 |
| validation | 6,781 | 5,988 | 793 | 0.116944 |
| test | 6,783 | 5,989 | 794 | 0.117057 |

## Conformance checks

| Check | Result |
|---|---|
| `sizes_exact` | PASS |
| `complete_coverage` | PASS |
| `pairwise_disjoint` | PASS |
| `positive_counts_exact` | PASS |
| `negative_counts_exact` | PASS |
| `chronological_test_overlap_exact` | PASS |
| `matches_authorized_reference` | PASS |
| `reference_fingerprint_matches` | PASS |
| `internally_reproducible` | PASS |

**All checks passed: True.**

## Overlap with the primary chronological blocks

| Random block | Chrono train | Chrono validation | Chrono test |
|---|---:|---:|---:|
| random_train | 22,138 | 4,792 | 4,717 |
| random_validation | 4,715 | 1,022 | 1,044 |
| random_test | 4,794 | 967 | 1,022 |

Random-test overlap with chronological test: **1,022** (expected 1,022).

## Comparison with the erroneous run (evidence only)

Source: `results/sensitivity_erroneous_v1/random_split_v1.json` (split hash `e43308ca7a0485fdf114be43372be3c0cc07017897ed766895de47e02f6eca3b`).

| Block | Shared rows with erroneous split |
|---|---:|
| train | 26,036 |
| validation | 563 |
| test | 3,381 |

These shared-row counts are evidence of the discrepancy. They are
not used for the corrected analysis.

## Status

- Models constructed: **none**
- Models fitted: **none**
- Predictions generated: **none**
- Metrics computed: **none**
- Bootstrap executed: **none**
- Human verification of exact index sets: **required before model fitting**

