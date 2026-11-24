# Random split v1 — guided sensitivity analysis

Created (UTC): 2026-11-14T09:56:04+00:00
Script: `src/my_analysis/guided_random_split_sensitivity.py`
Dataset SHA-256: `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f`
Split SHA-256: `e43308ca7a0485fdf114be43372be3c0cc07017897ed766895de47e02f6eca3b`

**Role.** Secondary post-test sensitivity analysis. The split is
optimistic because random splitting mixes temporal information. It is
**not** an independent confirmation and **not** a replacement for the
primary chronological evaluation.

## Generation

- Method: `sklearn.model_selection.train_test_split`, seed 42, stratified on `y`.
- Step 1: train_test_split(all_indices, test_size=13564, stratify=y, random_state=42)
- Step 2: train_test_split(remaining_indices, test_size=6783, stratify=y[remaining], random_state=42)
- Indices: sorted ascending after generation; membership unchanged.
- Labels are used only to assign split membership.

## Sizes and class counts

| Block | Rows | Negative | Positive | Prevalence |
|---|---|---|---|---|
| train | 31,647 | 27,945 | 3,702 | 0.116978 |
| validation | 6,781 | 5,988 | 793 | 0.116944 |
| test | 6,783 | 5,989 | 794 | 0.117057 |

## Verification

- Complete coverage of all 45,211 rows: True
- Pairwise disjoint blocks: True
- Reproducible from seed 42: True
- Covered index range: [0, 45210]

## Overlap with the primary chronological blocks

| Random block | Chronological train [0, 31,647) | Chronological validation [31,647, 38,428) | Chronological test [38,428, 45,211) |
|---|---|---|---|
| random_train | 22,121 | 4,800 | 4,726 |
| random_validation | 4,758 | 974 | 1,049 |
| random_test | 4,768 | 1,007 | 1,008 |

The overlap is the mechanism of the sensitivity analysis: every random
block draws rows from all chronological periods, so temporal
information is mixed between training and evaluation.

