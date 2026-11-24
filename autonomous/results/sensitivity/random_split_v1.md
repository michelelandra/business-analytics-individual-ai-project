# random_split_v1 — autonomous random-split sensitivity analysis

Generated (UTC): 2026-11-14T10:04:13+00:00
Script: `src/my_analysis/autonomous_random_split_sensitivity.py`
Dataset: `data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet` (SHA-256 `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f`)

## Role

Secondary sensitivity analysis (plan.md §5.5), created AFTER the
primary chronological analysis and the authorized one-shot test
evaluation. It does **not** replace or alter the frozen primary
chronological results.

## Procedure (pre-specified, exactly as authorized)

1. `train_test_split(arange(45211), test_size=6783, stratify=y,
   random_state=42)` → rest (38,428) + test (6,783)
2. `train_test_split(rest_idx, test_size=6781, stratify=y_rest,
   random_state=42)` → train (31,647) + validation (6,781)

Original row indices are preserved (indices into the raw Parquet,
0-based).

## Sizes and class counts

| Block | n | positives | prevalence |
|---|---|---|---|
| train | 31,647 | 3,702 | 0.116978 |
| validation | 6,781 | 793 | 0.116944 |
| test | 6,783 | 794 | 0.117057 |

## Verification

- sizes_match_protocol: **True**
- no_duplicate_indices: **True**
- full_coverage_exactly_once: **True**
- mutually_disjoint: **True**
- all_indices_within_range: **True**

## Overlap with the frozen chronological blocks

| Random block \ Chronological block | train | validation | test |
|---|---|---|---|
| train | 22,138 | 4,792 | 4,717 |
| validation | 4,715 | 1,022 | 1,044 |
| test | 4,794 | 967 | 1,022 |

## Hashes

- Split-file SHA-256 (serialization-dependent):
  `12177a2662ccbe6ca66c726a0877e5501bcd71f6098a07a6237769e9d1c20513`
- Canonical index fingerprint (order-insensitive, defined in the
  JSON): `6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`

The independently completed Guided run reported 3,702 / 793 / 794 positives and JSON hash e43308ca7a0485fdf114be43372be3c0cc07017897ed766895de47e02f6eca3b. Treated as reference only; no Guided file was read and no procedure was changed to force a hash match.

## Disclosures

- This random split uses observations from **all temporal periods**
  (May 2008 – November 2010) in every block; it removes the
  chronological separation that defines the primary analysis.
- The blocks overlap substantially with the frozen chronological
  blocks (table above); the random test set is **not** an
  independent confirmation of the primary chronological test.
- Hyperparameters were **frozen from the original autonomous
  analysis** (C=0.01 for all models); no new search was performed.
- All implementation choices in this script are **post-test
  decisions**.
- Whole-file target prevalence (11.70%) is preserved by the
  stratification, so every block has the same prevalence by
  construction — unlike the primary chronological split, where
  prevalence drifts between blocks.

