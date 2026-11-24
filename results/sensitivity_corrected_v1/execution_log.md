# Corrected guided random-split sensitivity — execution log (split-only)

## 2026-11-14T10:26:25+00:00 — corrected split generation

### Scope

Phase 2 only: generate and verify the corrected split. No model was
constructed, fitted or evaluated; no predictions, metrics or bootstrap were
produced. The original erroneous run is preserved at
`results/sensitivity_erroneous_v1/` and was not modified.

### Approvals in force

Phases 0, 1 and 2 of the controlled partial correction. Model fitting and
evaluation are **not** authorized. The authorizing prompt is recorded verbatim
in `results/sensitivity_corrected_v1/prompt_log.md`.

### Authorized procedure

```python
remaining_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=6783, stratify=y, random_state=42
)
train_idx, val_idx = train_test_split(
    remaining_idx, test_size=6781, stratify=y[remaining_idx], random_state=42
)
```

`train_test_split` returns `(train, test)` in that order.

### Implemented procedure

`generate_corrected_split` uses the exact authorized calls and return order.
`authorized_reference_split` is a separately coded reference implementation.

### Command

```
$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m my_analysis.guided_random_split_sensitivity_corrected --mode split-only
```

### Split

- Sizes: train 31,647 / validation 6,781 / test 6,783.
- Positive counts: train 3,702; validation 793; test 794.
- Negative counts: train 27,945; validation 5,988; test 5,989.
- Canonical fingerprint: `6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`.
- Chronological-test overlap: 1,022 (expected 1,022).

### Conformance checks

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

### Static no-model audit

- Method: ast.parse; scans actual Call and Import/ImportFrom nodes
- Forbidden calls found: []
- Forbidden imports found: []
- No model construction or evaluation code: True

Attempt 1 of this script used a substring scan over its own source and reported
a false failure: the forbidden-token list contained the strings being searched
for. The AST-based check above replaces it. The attempt-1 console output is
preserved in this directory as `console_output_attempt1.txt`.

### Preservation

- Tracked files hashed before: 98
- Tracked files hashed after: 98
- Added: []
- Removed: []
- Changed: []
- Unexpected modification: False

The baseline includes the original erroneous run, the archive and the audit
document. It excludes only this run's own output directory, `.git`, `.venv`,
`__pycache__` and `data/raw` (the raw data is verified by SHA-256).

### Errors, corrections and attempts

Attempt 1 (2026-11-14T10:25:12Z) generated the same split and passed every split-conformance check, but its static no-model audit used a substring scan over the script source. The scan matched its own list of forbidden token strings and therefore reported a false failure. The console output of attempt 1 is preserved as `console_output_attempt1.txt`. The audit was replaced with an AST-based check of actual Call and Import nodes, and the script was rerun. No model was constructed, fitted or evaluated in either attempt. The original erroneous guided sensitivity run was archived before this run; see `results/sensitivity_erroneous_v1/`.

### Status

- Models constructed: **none**
- Models fitted: **none**
- Predictions generated: **none**
- Metrics computed: **none**
- Bootstrap executed: **none**
- Human verification of exact index sets: **required before model fitting**
