# Initial Independent Verification of Exploratory Claims

Date: 2026-11-13
Verifier: project author (independent of the OpenCode exploratory script)
Dataset file: data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet

## 1. Claims under verification

These are claims made by OpenCode's exploratory analysis
(`src/my_analysis/explore_bank_marketing.py`, outputs in
`results/explore_bank_marketing/`):

| # | Claim | OpenCode source file |
|---|-------|----------------------|
| 1 | Dataset shape is (45,211 rows, 17 columns) | `01_shape.txt` |
| 2 | Target distribution is {0: 39,922, 1: 5,289} | `03_target.csv` |
| 3 | Number of observations with `pdays == 999` is 36,954 | `07_pdays_sentinel_stats.csv` |

## 2. Independent verification method

The verifier ran Python from the project directory using pandas to read the
original Parquet file directly. The OpenCode exploratory script was **not**
used, imported, or executed.

Exact command used (run from the project directory):

```powershell
uv run python -c "import pandas as pd; from pathlib import Path; d=pd.read_parquet(next(Path('data/raw').glob('*.parquet'))); print('Shape:', d.shape); print('Target:', d['y'].value_counts().sort_index().to_dict()); print('pdays=999:', int(d['pdays'].eq(999).sum()))"
```

The command performs three checks:

- `d.shape` — dataset dimensions
- `d['y'].value_counts().sort_index().to_dict()` — target distribution
- `d['pdays'].eq(999).sum()` — count of the `pdays` sentinel

## 3. Exact results obtained

| # | Independent result |
|---|--------------------|
| 1 | `(45211, 17)` |
| 2 | `{0: 39922, 1: 5289}` |
| 3 | `36954` |

## 4. Comparison and outcome

| # | Claim (OpenCode) | Independent result | Match |
|---|------------------|--------------------|-------|
| 1 | (45,211, 17) | (45,211, 17) | Yes |
| 2 | {0: 39,922; 1: 5,289} | {0: 39,922; 1: 5,289} | Yes |
| 3 | 36,954 | 36,954 | Yes |

Outcome: all three checks match.

## 5. Limitation

Matching counts confirm that the reported numbers were reproduced from the
same raw file. They do **not** independently validate the interpretation of
encoded variables — for example, that `y = 0` means "no" and `y = 1` means
"yes", that `pdays = 999` is a "never previously contacted" sentinel rather
than a measured value, or that integer codes map to the documented category
labels. Those interpretations rest on the Hugging Face dataset card and the
UCI documentation, not on this count-based verification.

## 6. Execution note: first attempt failed

The first verification attempt failed with
`ModuleNotFoundError: No module named 'pandas'`. The command was initially
executed from `C:\WINDOWS\system32`, outside the project's uv environment, so
pandas was not available and execution stopped before attempting to load the
Parquet dataset. After switching to the correct project directory, the
verification succeeded. This was a user-side execution issue, not an
analytical error made by OpenCode.

No additional checks were run and no results beyond those listed above are
reported in this document.
