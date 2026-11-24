# Descriptive subgroup analysis - chronological final test

## Provenance
- Date: 2026-11-14
- Performed independently by the human project author in PowerShell.
- Primary test: frozen chronological rows [38428, 45211), n=6783.
- Predictions: results/guided/test/test_predictions.csv
- Dataset: data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet
- Group definition: pdays == 999 (never contacted) versus pdays != 999 (previously contacted).
- Both source labels and frozen row indices were checked.
- No fitting, tuning or new hypothesis testing was performed.

## Results

| Metric | Never contacted | Previously contacted |
|---|---:|---:|
| N | 3343 | 3440 |
| Positive outcomes | 1108 | 1561 |
| Prevalence | 0.331439 | 0.453779 |
| Model A ROC-AUC | 0.573773 | 0.545890 |
| Model B ROC-AUC | 0.573916 | 0.615015 |
| Delta ROC-AUC B-A | 0.000144 | 0.069125 |
| Model A AP | 0.378283 | 0.501687 |
| Model B AP | 0.378320 | 0.568544 |
| Model A mean predicted | 0.069584 | 0.068123 |
| Model B mean predicted | 0.070077 | 0.065311 |
| Model B calibration gap | 0.261361 | 0.388468 |

## Interpretation

Previous-campaign features add negligible within-group discrimination
among never-contacted clients, but substantially improve descriptive
discrimination among previously contacted clients.

Strong underprediction occurs in both groups, especially among
previously contacted clients.

These findings are descriptive context only, as required by plan.md
section 6.7(8). There is no subgroup hypothesis test, no causal
inference and no new model selection.

The primary chronological full-test conclusion remains unchanged.
