# Independent Verification of Guided Validation Metrics

Date: 2026-11-14
Verifier: Project author (human)
Subject: validation metrics reported by the guided training/validation run
Source of claims: `results/guided/validation_metrics.csv`, produced by
`src/my_analysis/guided_train_validate.py`
Input to this check: `results/guided/validation_predictions.csv`

## 0. Authorship and scope of this document

**The check described here was performed by the project author, not by the AI
agent.** The command in section 2 was typed and executed by the human in
PowerShell. This document merely *records the output the human provided*. It
does not claim that the agent ran the command, and the agent did not re-execute
it while writing this file.

The agent's role was limited to writing this markdown file from the
human-supplied results.

## 1. What is being verified

The guided run reported, for Models A, B and C on the 6,781-row validation
block, four metrics each: ROC-AUC, average precision, Brier score and lift at
the top 10%. The purpose of this check is to recompute those four metrics
**from the saved prediction file** using an implementation written independently
of the agent's script, and to confirm that the validation rows are exactly the
frozen validation block.

## 2. Independent verification method

A standalone Python command, run from the project directory through `uv`, reads
only `results/guided/validation_predictions.csv` and recomputes the metrics with
`scikit-learn` metric functions. The agent's analysis script
(`src/my_analysis/guided_train_validate.py`) was **not** imported, executed or
consulted.

Exact command used (as supplied by the verifier):

```powershell
uv run python -c "import pandas as pd, numpy as np; from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss; d=pd.read_csv('results/guided/validation_predictions.csv'); y=d['y'].to_numpy(); assert d['row_index'].tolist()==list(range(31647,38428)); k=int(len(y)*0.10); print('Rows:',len(y),'Prevalence:',round(y.mean(),6),'Top k:',k); [(print(m,'ROC-AUC:',round(roc_auc_score(y,d['p_'+m]),6),'AP:',round(average_precision_score(y,d['p_'+m]),6),'Brier:',round(brier_score_loss(y,d['p_'+m]),6),'Lift@10%:',round(y[np.argsort(-d['p_'+m].to_numpy(),kind='stable')[:k]].mean()/y.mean(),6))) for m in ['A','B','C']]"
```

The command performs the following operations:

| Step | Operation |
|---|---|
| 1 | Load `results/guided/validation_predictions.csv` |
| 2 | **Assert** `row_index` equals `range(31647, 38428)` exactly, in order |
| 3 | Extract `y` and the three probability columns `p_A`, `p_B`, `p_C` |
| 4 | Set `k = int(len(y) * 0.10)` — i.e. 678 |
| 5 | For each model, compute `roc_auc_score`, `average_precision_score`, `brier_score_loss` |
| 6 | Compute lift as `mean(y[stable-descending-top-k]) / mean(y)`, i.e. precision@k ÷ prevalence |

The lift formula and the stable descending ranking rule are the same
definitions the guided run used, so the two computations are comparable.

## 3. Exact results obtained by the verifier

Common values:

| Quantity | Value |
|---|---|
| Rows | 6,781 |
| Validation prevalence | 0.115027 |
| Top-k observations | 678 |

| Model | ROC-AUC | Average precision | Brier score | Lift@10% |
|---|---|---|---|---|
| A | 0.640340 | 0.208401 | 0.103874 | 2.320855 |
| B | 0.659535 | 0.228795 | 0.103818 | 2.474724 |
| C | 0.868365 | 0.472761 | 0.090059 | 4.641710 |

The `assert` on the row indices passed, i.e. the validation rows in the saved
predictions are exactly `31647 … 38427` inclusive (6,781 rows), matching the
frozen split.

## 4. Comparison with the guided-run reported values

| Model | Metric | Guided run (full precision, `validation_metrics.csv`) | Independent (6 dp) | Match at 6 dp |
|---|---|---|---|---|
| A | ROC-AUC | 0.6403399647 | 0.640340 | Yes |
| A | Average precision | 0.2084012934 | 0.208401 | Yes |
| A | Brier | 0.1038743345 | 0.103874 | Yes |
| A | Lift@10% | 2.3208550790 | 2.320855 | Yes |
| B | ROC-AUC | 0.6595350561 | 0.659535 | Yes |
| B | Average precision | 0.2287946848 | 0.228795 | Yes |
| B | Brier | 0.1038179566 | 0.103818 | Yes |
| B | Lift@10% | 2.4747239241 | 2.474724 | Yes |
| C | ROC-AUC | 0.8683653152 | 0.868365 | Yes |
| C | Average precision | 0.4727607524 | 0.472761 | Yes |
| C | Brier | 0.0900593759 | 0.090059 | Yes |
| C | Lift@10% | 4.6417101581 | 4.641710 | Yes |

Also matched:

| Quantity | Guided run | Independent | Match |
|---|---|---|---|
| Rows | 6,781 | 6,781 | Yes |
| Validation prevalence | 0.1150272821 | 0.115027 | Yes |
| Top-k | 678 | 678 | Yes |

**Outcome: all twelve metric values and the three common quantities match to
the displayed precision (6 decimal places).**

## 5. What this check does and does not establish

**Does establish**

- The four reported validation metrics are reproducible from the saved
  prediction file by a separate implementation. The reported numbers were not
  mistyped or transcribed from a different source.
- The validation predictions cover exactly the frozen validation index range
  `[31647, 38428)` and no other rows.
- The metric definitions used in the guided run (including the lift definition
  and the no-skill AP reference equal to validation prevalence) are the ones
  that reproduce the published figures.

**Does not establish**

- **Correctness of model fitting.** The check recomputes metrics *from the
  saved predictions*. It does not re-fit any model, and therefore cannot detect
  an error in feature construction, encoding, hyperparameter selection, or the
  train/validation boundary *if such an error were already baked into the saved
  predictions*. A metric recomputation is a consistency check, not a
  specification check.
- **Correctness of the codebook interpretation.** Whether integer codes map to
  the documented category labels rests on the Hugging Face dataset card and the
  UCI documentation, not on this check.
- **That the split is chronologically valid.** That rests on the chronology
  checks and the human adjudication recorded in `results/splits/`.
- **Anything about the test block.** No test prediction exists and no test
  metric was computed or inspected here.
- **Calibration quality.** The check reproduces the Brier score; it does not
  address the known level miscalibration documented in
  `results/guided/validation_report.md` (models under-predict the validation
  base rate by roughly half).

## 6. Isolation guarantees

- The verification command read **only** `results/guided/validation_predictions.csv`.
- **No test-set rows, labels or predictions were read**, and no test metric was
  computed.
- **No model was refitted** and no new prediction was generated.
- The verification is read-only with respect to the dataset and the split; the
  only file created in this step is this document.

## 7. Files referenced

| File | Role |
|---|---|
| `results/guided/validation_predictions.csv` | input to the independent recomputation |
| `results/guided/validation_metrics.csv` | source of the claims being verified |
| `src/my_analysis/guided_train_validate.py` | producing script of the claims (not used by the verifier) |
| `results/splits/temporal_split_v1.json` | defines the frozen validation range `[31647, 38428)` |
| `results/guided/validation_report.md` | guided-run narrative, including the calibration caveat |
