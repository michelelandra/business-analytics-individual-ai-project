# Input schema — inference for the frozen Guided Models A and B

**Status: post-test usability artifact.** The inference interface described here
was added *after* the experiment was completed. It is not an original
experimental deliverable, and it changes none of the reported results. See
`models/export_manifest.json` (models) and
`results/verification/model_export_validation.md` (how the models were
reconstructed).

Models available: **A** and **B** only. Model C is a non-deployable leakage
diagnostic and is deliberately not shipped.

---

## 1. File format

A plain CSV with a header row and one row per customer, comma-separated.

```csv
customer_id,age,job,marital,education,default,balance,housing,loan,previous,pdays,poutcome
SYNTHETIC-001,34,4,1,2,0,1250,1,0,0,999,4
```

- `customer_id` is optional and used only as a row label.
- Any additional column is ignored, but `y` / `deposit` and `duration` are
  ignored *on purpose* (see §6).
- Scores are printed to the terminal. Nothing is written to disk.

---

## 2. Required columns

All 11 columns below are required. Categorical columns take **integer codes**
from the codebook in §3, not the readable words.

| Column | Type | Meaning | Used by A | Used by B |
|---|---|---|:--:|:--:|
| `age` | number | Customer age in years | yes | yes |
| `balance` | number | Average yearly account balance in euros; **may be negative** | yes | yes |
| `job` | integer code | Type of job | yes | yes |
| `marital` | integer code | Marital status | yes | yes |
| `education` | integer code | Education level | yes | yes |
| `default` | integer code | Has credit in default? | yes | yes |
| `housing` | integer code | Has a housing loan? | yes | yes |
| `loan` | integer code | Has a personal loan? | yes | yes |
| `previous` | number | Number of contacts performed before this campaign for this client (`0` if none) | no | yes |
| `pdays` | number | Days since the client was last contacted in a previous campaign; `999` means never contacted. **Not a model feature** — it only builds `poutcome_derived` (see §4) | no | indirectly |
| `poutcome` | integer code | Outcome of the previous marketing campaign | no | indirectly |

**Not required, and never used:** `duration` (the last contact duration). It is
only known after a call ends, so using it would leak information that does not
exist at prediction time. Models A and B exclude it by design. If a CSV
contains `duration`, the script prints a note and ignores the column.

---

## 3. Categorical codebook (frozen)

These codes are the dataset's own integer codes, relabelled with the frozen
`CODEBOOK` in `src/my_analysis/guided_test_evaluate.py`. They are **not**
invented for this interface and must not be changed.

**`job`**

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| admin. | blue-collar | entrepreneur | housemaid | management | retired | self-employed | services | student | technician | unemployed | unknown |

**`marital`**

| 0 | 1 | 2 | 3 |
|---|---|---|---|
| divorced | married | single | unknown |

**`education`**

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| primary | secondary | tertiary | unknown | basic.4y | basic.6y | basic.9y | high.school | illiterate | professional.course | university.degree |

**`default`** (credit in default)

| 0 | 1 | 2 |
|---|---|---|
| no | yes | unknown |

**`housing`** (housing loan)

| 0 | 1 | 2 |
|---|---|---|
| no | yes | unknown |

**`loan`** (personal loan)

| 0 | 1 | 2 |
|---|---|---|
| no | yes | unknown |

**`poutcome`** (previous campaign outcome, before the `pdays` rule is applied)

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| failure | other | success | unknown | nonexistent |

### Codes that the frozen models never saw

The saved encoders were fitted on the 31,647 training rows, which contained a
narrower set of labels than the codebook:

| Column | Labels present in training | Codebook labels **never seen in training** |
|---|---|---|
| `job` | all 12 | none |
| `marital` | divorced, married, single | `3` = unknown |
| `education` | primary, secondary, tertiary, unknown | `4`–`10` (basic.4y … university.degree) |
| `default` | no, yes | `2` = unknown |
| `housing` | no, yes | `2` = unknown |
| `loan` | no, yes | `2` = unknown |
| `poutcome` | failure, other, success + never_contacted | `3` = unknown, `4` = nonexistent (unless `pdays == 999`) |

The encoder uses `handle_unknown="ignore"`, so an unseen label is encoded as an
all-zero block: it is accepted, produces a score, and has **no effect** on that
column. That is a property of the frozen model, not of this interface. If you
score a customer with `education = 10`, the reported score is computed without
any education information. Prefer codes the model actually learned.

---

## 4. The `pdays == 999` convention

The original pipeline derives one extra column, in `prepare_frame`:

```python
poutcome_derived = "never_contacted" if pdays == 999 else poutcome
```

- `pdays == 999` means *the client was never contacted before*.
- In the raw data, such clients carry `poutcome = 4` ("nonexistent"); the rule
  turns that into the explicit label `never_contacted`.
- Model B's categorical block contains `poutcome_derived`; Model A's does not.
  Both models nevertheless require `pdays` and `poutcome` to be present,
  because the frozen preparation step builds the derived column for every row.

`pdays` values other than `999` are passed through as-is by the rule; they are
not otherwise used as a feature.

---

## 5. Validation, unknown categories and missing values

The script (`src/my_analysis/predict.py`) checks, before any scoring:

| Situation | Behaviour |
|---|---|
| Missing required column | Stops with exit code 2 and names the column |
| Duplicated column name | Stops and names it |
| Blank / missing cell | Stops and names the row and column |
| Non-numeric value in `age`, `balance`, `previous`, `pdays` | Stops, shows the offending value |
| Non-integer or unknown code in a categorical column | Stops and prints the valid codes for that column |
| Code valid in the book but unseen in training | Accepted; contributes nothing (see §3) |
| `duration`, `y`, `deposit` present | Accepted, ignored, with a printed note |
| Extra unknown column | Accepted, ignored, with a printed note |
| Model file hash differs from `models/export_manifest.json` | Stops; refuses to score with an altered model |
| scikit-learn version differs from the export | Stops, rather than silently reporting numbers |

**No imputation is performed anywhere.** A missing value is an error, not a
zero. There are also **no plausibility limits**: `age = 500` is a valid number
and will be scored. The interface validates types and codes, not realism.

---

## 6. Why the original schema must be respected

1. **The codebook is part of the model.** Categories were converted to words by
   the frozen mapping and one-hot encoded. Feeding `job = 4` when you meant
   `management` under a different numbering produces a different person, and
   the model will not complain.
2. **The preprocessing is frozen.** `StandardScaler` means and scales, and the
   encoder category lists, were learned from the 31,647 training rows and are
   stored inside the `.joblib` files. They cannot be re-estimated for new data
   without changing the model.
3. **The pipeline selects columns by name.** Extra columns are dropped, but a
   renamed or missing feature column breaks the contract.
4. **The outcome must stay out.** A `y` column present in a CSV is ignored:
   scoring must never depend on the true result.

---

## 7. Why the scores are not reliable individual probabilities

On the frozen chronological test block the mean predicted probability is about
**6.8%**, while about **39.3%** of customers actually subscribed
(`results/guided/test/test_metrics.csv`; discussion in
`results/guided/test/calibration_interpretation.md`). The models systematically
under-predict the level.

Consequences:

- A score of "5%" does **not** mean a 5% chance of subscribing. If anything it
  means "this customer sits low in the ranking".
- The useful information is **relative order**: which customers look more
  promising than others.
- A low-scoring customer can still subscribe. Individual outcomes are binary and
  noisy; roughly two in five customers in that period did subscribe.
- These models were **not calibrated** after fitting, and no calibration is
  applied here. Calibration would be a modelling change and is out of scope for
  a post-test usability addition.

Do not use these numbers for pricing, targeting commitments, credit decisions,
or statements about an identifiable person.

---

## 8. Quick check

```powershell
.venv\Scripts\python.exe -m my_analysis.predict --input models\example_customer.csv
```

The example file contains two clearly labelled synthetic customers. They are
invented, not taken from any real record.
