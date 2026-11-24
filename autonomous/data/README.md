# Data documentation — Bank Marketing (UCI `bank-full.csv`, Hugging Face mirror)

This document describes the single dataset used by the project in `plan.md`
v1.2. It covers identity, provenance, the complete data dictionary, special
encodings, data quality, leakage risk, and the chronology adjudication that
justifies the frozen temporal split.

**The dataset is read-only.** No row was filtered, merged, recoded or dropped in
producing this document: 45,211 rows before and 45,211 rows after, with
`filters_applied: []` and `rows_dropped: 0` recorded in
`results/explore_bank_marketing/manifest.json`. No model has been trained and
no test-set outcome has been inspected.

## 0. How to read this document

Every claim below is tagged with one of four evidence classes. The tags are not
decoration: they determine how much weight the claim can carry.

| Tag | Meaning | Typical source |
|---|---|---|
| **Documented** | Stated explicitly by UCI, the Hugging Face (HF) dataset card, or the HF build script | `https://archive.ics.uci.edu/dataset/222/bank+marketing`, `https://huggingface.co/datasets/cestwc/bank-marketing` |
| **Empirical** | Measured from the local Parquet file by a named script; the number is stored in `results/` | `src/my_analysis/explore_bank_marketing.py`, `src/my_analysis/validate_chronology.py` |
| **Derived** | Computed from documented/empirical facts by an explicit rule stated in `plan.md` | e.g. `never_contacted := (pdays == 999)` |
| **Assumption** | Neither documented nor measurable from this file; stated as uncertain and carried as a limitation | e.g. `balance` snapshot timing |

Where a statement mixes classes, the weaker class governs.

---

## 1. Dataset identity

### 1.1 Identity

| Property | Value |
|---|---|
| Local file | `data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet` |
| Format | Apache Parquet (single file, single `train` split) |
| Hugging Face dataset | `cestwc/bank-marketing`, config `default`, split `train` |
| Upstream file | UCI `bank-full.csv` (the 17-column, 45,211-row version) |
| Related but **not used** | `bank-additional-full.csv` (41,188 × 21) — a different UCI file with extra columns |
| Target | `y` — "has the client subscribed a term deposit?" |

The HF mirror is described on its card as "a precise version of Bank Marketing"
from UCI. It is a redistribution, not a re-collection.

### 1.2 Provenance chain

Three steps, all **documented**:

1. **UCI** publishes `bank-full.csv` (45,211 rows × 17 columns), described as
   "ordered by date (older version of this dataset with less inputs)".
2. **HF build script** (`cestwc/bank-marketing` card) reads
   `bank+marketing/bank/bank-full.csv` with `sep=";"`, then applies exactly one
   value change:

   ```python
   df_bank["pdays"] = df_bank["pdays"].replace(-1, 999)
   ```

   It then converts the ten categorical columns to `ClassLabel` with an explicit
   name list and calls `Dataset.from_pandas(df_bank, features=hf_features)`.
   **No sorting, reindexing or row-dropping call appears in the script.**
3. **Local Parquet** is the HF artifact as downloaded. Its SHA-256 is re-verified
   by every script before use.

Two consequences worth stating plainly:

- The only change to a numeric column's values is `pdays: -1 → 999`; the ten
  categorical columns are additionally converted from strings to integer
  `ClassLabel` codes (§4.4). All other columns are passed through unchanged.
- The categorical integer codes are **not** part of the original UCI CSV. They
  were assigned by the HF build script. Their exact derivation is documented in
  §4.4 and is essential to reading the file correctly.

`Dataset.from_pandas` preserving input row order is **assumed**, not proven by
the card; this assumption is part of the chronology evidence in §10.

### 1.3 Authors, licence, citation

**Documented.**

- **Creators (UCI):** S. Moro, P. Rita, P. Cortez.
- **Institution:** a Portuguese banking institution (unnamed in the
  documentation).
- **Donated to UCI:** 2012-02-13.
- **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0),
  as stated on the UCI dataset page. Reuse and adaptation are permitted with
  attribution.

**Uncertainty:** the HF dataset card does not declare a licence — the HF
dataset-info API returns an empty `license` field. The licence above is the
licence of the **original** dataset; the mirror adds no separate terms that I
could verify.

**Citations.**

> Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset].
> UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

> Moro, S., Cortez, P., & Rita, P. (2014). A data-driven approach to predict
> the success of bank telemarketing. *Decision Support Systems*.
> (Volume and page numbers are not reproduced here because they were not
> verified from the sources consulted; the DOI above is the dataset DOI.)

### 1.4 Local file integrity

**Empirical**, from `results/explore_bank_marketing/manifest.json` and
`01_shape.txt`:

| Property | Value |
|---|---|
| Size | 692,795 bytes |
| SHA-256 | `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f` |
| Rows | 45,211 |
| Columns | 17 |
| File last modified | 2026-11-05 (unchanged since download) |

Every script in `src/my_analysis/` recomputes this hash and refuses to proceed
on a mismatch. The file has never been written to by this project.

---

## 2. Shape and storage

**Empirical**, from `01_shape.txt` and `02_schema.csv`:

| Property | Value |
|---|---|
| Observations | 45,211 |
| Variables | 17 (16 features + 1 target) |
| Stored dtypes | `int64` for **all 17 columns** |
| Genuinely numeric columns | 7 (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) |
| Label-encoded categorical columns | 10 (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `y`) |
| Missing cells (NaN) | 0 |
| Exact duplicate rows | 0 |

**Documentation inconsistency (minor, documented):** UCI describes
`bank-full.csv` as having "17 inputs". The file has 16 inputs plus the target
`y`, i.e. 17 **columns**. The UCI wording counts columns, not features. The
project therefore refers to 16 features + 1 target throughout, and `plan.md`
§4.2 records the shape as 45,211 × 17.

---

## 3. Unit of observation and collection period

### 3.1 Unit of observation

**Documented:** the data come from direct marketing campaigns (phone calls) of a
Portuguese banking institution. The classification goal is whether the client
subscribes a term deposit. UCI notes that "often, more than one contact to the
same client was required".

**Operational definition used here:** one row = one client's record for the last
contact of the current campaign, with `y` recording the campaign outcome for
that client.

**Uncertainty:** the dataset contains **no client identifier**. Whether a client
can appear in more than one row cannot be determined from the file, so
observations must be treated as potentially non-independent. This is recorded
as a limitation in `plan.md` §9 and in §8 below. It is **not** corrected or
adjusted for.

### 3.2 Collection period

| Statement | Class |
|---|---|
| The campaign data cover **May 2008 – November 2010** | **Documented** for `bank-additional-full.csv`; **inherited by inference** for `bank-full.csv`, which UCI calls an "older version of this dataset with less inputs" |
| `bank-full.csv` is **ordered by date** | **Documented** (UCI) |
| The file contains exactly **30 month blocks** spanning May → November over two cycles | **Empirical** (`chronology_month_blocks.csv`) |
| The span is therefore consistent with 31 calendar months minus one zero-contact month | **Empirical + inference** (§10.3) |
| Exact calendar dates (which year each block belongs to) | **Unknown** — there is no year variable |

The period is used only to describe the data. No seasonal or macroeconomic
adjustment is applied anywhere in the project.

---

## 4. Data dictionary

### 4.1 Conventions

- **Type** = storage type in the Parquet file, followed by the documented
  semantic type.
- **Unit / domain** = the unit or the full categorical domain. Ranges are
  observed ranges from `06_describe_numeric.csv`, not assumed bounds.
- **Pre-call** = whether the value is knowable at the decision point defined in
  `plan.md` §5.1 ("immediately before dialling the client in the current
  campaign"). This is a *design* judgement, graded in §4.5.
- **A / B / C** = role in the three models of `plan.md` §5.2.
- All categorical columns are `int64` codes; the **exact** code → label mapping
  is given in §4.3 and verified in §4.4.

### 4.2 Numeric quantities (7)

| Variable | Meaning (documented) | Type | Unit / observed range | Pre-call? | A | B | C |
|---|---|---|---|---|---|---|---|
| `age` | Age of the client | `int64` / numeric | years; 18–95 | Yes — client attribute | ✅ | ✅ | ✅ |
| `balance` | Average yearly balance | `int64` / numeric | euros; −8,019 to 102,127 | **Assumed** — see §4.5 | ✅ | ✅ | ✅ |
| `day` | Last contact day of the month | `int64` / numeric | day of month; 1–31 | No — describes the current contact | — | — | — |
| `duration` | Last contact duration | `int64` / numeric | seconds; 0–4,918 | **No — post-call** (§9.1) | — | — | ✅ diagnostic |
| `campaign` | Number of contacts performed during this campaign for this client, **including the last contact** | `int64` / count | count; 1–63 | No — includes the current contact | — | — | — |
| `pdays` | Days since the client was last contacted from a **previous** campaign | `int64` / numeric with sentinel | days, 1–871, **plus 999 sentinel** (§5.1) | Yes — prior campaign | — | — † | — † |
| `previous` | Number of contacts performed **before** this campaign for this client | `int64` / count | count; 0–275 | Yes — prior campaign | — | ✅ | ✅ |

† `pdays` is **not a predictor** in any model under `plan.md` v1.2. It is used
only to construct the derived `poutcome` level `never_contacted` (§5.2) and for
diagnostic consistency checks. The raw column is never entered as a number,
because doing so would treat "never contacted" as "999 days ago".

`age` unit note: UCI states only "age (numeric)" and gives no explicit unit;
"years" is the natural reading and is marked as such, not as documented.

### 4.3 Categorical variables (9) and target (1)

All are `int64` label codes. The mapping column below is **exact** and
verified in §4.4. "Observed" lists the codes actually present in this file.

| Variable | Meaning (documented) | Codes (integer = label) | Observed | Pre-call? | A | B | C |
|---|---|---|---|---|---|---|---|
| `job` | Type of job | `0=admin.`, `1=blue-collar`, `2=entrepreneur`, `3=housemaid`, `4=management`, `5=retired`, `6=self-employed`, `7=services`, `8=student`, `9=technician`, `10=unemployed`, `11=unknown` | 0–11 | Yes — client attribute | ✅ | ✅ | ✅ |
| `marital` | Marital status (`divorced` means divorced **or widowed**) | `0=divorced`, `1=married`, `2=single`, `3=unknown` | 0–2 | Yes — client attribute | ✅ | ✅ | ✅ |
| `education` | Education level | `0=primary`, `1=secondary`, `2=tertiary`, `3=unknown`, `4=basic.4y`, `5=basic.6y`, `6=basic.9y`, `7=high.school`, `8=illiterate`, `9=professional.course`, `10=university.degree` | **0–3 only** (§5.4) | Yes — client attribute | ✅ | ✅ | ✅ |
| `default` | Has credit in default? | `0=no`, `1=yes`, `2=unknown` | 0–1 | Yes — client attribute | ✅ | ✅ | ✅ |
| `housing` | Has housing loan? | `0=no`, `1=yes`, `2=unknown` | 0–1 | Yes — client attribute | ✅ | ✅ | ✅ |
| `loan` | Has personal loan? | `0=no`, `1=yes`, `2=unknown` | 0–1 | Yes — client attribute | ✅ | ✅ | ✅ |
| `contact` | Contact communication type | `0=cellular`, `1=telephone`, `2=unknown` | 0–2 | No — chosen at contact time | — | — | — |
| `month` | Last contact month of year | `0=jan`, `1=feb`, `2=mar`, `3=apr`, `4=may`, `5=jun`, `6=jul`, `7=aug`, `8=sep`, `9=oct`, `10=nov`, `11=dec` | 0–11 | No — campaign operations | — | — | — |
| `poutcome` | Outcome of the **previous** marketing campaign | File: `0=failure`, `1=other`, `2=success`, `3=unknown`, `4=nonexistent` (0 rows). Model: five levels — `failure`, `other`, `success`, `unknown`, `never_contacted` (derived, §5.2) | 0–3 | Yes — prior campaign | — | ✅ | ✅ |
| `y` | Has the client subscribed a term deposit? | `0=no`, `1=yes` | 0–1 | No — outcome | target | target | target |

**Target distribution** (**empirical**, `03_target.csv`): `y=0` "no" 39,922
(88.3015%); `y=1` "yes" 5,289 (11.6985%). Positive prevalence 11.6985%. This is
a whole-file property and is **not** assumed to hold in any evaluation set
(`plan.md` §4.2, §5.7).

### 4.4 Codebook provenance and verification

This is the part most easily got wrong, so it is documented in full.

**How the codes were created (documented, HF build script).** The script does
*not* use a hand-written mapping. It takes, for each categorical column, the
union of:

- `df_bank[col].astype("category").cat.categories` — pandas sorts these
  **alphabetically**, and
- the same list from `bank-additional-full.csv`, appended after it,

then passes the merged ordered list to `ClassLabel(names=...)`. `month` is the
one exception: it is **manually overridden** to `['jan', 'feb', ..., 'dec']`.

This single fact explains all the codebook oddities below. In every column the
resulting order is: the **alphabetical order of the levels that actually occur
in `bank-full`**, followed by any level that occurs only in `bank-additional`.
That is exactly what `astype("category").cat.categories` produces (pandas sorts
string categories lexicographically), and the observed name lists agree with
this rule in all 10 categorical columns:

- `job` 0–11: alphabetical over the 12 levels present in `bank-full`
  (`admin.`, `blue-collar`, …, `unknown`).
- `marital` 0–2: alphabetical over the three levels present in `bank-full`
  (`divorced`, `married`, `single`); `unknown` is appended at 3 because no
  `bank-full` row uses it.
- `education` 0–3: alphabetical over the four levels present in `bank-full`
  (`primary`, `secondary`, `tertiary`, `unknown`); codes 4–10 are
  `bank-additional`-only levels appended afterwards, which is why they never
  occur in this file.
- `default`, `housing`, `loan`: `0=no`, `1=yes` (the two levels present in
  `bank-full`), with `unknown` appended at 2 from `bank-additional`.
- `poutcome` 0–3: alphabetical over the four `bank-full` levels
  (`failure`, `other`, `success`, `unknown`); `nonexistent` is appended at 4
  and has 0 rows here.
- `month` 0–11: calendar order **by explicit override**, not alphabetical.

**Verification performed for this document** (2026-11-13):

| Check | Result |
|---|---|
| HF `dataset_info.features` ordered `ClassLabel` names match the mapping in §4.3 exactly, for all 10 categorical columns | ✅ match (`https://datasets-server.huggingface.co/info?dataset=cestwc/bank-marketing`) |
| HF card `README.md` YAML `class_label.names` matches the same mapping | ✅ match (`https://huggingface.co/datasets/cestwc/bank-marketing/raw/main/README.md`) |
| Every observed integer code is a valid index into those name lists | ✅ 0 undocumented codes (`10_validation_checks.csv`) |
| UCI's documented `bank-full` level *sets* agree with the labels used at the corresponding codes (order differs, and HF appends `bank-additional` levels) | ✅ for `job`, `marital`, `education`, `contact`, `poutcome`; ✅ for `default`/`housing`/`loan` at codes 0–1 |
| No mapping was inferred from the apparent numeric order alone | ✅ the order is taken from the HF name lists, and the *reason* for the order is documented in the build script |

**Where UCI and HF differ, deliberately:**

| Variable | UCI `bank-full` documented levels | HF card levels | Explanation |
|---|---|---|---|
| `education` | `unknown`, `secondary`, `primary`, `tertiary` | 11 levels incl. `basic.4y` … `university.degree` | HF unions with `bank-additional`; codes 4–10 are absent here |
| `poutcome` | `unknown`, `other`, `failure`, `success` | 5 levels incl. `nonexistent` | `nonexistent` is a `bank-additional` level; code 4 has 0 rows here |
| `default`, `housing`, `loan` | binary `yes`/`no` | 3 levels incl. `unknown` | `unknown` comes from the union; 0 rows use it here |

None of these is a data error. They are consequences of the mirror's
category-union construction, and they are why the file's codes must be read
through the HF name lists rather than through UCI's category wording.

### 4.5 Pre-call availability: what is documented and what is assumed

| Variable | Basis for the "pre-call" judgement |
|---|---|
| `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` | **Documented** as client attributes; available before a call by construction |
| `previous`, `poutcome` | **Documented** as prior-campaign history |
| `pdays` | **Documented** as prior-campaign history; used only to derive `never_contacted` |
| `balance` | **Assumption.** UCI documents `balance` as "average yearly balance, in euros" but does **not** say when the snapshot is taken relative to the call. It is treated as pre-call because it is a client attribute; if it were captured after the call, it would be contaminated. Retained in A/B/C per approved decision, with the assumption recorded here and in `plan.md` §9 |
| `contact`, `day`, `month`, `campaign`, `duration` | **Documented** as properties of the current contact → excluded (see §9) |
| `y` | **Documented** outcome |

---

## 5. Special encodings and anomalies

### 5.1 The `pdays = 999` sentinel

**Documented + empirical.**

- UCI stores `pdays = -1` for "client was not previously contacted".
- The HF build script replaces `-1` with `999`
  (`df_bank["pdays"].replace(-1, 999)`).
- Therefore in this file **`999` means "never previously contacted"**. It is
  **not** missing data and **not** 999 days.

Counts (**empirical**, `07_pdays_sentinel_stats.csv`):

| Quantity | Value |
|---|---|
| Rows with `pdays = 999` | 36,954 (81.7367%) |
| Rows with an actual day count | 8,257 (18.2633%) |
| Actual days: min / median / mean / max | 1 / 194 / 224.5777 / 871 |
| Rows with negative `pdays` | 0 |

Consistency (**empirical**, `07b_pdays_consistency.csv`): the sentinel is
exactly equivalent to `previous == 0`.

| Check | Rows |
|---|---|
| `pdays == 999` and `previous == 0` | 36,954 |
| `pdays == 999` and `previous > 0` | 0 |
| `pdays != 999` and `previous == 0` | 0 |
| `pdays != 999` and `previous >= 1` | 8,257 |

Raw `pdays` statistics are uninterpretable (mean 857.57, SD 303.25 —
`06_describe_numeric.csv`), because 81.7% of the column is a code. This is why
`plan.md` v1.2 excludes `pdays` as a predictor.

### 5.2 `poutcome` codes and the derived `never_contacted` level

**File codes** (documented in the HF card, verified in §4.4):

| Code | Label | Rows | Source file |
|---|---|---|---|
| 0 | `failure` | 4,901 | `08_frequency_categorical.csv` |
| 1 | `other` | 1,840 | `08_frequency_categorical.csv` |
| 2 | `success` | 1,511 | `08_frequency_categorical.csv` |
| 3 | `unknown` | 36,959 | `08_frequency_categorical.csv` |
| 4 | `nonexistent` | **0** | `10_notes.txt` |

**Model levels** (**derived**, `plan.md` §5.3): because the file has no
"never contacted" level, `poutcome` is recoded to five levels:

| Derived level | Rule | Rows |
|---|---|---|
| `never_contacted` | `pdays == 999` | 36,954 |
| `unknown` | `poutcome == 3` **and** `pdays != 999` | 5 |
| `failure` | `poutcome == 0` | 4,901 |
| `other` | `poutcome == 1` | 1,840 |
| `success` | `poutcome == 2` | 1,511 |

The derived level is what carries the "never previously contacted" information
into Models B and C. It is **not** in the raw file, and it is **not** a
replacement for a `pdays` predictor: no `pdays` value is entered into any
model.

### 5.3 The five unusual observations

**Empirical**, `07c_poutcome_by_pdays_sentinel.csv`:

| `poutcome` | `pdays != 999` | `pdays = 999` |
|---|---|---|
| `failure` | 4,901 | 0 |
| `other` | 1,840 | 0 |
| `success` | 1,511 | 0 |
| `unknown` | **5** | 36,954 |

The five rows in the bottom-left cell are internally inconsistent with the
documented semantics: they are marked `poutcome = unknown` although `pdays`
shows the client **was** previously contacted. Possible explanations (a data
entry artefact, a prior contact whose outcome was never recorded) cannot be
distinguished from this file.

**Decision (`plan.md` §5.3):** the five rows are **preserved as `unknown`**.
They are not merged into `never_contacted`, not imputed, not corrected and not
dropped. Under the derived rule in §5.2 they stay in the `unknown` level, which
keeps them visible rather than silently absorbing them into another category.

### 5.4 Documented but absent codes

**Empirical**, `02_schema.csv` and `10_notes.txt`:

| Variable | Codes documented | Codes observed | Interpretation |
|---|---|---|---|
| `education` | 0–10 | 0–3 | Codes 4–10 (`basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`) belong to `bank-additional`; their absence is expected for `bank-full`, not a defect |
| `poutcome` | 0–4 | 0–3 | Code 4 `nonexistent` is a `bank-additional` level; 0 rows here |

### 5.5 Documented but unused `unknown` codes

**Empirical**, `04_missingness.csv`: `marital` (code 3), `default` (code 2),
`housing` (code 2) and `loan` (code 2) all document an `unknown` level that
**no observation uses**. The level is retained in the encoder for consistency
with the documented schema; it simply has zero support. No rows are affected.

---

## 6. Missing values, "unknown" categories and duplicates

### 6.1 Missing values

**Empirical**, `04_missingness.csv`: **0 NaN cells** in all 17 columns
(0.0%). UCI also declares "Has Missing Values? No". Missingness is therefore
not an issue in this file — but "no NaN" does **not** mean "no missing
information", because missingness is encoded as the `unknown` category (§6.2).

### 6.2 "unknown" categories are not missing values

**Empirical**, `04_missingness.csv`. `unknown` is a documented category, kept
as an explicit level and never imputed (`plan.md` §5.4).

| Variable | `unknown` code | Rows | Share |
|---|---|---|---|
| `job` | 11 | 288 | 0.6370% |
| `education` | 3 | 1,857 | 4.1074% |
| `contact` | 2 | 13,020 | 28.7983% |
| `poutcome` | 3 | 36,959 | 81.7478% |
| `marital` | 3 | 0 | 0.0% |
| `default` | 2 | 0 | 0.0% |
| `housing` | 2 | 0 | 0.0% |
| `loan` | 2 | 0 | 0.0% |

The `poutcome = unknown` share is large (81.75%) **by construction**: clients
never previously contacted have no prior outcome to report. The derived
`never_contacted` level (§5.2) separates 36,954 of those 36,959 rows from the
5 genuinely anomalous ones. This distinction matters — treating all 36,959 as
one "unknown" bucket would conflate "no prior campaign" with "prior campaign of
unrecorded outcome".

### 6.3 Duplicates

**Empirical**, `05_duplicates.txt`:

| Check | Result |
|---|---|
| Exact duplicate rows on all 17 columns | 0 |
| Duplicate rows on the 16 features (ignoring `y`) | 0 |
| Rows before / after | 45,211 / 45,211 (nothing dropped) |

No deduplication was applied or is needed. The absence of exact duplicates does
**not** rule out the same client appearing twice with different values; there
is no client identifier (§3.1).

---

## 7. Numeric quantities vs label-encoded categorical variables

This distinction is the most common source of analysis error in this dataset,
so it is stated explicitly.

| | Numeric quantities (7) | Label-encoded categoricals (10) |
|---|---|---|
| Columns | `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous` | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`, `y` |
| Storage | `int64` | `int64` — **same storage type, different meaning** |
| Arithmetic meaningful? | Yes (within documented units) | **No** |
| Ordering meaningful? | Yes (except the `pdays` sentinel) | **Only for `month`** (calendar order by explicit override); for all others the order is alphabetical or an artefact of the category union, and carries no meaning |
| Distance meaningful? | Yes (except the `pdays` sentinel) | **No** — `job = 10` is not "twice" `job = 5` |

`pdays` is the exception inside the numeric block: while the 999 sentinel is
present, neither its ordering nor its distances are meaningful (§5.1).

**Concrete pitfalls.**

- `df.describe()` on the full frame silently reports means and standard
  deviations of category codes. Those numbers are meaningless. The EDA script
  restricts descriptive statistics to the 7 numeric columns
  (`06_describe_numeric.csv`).
- A correlation matrix over all 17 columns would report correlations between
  arbitrary integers. Such a matrix is not reported anywhere in this project
  and should not be computed from the raw codes.
- A model fitted on the raw codes would impose a false linear ordering on
  `job`, `education`, `marital`, etc. The pipeline therefore one-hot encodes
  every categorical column, with levels learned from the training block only
  and `handle_unknown="ignore"` (`plan.md` §5.4).
- `month` is the one column where the integer order *is* meaningful, but that
  is a property of the HF override, not of the integers themselves. It is
  excluded from all models regardless (§9.3).

---

## 8. Data quality limitations

Each limitation is paired with its consequence and the mitigation actually in
place. Nothing here is corrected in the data.

| # | Limitation | Class | Consequence | Mitigation in this project |
|---|---|---|---|---|
| 1 | No year variable | Documented absence | Dec→Jan boundaries cannot be dated; exact dates unknown | Chronology adjudication (§10); year ambiguity carried into reporting |
| 2 | No client identifier | Documented absence | Possible repeated clients → observations may be non-independent | Documented (`plan.md` §9); no adjustment claimed |
| 3 | `balance` snapshot timing unconfirmed | **Assumption** | If post-call, `balance` would be contaminated | Retained in A/B/C as approved, with the assumption stated; flagged in §4.5 and §12 |
| 4 | `pdays = 999` sentinel | Documented transformation | Raw column unusable as a number | Excluded as a predictor; used only to derive `never_contacted` (§5.1, §5.2) |
| 5 | 5 rows with `poutcome = unknown` but `pdays != 999` | Empirical anomaly | Internally inconsistent prior-campaign record | Preserved as `unknown`; not merged, imputed or dropped (§5.3) |
| 6 | `education` codes 4–10 documented but absent | Empirical | Encoder has unused levels | Harmless; documented (§5.4) |
| 7 | `poutcome = nonexistent` (code 4) has 0 rows | Empirical | Unused level | Documented (§5.4) |
| 8 | `unknown` documented but unused in `marital`, `default`, `housing`, `loan` | Empirical | Unused levels | Retained; documented (§5.5) |
| 9 | Class imbalance (11.70% positive) | Empirical | ROC-AUC alone can look good while positive-class performance is poor | PR-AUC with set-specific baselines, Brier, lift, calibration (`plan.md` §5.7) |
| 10 | Prior-campaign variables are not randomly assigned | Design | Selection effects in who was contacted before | Associational scope only; no causal claims (`plan.md` §2) |
| 11 | External validity: one Portuguese bank, 2008–2010 | Documented | Results may not transfer to other markets or periods | Claims scoped to this dataset (`plan.md` §9) |
| 12 | Quasi-identifiers (`age`, `job`, `marital`, `education`, `balance`, `day`, `month`) | Empirical | Re-identification risk in combination; `balance` is financial data | Aggregates only; no row-level records published; no external joins (`results/explore_bank_marketing/11_leakage_and_privacy.md` §5) |
| 13 | UCI says "17 inputs"; file has 16 features + target | Documentation inconsistency | Ambiguity in feature counting | Resolved as 16 + 1 (§2) |
| 14 | HF card declares no licence | Documentation gap | Mirror's terms unverified | Rely on UCI CC BY 4.0 (§1.3) |
| 15 | HF category lists union `bank-full` with `bank-additional` | Documented construction | Codes are not the UCI category order | Verified mapping (§4.4); never inferred from apparent order |

---

## 9. Leakage risks

Leakage here means: **using information that is not available at the decision
point, in a way that inflates apparent performance.** The decision point is
`plan.md` §5.1 — immediately before dialling the client.

### 9.1 `duration` — the primary leakage risk

**Documented:** "last contact duration, in seconds". It is known only **after
the call ends**, and the target is realised at the end of that same call.

**Empirical** (`09_duration_by_target.csv`, descriptive only — no model):

| `y` | n | mean (s) | median (s) | min | max |
|---|---|---|---|---|---|
| 0 "no" | 39,922 | 221.18 | 164.0 | 0 | 4,918 |
| 1 "yes" | 5,289 | 537.29 | 426.0 | 8 | 3,881 |

The separation is large, which is exactly the problem: a model using `duration`
partly predicts the outcome it is trying to explain.

**Decision:** `duration` is excluded from Models A and B. Model C
(B + `duration`) exists **only** to quantify the inflation and is labelled
"diagnostic — leakage" in every table (`plan.md` §3.3, §5.6).

**Honest caveat:** Moro et al. (2014) include `duration` in their published
benchmarks, so the applied literature is not uniform. This project follows the
pre-call decision framing, and states that choice rather than presenting it as
the only defensible one.

### 9.2 `campaign` — contemporaneous, includes the current contact

**Documented:** "number of contacts performed during this campaign and for this
client, **includes last contact**". It is therefore not available before the
first call and changes with every attempt; it can correlate with the outcome
mechanically (clients who need many attempts may differ). Excluded from A/B/C.

### 9.3 `contact`, `day`, `month` — operational variables

These describe **how and when the current contact happened**, not the client's
state before it. `month` and `day` also encode campaign scheduling decisions.
Excluded from A/B/C as "campaign operations" (`plan.md` §5.2). They remain in
the file and are used for the chronology checks, which is a legitimate,
non-predictive use.

### 9.4 `y` — the outcome

Never a feature. Additionally, `y` was **not** loaded at all by the chronology
validation script (`columns_loaded` in `chronology_checks.json`), so the split
decision could not be influenced by the target.

### 9.5 Prior-campaign variables — selection, not leakage

`previous` and `poutcome` are legitimately pre-call. The risk is different:
previously contacted clients were selected by the bank's own campaign logic, so
these variables are not randomly assigned. Any association involving them is
descriptive, and no causal effect is estimated (`plan.md` §2).

### 9.6 Sentinel-as-number — an encoding trap, not classic leakage

Entering raw `pdays` would encode "never contacted" as "999 days ago" and
distort any coefficient. This is why the v1.2 plan removes `pdays` as a
predictor entirely (§5.1, §5.2).

### 9.7 No test-set leakage through iteration

The split is frozen and hashed before any model exists; the test block is
opened once. Preprocessing is fitted inside pipelines on the training block
only. These are audit criteria 3–5 in `plan.md` §8.2 and are verified by the
independent recomputation step.

---

## 10. Chronology, adjudication and the frozen split

### 10.1 What the file can and cannot tell us

**Documented:** UCI describes `bank-full.csv` as "ordered by date". **Empirical:**
the file has no year column, so row order is the only date-like signal. The
chronology protocol (`plan.md` §4.3) was therefore run **before** any model and
**without reading `y`**.

### 10.2 Automated checks

**Empirical**, `results/splits/chronology_checks.md` and
`results/splits/chronology_checks.json`:

| ID | Evidence class | Observed | Status |
|---|---|---|---|
| C1 Provenance | Documentary | HF build script has no sort/reorder call | DOCUMENTARY-PASS |
| C2 Identity | Empirical | 45,211 × 17; all 17 names match | PASS |
| C3 Month blocks | Direct ordering | **30 runs** (random-order reference 37,711; threshold ≤ 60) | PASS |
| C4 Transitions | Direct ordering | **1 forward skip**, 0 reversals | **INCONCLUSIVE** |
| C5 Day monotonicity | Direct ordering | **1.000000** over 45,181 adjacent pairs; tie-aware random reference 0.518455 | PASS |
| C6 Transformation | Empirical | `pdays == -1`: 0; 999 ⇔ `previous == 0`; `poutcome == 4`: 0 | PASS |
| C7 Year ambiguity | Unresolved | no year variable | LIMITATION |

**Automated verdict: `INCONCLUSIVE`.** It was not rewritten to say otherwise.

### 10.3 The single anomaly

There are **30 month blocks and therefore 29 transitions**: 28 consecutive
(+1) transitions and **one forward skip**.

- Block 4: `aug`, rows 18,823–24,037 (5,215 rows).
- Block 5: `oct`, rows 24,038–24,117 (80 rows).
- Transition: `aug → oct`, delta 2 mod 12, classified `skip`.
- September appears later, as block 16 (282 rows) and block 28 (297 rows).
- **Zero reversals** anywhere; within every block `day` is 100% non-decreasing.

The absent month is consistent with **September 2008 having zero contacts**:
the blocks run May→April then May→November, and the documented span
(May 2008 – November 2010) contains 31 calendar months against 30 observed
blocks. Without a year variable, the identity of the missing September cannot
be *proved*, only inferred.

### 10.4 Human adjudication

Recorded in `results/splits/chronology_adjudication.md` (2026-11-13). The
project author chose **option (a): accept the chronological ordering for the
primary split**, on the basis of:

1. UCI's statement that `bank-full.csv` is ordered by date.
2. The HF transformation containing no reordering step.
3. C3 (30 blocks vs 37,711 expected under random order) and C5 (100%
   non-decreasing days) being strongly consistent with date order.
4. The single C4 anomaly being a **forward skip**, not a reversal, and
   consistent with a zero-contact month.

This is an **evidence-based adjudication, not absolute proof of exact calendar
dates**. The automated verdict remains `INCONCLUSIVE` on record, and the
adjudication document is additive: it explains why a human accepted the
ordering, and it discloses in full that the first C4 implementation wrongly
treated any skip as a hard failure (returning "NOT CHRONOLOGICAL"), that this
was corrected to separate skips from reversals, and that the script was re-run.
No threshold was changed (C3 ≤ 60 runs; C5 ≥ 0.99 share).

### 10.5 The frozen split

**Empirical**, `results/splits/temporal_split_v1.json`, generated by
`src/my_analysis/create_temporal_split.py` using the exact boundaries
pre-registered in `plan.md` v1.1 §5.5 and preserved in v1.2:

| Split | Row-index range | n | Share |
|---|---|---|---|
| train | `[0, 31,647)` | 31,647 | 69.9985% |
| validation | `[31,647, 38,428)` | 6,781 | 14.9986% |
| test | `[38,428, 45,211)` | 6,783 | 15.0030% |

- Dataset SHA-256:
  `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f`
- Split SHA-256:
  `d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c`
  (hash of the canonical JSON of the exact index assignment; method recorded in
  the file)
- Verified: no duplicate indices; every row covered exactly once (union =
  45,211); each index list strictly increasing; train < validation < test by row
  index; sizes match the plan.
- Created **after** the chronology checks and **before** any model run. Both
  execution runs load this file and never regenerate it.

### 10.6 What the split does and does not license

- **Licensed:** a chronological (temporal hold-out) evaluation, with the year
  ambiguity disclosed.
- **Not licensed:** claims about exact calendar dates, seasonality, or
  macroeconomic change; claims that the test block is a random sample of the
  population; any statement that the ordering is *proven*.
- The secondary stratified random split (seed 42) is a clearly-labelled
  optimistic sensitivity analysis, not a replacement for the primary result.

---

## 11. Supporting results files

Every number in this document is traceable to one of these files.

| Claim | File | Producing script |
|---|---|---|
| Shape 45,211 × 17, hash, path | `results/explore_bank_marketing/01_shape.txt` | `explore_bank_marketing.py` |
| dtypes, roles, observed vs documented codes | `results/explore_bank_marketing/02_schema.csv` | `explore_bank_marketing.py` |
| Target counts and percentages | `results/explore_bank_marketing/03_target.csv` | `explore_bank_marketing.py` |
| Target code verification | `results/explore_bank_marketing/03_target_verification.txt` | `explore_bank_marketing.py` |
| NaN and `unknown` counts, sentinel counts | `results/explore_bank_marketing/04_missingness.csv` | `explore_bank_marketing.py` |
| Duplicate checks, rows before/after | `results/explore_bank_marketing/05_duplicates.txt` | `explore_bank_marketing.py` |
| Numeric descriptive statistics and ranges | `results/explore_bank_marketing/06_describe_numeric.csv` | `explore_bank_marketing.py` |
| `pdays` sentinel and actual-day statistics | `results/explore_bank_marketing/07_pdays_sentinel_stats.csv` | `explore_bank_marketing.py` |
| Sentinel ⇔ `previous == 0` consistency | `results/explore_bank_marketing/07b_pdays_consistency.csv` | `explore_bank_marketing.py` |
| `poutcome` × sentinel cross-tab (5 anomalies) | `results/explore_bank_marketing/07c_poutcome_by_pdays_sentinel.csv` | `explore_bank_marketing.py` |
| Category frequencies and labels | `results/explore_bank_marketing/08_frequency_categorical.csv` | `explore_bank_marketing.py` |
| `duration` by target (descriptive) | `results/explore_bank_marketing/09_duration_by_target.csv` | `explore_bank_marketing.py` |
| Anomaly notes, absent/unused codes | `results/explore_bank_marketing/10_notes.txt` | `explore_bank_marketing.py` |
| Validation checks (0 failures) | `results/explore_bank_marketing/10_validation_checks.csv` | `explore_bank_marketing.py` |
| Leakage and privacy assessment | `results/explore_bank_marketing/11_leakage_and_privacy.md` | `explore_bank_marketing.py` |
| Consolidated EDA report | `results/explore_bank_marketing/REPORT.md` | `explore_bank_marketing.py` |
| Run manifest: hash, versions, no filters | `results/explore_bank_marketing/manifest.json` | `explore_bank_marketing.py` |
| Chronology checks C1–C7 | `results/splits/chronology_checks.md`, `.json` | `validate_chronology.py` |
| Month blocks and transitions | `results/splits/chronology_month_blocks.csv` | `validate_chronology.py` |
| Within-block day monotonicity | `results/splits/chronology_day_monotonicity.csv` | `validate_chronology.py` |
| Human adjudication of the INCONCLUSIVE verdict | `results/splits/chronology_adjudication.md` | (hand-written record) |
| Frozen split indices, sizes, hash, verification | `results/splits/temporal_split_v1.json` | `create_temporal_split.py` |
| Independent recomputation of shape/target/sentinel counts | `results/verification/initial_independent_checks.md` | (independent, hand-run) |

Category frequencies are **not** duplicated in this document; they live in
`08_frequency_categorical.csv` and `03_target.csv`. Only the counts needed to
explain an encoding or an anomaly are quoted inline, always with the source
file named.

---

## 12. Summary: documented, measured, assumed

### Documented facts

- UCI `bank-full.csv`: 45,211 rows, 16 features + target, date-ordered,
  no missing values, Portuguese bank, May 2008 – November 2010 (period stated
  for `bank-additional-full`, inherited here by inference).
- CC BY 4.0 licence; Moro, Rita & Cortez (2014); DOI 10.24432/C5K306.
- HF transformation: `pdays: -1 → 999`, categorical columns converted to
  `ClassLabel` from a union of `bank-full` and `bank-additional` category lists,
  `month` forced to calendar order, no sort/reorder call.
- Variable meanings, units and the `divorced` = divorced-or-widowed note.

### Empirical findings (from this file, in `results/`)

- Shape 45,211 × 17; 0 NaN; 0 exact duplicates.
- Target 39,922 / 5,289 (88.3015% / 11.6985%).
- `pdays = 999` in 36,954 rows (81.7367%), exactly equivalent to
  `previous == 0`.
- `poutcome`: 4,901 failure / 1,840 other / 1,511 success / 36,959 unknown;
  `nonexistent` 0 rows.
- 5 rows with `poutcome = unknown` and `pdays != 999`.
- `education` uses only codes 0–3; codes 4–10 absent.
- 30 month blocks, 29 transitions, 28 consecutive + 1 forward skip (Aug→Oct);
  100% within-block day monotonicity.
- All observed categorical codes are valid indices into the documented HF
  name lists (0 undocumented).

### Unresolved assumptions and uncertainties

- `balance` is **assumed** to be captured before the call; the documentation
  does not confirm this.
- `Dataset.from_pandas` is **assumed** to preserve row order.
- The missing September is **inferred** to be September 2008 with zero
  contacts; no year variable exists to confirm it.
- The five anomalous `poutcome` rows have **no documented explanation**.
- Whether clients repeat across rows is **unknowable** without a client ID.
- The HF mirror declares **no licence**; only the upstream UCI licence is
  verified.
- The exact volume/pages of the Moro et al. (2014) journal article were **not
  verified** from the sources consulted.

---

## 13. Change log

| Date | Change |
|---|---|
| 2026-11-13 | Initial version. Replaces the placeholder template. Aligned with `plan.md` v1.2: `pdays` not a predictor; `poutcome = never_contacted` derived; five anomalies preserved as `unknown`; `balance` retained with availability recorded as an assumption; codebook verified against the HF dataset card and UCI documentation; chronology adjudication and frozen split documented. |
