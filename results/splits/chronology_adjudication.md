# Chronology adjudication

Date: 2026-11-13
Adjudicator: project author (human)
Related files (preserved unchanged):
`results/splits/chronology_checks.md`,
`results/splits/chronology_checks.json`,
`results/splits/chronology_month_blocks.csv`,
`results/splits/chronology_day_monotonicity.csv`.
Script that produced them: `src/my_analysis/validate_chronology.py`.

This document records a human adjudication of an **INCONCLUSIVE** automated
verdict. It does **not** rewrite or upgrade the original evidence: those files
still report `INCONCLUSIVE` and must be read together with this document.

---

## 1. The original verdict

The pre-registered chronology validation (plan.md §4.3) returned:

| Check | Evidence class | Observed | Status |
|---|---|---|---|
| C1 Provenance | documentary | No sort/reorder call in the HF build script | DOCUMENTARY-PASS |
| C2 Identity | empirical | 45,211 × 17, names match | PASS |
| C3 Month blocks | direct ordering | 30 runs (random reference 37,711; threshold ≤ 60) | PASS |
| C4 Transitions | direct ordering | 1 forward skip, 0 reversals | **INCONCLUSIVE** |
| C5 Day monotonicity | direct ordering | 1.000000 over 45,181 pairs; tie-aware random reference 0.518455 | PASS |
| C6 Transformation | empirical | `pdays==-1`: 0; 999 ⇔ `previous==0`; `poutcome==4`: 0 | PASS |
| C7 Year ambiguity | unresolved | no year variable | LIMITATION |

**Automated verdict: `INCONCLUSIVE`** (`chronology_checks.json`,
`generated_utc` 2026-11-13T19:01:35+00:00).

## 2. The August-to-October forward skip

The month sequence contains **30 blocks and therefore 29 transitions**:
**28 consecutive (+1) transitions** and **one forward skip**.

- Block 4: `aug`, rows 18,823–24,037 (5,215 rows).
- Block 5: `oct`, rows 24,038–24,117 (80 rows).
- Transition: `aug -> oct`, delta = 2 mod 12, classified as `skip`.
- **No reversal** occurs anywhere in the sequence.
- September appears later as block 16 (282 rows) and block 28 (297 rows).
- Within every block, `day` is non-decreasing (45,181 / 45,181 adjacent pairs).

The absent month is consistent with **September 2008 having zero contacts**:
the sequence runs May→April in the first cycle and May→November in the second,
and the documented span May 2008 – November 2010 contains exactly 31 calendar
months against 30 observed blocks.

## 3. Supporting documentation

### UCI Machine Learning Repository — Bank Marketing

URL: https://archive.ics.uci.edu/dataset/222/bank+marketing
(accessed 2026-11-13)

Relevant statement: `bank-full.csv` is described as the dataset with
"all examples and 17 inputs, **ordered by date** (older version of this dataset
with less inputs)". The documented period is May 2008 – November 2010.

This is the primary documentary basis for expecting chronological row order.

### Hugging Face dataset card — `cestwc/bank-marketing`

URL: https://huggingface.co/datasets/cestwc/bank-marketing
(accessed 2026-11-13)

The published build script:

```python
df_bank = pd.read_csv("bank+marketing/bank/bank-full.csv", sep=";")
...
df_bank["pdays"] = df_bank["pdays"].replace(-1, 999)
...
hf_dataset = DatasetDict({
    "train": Dataset.from_pandas(df_bank, features=hf_features),
})
```

No sorting, reindexing or reordering call appears. `Dataset.from_pandas` is
assumed to preserve row order. This shows the transformation did not
deliberately reorder rows; it does **not** independently prove the source file
was date-ordered.

### Dataset paper

Moro, S., Rita, P., & Cortez, P. (2014). *A data-driven approach to predict the
success of bank telemarketing*. Decision Support Systems.
DOI: https://doi.org/10.24432/C5K306

## 4. Human decision

**Decision: option (a) — accept the chronological ordering for the primary
split.**

Basis:

1. UCI documents `bank-full.csv` as ordered by date.
2. The Hugging Face transformation contains no reordering step.
3. C3 (30 blocks vs 37,711 expected under random order) and C5 (100%
   non-decreasing days) are strongly consistent with date ordering.
4. The single C4 anomaly is a **forward skip**, not a reversal, and is
   consistent with a zero-contact month.

This is an **evidence-based adjudication, not absolute proof of exact calendar
dates**. The original automated verdict remains `INCONCLUSIVE`; this document
records why the human reviewer accepted the ordering anyway.

## 5. Remaining uncertainty

- **No year variable exists.** December→January boundaries cannot be dated, and
  the identity of the missing September as September 2008 is an inference from
  the documented span and block counts, not a directly observed fact.
- The zero-contact explanation for the skip is plausible but unproven.
- A different ordering that happens to produce contiguous month blocks cannot
  be ruled out with certainty, though it is not supported by any evidence here.
- Consequently, the split is treated as **chronological up to year ambiguity**,
  and this limitation is carried into all downstream reporting.

## 6. Initial C4 implementation error and correction (full disclosure)

1. **First implementation.** C4 was coded as `PASS` only when there were zero
   non-`+1` transitions. This ignored the frozen exemption "violations beyond
   months with zero contacts". The first execution therefore returned
   `NOT CHRONOLOGICAL`.
2. **Error identified.** The single violation is a forward skip, not a
   reversal, and the frozen criterion explicitly allows zero-contact months.
   Treating every skip as a hard failure overstated the evidence and did not
   implement plan.md §4.3 faithfully.
3. **Correction.** C4 was rewritten to distinguish `skip` from `reversal`, to
   check whether skipped months appear elsewhere, and to return
   `INCONCLUSIVE` when the exemption cannot be resolved from the data. No
   threshold changed (C3 ≤ 60 runs; C5 ≥ 0.99 share).
4. **Additional execution.** The script was re-run after the correction,
   producing the preserved outputs that report `INCONCLUSIVE`. A later
   cosmetic grammar fix in the generated report prompted one further run; it
   changed no result.
5. **No concealment.** The original evidence files were not rewritten to make
   the checks appear to have passed. This document is additive.

## 7. Split created under this adjudication

`results/splits/temporal_split_v1.json`, generated by
`src/my_analysis/create_temporal_split.py`, using the exact frozen boundaries
of plan.md v1.1 §5.5:

| Split | Row-index range | n | Share |
|---|---|---|---|
| train | [0, 31,647) | 31,647 | 69.9985% |
| validation | [31,647, 38,428) | 6,781 | 14.9986% |
| test | [38,428, 45,211) | 6,783 | 15.0030% |

- Dataset SHA-256:
  `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f`
- Split SHA-256:
  `d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c`
- Verification: no duplicates; full coverage exactly once; each index list
  strictly increasing; train < validation < test by row index.
- No target values were read; no model was fitted; no performance metric was
  computed.
