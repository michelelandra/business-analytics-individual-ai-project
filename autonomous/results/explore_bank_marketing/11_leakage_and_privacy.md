# Data leakage and privacy assessment

Script: `src/my_analysis/explore_bank_marketing.py`
Generated: 2026-11-13T18:11:37+00:00
Every number below is produced by `src/my_analysis/explore_bank_marketing.py` and stored in this folder.

## 1. `duration` — post-call information (leakage for pre-call targeting)

`duration` is the duration of the last contact, in seconds. It is only known
**after the call has ended**. For any model intended to decide *whom to call*,
using `duration` leaks the outcome: the target is realised at the end of the
same call whose length is measured.

Descriptive evidence (not a model) — `09_duration_by_target.csv`:
- y=0 ("no"): n=39,922, median duration 164s
- y=1 ("yes"): n=5,289, median duration 426s

Decision: `duration` may be described, but it is **excluded from future
pre-call predictive models**. Caveat recorded honestly: Moro et al. (2014)
benchmark `duration` in their published models, so the applied literature is
not uniform on this point. We follow the pre-call decision framing.

## 2. `pdays` — sentinel, not a measurement

In this HF mirror the builder replaced UCI's `-1` with `999`
(https://huggingface.co/datasets/cestwc/bank-marketing). Therefore:
- `999` means "client was not previously contacted" — a **sentinel**, not
  missing data and not 999 days.
- Frequency: 36,954 of 45,211 rows (81.7367%) —
  `07_pdays_sentinel_stats.csv`.
- Actual-day statistics (mean/median/etc.) exclude 999 and are computed on a
  temporary view only; no observation is modified or dropped.
- Using raw `pdays` as a continuous feature would encode "never contacted" as
  "999 days ago" and distort any model. A future step (not done here) would
  need an explicit `was_previously_contacted` indicator plus a days-since
  variable with an explicit missing-by-design code.

## 3. `campaign` — contemporaneous contact count

`campaign` counts contacts during the current campaign *including the last
contact*. It is known at contact time but is partly an artefact of how many
attempts the client needed, so it can correlate with the outcome mechanically.
Flagged; not excluded at this stage.

## 4. Safe pre-call candidates (tentative)

Client attributes (`age`, `job`, `marital`, `education`, `default`, `balance`,
`housing`, `loan`) and prior-campaign history (`previous`, `poutcome`, and a
properly recoded `pdays`) are knowable before the call. `day`/`month` describe
when the contact happened and may encode campaign scheduling; they are
pre-call only in a narrow sense and should be treated with care.

## 5. Privacy

- No direct identifiers in the schema: no names, IDs, phone numbers, addresses
  (`02_schema.csv`).
- Quasi-identifiers exist: `age`, `job`, `marital`, `education`, `balance`,
  `day`, `month`. In combination these can be highly distinctive.
- This report contains **univariate aggregates only** — no row-level records
  and no small-cell cross-tabs. If cross-tabs are added later, suppress cells
  with fewer than 10 observations.
- `balance` is financial information and should not be joined to any external
  data.
- Licence: CC BY 4.0 (https://archive.ics.uci.edu/dataset/222/bank+marketing); cite Moro, Rita & Cortez
  (2014), DOI https://doi.org/10.24432/C5K306.
