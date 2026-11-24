# Guided random-split protocol discrepancy — root-cause audit

**Date:** 2026-11-14  
**Status:** Erroneous run archived; corrected split generated in split-only mode; no models fitted.  
**Author:** OpenCode agent (`opencode/big-pickle`), guided workspace only.

> **Status update (2026-11-16).** The **Status** line above, and §9 below,
> describe the split-only state at the time of this audit. They are retained as
> historical record and are no longer current. The corrected evaluation was
> subsequently authorised (PHASE 4) and executed on 2026-11-14T11:26:00Z by
> `src/my_analysis/guided_random_split_sensitivity_evaluate.py`; its artifacts
> are under `results/sensitivity_corrected_v1/` (`report_evaluation.md`,
> `run_manifest_evaluation.json`, `metrics.csv`, `bootstrap.json`). Models were
> fitted and evaluated on the corrected random blocks after that authorisation.
> The original erroneous run remains preserved and labelled invalid under
> `results/sensitivity_erroneous_v1/`. Nothing else in this document has been
> altered.

---

## 1. Summary

The guided random-split sensitivity run archived at
`results/sensitivity_erroneous_v1/` implemented a **train-first** two-step
split. The authorized procedure was **test-first**. The block sizes were
correct, and the split was internally reproducible, but the index membership
was not the authorized membership. The error was found by independent human
regeneration and comparison against the saved guided and autonomous splits.

The erroneous run is preserved byte-for-byte as evidence and is labelled:

> **INVALID FOR THE AUTHORIZED SENSITIVITY ANALYSIS — PRESERVED AS AGENT ERROR EVIDENCE.**

No model has been fitted on the corrected split. The corrected script currently
supports only `--mode split-only`; model fitting and evaluation require a
separate authorization after human verification of the corrected index sets.

---

## 2. Authorized procedure

The exact authorized implementation is:

```python
remaining_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=6783,
    stratify=y,
    random_state=42
)

train_idx, val_idx = train_test_split(
    remaining_idx,
    test_size=6781,
    stratify=y[remaining_idx],
    random_state=42
)
```

`train_test_split` returns `(train, test)` in that order. The first call
therefore returns the 38,428-row remainder as `remaining_idx` and the
6,783-row held-out block as `test_idx`. The second call returns the 31,647-row
training block as `train_idx` and the 6,781-row validation block as `val_idx`.

Required final sizes: **31,647 training / 6,781 validation / 6,783 test**.

---

## 3. Implemented erroneous procedure

The erroneous script
`src/my_analysis/guided_random_split_sensitivity.py` (archived at
`results/sensitivity_erroneous_v1/guided_random_split_sensitivity.py`)
implemented:

```python
idx = np.arange(len(y))
train_idx, temp_idx = train_test_split(
    idx,
    test_size=N_VAL + N_TEST,   # 6_781 + 6_783 = 13_564
    stratify=y,
    random_state=SEED,
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=N_TEST,           # 6_783
    stratify=y[temp_idx],
    random_state=SEED,
)
```

Relevant source lines in the erroneous script:

- Lines 339–363: `generate_random_split`.
- Lines 342–343: docstring explicitly says “Step 1 removes the training block
  (31,647 rows), leaving 13,564 rows. Step 2 divides those 13,564 rows into
  validation (6,781) and test (6,783).”
- Lines 347–352: first call uses `test_size=N_VAL + N_TEST`.
- Lines 353–358: second call uses `test_size=N_TEST`.
- Lines 592–598: `build_split_document` records the same wrong order:
  `test_size=13564` first, then `test_size=6783`.

The erroneous split hash was:

`e43308ca7a0485fdf114be43372be3c0cc07017897ed766895de47e02f6eca3b`

The erroneous script SHA-256 was:

`3806a2b0eeedbfcd480005f754bf345417fc62a228b9d4a91fc9063d0485cc54`

### Why the two procedures differ

The first partition is different:

- Authorized: hold out 6,783 test rows first; remainder = 38,428 rows.
- Implemented: hold out 13,564 rows first; remainder = 31,647 training rows.

`train_test_split` with `stratify=y` uses a stratified shuffle whose exact
membership depends on the requested `test_size` and the seed. Changing the
first `test_size` from 6,783 to 13,564 changes which rows are held out. The
second call then operates on a different remainder (38,428 vs 13,564) and uses
a different `test_size` (6,781 vs 6,783). The final block sizes are the same,
but the index sets are different.

---

## 4. Independent human discovery

The project author independently regenerated the authorized test-first
procedure and compared its index sets with the saved guided and autonomous
sensitivity splits.

Verified findings:

- Autonomous matches the authorized procedure: **TRUE** for train, validation
  and test.
- Guided matches the authorized procedure: **FALSE** for train, validation and
  test.
- Guided versus autonomous shared rows:
  - Training: **26,036 of 31,647**
  - Validation: **563 of 6,781**
  - Test: **3,381 of 6,783**
- Random test overlap with the chronological test block:
  - Guided: **1,008**
  - Autonomous and independently generated reference: **1,022**

These findings are consistent with the code-level difference described above.
The agent did not access the autonomous workspace during this audit.

---

## 5. Why the agent’s checks failed to detect nonconformance

`verify_random_split` checked only:

- exact block sizes;
- complete coverage of all 45,211 rows;
- pairwise disjointness;
- reproducibility by calling `generate_random_split(y)` again and comparing the
  result with itself.

That is a reproducibility check, not a conformance check. It confirms that the
same wrong function returns the same wrong split on repeated calls. The
manifest therefore recorded `reproducible_from_seed: true`, which was true but
did not mean the split matched the authorized procedure.

The narrative artifacts also omitted the step order:

- `execution_log.md`: “Method: `train_test_split`, stratified on `y`, seed 42.”
- `report.md` §8: “Two-step stratified `train_test_split` with seed 42 and
  exact sizes …”
- `prompt_log.md`: quoted the authorization message, which says “Follow the
  protocol you just proposed,” but did not quote the exact two-step order.

The split JSON did record the wrong order, but it was produced by the same
wrong code and was not compared against an independent reference.

### Additional error in the proposed correction snippet

The read-only root-cause audit proposed a correction snippet with the return
assignments reversed:

```python
# WRONG — reversed sklearn return order
test_idx, remaining_idx = train_test_split(...)
val_idx, train_idx = train_test_split(...)
```

`train_test_split` returns `(train, test)`, so that snippet would have assigned
the 38,428-row training remainder to `test_idx` and the 6,783-row held-out
block to `remaining_idx`. The human project author caught this before
execution. The corrected script uses the authorized order:

```python
remaining_idx, test_idx = train_test_split(...)
train_idx, val_idx = train_test_split(...)
```

This second error is documented here rather than minimized.

---

## 6. Affected sensitivity outputs

All files in the erroneous `results/sensitivity/` run are affected:

| Artifact | Effect |
|---|---|
| `random_split_v1.json` | Wrong indices, split hash, class counts, overlap and generation steps. |
| `random_split_v1.md` | Wrong split summary. |
| `metrics.csv` | Metrics computed on the wrong random test block. |
| `predictions.csv` | Predictions for the wrong random test block. |
| `calibration.csv` | Calibration on the wrong random test block. |
| `bootstrap.json` | Bootstrap on the wrong random test block. |
| `report.md` | Wrong split and all derived sensitivity results. |
| `run_manifest_sensitivity.json` | Wrong split hash, class counts, overlap, model calls and artifact hashes. |
| `console_output.txt` | Wrong split and metric log. |
| `execution_log.md` | Wrong split hash and results; omitted the step order. |
| `prompt_log.md` | Quoted the authorization but not the exact two-step procedure. |
| `guided_random_split_sensitivity.py` | Contained the train-first split function. |

None of these files are valid for the authorized sensitivity analysis.

---

## 7. Primary chronological analysis is unchanged

The primary chronological analysis was not modified. Evidence:

- The erroneous run’s manifest records 69 tracked files before and after, with
  no added, removed or changed files and
  `unexpected_modification: false`.
- The erroneous script verified the frozen chronological split hash at startup:
  `d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c`.
- The erroneous script writes only under `results/sensitivity/`.
- It read `results/guided/test/test_metrics.csv` and
  `results/guided/test/hypothesis_tests_test.json` only for the descriptive
  comparison; it did not write to them.
- The raw dataset hash was verified as
  `0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f`.

The corrected split-only run also verifies that the primary artifacts and the
erroneous archive are unchanged.

---

## 8. Evidence archive

- Archive: `results/sensitivity_erroneous_v1/`
- Banner: `ARCHIVE_NOTICE.md`
- Inventory: `SHA256SUMS.txt`
- Archive summary: `archive_manifest.json`
- Archived original files: 11 sensitivity files plus the exact erroneous
  script, all verified byte-for-byte against their sources.
- Original `results/sensitivity/` directory: not deleted, not overwritten, not
  altered.

---

## 9. Corrected split-only status

- Corrected script:
  `src/my_analysis/guided_random_split_sensitivity_corrected.py`
- Corrected outputs:
  `results/sensitivity_corrected_v1/`
- Mode executed: `--mode split-only`
- Models fitted: **none**
- Predictions generated: **none**
- Metrics computed: **none**
- Bootstrap executed: **none**
- Human verification of exact corrected index sets is required before any model
  fitting is authorized.

The corrected run records exact equality against a separately implemented
authorized reference, exact positive counts (3,702 / 793 / 794), complete
coverage, pairwise disjointness, and random-test overlap with the chronological
test block of exactly 1,022 rows.

---

## 10. References

- `results/sensitivity_erroneous_v1/ARCHIVE_NOTICE.md`
- `results/sensitivity_erroneous_v1/SHA256SUMS.txt`
- `results/sensitivity_corrected_v1/random_split_v1.json`
- `results/sensitivity_corrected_v1/conformance_check.json`
- `results/sensitivity_corrected_v1/execution_log.md`
- `mistakes.md` entry A-5
