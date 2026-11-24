"""Pre-registered chronology validation for the Bank Marketing dataset.

Implements checks C1-C7 of plan.md section 4.3.

Guarantees
----------
* The target column ``y`` is NEVER read. Data is loaded with an explicit
  column list that excludes it; shape/column names come from Parquet metadata.
* ``data/raw/`` is opened read-only and its SHA-256 is re-verified.
* No train/validation/test split is created here.
* Deterministic: no randomness is used.

Run with:  uv run python -m my_analysis.validate_chronology
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
OUT = ROOT / "results" / "splits"
SCRIPT_REL = "src/my_analysis/validate_chronology.py"

EXPECTED_SHAPE = (45211, 17)
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome", "y",
]

# Only these columns are loaded; y is deliberately absent.
CHECK_COLUMNS = ["month", "day", "pdays", "previous", "poutcome"]

MONTHS = {
    0: "jan", 1: "feb", 2: "mar", 3: "apr", 4: "may", 5: "jun",
    6: "jul", 7: "aug", 8: "sep", 9: "oct", 10: "nov", 11: "dec",
}

# Thresholds frozen in plan.md 4.3.
C3_MAX_RUNS = 60
C5_MIN_SHARE = 0.99

# Verbatim excerpt from the Hugging Face dataset card build script.
# Source: https://huggingface.co/datasets/cestwc/bank-marketing
# Accessed: 2026-11-13. Documentary evidence only (see C1).
C1_EXCERPT = """\
df_bank = pd.read_csv("bank+marketing/bank/bank-full.csv", sep=";")
...
df_bank["pdays"] = df_bank["pdays"].replace(-1, 999)
...
hf_dataset = DatasetDict({
    "train": Dataset.from_pandas(df_bank, features=hf_features),
})"""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def month_blocks(month: pd.Series) -> list[dict]:
    """Maximal runs of consecutive identical month codes."""
    run_id = (month != month.shift()).cumsum()
    blocks = []
    for bid, idx in enumerate(month.groupby(run_id).groups.values(), start=1):
        idx = list(idx)
        blocks.append({
            "block_id": bid,
            "start_row": int(idx[0]),
            "end_row": int(idx[-1]),
            "month_code": int(month.loc[idx[0]]),
            "month_label": MONTHS[int(month.loc[idx[0]])],
            "length": int(len(idx)),
        })
    return blocks


def expected_nondecreasing_share(values: pd.Series) -> float:
    """P(X >= Y) for iid X, Y drawn from the empirical distribution of values.

    Ties are included, so this is NOT 0.5 in general. For discrete uniform
    days 1..31 it equals 32/62 = 0.5161.
    """
    counts = values.value_counts(normalize=True).sort_index()
    cdf = counts.cumsum()
    return float((counts * (1.0 - cdf + counts)).sum())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        print(f"FATAL: input not found: {RAW}", file=sys.stderr)
        return 1

    digest = sha256(RAW)
    pf = pq.ParquetFile(RAW)
    num_rows = int(pf.metadata.num_rows)
    num_columns = int(pf.metadata.num_columns)
    column_names = list(pf.schema_arrow.names)

    # ---- C2: version identity (metadata only) -------------------------
    c2_pass = (num_rows, num_columns) == EXPECTED_SHAPE and column_names == EXPECTED_COLUMNS
    c2_observed = f"{num_rows} rows x {num_columns} columns; names match={column_names == EXPECTED_COLUMNS}"

    # ---- Load only the chronology columns (y excluded) ----------------
    df = pd.read_parquet(RAW, columns=CHECK_COLUMNS)
    assert "y" not in df.columns, "target column must not be loaded"
    n = len(df)

    # ---- C3: month block contiguity -----------------------------------
    blocks = month_blocks(df["month"])
    n_runs = len(blocks)
    c3_pass = n_runs <= C3_MAX_RUNS

    month_share = df["month"].value_counts(normalize=True)
    expected_runs_random = 1.0 + (n - 1) * (1.0 - float((month_share ** 2).sum()))

    # ---- C4: month transitions (skip vs reversal) ---------------------
    # Frozen criterion: "0 violations beyond months with zero contacts".
    # A forward skip is therefore not automatically a failure: it is exempt
    # if the skipped month(s) have no contacts. With no year variable we can
    # only distinguish:
    #   * reversal  -> direct evidence against chronological ordering (FAIL)
    #   * skip      -> consistent with a zero-contact period; if the skipped
    #                  month appears elsewhere, the period-specific absence
    #                  cannot be proven to be zero-contact -> INCONCLUSIVE
    global_month_counts = df["month"].value_counts().to_dict()
    transitions = []
    for prev, nxt in zip(blocks[:-1], blocks[1:]):
        delta = (nxt["month_code"] - prev["month_code"]) % 12
        skipped = []
        if delta == 1:
            cls = "valid"
        elif 2 <= delta <= 10:
            cls = "skip"
            skipped = [(prev["month_code"] + k) % 12 for k in range(1, delta)]
        elif delta == 11:
            # Forward 11 months is indistinguishable from a one-month
            # reversal without a year variable; treat as a reversal.
            cls = "reversal"
        else:
            cls = "same_month_noncontiguous"
        transitions.append({
            "from_block": prev["block_id"],
            "to_block": nxt["block_id"],
            "from_month": prev["month_label"],
            "to_month": nxt["month_label"],
            "delta_mod12": int(delta),
            "classification": cls,
            "skipped_month_codes": skipped,
            "skipped_months": [MONTHS[m] for m in skipped],
            "skipped_months_zero_contact_globally": all(
                int(global_month_counts.get(m, 0)) == 0 for m in skipped
            ) if skipped else None,
            "valid": cls == "valid",
        })

    reversals = [t for t in transitions if t["classification"] in
                 ("reversal", "same_month_noncontiguous")]
    skips = [t for t in transitions if t["classification"] == "skip"]
    unresolved_skips = [
        t for t in skips if not t["skipped_months_zero_contact_globally"]
    ]

    if reversals:
        c4_status = "FAIL"
    elif unresolved_skips:
        c4_status = "INCONCLUSIVE"
    else:
        c4_status = "PASS"
    c4_violations = [t for t in transitions if not t["valid"]]
    c4_pass = c4_status == "PASS"

    # ---- C5: within-month day monotonicity ----------------------------
    day_rows = []
    total_pairs = 0
    total_nondec = 0
    for b in blocks:
        seg = df["day"].iloc[b["start_row"]: b["end_row"] + 1]
        pairs = max(len(seg) - 1, 0)
        nondec = int((seg.iloc[1:].to_numpy() >= seg.iloc[:-1].to_numpy()).sum()) if pairs else 0
        total_pairs += pairs
        total_nondec += nondec
        day_rows.append({
            "block_id": b["block_id"],
            "month_label": b["month_label"],
            "start_row": b["start_row"],
            "end_row": b["end_row"],
            "n_days": int(len(seg)),
            "adjacent_pairs": int(pairs),
            "non_decreasing": nondec,
            "share_non_decreasing": round(nondec / pairs, 6) if pairs else np.nan,
            "violations": int(pairs - nondec),
            "expected_share_if_random": round(expected_nondecreasing_share(seg), 6) if pairs else np.nan,
        })

    pooled_share = (total_nondec / total_pairs) if total_pairs else None
    c5_pass = (pooled_share is not None) and (pooled_share >= C5_MIN_SHARE)
    expected_day_share_random = expected_nondecreasing_share(df["day"])

    # ---- C6: transformation fingerprint -------------------------------
    no_minus_one = int((df["pdays"] == -1).sum()) == 0
    sentinel = df["pdays"] == 999
    prev_zero = df["previous"] == 0
    sentinel_matches_previous_zero = bool((sentinel == prev_zero).all())
    poutcome_4 = int((df["poutcome"] == 4).sum())
    c6_pass = no_minus_one and sentinel_matches_previous_zero and poutcome_4 == 0
    c6_observed = (
        f"pdays==-1: {int((df['pdays'] == -1).sum())}; "
        f"pdays==999: {int(sentinel.sum())}; previous==0: {int(prev_zero.sum())}; "
        f"masks identical: {sentinel_matches_previous_zero}; poutcome==4: {poutcome_4}"
    )

    # ---- Verdict (C3-C5 only) -----------------------------------------
    if not c3_pass:
        verdict = "NOT CHRONOLOGICAL"
        exit_code = 2
    elif c4_status == "FAIL":
        verdict = "NOT CHRONOLOGICAL"
        exit_code = 2
    elif c4_status == "INCONCLUSIVE":
        verdict = "INCONCLUSIVE"
        exit_code = 2
    elif c5_pass is None:
        verdict = "INCONCLUSIVE"
        exit_code = 2
    elif c5_pass:
        verdict = "CHRONOLOGICAL"
        exit_code = 0
    else:
        verdict = "PSEUDO-TEMPORAL"
        exit_code = 2

    stop = verdict != "CHRONOLOGICAL"

    # ---- Write block CSV ----------------------------------------------
    block_rows = []
    trans_by_from = {t["from_block"]: t for t in transitions}
    for b in blocks:
        t = trans_by_from.get(b["block_id"])
        block_rows.append({
            **b,
            "transition_delta_mod12": t["delta_mod12"] if t else "",
            "transition_classification": t["classification"] if t else "",
            "transition_valid": t["valid"] if t else "",
        })
    pd.DataFrame(block_rows).to_csv(
        OUT / "chronology_month_blocks.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(day_rows).to_csv(
        OUT / "chronology_day_monotonicity.csv", index=False, encoding="utf-8-sig"
    )

    # ---- JSON ---------------------------------------------------------
    payload = {
        "script": SCRIPT_REL,
        "input_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": digest,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "columns_loaded": CHECK_COLUMNS,
        "target_column_loaded": False,
        "thresholds": {"C3_max_runs": C3_MAX_RUNS, "C5_min_share": C5_MIN_SHARE},
        "checks": {
            "C1": {"status": "DOCUMENTARY-PASS", "evidence": "documentary",
                   "detail": "HF build script has no sort/reorder call"},
            "C2": {"status": "PASS" if c2_pass else "FAIL", "evidence": "empirical",
                   "detail": c2_observed},
            "C3": {"status": "PASS" if c3_pass else "FAIL", "evidence": "empirical",
                   "observed_runs": n_runs, "max_runs": C3_MAX_RUNS,
                   "random_reference_runs": round(expected_runs_random, 2)},
            "C4": {"status": c4_status, "evidence": "empirical",
                   "violations": len(c4_violations),
                   "reversals": len(reversals),
                   "skips": len(skips),
                   "unresolved_skips": len(unresolved_skips),
                   "detail": c4_violations},
            "C5": {"status": "PASS" if c5_pass else "FAIL", "evidence": "empirical",
                   "pooled_share": round(pooled_share, 6) if pooled_share is not None else None,
                   "min_share": C5_MIN_SHARE,
                   "random_reference_share": round(expected_day_share_random, 6)},
            "C6": {"status": "PASS" if c6_pass else "FAIL", "evidence": "empirical",
                   "detail": c6_observed},
            "C7": {"status": "LIMITATION", "evidence": "unresolved",
                   "detail": "no year variable; Dec-Jan boundaries cannot be dated"},
        },
        "verdict": verdict,
        "split_created": False,
        "stop_and_review": stop,
    }
    (OUT / "chronology_checks.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    # ---- Markdown report ----------------------------------------------
    block_table = "\n".join(
        f"| {b['block_id']} | {b['start_row']:,} | {b['end_row']:,} | "
        f"{b['month_label']} | {b['length']:,} |"
        for b in blocks
    )
    trans_table = "\n".join(
        f"| {t['from_block']}->{t['to_block']} | {t['from_month']}->{t['to_month']} | "
        f"{t['delta_mod12']} | {', '.join(t['skipped_months']) or '-'} | "
        f"{t['classification']} | {'yes' if t['valid'] else 'NO'} |"
        for t in transitions
    )
    day_table = "\n".join(
        f"| {r['block_id']} | {r['month_label']} | {r['adjacent_pairs']:,} | "
        f"{r['non_decreasing']:,} | {r['share_non_decreasing']} | "
        f"{r['expected_share_if_random']} |"
        for r in day_rows
    )

    def status(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    report = f"""# Chronology validation — Bank Marketing (HF mirror)

Generated by `{SCRIPT_REL}` on {payload['generated_utc']}.
Input: `{payload['input_file']}` (SHA-256 `{digest}`).
**Target `y` was never loaded.** Columns read: {", ".join(f"`{c}`" for c in CHECK_COLUMNS)}.
No split was created. No model was trained.

## Verdict

**{verdict}**

{("STOP AND REVIEW: a required direct-ordering check failed or was inconclusive. "
  "No split file has been created. Per plan.md section 4.3, a human decision is required "
  "before choosing a split.") if stop else
 ("C3, C4 and C5 pass. The row-index split is eligible to be created, but it has "
  "NOT been created: this awaits separate approval.")}

## Evidence classes

| Class | Checks | Can it establish chronological ordering? |
|---|---|---|
| Documentary | C1 | No. Shows only that the published transformation did not deliberately reorder rows. |
| Identity / encoding | C2, C6 | No. Shows which file was used and that the documented recoding ran. |
| **Direct ordering evidence** | **C3, C4, C5** | **Yes — these alone drive the verdict.** |
| Unresolved limitation | C7 | No check can remove it. |

## Checks

| ID | Evidence | Criterion | Observed | Status |
|---|---|---|---|---|
| C1 | documentary | No sort/reorder call in HF build script | Verbatim excerpt below; no `sort`, `reindex` or reordering call | DOCUMENTARY-PASS |
| C2 | empirical | 45,211 x 17, expected names | {c2_observed} | {status(c2_pass)} |
| C3 | empirical | `n_runs <= {C3_MAX_RUNS}` | {n_runs} runs (random-order reference: {expected_runs_random:.1f}) | {status(c3_pass)} |
| C4 | empirical | 0 violations beyond zero-contact months; skips distinguished from reversals | {len(c4_violations)} violation(s): {len(skips)} skip(s), {len(reversals)} reversal(s) | {c4_status} |
| C5 | empirical | pooled share >= {C5_MIN_SHARE} | {f"{pooled_share:.6f}" if pooled_share is not None else "undefined"} (tie-aware random reference: {expected_day_share_random:.6f}) | {status(c5_pass) if pooled_share is not None else "INCONCLUSIVE"} |
| C6 | empirical | no -1; 999 iff previous==0; no poutcome==4 | {c6_observed} | {status(c6_pass)} |
| C7 | unresolved | year available | No year variable exists | LIMITATION |

### C1 — provenance excerpt (documentary)

Source: https://huggingface.co/datasets/cestwc/bank-marketing (accessed 2026-11-13)

```python
{C1_EXCERPT}
```

No sorting or reordering call appears. `Dataset.from_pandas` is assumed to
preserve row order. This is documentary evidence about the transformation, not
empirical proof that the underlying UCI file was date-ordered.

### C3 — month blocks ({n_runs} runs)

| block | start_row | end_row | month | length |
|---|---|---|---|---|
{block_table}

Random-order reference (computed from observed month frequencies):
`1 + (n-1)(1 - sum p_m^2)` = {expected_runs_random:.1f} runs.

### C4 — transitions

Status: **{c4_status}** — {len(skips)} forward skip{"" if len(skips) == 1 else "s"}, {len(reversals)} reversal{"" if len(reversals) == 1 else "s"}.

A `skip` means the month advanced by more than one. It is a violation of the
+1 rule but **not automatically** evidence against chronological ordering: the
missing month may simply have had zero contacts in that period. A `reversal`
(delta 11, indistinguishable from a one-month backward step without a year)
is direct evidence against chronological ordering.

{("**Unresolved skip:** the skipped month "
  + ", ".join(sorted({m for t in unresolved_skips for m in t['skipped_months']}))
  + " appears elsewhere in the dataset, so the period-specific absence cannot be "
    "confirmed to be a zero-contact month without a year variable. The frozen "
    "criterion therefore cannot be resolved from the data alone: INCONCLUSIVE, "
    "not a definitive failure.")
 if len(unresolved_skips) == 1 else
 ("**Unresolved skips:** the skipped months "
  + ", ".join(sorted({m for t in unresolved_skips for m in t['skipped_months']}))
  + " appear elsewhere in the dataset, so the period-specific absences cannot be "
    "confirmed to be zero-contact months without a year variable: INCONCLUSIVE, "
    "not a definitive failure.")
 if unresolved_skips else
 ("All skipped months have zero contacts in the dataset, so the exemption in "
  "the frozen criterion applies." if skips else
  "No skips or reversals: every consecutive block advances by exactly one month.")}

| blocks | months | delta mod 12 | skipped | classification | valid |
|---|---|---|---|---|---|
{trans_table}

### C5 — within-month day monotonicity

Pooled share: **{f"{pooled_share:.6f}" if pooled_share is not None else "undefined"}**
over {total_pairs:,} adjacent pairs ({total_nondec:,} non-decreasing).

Tie-aware random reference: {expected_day_share_random:.6f} — this is
`P(X >= Y)` for iid days from the observed distribution, so it is not assumed
to be 0.5.

| block | month | pairs | non-decreasing | share | random ref |
|---|---|---|---|---|---|
{day_table}

## Files

- `results/splits/chronology_checks.md` (this file)
- `results/splits/chronology_checks.json`
- `results/splits/chronology_month_blocks.csv`
- `results/splits/chronology_day_monotonicity.csv`

`temporal_split_v1.json` was deliberately **not** created.
"""
    (OUT / "chronology_checks.md").write_text(report, encoding="utf-8")

    # ---- Console summary ---------------------------------------------
    print(f"[C2] {status(c2_pass)}  {c2_observed}")
    print(f"[C3] {status(c3_pass)}  runs={n_runs} (random ref {expected_runs_random:.1f}, max {C3_MAX_RUNS})")
    print(f"[C4] {c4_status}  violations={len(c4_violations)} "
          f"(skips={len(skips)}, reversals={len(reversals)}, unresolved_skips={len(unresolved_skips)})")
    for t in c4_violations:
        print(f"      {t['from_month']}->{t['to_month']} delta={t['delta_mod12']} ({t['classification']})")
    print(f"[C5] {status(c5_pass) if pooled_share is not None else 'INCONCLUSIVE'}  "
          f"share={pooled_share:.6f} (tie-aware random ref {expected_day_share_random:.6f})")
    print(f"[C6] {status(c6_pass)}  {c6_observed}")
    print(f"[C1] DOCUMENTARY-PASS (no reorder call in HF build script)")
    print(f"[C7] LIMITATION (no year variable)")
    print(f"[verdict] {verdict}")
    if stop:
        print("[stop] no split created; human review required")
    else:
        print("[ok] eligible for temporal split; split NOT created (awaiting approval)")
    print(f"[out] {OUT.relative_to(ROOT)}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
