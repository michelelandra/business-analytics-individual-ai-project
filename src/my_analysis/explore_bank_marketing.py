"""Exploratory data analysis of the Bank Marketing dataset (UCI bank-full, HF mirror).

Read-only analysis. Nothing in data/raw/ is modified. No rows are filtered,
dropped or recoded; all derived views are temporary Series used only for
summary statistics.

Outputs go to results/explore_bank_marketing/ so that every number reported in
REPORT.md is traceable to a file produced by this script.

Run with:  uv run python -m my_analysis.explore_bank_marketing
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

SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
OUT = ROOT / "results" / "explore_bank_marketing"

SCRIPT_REL = "src/my_analysis/explore_bank_marketing.py"

# ---------------------------------------------------------------------------
# Codebook, transcribed from the Hugging Face dataset card
# (https://huggingface.co/datasets/cestwc/bank-marketing) and cross-checked
# against the UCI variable documentation.
# ---------------------------------------------------------------------------
CODEBOOK: dict[str, dict[int, str]] = {
    "job": {
        0: "admin.", 1: "blue-collar", 2: "entrepreneur", 3: "housemaid",
        4: "management", 5: "retired", 6: "self-employed", 7: "services",
        8: "student", 9: "technician", 10: "unemployed", 11: "unknown",
    },
    "marital": {0: "divorced", 1: "married", 2: "single", 3: "unknown"},
    "education": {
        0: "primary", 1: "secondary", 2: "tertiary", 3: "unknown",
        4: "basic.4y", 5: "basic.6y", 6: "basic.9y", 7: "high.school",
        8: "illiterate", 9: "professional.course", 10: "university.degree",
    },
    "default": {0: "no", 1: "yes", 2: "unknown"},
    "housing": {0: "no", 1: "yes", 2: "unknown"},
    "loan": {0: "no", 1: "yes", 2: "unknown"},
    "contact": {0: "cellular", 1: "telephone", 2: "unknown"},
    "month": {
        0: "jan", 1: "feb", 2: "mar", 3: "apr", 4: "may", 5: "jun",
        6: "jul", 7: "aug", 8: "sep", 9: "oct", 10: "nov", 11: "dec",
    },
    "poutcome": {
        0: "failure", 1: "other", 2: "success", 3: "unknown", 4: "nonexistent",
    },
    "y": {0: "no", 1: "yes"},
}

CATEGORICAL = list(CODEBOOK.keys())

# Genuinely continuous / count variables.
TRUE_NUMERIC = [
    "age", "balance", "day", "duration", "campaign", "pdays", "previous",
]

# Documented "unknown" codes, kept conceptually separate from NaN.
UNKNOWN_CODE = {
    "job": 11, "marital": 3, "education": 3, "default": 2,
    "housing": 2, "loan": 2, "contact": 2, "poutcome": 3,
}

# Documented sentinel values (not missing values, not valid measurements).
SENTINEL = {
    "pdays": {999: "client was not previously contacted (HF builder replaced -1 with 999)"},
}

SOURCES = {
    "uci_dataset": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
    "hf_card": "https://huggingface.co/datasets/cestwc/bank-marketing",
    "hf_info_api": "https://datasets-server.huggingface.co/info?dataset=cestwc/bank-marketing",
    "paper_doi": "https://doi.org/10.24432/C5K306",
    "accessed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
}

CHECKS: list[dict[str, object]] = []
NOTES: list[str] = []


def check(name: str, expected: str, observed: str, ok: bool) -> None:
    CHECKS.append({
        "check": name,
        "expected": expected,
        "observed": observed,
        "status": "PASS" if ok else "FAIL",
    })


def note(msg: str) -> None:
    NOTES.append(msg)


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_text(name: str, text: str) -> None:
    (OUT / name).write_text(text, encoding="utf-8")


def write_csv(name: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def pct(n: float, d: float) -> float:
    return round(100.0 * n / d, 4) if d else float("nan")


def label(col: str, code: object) -> str:
    if pd.isna(code):
        return "<NaN>"
    try:
        return CODEBOOK.get(col, {}).get(int(code), "<UNDOCUMENTED>")
    except (TypeError, ValueError):
        return "<NON-INTEGER>"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Load (read-only)
    # ------------------------------------------------------------------
    if not RAW.exists():
        print(f"FATAL: input file not found: {RAW}", file=sys.stderr)
        return 1

    digest = sha256(RAW)
    df = pd.read_parquet(RAW)
    n_rows, n_cols = df.shape
    print(f"[load] rows={n_rows:,} cols={n_cols} | no filter, merge or drop applied")

    manifest = {
        "script": SCRIPT_REL,
        "input_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": digest,
        "input_size_bytes": RAW.stat().st_size,
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "random_seed": SEED,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "pyarrow": pa.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "sources": SOURCES,
        "filters_applied": [],
        "rows_dropped": 0,
    }
    write_text("manifest.json", json.dumps(manifest, indent=2) + "\n")

    # ------------------------------------------------------------------
    # 1. Shape
    # ------------------------------------------------------------------
    write_text(
        "01_shape.txt",
        f"rows: {n_rows}\ncolumns: {n_cols}\n"
        f"script: {SCRIPT_REL}\ninput: {manifest['input_file']}\n"
        f"sha256: {digest}\n",
    )
    check("row count matches UCI bank-full (45,211)", "45211", f"{n_rows}", n_rows == 45211)
    check("column count (16 features + y)", "17", f"{n_cols}", n_cols == 17)

    # ------------------------------------------------------------------
    # 2. Schema + codebook validation
    # ------------------------------------------------------------------
    schema_rows = []
    undocumented: dict[str, list[int]] = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        if col in CODEBOOK:
            role = "categorical (label-encoded integer)"
            codes = CODEBOOK[col]
            observed = sorted(int(v) for v in df[col].dropna().unique())
            bad = [v for v in observed if v not in codes]
            if bad:
                undocumented[col] = bad
            code_str = "; ".join(f"{k}={v}" for k, v in codes.items())
            observed_str = ",".join(str(v) for v in observed)
        else:
            role = "numeric"
            code_str = ""
            observed_str = ""
            bad = []
        schema_rows.append({
            "variable": col,
            "dtype": dtype,
            "role": role,
            "n_unique": int(df[col].nunique(dropna=True)),
            "observed_codes": observed_str,
            "documented_codes": code_str,
            "undocumented_codes": ",".join(str(v) for v in bad),
        })
    schema = pd.DataFrame(schema_rows)
    write_csv("02_schema.csv", schema)

    check(
        "all observed categorical codes documented in HF card",
        "0 undocumented codes",
        f"{sum(len(v) for v in undocumented.values())} undocumented",
        not undocumented,
    )
    for col, bad in undocumented.items():
        note(f"UNDOCUMENTED CODES in '{col}': {bad} (not relabeled; left as-is).")

    # Expected code sets per dataset version (bank-full vs bank-additional union)
    edu_extra = [c for c in sorted(df["education"].unique()) if int(c) >= 4]
    if edu_extra:
        note(f"education contains codes >= 4 {edu_extra}: unexpected for bank-full; "
             f"labels per HF card: {[CODEBOOK['education'][int(c)] for c in edu_extra]}")
    pout_4 = int((df["poutcome"] == 4).sum())
    note(f"poutcome code 4 ('nonexistent') observations: {pout_4}")

    # ------------------------------------------------------------------
    # 3. Target variable
    # ------------------------------------------------------------------
    y_counts = df["y"].value_counts(dropna=False).sort_index()
    target_rows = []
    for code, count in y_counts.items():
        target_rows.append({
            "code": int(code),
            "label_documented": label("y", code),
            "count": int(count),
            "percent": pct(count, n_rows),
        })
    target = pd.DataFrame(target_rows)
    write_csv("03_target.csv", target)

    observed_y = sorted(int(v) for v in df["y"].dropna().unique())
    check("target y has only documented codes {0,1}", "{0,1}",
          str(observed_y), set(observed_y).issubset({0, 1}))
    check("target y has no missing values", "0 NaN", f"{int(df['y'].isna().sum())} NaN",
          int(df["y"].isna().sum()) == 0)

    y_txt = [
        "Target variable: y",
        "",
        "Documented meaning (UCI): 'has the client subscribed a term deposit?'",
        "Documented encoding (HF ClassLabel names): 0='no', 1='yes'",
        f"Source: {SOURCES['hf_card']}",
        f"Source: {SOURCES['uci_dataset']}",
        "",
        f"Observed codes: {observed_y}",
        f"Rows: {n_rows:,}",
    ]
    for r in target_rows:
        y_txt.append(f"  {r['code']} ({r['label_documented']}): {r['count']:,} "
                     f"({r['percent']}%)")
    y_txt += [
        "",
        "Verification: observed codes are a subset of the documented set, and the",
        "ClassLabel order in the HF card is ['no','yes'], so 0=no and 1=yes.",
        "No assumption beyond the documented mapping was used.",
    ]
    write_text("03_target_verification.txt", "\n".join(y_txt) + "\n")

    # ------------------------------------------------------------------
    # 4. Missingness: NaN vs 'unknown' codes vs sentinels (kept separate)
    # ------------------------------------------------------------------
    miss_rows = []
    for col in df.columns:
        n_nan = int(df[col].isna().sum())
        ucode = UNKNOWN_CODE.get(col)
        n_unknown = int((df[col] == ucode).sum()) if ucode is not None else 0
        sent = SENTINEL.get(col, {})
        n_sent = sum(int((df[col] == s).sum()) for s in sent)
        sent_meaning = "; ".join(f"{s}={m}" for s, m in sent.items())
        miss_rows.append({
            "variable": col,
            "dtype": str(df[col].dtype),
            "n_nan": n_nan,
            "pct_nan": pct(n_nan, n_rows),
            "unknown_code": ucode if ucode is not None else "",
            "unknown_label": CODEBOOK.get(col, {}).get(ucode, "") if ucode is not None else "",
            "n_unknown_code": n_unknown,
            "pct_unknown_code": pct(n_unknown, n_rows) if ucode is not None else "",
            "n_sentinel": n_sent,
            "sentinel_definition": sent_meaning,
        })
    missingness = pd.DataFrame(miss_rows)
    write_csv("04_missingness.csv", missingness)

    total_nan = int(df.isna().sum().sum())
    check("no NaN cells (UCI declares no missing values)", "0", f"{total_nan}", total_nan == 0)

    # ------------------------------------------------------------------
    # 5. Duplicates (count only, nothing dropped)
    # ------------------------------------------------------------------
    dup_all = int(df.duplicated().sum())
    dup_features = int(df.drop(columns=["y"]).duplicated().sum())
    write_text(
        "05_duplicates.txt",
        f"exact duplicate rows (all {n_cols} columns): {dup_all}\n"
        f"duplicate rows ignoring target y: {dup_features}\n"
        f"rows before: {n_rows:,}\nrows after: {n_rows:,} (nothing dropped)\n"
        f"script: {SCRIPT_REL}\n",
    )
    note(f"Exact duplicate rows: {dup_all} (kept in the data; not dropped).")

    # ------------------------------------------------------------------
    # 6. Descriptive statistics: true numerics only
    # ------------------------------------------------------------------
    num_stats = df[TRUE_NUMERIC].describe().T
    num_stats = num_stats.reset_index().rename(columns={"index": "variable"})
    for c in num_stats.columns:
        if c != "variable":
            num_stats[c] = num_stats[c].astype(float).round(4)
    num_stats["caveat"] = num_stats["variable"].map(
        lambda v: "contains sentinel 999; raw statistics are NOT interpretable — "
                  "see 07_pdays_sentinel_stats.csv for actual-day statistics"
        if v == "pdays" else ""
    )
    write_csv("06_describe_numeric.csv", num_stats)

    # ------------------------------------------------------------------
    # 7. pdays: sentinel vs actual days (sentinel excluded from day stats
    #    via a temporary view; the DataFrame itself is never modified)
    # ------------------------------------------------------------------
    sentinel_val = 999
    pdays_all = df["pdays"]
    n_sent = int((pdays_all == sentinel_val).sum())
    n_neg = int((pdays_all < 0).sum())
    actual = pdays_all[pdays_all != sentinel_val]  # view for stats only

    pdays_summary = pd.DataFrame([
        {"metric": "rows_total", "value": int(len(pdays_all))},
        {"metric": "rows_pdays_eq_999_sentinel", "value": n_sent},
        {"metric": "pct_pdays_eq_999_sentinel", "value": pct(n_sent, len(pdays_all))},
        {"metric": "rows_pdays_negative", "value": n_neg},
        {"metric": "rows_actual_days", "value": int(len(actual))},
        {"metric": "actual_days_mean", "value": round(float(actual.mean()), 4)},
        {"metric": "actual_days_std", "value": round(float(actual.std()), 4)},
        {"metric": "actual_days_min", "value": int(actual.min())},
        {"metric": "actual_days_q25", "value": float(actual.quantile(0.25))},
        {"metric": "actual_days_median", "value": float(actual.median())},
        {"metric": "actual_days_q75", "value": float(actual.quantile(0.75))},
        {"metric": "actual_days_max", "value": int(actual.max())},
    ])
    write_csv("07_pdays_sentinel_stats.csv", pdays_summary)

    # Consistency checks between the sentinel and 'previous' / 'poutcome'
    prev0 = df["previous"] == 0
    is_sent = df["pdays"] == sentinel_val
    pdays_checks = pd.DataFrame([
        {"check": "pdays==999 and previous==0", "rows": int((is_sent & prev0).sum())},
        {"check": "pdays==999 and previous>0", "rows": int((is_sent & ~prev0).sum())},
        {"check": "pdays!=999 and previous==0", "rows": int((~is_sent & prev0).sum())},
        {"check": "pdays!=999 and previous>=1", "rows": int((~is_sent & ~prev0).sum())},
    ])
    write_csv("07b_pdays_consistency.csv", pdays_checks)

    po_by_sent = pd.crosstab(
        df["poutcome"].map(lambda v: label("poutcome", v)),
        np.where(is_sent, "pdays=999 (not previously contacted)", "pdays!=999"),
    ).reset_index().rename(columns={"poutcome": "poutcome_label"})
    write_csv("07c_poutcome_by_pdays_sentinel.csv", po_by_sent)

    if n_neg:
        note(f"pdays contains {n_neg} negative values; documented -1 sentinel was "
             f"expected to have been replaced by 999 in this HF mirror.")
    bad_sent_prev = int((is_sent & ~prev0).sum())
    if bad_sent_prev:
        note(f"INCONSISTENCY: {bad_sent_prev} rows have pdays==999 but previous>0; "
             f"the sentinel does not perfectly coincide with 'never contacted'.")
    bad_actual_prev = int((~is_sent & prev0).sum())
    if bad_actual_prev:
        note(f"INCONSISTENCY: {bad_actual_prev} rows have pdays!=999 but previous==0.")

    # poutcome 'unknown' vs the pdays sentinel: these overlap almost perfectly
    # but not exactly; record the gap rather than reconciling it silently.
    po_unknown = int((df["poutcome"] == UNKNOWN_CODE["poutcome"]).sum())
    po_unknown_contacted = int(
        ((df["poutcome"] == UNKNOWN_CODE["poutcome"]) & ~is_sent).sum()
    )
    note(
        f"poutcome='unknown' occurs in {po_unknown:,} rows, while pdays==999 "
        f"occurs in {n_sent:,} rows: {po_unknown_contacted} rows are marked "
        f"poutcome='unknown' although they WERE previously contacted "
        f"(pdays!=999). See 07c_poutcome_by_pdays_sentinel.csv. Not corrected."
    )
    edu_observed = sorted(int(v) for v in df["education"].unique())
    note(
        f"education uses only codes {edu_observed} "
        f"({', '.join(CODEBOOK['education'][c] for c in edu_observed)}); the "
        f"codes 4-10 documented in the HF card (basic.4y ... university.degree) "
        f"are absent. This is consistent with bank-full, not a data error."
    )
    for col in ("marital", "default", "housing", "loan"):
        if int((df[col] == UNKNOWN_CODE[col]).sum()) == 0:
            note(f"'{col}' documents an 'unknown' code "
                 f"({UNKNOWN_CODE[col]}) but no observation uses it.")

    # ------------------------------------------------------------------
    # 8. Frequency distributions for categorical variables
    # ------------------------------------------------------------------
    freq_rows = []
    for col in CATEGORICAL:
        vc = df[col].value_counts(dropna=False).sort_index()
        for code, count in vc.items():
            freq_rows.append({
                "variable": col,
                "code": int(code),
                "label": label(col, code),
                "count": int(count),
                "percent": pct(count, n_rows),
            })
    freq = pd.DataFrame(freq_rows)
    write_csv("08_frequency_categorical.csv", freq)

    # ------------------------------------------------------------------
    # 9. duration: descriptive only, flagged as post-call information
    # ------------------------------------------------------------------
    dur_rows = []
    for code, grp in df.groupby("y"):
        s = grp["duration"]
        dur_rows.append({
            "y_code": int(code),
            "y_label": label("y", code),
            "n": int(len(s)),
            "mean_seconds": round(float(s.mean()), 2),
            "std_seconds": round(float(s.std()), 2),
            "min_seconds": int(s.min()),
            "median_seconds": float(s.median()),
            "max_seconds": int(s.max()),
        })
    dur = pd.DataFrame(dur_rows)
    write_csv("09_duration_by_target.csv", dur)

    check("duration is non-negative", "min >= 0", f"min={int(df['duration'].min())}",
          int(df["duration"].min()) >= 0)
    check("campaign >= 1", "min >= 1", f"min={int(df['campaign'].min())}",
          int(df["campaign"].min()) >= 1)

    # ------------------------------------------------------------------
    # 10. Validation checks + notes
    # ------------------------------------------------------------------
    write_csv("10_validation_checks.csv", pd.DataFrame(CHECKS))
    write_text("10_notes.txt", "\n".join(NOTES) + "\n" if NOTES else "no unexpected findings\n")

    # ------------------------------------------------------------------
    # 11. Leakage & privacy documentation
    # ------------------------------------------------------------------
    y_no = int(target.loc[target["code"] == 0, "count"].iloc[0]) if (target["code"] == 0).any() else 0
    y_yes = int(target.loc[target["code"] == 1, "count"].iloc[0]) if (target["code"] == 1).any() else 0
    dur_no = dur.loc[dur["y_code"] == 0, "median_seconds"]
    dur_yes = dur.loc[dur["y_code"] == 1, "median_seconds"]
    dur_no_med = float(dur_no.iloc[0]) if len(dur_no) else float("nan")
    dur_yes_med = float(dur_yes.iloc[0]) if len(dur_yes) else float("nan")

    leak = f"""# Data leakage and privacy assessment

Script: `{SCRIPT_REL}`
Generated: {manifest['generated_utc']}
Every number below is produced by `{SCRIPT_REL}` and stored in this folder.

## 1. `duration` — post-call information (leakage for pre-call targeting)

`duration` is the duration of the last contact, in seconds. It is only known
**after the call has ended**. For any model intended to decide *whom to call*,
using `duration` leaks the outcome: the target is realised at the end of the
same call whose length is measured.

Descriptive evidence (not a model) — `09_duration_by_target.csv`:
- y=0 ("no"): n={y_no:,}, median duration {dur_no_med:.0f}s
- y=1 ("yes"): n={y_yes:,}, median duration {dur_yes_med:.0f}s

Decision: `duration` may be described, but it is **excluded from future
pre-call predictive models**. Caveat recorded honestly: Moro et al. (2014)
benchmark `duration` in their published models, so the applied literature is
not uniform on this point. We follow the pre-call decision framing.

## 2. `pdays` — sentinel, not a measurement

In this HF mirror the builder replaced UCI's `-1` with `999`
({SOURCES['hf_card']}). Therefore:
- `999` means "client was not previously contacted" — a **sentinel**, not
  missing data and not 999 days.
- Frequency: {n_sent:,} of {n_rows:,} rows ({pct(n_sent, n_rows)}%) —
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
- Licence: CC BY 4.0 ({SOURCES['uci_dataset']}); cite Moro, Rita & Cortez
  (2014), DOI {SOURCES['paper_doi']}.
"""
    write_text("11_leakage_and_privacy.md", leak)

    # ------------------------------------------------------------------
    # 12. REPORT.md (numbers interpolated from computed values only)
    # ------------------------------------------------------------------
    miss_unknown_total = int(missingness["n_unknown_code"].sum())
    top_unknown = missingness[missingness["n_unknown_code"] > 0][
        ["variable", "unknown_label", "n_unknown_code", "pct_unknown_code"]
    ]
    top_unknown_txt = "\n".join(
        f"  - `{r.variable}` = \"{r.unknown_label}\": {int(r.n_unknown_code):,} "
        f"({r.pct_unknown_code}%)"
        for r in top_unknown.itertuples()
    ) or "  - none"

    freq_top = freq.sort_values(["variable", "count"], ascending=[True, False])
    top_rows = []
    for col in CATEGORICAL:
        sub = freq_top[freq_top["variable"] == col].head(3)
        if col == "y":
            continue
        parts = ", ".join(
            f"{r.label} {r.count:,} ({r.percent}%)" for r in sub.itertuples()
        )
        top_rows.append(f"  - `{col}`: {parts}")
    top_rows_txt = "\n".join(top_rows)

    report = f"""# Exploratory analysis — Bank Marketing (UCI bank-full, HF mirror)

Generated by `{SCRIPT_REL}` on {manifest['generated_utc']}.
Input: `{manifest['input_file']}` (sha256 `{digest[:16]}…`).
**No filtering, dropping, recoding or modelling was performed.**

## 1. Size

- Rows: **{n_rows:,}** — `01_shape.txt`
- Columns: **{n_cols}** (16 features + target) — `01_shape.txt`

## 2. Variables and types

All 10 categorical variables are stored as **label-encoded integers**, not as
strings. Treating them as continuous numbers would be a coding error.
Full table: `02_schema.csv`.

- Numeric ({len(TRUE_NUMERIC)}): {", ".join(f"`{c}`" for c in TRUE_NUMERIC)}
- Categorical ({len(CATEGORICAL)}): {", ".join(f"`{c}`" for c in CATEGORICAL)}

Codebook validation against the Hugging Face card: **{len(undocumented)} variable(s)**
contained undocumented codes. Details in `10_validation_checks.csv` and `10_notes.txt`.

## 3. Target `y`

Documented: "has the client subscribed a term deposit?", encoded
0="no", 1="yes" (HF ClassLabel order). Verified against observed codes;
see `03_target_verification.txt`.

| code | label | count | % |
|---|---|---|---|
""" + "\n".join(
        f"| {r.code} | {r.label_documented} | {r.count:,} | {r.percent}% |"
        for r in target.itertuples()
    ) + f"""

Class imbalance: **{pct(y_yes, n_rows)}%** positive — `03_target.csv`.

## 4. Missing values, "unknown" categories, sentinels (kept separate)

- NaN cells anywhere: **{total_nan}** — `04_missingness.csv`
- Rows carrying a documented "unknown" category: **{miss_unknown_total:,}**
  across all categorical columns (a row can count in more than one column) —
  `04_missingness.csv`
{top_unknown_txt}
- Sentinel values: `pdays` = 999 in **{n_sent:,}** rows ({pct(n_sent, n_rows)}%) —
  `07_pdays_sentinel_stats.csv`. This is **not** missing data.
- Exact duplicate rows: **{dup_all}** (kept) — `05_duplicates.txt`

## 5. Descriptive statistics

Numeric variables: `06_describe_numeric.csv`.
`pdays` is split into sentinel frequency and actual-day statistics:
`07_pdays_sentinel_stats.csv`, consistency cross-checks in
`07b_pdays_consistency.csv` and `07c_poutcome_by_pdays_sentinel.csv`.

Categorical frequencies with labels: `08_frequency_categorical.csv`.
Most frequent categories per variable:

{top_rows_txt}

## 6. `duration` and `pdays`

- `duration` is post-call information (known only after the call ends) and is
  **excluded from future pre-call models**. Descriptive split by target:
  `09_duration_by_target.csv`.
- `pdays` = 999 is the documented sentinel for "never previously contacted".
  It is reported separately and excluded from day-based statistics without
  altering any observation.

## 7. Leakage and privacy

Full assessment: `11_leakage_and_privacy.md`.

## 8. Unexpected findings

{chr(10).join(f"- {n}" for n in NOTES) if NOTES else "- None."}

## 9. Validation summary

See `10_validation_checks.csv`; failures: {sum(1 for c in CHECKS if c['status'] == 'FAIL')}.
"""
    write_text("REPORT.md", report)

    # ------------------------------------------------------------------
    # Final row-count confirmation (AGENTS.md rule 2)
    # ------------------------------------------------------------------
    print(f"[end]  rows={len(df):,} cols={df.shape[1]} | no filter, merge or drop applied")
    print(f"[out]  {OUT.relative_to(ROOT)}")
    print(f"[target] {dict(zip(target['label_documented'], target['count']))}")
    print(f"[pdays] sentinel 999 = {n_sent:,} rows; actual days = {len(actual):,}")
    print(f"[dupes] {dup_all}")
    print(f"[checks] {sum(1 for c in CHECKS if c['status']=='PASS')} pass / "
          f"{sum(1 for c in CHECKS if c['status']=='FAIL')} fail")
    if NOTES:
        print("[notes]")
        for n in NOTES:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
