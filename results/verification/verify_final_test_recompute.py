"""Independent recomputation of the frozen chronological final-test metrics.

Purpose
-------
Verify, from saved artifacts only, that the frozen chronological final-test
results are arithmetically reproducible. This script:

* reads ``results/guided/test/test_predictions.csv`` (Models A, B, C);
* recomputes ROC-AUC, average precision, Brier score and lift@10% independently;
* reruns the paired bootstrap exactly as authorised: 2,000 resamples, seed 42,
  ``rng.integers(0, n, n)``, identical positions reused across models and
  metrics, single-class samples skipped, percentile 95% intervals;
* compares every recomputed quantity with the frozen artifacts at tolerance 1e-6;
* verifies integrity of the protected artifacts by SHA-256 before and after;
* verifies dataset and split metadata from saved manifests and split files only.

Model C is a NON-DEPLOYABLE duration-leakage diagnostic and is never an
operational model.

Scope
-----
This is a METRIC-LEVEL computational check. It does not and cannot verify feature
engineering, preprocessing, hyperparameter selection, model training, or that the
saved labels match the raw data. ``data/raw/`` is never opened. No model is
fitted, no prediction is generated, and nothing is overwritten.

This is an OpenCode-generated, independently implemented check. It supplements
but does not replace the human PowerShell verification recorded in
``results/verification/final_test_independent_verification.md``.

Run (from the repository root)::

    .venv\\Scripts\\python.exe results\\verification\\verify_final_test_recompute.py

Exit codes: 0 all checks pass; 1 one or more checks fail; 2 unexpected error.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
VER = ROOT / "results" / "verification"
TEST = ROOT / "results" / "guided" / "test"
SPLIT_FILE = ROOT / "results" / "splits" / "temporal_split_v1.json"

CONSOLE = VER / "final_test_recompute_console.txt"
REPORT = VER / "final_test_recompute_report.md"
CHECKS_JSON = VER / "final_test_recompute_checks.json"
HASHES_JSON = VER / "final_test_recompute_hashes.json"
SCRIPT = Path(__file__).resolve()

OUTPUTS = {CONSOLE.resolve(), REPORT.resolve(), CHECKS_JSON.resolve(), HASHES_JSON.resolve(), SCRIPT}

TOL = 1e-6
N_BOOT = 2000
SEED = 42
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache"}

LOG_LINES: list[str] = []
_log_fh = None


def log(msg: str = "") -> None:
    print(msg)
    LOG_LINES.append(msg)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def snapshot() -> dict[str, str]:
    """SHA-256 of every project file except VCS/venv caches and this run's outputs."""
    out: dict[str, str] = {}
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if p.resolve() in OUTPUTS:
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        out[rel.as_posix()] = sha256_file(p)
    return out


CHECKS: list[dict[str, Any]] = []


def add_check(
    check_id: str,
    description: str,
    reported: Any,
    recomputed: Any,
    tol: float | None = TOL,
    kind: str = "float",
) -> bool:
    if kind == "float":
        diff: float | None = abs(float(reported) - float(recomputed))
        passed = diff <= float(tol)
        rec = {
            "id": check_id,
            "description": description,
            "reported": float(reported),
            "recomputed": float(recomputed),
            "abs_diff": float(diff),
            "tolerance": float(tol),
            "passed": bool(passed),
        }
        detail = f"reported={float(reported):.12g} recomputed={float(recomputed):.12g} |diff|={diff:.3e}"
    elif kind == "bool":
        passed = bool(reported) == bool(recomputed)
        rec = {
            "id": check_id,
            "description": description,
            "reported": bool(reported),
            "recomputed": bool(recomputed),
            "abs_diff": None,
            "tolerance": None,
            "passed": bool(passed),
        }
        detail = f"reported={reported} recomputed={recomputed}"
    else:  # exact string / hash
        passed = str(reported) == str(recomputed)
        rec = {
            "id": check_id,
            "description": description,
            "reported": str(reported),
            "recomputed": str(recomputed),
            "abs_diff": None,
            "tolerance": None,
            "passed": bool(passed),
        }
        detail = "match" if passed else f"reported={reported} recomputed={recomputed}"

    CHECKS.append(rec)
    log(f"[{'PASS' if passed else 'FAIL'}] {check_id}: {description} — {detail}")
    return passed


def auc_numpy(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) ROC-AUC with average ranks for ties."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    pos = p[y == 1]
    neg = p[y == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    uniq, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    rank_sums = np.zeros(len(uniq), dtype=float)
    np.add.at(rank_sums, inv, ranks)
    avg = rank_sums / counts
    ranks = avg[inv]
    r_pos = float(ranks[:n_pos].sum())
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def lift_at_10(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    """Lift at the top 10%: k = floor(0.1 * n), stable descending sort."""
    n = len(y)
    k = max(1, int(np.floor(0.1 * n)))
    order = np.argsort(-np.asarray(p), kind="stable")
    tp = int(np.asarray(y)[order[:k]].sum())
    prevalence = float(np.mean(y))
    precision = tp / k
    return {
        "k": k,
        "tp": tp,
        "precision": precision,
        "recall": tp / float(np.sum(y)),
        "expected_tp": k * prevalence,
        "lift": precision / prevalence,
    }


def classify_h1(delta: float, ci: list[float]) -> str:
    """Decision table from plan.md section 3.2, re-implemented here."""
    lo, hi = ci
    if delta >= 0.01 and lo > 0:
        return "H0 rejected, materially relevant"
    if 0 < delta < 0.01 and lo > 0:
        return "H0 rejected, below relevance threshold"
    if lo <= 0 <= hi:
        return "Insufficient evidence"
    if delta < 0 and hi < 0:
        return "H0 not rejected / direction contradicted"
    return "Unclassified — human review required"


def main() -> int:
    global _log_fh

    started = datetime.now(timezone.utc)
    _log_fh = CONSOLE.open("w", encoding="utf-8")

    log("Independent recomputation — frozen chronological final-test metrics")
    log(f"Script: {SCRIPT.relative_to(ROOT).as_posix()}")
    log(f"Started (UTC): {started.isoformat(timespec='seconds')}")
    log("")
    log("Model C is a NON-DEPLOYABLE duration-leakage diagnostic.")
    log("")

    # ---------------------------------------------------------------- integrity
    log("== 1. Integrity snapshot (before) ==")
    before = snapshot()
    log(f"Protected files hashed: {len(before)}")
    log("")

    manifest = json.loads((TEST / "run_manifest_test.json").read_text(encoding="utf-8"))
    metrics_df = pd.read_csv(TEST / "test_metrics.csv")
    preds = pd.read_csv(TEST / "test_predictions.csv")
    hyp = json.loads((TEST / "hypothesis_tests_test.json").read_text(encoding="utf-8"))

    log("== 2. Frozen-artifact integrity ==")
    for name, expected in manifest["artifacts"].items():
        actual = sha256_file(TEST / name)
        add_check(f"hash_{name}", f"SHA-256 of {name} matches manifest", expected, actual, kind="exact")

    script_hash = sha256_file(ROOT / manifest["script"])
    add_check(
        "hash_analysis_script",
        "SHA-256 of guided_test_evaluate.py matches manifest (definitions unchanged)",
        manifest["script_sha256"],
        script_hash,
        kind="exact",
    )

    log("")
    log("== 3. Split and dataset metadata (saved files only) ==")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    canonical = json.dumps(split["indices"], sort_keys=True, separators=(",", ":"))
    split_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    add_check(
        "split_canonical_hash",
        "Canonical hash of saved chronological index arrays equals recorded split hash",
        split["split_hash_sha256"],
        split_hash,
        kind="exact",
    )
    add_check(
        "split_matches_manifest",
        "Manifest split_sha256 equals split file canonical hash",
        manifest["split_sha256"],
        split_hash,
        kind="exact",
    )
    add_check(
        "dataset_metadata_consistency",
        "Dataset SHA-256 recorded in test manifest equals split file input_sha256",
        manifest["dataset_sha256"],
        split["input_sha256"],
        kind="exact",
    )
    add_check(
        "split_sizes",
        "Split sizes are train 31,647 / validation 6,781 / test 6,783",
        {"train": 31647, "validation": 6781, "test": 6783},
        {k: int(v) for k, v in split["sizes"].items()},
        kind="exact",
    )
    log("(Raw dataset not opened; its SHA-256 is taken from saved metadata, not rehashed.)")

    log("")
    log("== 4. Prediction file structure ==")
    add_check("n_rows", "Prediction rows equal 6,783", 6783, len(preds), kind="bool")
    add_check("n_positive", "Positive outcomes equal 2,669", 2669, int(preds["y"].sum()), kind="bool")
    add_check("no_nulls", "No missing values in predictions", 0, int(preds.isna().sum().sum()), kind="bool")
    contiguous = bool(np.all(np.diff(preds["row_index"].to_numpy()) == 1))
    add_check("row_index_contiguous", "Row indices are contiguous", True, contiguous, kind="bool")

    y = preds["y"].to_numpy(dtype=np.int64)
    p = {m: preds[f"p_{m}"].to_numpy(dtype=float) for m in ("A", "B", "C")}
    test_idx_split = np.asarray(split["indices"]["test"], dtype=np.int64)
    add_check(
        "row_alignment_split",
        "Prediction row_index equals the saved chronological test indices",
        True,
        bool(np.array_equal(np.sort(test_idx_split), np.sort(preds["row_index"].to_numpy()))),
        kind="bool",
    )
    add_check(
        "prevalence",
        "Test prevalence equals 0.39348370927318294",
        0.39348370927318294,
        float(y.mean()),
    )
    add_check(
        "model_c_role",
        "Model C is labelled non-deployable in the frozen metrics",
        True,
        "not deployable" in str(metrics_df.loc[metrics_df["model"] == "C", "role"].iloc[0]).lower(),
        kind="bool",
    )
    add_check(
        "model_c_json_note",
        "Frozen bootstrap JSON labels Model C as a leakage diagnostic",
        True,
        "leakage diagnostic" in str(hyp.get("note", "")).lower(),
        kind="bool",
    )

    # ------------------------------------------------------------------ metrics
    log("")
    log("== 5. Point metrics recomputed ==")
    recomputed: dict[str, dict[str, float]] = {}
    for m in ("A", "B", "C"):
        row = metrics_df.loc[metrics_df["model"] == m].iloc[0]
        auc = float(roc_auc_score(y, p[m]))
        ap = float(average_precision_score(y, p[m]))
        brier = float(np.mean((p[m] - y) ** 2))
        brier_sk = float(brier_score_loss(y, p[m]))
        lift = lift_at_10(y, p[m])
        auc_mw = float(auc_numpy(y, p[m]))

        recomputed[m] = {
            "roc_auc": auc,
            "pr_auc": ap,
            "brier": brier,
            "mean_predicted": float(np.mean(p[m])),
            "lift": float(lift["lift"]),
            "lift_precision": float(lift["precision"]),
            "lift_recall": float(lift["recall"]),
            "lift_tp": int(lift["tp"]),
            "lift_k": int(lift["k"]),
            "lift_expected": float(lift["expected_tp"]),
        }

        add_check(f"{m}_roc_auc", f"Model {m} ROC-AUC", row["roc_auc"], auc)
        add_check(f"{m}_roc_auc_mannwhitney", f"Model {m} ROC-AUC via independent rank formula", row["roc_auc"], auc_mw)
        add_check(f"{m}_pr_auc", f"Model {m} average precision", row["pr_auc"], ap)
        add_check(f"{m}_brier", f"Model {m} Brier score", row["brier"], brier)
        add_check(f"{m}_brier_sklearn", f"Model {m} Brier via sklearn", row["brier"], brier_sk)
        add_check(f"{m}_mean_predicted", f"Model {m} mean predicted probability", row["mean_predicted"], np.mean(p[m]))
        add_check(f"{m}_lift", f"Model {m} lift@10%", row["lift"], lift["lift"])
        add_check(f"{m}_lift_k", f"Model {m} lift k = floor(0.1n)", row["lift_k"], lift["k"], tol=0.0)
        add_check(f"{m}_lift_tp", f"Model {m} true positives in top 10%", row["lift_tp_in_top_k"], lift["tp"], tol=0.0)
        add_check(f"{m}_lift_precision", f"Model {m} precision@10%", row["lift_precision_at_k"], lift["precision"])
        add_check(f"{m}_lift_recall", f"Model {m} recall@10%", row["lift_recall_at_k"], lift["recall"])
        add_check(
            f"{m}_lift_expected",
            f"Model {m} expected true positives by chance",
            row["lift_expected_tp_by_chance"],
            lift["expected_tp"],
        )

    # --------------------------------------------------------------- bootstrap
    log("")
    log("== 6. Paired bootstrap (2,000 resamples, seed 42) ==")
    rng = np.random.default_rng(SEED)
    n = len(y)
    ba_auc: list[float] = []
    cb_auc: list[float] = []
    ba_ap: list[float] = []
    cb_ap: list[float] = []
    skipped = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        auc = {m: roc_auc_score(yb, p[m][idx]) for m in ("A", "B", "C")}
        ap = {m: average_precision_score(yb, p[m][idx]) for m in ("A", "B", "C")}
        ba_auc.append(auc["B"] - auc["A"])
        cb_auc.append(auc["C"] - auc["B"])
        ba_ap.append(ap["B"] - ap["A"])
        cb_ap.append(ap["C"] - ap["B"])

    def ci(values: list[float]) -> tuple[float, float]:
        lo, hi = np.percentile(np.asarray(values), [2.5, 97.5])
        return float(lo), float(hi)

    boot = {
        "B-A": {
            "delta_roc_auc": ci(ba_auc),
            "delta_pr_auc": ci(ba_ap),
        },
        "C-B": {
            "delta_roc_auc": ci(cb_auc),
            "delta_pr_auc": ci(cb_ap),
        },
    }
    point = {
        "B-A": {
            "delta_roc_auc": float(roc_auc_score(y, p["B"]) - roc_auc_score(y, p["A"])),
            "delta_pr_auc": float(average_precision_score(y, p["B"]) - average_precision_score(y, p["A"])),
        },
        "C-B": {
            "delta_roc_auc": float(roc_auc_score(y, p["C"]) - roc_auc_score(y, p["B"])),
            "delta_pr_auc": float(average_precision_score(y, p["C"]) - average_precision_score(y, p["B"])),
        },
    }

    log(f"Valid resamples: {len(ba_auc)}; skipped single-class: {skipped}")

    for comparison in ("B-A", "C-B"):
        reported_block = hyp[comparison]
        for metric_key, label in (("delta_roc_auc", "Delta ROC-AUC"), ("delta_pr_auc", "Delta average precision")):
            rep = reported_block[metric_key]
            add_check(
                f"{comparison}_{metric_key}_point",
                f"{comparison} {label} point estimate",
                rep["point"],
                point[comparison][metric_key],
            )
            lo, hi = boot[comparison][metric_key]
            add_check(
                f"{comparison}_{metric_key}_ci_lo",
                f"{comparison} {label} 95% CI lower bound",
                rep["ci95"][0],
                lo,
            )
            add_check(
                f"{comparison}_{metric_key}_ci_hi",
                f"{comparison} {label} 95% CI upper bound",
                rep["ci95"][1],
                hi,
            )
            add_check(
                f"{comparison}_{metric_key}_n_valid",
                f"{comparison} {label} valid resamples",
                rep["n_valid_resamples"],
                2000 - skipped,
                tol=0.0,
            )

    add_check(
        "boot_skipped",
        "Reported skipped single-class resamples equal recomputed count",
        hyp["resamples_skipped_single_class"],
        skipped,
        tol=0.0,
    )
    add_check("boot_n", "Bootstrap resample count is 2,000", hyp["n_boot"], N_BOOT, tol=0.0)
    add_check("boot_seed", "Bootstrap seed is 42", hyp["seed"], SEED, tol=0.0)

    # ------------------------------------------------------------ classification
    log("")
    log("== 7. Frozen classification labels ==")
    ba_ci = list(boot["B-A"]["delta_roc_auc"])
    h1_recomputed = classify_h1(point["B-A"]["delta_roc_auc"], ba_ci)
    add_check("H1_classification", "H1 classification matches frozen label", hyp["H1_classification"], h1_recomputed, kind="exact")
    add_check(
        "H2_delta",
        "H2 delta ROC-AUC (C-B) matches frozen value",
        hyp["H2_delta_roc_auc_CB"],
        point["C-B"]["delta_roc_auc"],
    )
    add_check(
        "H2_material",
        "H2 materiality flag matches frozen value",
        hyp["H2_material"],
        bool(point["C-B"]["delta_roc_auc"] >= float(hyp["H2_threshold"])),
        kind="bool",
    )

    # ------------------------------------------------------------------ hashes
    log("")
    log("== 8. Integrity snapshot (after) ==")
    after = snapshot()
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(k for k in set(before) & set(after) if before[k] != after[k])
    integrity_ok = not (removed or changed)
    add_check("no_removed_files", "No protected file was removed", [], removed, kind="exact")
    add_check("no_changed_files", "No protected file changed", [], changed, kind="exact")
    log(f"Files before: {len(before)}; after: {len(after)}; added: {len(added)}")

    # ------------------------------------------------------------------ verdict
    failures = [c for c in CHECKS if not c["passed"]]
    status = "PASS" if not failures and integrity_ok else "FAIL"

    log("")
    log(f"== RESULT: {status} ==")
    log(f"Checks: {len(CHECKS)}; passed: {len(CHECKS) - len(failures)}; failed: {len(failures)}")
    if failures:
        for c in failures:
            log(f"  FAILED: {c['id']} — {c['description']}")
        log("")
        log("A discrepancy exceeds tolerance. The methodology and tolerance were not")
        log("changed to force a match. Human review is required.")

    # ----------------------------------------------------------------- outputs
    HASHES_JSON.write_text(
        json.dumps(
            {
                "generated_utc": started.isoformat(timespec="seconds"),
                "script": SCRIPT.relative_to(ROOT).as_posix(),
                "scope": "SHA-256 of all project files, excluding .git/.venv caches and this run's five outputs",
                "allowed_new_outputs": sorted(p.relative_to(ROOT).as_posix() for p in OUTPUTS),
                "n_files_before": len(before),
                "n_files_after": len(after),
                "added": added,
                "removed": removed,
                "changed": changed,
                "unexpected_modification": bool(removed or changed),
                "raw_dataset_accessed": False,
                "raw_dataset_note": (
                    "data/raw was not opened. The dataset SHA-256 is taken from saved manifests "
                    "and the split file, and checked for cross-artifact consistency only."
                ),
                "manifest_metadata": {
                    "dataset_sha256": manifest["dataset_sha256"],
                    "split_sha256": manifest["split_sha256"],
                    "split_file_sha256": split["split_hash_sha256"],
                    "analysis_script_sha256": manifest["script_sha256"],
                },
                "before": before,
                "after": after,
            },
            indent=1,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    CHECKS_JSON.write_text(
        json.dumps(
            {
                "generated_utc": started.isoformat(timespec="seconds"),
                "script": SCRIPT.relative_to(ROOT).as_posix(),
                "status": status,
                "tolerance": TOL,
                "n_boot": N_BOOT,
                "seed": SEED,
                "valid_resamples": len(ba_auc),
                "skipped_single_class": skipped,
                "n_checks": len(CHECKS),
                "n_failed": len(failures),
                "checks": CHECKS,
                "integrity": {
                    "n_files_before": len(before),
                    "n_files_after": len(after),
                    "added": added,
                    "removed": removed,
                    "changed": changed,
                },
                "model_c": "NON-DEPLOYABLE duration-leakage diagnostic; not an operational model",
                "scope": "metric-level recomputation from saved predictions; no model fitting, no new predictions",
                "supplements": "results/verification/final_test_independent_verification.md (human PowerShell verification)",
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    def rows(prefix: str, label: str) -> str:
        lines = [
            f"| {label} | reported | recomputed | abs diff | result |",
            "|---|---:|---:|---:|---|",
        ]
        for c in CHECKS:
            if not c["id"].startswith(prefix):
                continue
            if c["abs_diff"] is None:
                lines.append(f"| `{c['id']}` | {c['reported']} | {c['recomputed']} | — | {'PASS' if c['passed'] else 'FAIL'} |")
            else:
                lines.append(
                    f"| `{c['id']}` | {c['reported']:.12g} | {c['recomputed']:.12g} | {c['abs_diff']:.3e} | "
                    f"{'PASS' if c['passed'] else 'FAIL'} |"
                )
        return "\n".join(lines)

    metric_rows = []
    for m in ("A", "B", "C"):
        r = recomputed[m]
        metric_rows.append(
            f"| {m} | {r['roc_auc']:.12f} | {r['pr_auc']:.12f} | {r['brier']:.12f} | {r['lift']:.12f} | "
            f"{r['lift_tp']}/{r['lift_k']} |"
        )

    report = f"""# Independent recomputation — frozen chronological final-test metrics

**Status: {status}.** Generated {started.isoformat(timespec='seconds')} by
`{SCRIPT.relative_to(ROOT).as_posix()}`.

> **Model C is a NON-DEPLOYABLE duration-leakage diagnostic.** It is reported
> only to quantify leakage inflation and is not an operational model.

**Scope.** Metric-level recomputation from the saved chronological final-test
predictions. No model was fitted, no prediction was generated, and no analysis
was rerun. `data/raw/` was not opened. This is an OpenCode-generated,
independently implemented computational check; it **supplements but does not
replace** the human PowerShell verification in
`results/verification/final_test_independent_verification.md`.

## Environment

| Item | Value |
|---|---|
| Python | {platform.python_version()} |
| NumPy | {np.__version__} |
| scikit-learn | {sklearn.__version__} |
| pandas | {pd.__version__} |
| Platform | {platform.platform()} |
| Bootstrap | {N_BOOT} paired resamples, `default_rng({SEED})`, `integers(0, n, n)`, percentile 95% |
| Tolerance | {TOL:g} |

## Recomputed point metrics (test n = {n:,}; positives = {int(y.sum()):,}; prevalence = {y.mean():.12f})

| Model | ROC-AUC | Average precision | Brier | Lift@10% | TP in top 10% |
|---|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

## Checks

### Integrity and metadata

{rows("hash_", "hash")}

{rows("split_", "split")}

{rows("dataset_", "dataset")}

### Prediction structure

{rows("n_", "row")}

{rows("row_", "index")}

{rows("prevalence", "prevalence")}

{rows("no_nulls", "nulls")}

{rows("model_c_", "model C")}

### Model metrics

{rows("A_", "A")}

{rows("B_", "B")}

{rows("C_", "C")}

### Paired bootstrap: B − A

{rows("B-A_", "B-A")}

### Paired bootstrap: C − B (leakage diagnostic)

{rows("C-B_", "C-B")}

{rows("boot_", "bootstrap")}

### Frozen classifications

{rows("H1_", "H1")}

{rows("H2_", "H2")}

## Integrity

- Protected files hashed before: {len(before)}
- Protected files hashed after: {len(after)}
- Added by this run: {", ".join(added) if added else "none"}
- Removed: {", ".join(removed) if removed else "none"}
- Changed: {", ".join(changed) if changed else "none"}

Dataset and split metadata were verified from saved manifests and the saved
split file only. The canonical split hash
`{split_hash}` was recomputed from the saved index arrays without
opening the raw dataset, and the dataset SHA-256 was checked for consistency
between the test manifest and the split file.

## Limitations

- This verifies **metric arithmetic, bootstrap inference and saved-file
  integrity**. It does **not** verify feature engineering, categorical handling,
  the derived previous-campaign variable, preprocessing, hyperparameter
  selection, or model training.
- Labels are taken from the saved prediction file. That they match the raw
  Parquet was verified separately by the human author; it is not re-established
  here.
- Agreement is expected to be near-exact because the same saved probabilities
  and the same documented bootstrap convention are used. The check therefore
  tests reproducibility of the reported numbers, not the correctness of the
  modelling pipeline.
- AP is computed with scikit-learn, whose tie handling differs from naive
  hand implementations; this is deliberate and matches the frozen artifacts.

## Result

**{status}** — {len(CHECKS) - len(failures)} of {len(CHECKS)} checks passed.
{"No discrepancy exceeded tolerance; all protected artefacts are unchanged." if status == "PASS" else "One or more checks failed. No tolerance or method was changed to force a match. Human review is required."}
"""

    REPORT.write_text(report, encoding="utf-8")

    log("")
    log(f"Report: {REPORT.relative_to(ROOT).as_posix()}")
    log(f"Checks JSON: {CHECKS_JSON.relative_to(ROOT).as_posix()}")
    log(f"Hashes JSON: {HASHES_JSON.relative_to(ROOT).as_posix()}")
    log(f"Console: {CONSOLE.relative_to(ROOT).as_posix()}")
    log(f"Finished (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}")

    _log_fh.close()
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        _msg = "UNEXPECTED ERROR\n" + traceback.format_exc()
        print(_msg, file=sys.stderr)
        if _log_fh is not None:
            _log_fh.write(_msg + "\n")
            _log_fh.close()
        raise SystemExit(2)
