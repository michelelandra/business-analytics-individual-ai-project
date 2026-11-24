#!/usr/bin/env python
"""Build the self-contained project dashboard from frozen, saved artifacts.

This script only READS existing project files. It writes exactly one output:
dashboard/project_dashboard.html.

Design constraints:
  * no network calls, no CDN assets, no runtime fetch() in the HTML;
  * no raw data, no test labels, no model fitting;
  * every displayed metric is read from a saved CSV/JSON/Markdown artifact;
  * key numerical relationships are asserted before the HTML is written.

Run from the repository root:
    .venv\\Scripts\\python.exe dashboard\\build_dashboard.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "project_dashboard.html"

MODELS = ("A", "B", "C")

TEST_METRICS = "results/guided/test/test_metrics.csv"
HYPOTHESIS = "results/guided/test/hypothesis_tests_test.json"
CALIBRATION = "results/guided/test/test_calibration.csv"
CALIBRATION_NOTE = "results/guided/test/calibration_interpretation.md"
HYPERPARAMS = "results/guided/hyperparameter_selection.csv"
VALIDATION_REPORT = "results/guided/validation_report.md"
SPLIT = "results/splits/temporal_split_v1.json"
SUBGROUP = "results/verification/subgroup_descriptives_test.md"
SENS_METRICS = "results/sensitivity_corrected_v1/metrics.csv"
SENS_BOOT = "results/sensitivity_corrected_v1/bootstrap.json"
SENS_SPLIT = "results/sensitivity_corrected_v1/random_split_v1.json"
CHECKS = "results/verification/final_test_recompute_checks.json"
COMPARISON = "results/verification/guided_vs_autonomous_comparison.md"
EXPORT_MANIFEST = "models/export_manifest.json"
EXPORT_VALIDATION = "results/verification/model_export_validation.md"
INFERENCE_SMOKE = "results/verification/inference_smoke_test.md"
EXAMPLE_CUSTOMER = "models/example_customer.csv"
MISTAKES = "mistakes.md"
PLAN = "plan.md"
README = "README.md"
HUMAN_VERIFICATION = "results/verification/final_test_independent_verification.md"
PROVENANCE = "results/verification/provenance_update.md"
RECOMPUTE = "results/verification/final_test_recompute_report.md"
PROTOCOL_AUDIT = "results/audit/guided_random_split_protocol_discrepancy.md"
CONFORMANCE = "results/sensitivity_corrected_v1/conformance_check.json"


def read_text(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def read_json(rel: str):
    with (ROOT / rel).open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def read_rows(rel: str):
    with (ROOT / rel).open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def num(value, label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"VALIDATION FAILED: {label} is not numeric: {value!r}")
    if not (out == out and abs(out) != float("inf")):
        raise SystemExit(f"VALIDATION FAILED: {label} is not finite: {value!r}")
    return out


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"VALIDATION FAILED: {message}")


def close(a: float, b: float, tol: float, label: str) -> None:
    if abs(a - b) > tol:
        raise SystemExit(
            f"VALIDATION FAILED: {label}: {a!r} vs {b!r} differ by {abs(a-b):.3e} "
            f"(tolerance {tol:.1e})"
        )


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def strip_md(value: str) -> str:
    value = value.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", value).strip()


def parse_subgroup(doc: str, metric: str):
    for line in doc.splitlines():
        if line.startswith(f"| {metric} |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            require(len(cells) == 3, f"subgroup row {metric!r} has {len(cells)} cells")
            return num(cells[1], f"subgroup {metric} left"), num(
                cells[2], f"subgroup {metric} right"
            )
    raise SystemExit(f"VALIDATION FAILED: subgroup metric {metric!r} not found")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Load and validate primary chronological test artifacts
# ---------------------------------------------------------------------------
metric_rows = read_rows(TEST_METRICS)
require(len(metric_rows) == 3, f"test_metrics.csv should have 3 model rows, got {len(metric_rows)}")
metrics = {row["model"]: row for row in metric_rows}
require(set(metrics) == set(MODELS), f"test_metrics.csv models are {sorted(metrics)}")

n_test = int(num(metrics["A"]["n_test"], "n_test"))
n_positive = int(num(metrics["A"]["n_positive"], "n_positive"))
prevalence = num(metrics["A"]["prevalence"], "prevalence")
for model in MODELS:
    row = metrics[model]
    require(int(num(row["n_test"], f"{model} n_test")) == n_test, f"{model} n_test mismatch")
    require(int(num(row["n_positive"], f"{model} n_positive")) == n_positive, f"{model} positives mismatch")
    close(num(row["prevalence"], f"{model} prevalence"), prevalence, 1e-15, f"{model} prevalence")
    for field in ("roc_auc", "pr_auc", "brier", "mean_predicted", "lift"):
        value = num(row[field], f"{model} {field}")
        require(0.0 <= value <= 1.0 if field != "lift" else value > 0.0, f"{model} {field} out of range")
close(prevalence, n_positive / n_test, 1e-15, "prevalence == n_positive / n_test")
require(n_test == 6783 and n_positive == 2669, "unexpected primary test size")
require("NOT deployable" in metrics["C"]["role"], "Model C is not labelled non-deployable")

models_out = {}
for model in MODELS:
    row = metrics[model]
    models_out[model] = {
        "model": model,
        "role": row["role"],
        "diagnostic": model == "C",
        "roc_auc": num(row["roc_auc"], f"{model} roc_auc"),
        "pr_auc": num(row["pr_auc"], f"{model} pr_auc"),
        "pr_auc_no_skill": num(row["pr_auc_no_skill"], f"{model} pr_auc_no_skill"),
        "brier": num(row["brier"], f"{model} brier"),
        "mean_predicted": num(row["mean_predicted"], f"{model} mean_predicted"),
        "lift": num(row["lift"], f"{model} lift"),
        "lift_k": int(num(row["lift_k"], f"{model} lift_k")),
        "lift_tp_in_top_k": int(num(row["lift_tp_in_top_k"], f"{model} lift_tp")),
        "lift_precision_at_k": num(row["lift_precision_at_k"], f"{model} lift precision"),
        "lift_recall_at_k": num(row["lift_recall_at_k"], f"{model} lift recall"),
        "lift_expected_tp_by_chance": num(row["lift_expected_tp_by_chance"], f"{model} lift expected"),
        "selected_C": float(num(row["selected_C"], f"{model} selected_C")),
    }

# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------
hyp = read_json(HYPOTHESIS)
require(int(hyp["n_boot"]) == 2000, "bootstrap n_boot is not 2000")
require(int(hyp["seed"]) == 42, "bootstrap seed is not 42")
require(int(hyp["resamples_skipped_single_class"]) == 0, "single-class resamples were skipped")

primary = {}
for pair in ("B-A", "C-B"):
    primary[pair] = {}
    for metric in ("roc_auc", "pr_auc"):
        entry = hyp[pair][f"delta_{metric}"]
        point = num(entry["point"], f"{pair} {metric} point")
        lo, hi = (num(v, f"{pair} {metric} CI") for v in entry["ci95"])
        require(lo <= point <= hi, f"{pair} {metric} point outside its CI")
        require(int(entry["n_valid_resamples"]) == 2000, f"{pair} {metric} invalid resamples")
        primary[pair][metric] = {"point": point, "lo": lo, "hi": hi}
for pair, left, right in (("B-A", "B", "A"), ("C-B", "C", "B")):
    close(
        primary[pair]["roc_auc"]["point"],
        models_out[left]["roc_auc"] - models_out[right]["roc_auc"],
        1e-12,
        f"{pair} ROC-AUC point equals metric difference",
    )
    close(
        primary[pair]["pr_auc"]["point"],
        models_out[left]["pr_auc"] - models_out[right]["pr_auc"],
        1e-12,
        f"{pair} AP point equals metric difference",
    )
require(hyp["H1_classification"] == "H0 rejected, materially relevant", "H1 classification unexpected")
require(hyp["H2_material"] is True, "H2 not material")
close(num(hyp["H2_delta_roc_auc_CB"], "H2 delta"), primary["C-B"]["roc_auc"]["point"], 1e-15, "H2 delta")

# ---------------------------------------------------------------------------
# Calibration deciles
# ---------------------------------------------------------------------------
cal_rows = read_rows(CALIBRATION)
require(len(cal_rows) == 30, f"calibration should have 30 rows, got {len(cal_rows)}")
calibration = {model: [] for model in MODELS}
for row in cal_rows:
    model = row["model"]
    require(model in calibration, f"unexpected calibration model {model!r}")
    calibration[model].append(
        {
            "bin": row["bin"],
            "n": int(num(row["n"], f"calibration {model} n")),
            "mean_predicted": num(row["mean_predicted"], f"calibration {model} predicted"),
            "observed_rate": num(row["observed_rate"], f"calibration {model} observed"),
        }
    )
for model in MODELS:
    bins = calibration[model]
    require(len(bins) == 10, f"calibration {model} has {len(bins)} bins, expected 10")
    require(sum(b["n"] for b in bins) == n_test, f"calibration {model} bin counts do not sum to n_test")
    weighted = sum(b["n"] * b["mean_predicted"] for b in bins) / sum(b["n"] for b in bins)
    close(weighted, models_out[model]["mean_predicted"], 1e-12, f"calibration {model} weighted mean")
    for b in bins:
        require(0 <= b["observed_rate"] <= 1, f"calibration {model} observed rate out of range")
        require(0 <= b["mean_predicted"] <= 1, f"calibration {model} predicted out of range")
hindsight_brier = prevalence * (1 - prevalence)
close(hindsight_brier, 0.2386542798098002, 1e-15, "hindsight constant Brier")

# ---------------------------------------------------------------------------
# Hyperparameter selection (guided only; validation-side)
# ---------------------------------------------------------------------------
hp_rows = read_rows(HYPERPARAMS)
require(len(hp_rows) == 27, f"hyperparameter table should have 27 rows, got {len(hp_rows)}")
hyper = {model: [] for model in MODELS}
for row in hp_rows:
    model = row["model"]
    require(model in hyper, f"unexpected hyperparameter model {model!r}")
    hyper[model].append(
        {
            "C": float(num(row["C"], f"hyper {model} C")),
            "mean_cv_roc_auc": num(row["mean_cv_roc_auc"], f"hyper {model} cv"),
            "std_cv_roc_auc": num(row["std_cv_roc_auc"], f"hyper {model} cv std"),
            "rank": int(num(row["rank"], f"hyper {model} rank")),
        }
    )
for model in MODELS:
    rows = hyper[model]
    require(len(rows) == 9, f"hyperparameter model {model} has {len(rows)} rows, expected 9")
    best = [r for r in rows if r["rank"] == 1]
    require(len(best) == 1, f"hyperparameter model {model} has {len(best)} rank-1 rows")
    close(best[0]["C"], models_out[model]["selected_C"], 1e-15, f"{model} selected C")
    for r in rows:
        r["selected"] = r["rank"] == 1
    rows.sort(key=lambda r: r["C"])

# ---------------------------------------------------------------------------
# Sensitivity analysis (secondary, post-test, optimistic)
# ---------------------------------------------------------------------------
sens_rows = read_rows(SENS_METRICS)
require(len(sens_rows) == 3, "sensitivity metrics should have 3 rows")
sens_models = {}
for row in sens_rows:
    model = row["model"]
    require(model in MODELS, f"unexpected sensitivity model {model!r}")
    sens_models[model] = {
        "model": model,
        "role": row["role"],
        "diagnostic": model == "C",
        "roc_auc": num(row["roc_auc"], f"sens {model} roc"),
        "pr_auc": num(row["pr_auc"], f"sens {model} ap"),
        "pr_auc_no_skill": num(row["pr_auc_no_skill"], f"sens {model} no skill"),
        "brier": num(row["brier"], f"sens {model} brier"),
        "mean_predicted": num(row["mean_predicted"], f"sens {model} mean"),
        "lift": num(row["lift"], f"sens {model} lift"),
        "selected_C": float(num(row["selected_C"], f"sens {model} C")),
        "n_test": int(num(row["n_test"], f"sens {model} n_test")),
        "n_positive": int(num(row["n_positive"], f"sens {model} n_positive")),
        "prevalence": num(row["prevalence"], f"sens {model} prevalence"),
    }
require(len({(m["n_test"], m["n_positive"]) for m in sens_models.values()}) == 1, "sensitivity sizes differ by model")
sens_boot = read_json(SENS_BOOT)
require(int(sens_boot["n_boot"]) == 2000 and int(sens_boot["seed"]) == 42, "sensitivity bootstrap parameters")
sensitivity = {
    "n_test": sens_models["A"]["n_test"],
    "n_positive": sens_models["A"]["n_positive"],
    "prevalence": sens_models["A"]["prevalence"],
    "models": sens_models,
    "n_boot": int(sens_boot["n_boot"]),
    "seed": int(sens_boot["seed"]),
}
for pair in ("B-A", "C-B"):
    sensitivity[pair] = {}
    for metric in ("roc_auc", "pr_auc"):
        entry = sens_boot[pair][f"delta_{metric}"]
        point = num(entry["point"], f"sens {pair} {metric} point")
        lo, hi = (num(v, f"sens {pair} {metric} CI") for v in entry["ci95"])
        require(lo <= point <= hi, f"sens {pair} {metric} point outside CI")
        sensitivity[pair][metric] = {"point": point, "lo": lo, "hi": hi}
close(
    sensitivity["B-A"]["roc_auc"]["point"],
    sens_models["B"]["roc_auc"] - sens_models["A"]["roc_auc"],
    1e-12,
    "sensitivity B-A ROC-AUC point",
)
close(
    sensitivity["C-B"]["roc_auc"]["point"],
    sens_models["C"]["roc_auc"] - sens_models["B"]["roc_auc"],
    1e-12,
    "sensitivity C-B ROC-AUC point",
)
sens_split = read_json(SENS_SPLIT)
fingerprint = sens_split["canonical_fingerprint_sha256"]
require(re.fullmatch(r"[0-9a-f]{64}", fingerprint) is not None, "sensitivity fingerprint malformed")
require(
    sens_split["reference_fingerprint_sha256"] == fingerprint,
    "sensitivity reference fingerprint differs from canonical fingerprint",
)
require("secondary" in sens_split["analysis_role"], "sensitivity role is not marked secondary")

# ---------------------------------------------------------------------------
# Split and chronology
# ---------------------------------------------------------------------------
split = read_json(SPLIT)
sizes = {k: int(v) for k, v in split["sizes"].items()}
require(sizes == {"train": 31647, "validation": 6781, "test": 6783}, f"unexpected split sizes {sizes}")
require(sum(sizes.values()) == int(split["n_rows"]) == 45211, "split sizes do not sum to n_rows")
require(all(split["verification"].values()), "split verification flags are not all true")
split_hash = split["split_hash_sha256"]
require(re.fullmatch(r"[0-9a-f]{64}", split_hash) is not None, "split hash malformed")

val_doc = read_text(VALIDATION_REPORT)
m_train = re.search(r"prevalence\s+([0-9.]+)\s*\(([\d,]+)/([\d,]+)\)", val_doc)
require(m_train is not None, "training prevalence not found in validation report")
train_prev = float(m_train.group(1))
require(int(m_train.group(2).replace(",", "")) == 1840, "training positives changed")
m_val = re.search(r"validation block has\s+([0-9.]+)\s*\(([\d,]+)/([\d,]+)\)", val_doc, re.S)
require(m_val is not None, "validation prevalence not found in validation report")
val_prev = float(m_val.group(1))
require(int(m_val.group(2).replace(",", "")) == 780, "validation positives changed")
close(train_prev, 1840 / 31647, 1e-6, "training prevalence (reported to 6 dp)")
close(val_prev, 780 / 6781, 1e-6, "validation prevalence (reported to 6 dp)")

# ---------------------------------------------------------------------------
# Subgroup descriptives
# ---------------------------------------------------------------------------
sub_doc = read_text(SUBGROUP)
sub_never_n, sub_prev_n = parse_subgroup(sub_doc, "N")
sub_never_prev, sub_prev_prev = parse_subgroup(sub_doc, "Prevalence")
sub_never_delta, sub_prev_delta = parse_subgroup(sub_doc, "Delta ROC-AUC B-A")
require(sub_never_n + sub_prev_n == n_test, "subgroup N does not sum to n_test")
subgroups = {
    "never": {"n": int(sub_never_n), "prevalence": sub_never_prev, "delta_roc_auc": sub_never_delta},
    "previous": {"n": int(sub_prev_n), "prevalence": sub_prev_prev, "delta_roc_auc": sub_prev_delta},
}

# ---------------------------------------------------------------------------
# Independent verification
# ---------------------------------------------------------------------------
checks = read_json(CHECKS)
require(checks["status"] == "PASS", "verification status is not PASS")
require(int(checks["n_checks"]) == 77, "verification n_checks is not 77")
require(int(checks["n_failed"]) == 0, "verification has failed checks")
require(len(checks["checks"]) == 77, "verification check list length is not 77")
require(all(c["passed"] is True for c in checks["checks"]), "a verification check is not passed")
require(float(checks["tolerance"]) == 1e-06, "verification tolerance is not 1e-6")
require(int(checks["n_boot"]) == 2000 and int(checks["seed"]) == 42, "verification bootstrap parameters")


def check_group(cid: str) -> str:
    if cid.startswith(("hash_", "split_", "dataset_", "no_removed", "no_changed")):
        return "Integrity"
    if cid.startswith(("n_rows", "n_positive", "no_nulls", "row_", "prevalence")):
        return "Alignment"
    if re.match(r"^[ABC]_", cid):
        return "Metrics"
    if cid.startswith(("B-A_", "C-B_", "boot_", "H1_", "H2_")):
        return "Bootstrap & hypotheses"
    if cid.startswith("model_c"):
        return "Model role"
    return "Other"


checks_out = []
for c in checks["checks"]:
    checks_out.append(
        {
            "id": c["id"],
            "description": c["description"],
            "group": check_group(c["id"]),
            "passed": bool(c["passed"]),
        }
    )
require(not any(c["group"] == "Other" for c in checks_out), "unclassified verification checks present")
check_counts = Counter(c["group"] for c in checks_out)
require(sum(check_counts.values()) == 77, "verification group counts do not sum to 77")
integrity = checks["integrity"]
require(
    int(integrity["n_files_before"]) == int(integrity["n_files_after"]) == 112,
    "verification file counts are not 112/112",
)
require(not integrity["added"] and not integrity["removed"] and not integrity["changed"], "verification integrity lists are not empty")

# ---------------------------------------------------------------------------
# Model export validation
# ---------------------------------------------------------------------------
manifest = read_json(EXPORT_MANIFEST)
require(manifest["post_test_reconstruction"] is True, "manifest does not mark reconstruction")
require(manifest["test_labels_read"] is False, "manifest says test labels were read")
require(manifest["model_c_trained"] is False, "manifest says Model C was trained")
require(manifest["split"]["sizes"] == sizes, "manifest split sizes differ from split file")
for model in ("A", "B"):
    close(
        float(manifest["model_spec"]["C"][model]),
        models_out[model]["selected_C"],
        1e-15,
        f"manifest frozen C for {model}",
    )
export_doc = read_text(EXPORT_VALIDATION)
require("0.000e+00" in export_doc, "export validation does not report zero max difference")
require("1e-12" in export_doc, "export validation tolerance not found")
require("6,781" in export_doc or "6781" in export_doc, "export validation row count not found")
export_models = {}
for model in ("A", "B"):
    artifact = manifest["artifacts"][f"models/{model}.joblib"]
    require(re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"]) is not None, f"{model} model hash malformed")
    export_models[model] = {
        "sha256": artifact["sha256"],
        "bytes": int(artifact["bytes"]),
        "max_abs_diff": 0.0,
        "tolerance": 1e-12,
    }
require(set(manifest["artifacts"]) == {"models/A.joblib", "models/B.joblib"}, "unexpected exported artifacts")

# ---------------------------------------------------------------------------
# Synthetic inference demonstration
# ---------------------------------------------------------------------------
inference_doc = read_text(INFERENCE_SMOKE)
example_rows = read_rows(EXAMPLE_CUSTOMER)
require(len(example_rows) == 2, "example_customer.csv should contain exactly two rows")
inference = {
    "command": r".venv\Scripts\python.exe -m my_analysis.predict --input models\example_customer.csv",
    "command_note": "Windows PowerShell, run from the repository root (as documented).",
    "customers": [],
}
for idx, row in enumerate(example_rows):
    cid = row["customer_id"]
    pattern = re.compile(
        rf"{re.escape(cid)}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)%\s+([0-9.]+)%"
    )
    match = pattern.search(inference_doc)
    require(match is not None, f"synthetic scores for {cid} not found")
    p_a, p_b, pct_a, pct_b = (float(v) for v in match.groups())
    close(p_a * 100, pct_a, 0.0051, f"{cid} A percentage")
    close(p_b * 100, pct_b, 0.0051, f"{cid} B percentage")
    require(0 <= p_a <= 1 and 0 <= p_b <= 1, f"{cid} score outside [0,1]")
    inference["customers"].append(
        {
            "id": cid,
            "features": {k: row[k] for k in row if k != "customer_id"},
            "p_a": p_a,
            "p_b": p_b,
            "pct_a": pct_a,
            "pct_b": pct_b,
        }
    )

# ---------------------------------------------------------------------------
# Guided vs autonomous comparison (verified report only; no autonomous files read)
# ---------------------------------------------------------------------------
comp_doc = read_text(COMPARISON)
require("max |p_guided − p_autonomous| = 0.000e+00" in comp_doc, "bit-identical A/B claim not found")
require("max |Δp| = 0.2505" in comp_doc, "Model C max difference not found")
require("C = 0.001" in comp_doc and "C = 0.01" in comp_doc, "selected C values not found")
require("9 values vs 5 values" in comp_doc or "9 values vs 5" in comp_doc, "grid sizes not found")
require("num__age = -0.05842904778480015" in comp_doc, "identical coefficient not found")
comparison = {
    "ab_max_abs_diff": 0.0,
    "c_max_abs_diff": 0.2505,
    "guided_c": 0.001,
    "autonomous_c": 0.01,
    "grid_guided": 9,
    "grid_autonomous": 5,
    "identical_coefficient": -0.05842904778480015,
    "source": COMPARISON,
}

# ---------------------------------------------------------------------------
# Mistakes ledger (append-only)
# ---------------------------------------------------------------------------
mistake_doc = read_text(MISTAKES)
mistake_items = []
for line in mistake_doc.splitlines():
    if not line.startswith("| 2026-"):
        continue
    cells = re.split(r"\s\|\s", line.strip().strip("|"))
    if len(cells) != 6:
        raise SystemExit(f"VALIDATION FAILED: mistake row has {len(cells)} cells, expected 6")
    date, action, mistake, detection, correction, lesson = cells
    head = re.match(r"^\*\*\[([^\]]+)\]\s+([HAS])-(\d+)\*\*\s*[—–-]\s*(.*)$", action, re.S)
    require(head is not None, f"could not parse mistake heading: {action[:80]}")
    label, letter, number, title = head.groups()
    category = {"H": "Human design omission", "A": "Agent mistake", "S": "Self-correction"}[letter]
    if "Human/process" in label:
        category = "Human/process omission"
    if "Successful self-correction" in label:
        category = "Successful self-correction"
    if "Accepted, documented" in label:
        category = "Accepted, documented"
    mistake_items.append(
        {
            "id": f"{letter}-{number}",
            "letter": letter,
            "category": category,
            "date": date.strip(),
            "title": strip_md(title),
            "action": action,
            "mistake": mistake,
            "detection": detection,
            "correction": correction,
            "lesson": lesson,
            "historical": True,
        }
    )
require(len(mistake_items) == 16, f"expected 16 mistake rows, got {len(mistake_items)}")
counts = Counter(item["letter"] for item in mistake_items)
require(counts == {"H": 6, "A": 5, "S": 5}, f"unexpected mistake category counts {dict(counts)}")

status_start = mistake_doc.find("## Status update")
reflection_start = mistake_doc.find("## Reflection")
require(status_start > 0 and reflection_start > status_start, "status/reflection sections not found")
status_doc = mistake_doc[status_start:reflection_start]
reflection_doc = mistake_doc[reflection_start:]
fp_match = re.search(r"`([0-9a-f]{64})`", status_doc)
require(fp_match is not None, "status update fingerprint not found")
require(fp_match.group(1) == fingerprint, "status update fingerprint differs from sensitivity split")
delta_match = re.search(r"B−A random ΔROC-AUC ([0-9.]+)", status_doc)
require(delta_match is not None, "status update random delta not found")
random_delta = float(delta_match.group(1))
close(random_delta, 0.054570, 5e-7, "status update random delta")

updates = [
    {
        "id": "STATUS",
        "letter": "U",
        "category": "Later status update",
        "date": "2026-11-16",
        "title": "A-5 closed, not rewritten",
        "body": (
            "Append-only status entry. The original A-5 row remains exactly as written. "
            "The human author regenerated the authorised test-first split, PHASE 4 was "
            "authorised, and the corrected evaluation was executed without regenerating "
            "the split. It remains a post-test secondary sensitivity result, optimistic "
            "by construction, and does not replace the primary chronological test."
        ),
        "facts": [
            {"k": "Corrected split fingerprint", "v": fingerprint},
            {"k": "B−A random ΔROC-AUC", "v": f"{random_delta:.6f}"},
            {"k": "Role", "v": "Secondary sensitivity · not primary evidence"},
        ],
        "note": "The correction is a later record; the historical mistake is not relabelled as valid.",
    },
    {
        "id": "REFLECT",
        "letter": "R",
        "category": "Dated reflection",
        "date": "2026-11-16",
        "title": "Calibration risk: H-7",
        "body": (
            "Recorded as a dated reflection rather than a new table row, so no earlier "
            "row is rewritten. The warning was visible on validation before the test "
            "block was opened: prevalence rose and mean predictions were roughly half "
            "the observed rate. The test block repeated the pattern at larger magnitude. "
            "The defect was documented but not escalated into a stop-and-review decision."
        ),
        "facts": [
            {"k": "Validation prevalence", "v": f"{val_prev:.6f}"},
            {"k": "Test prevalence", "v": f"{prevalence:.6f}"},
            {"k": "Mean predicted on test", "v": f"A {models_out['A']['mean_predicted']:.4f} · B {models_out['B']['mean_predicted']:.4f}"},
        ],
        "note": "Documentation is not mitigation. The test block must not be used to recalibrate or reselect.",
    },
]

# ---------------------------------------------------------------------------
# Human-in-the-loop verification provenance
#
# Three different kinds of checking are documented in this project:
#   1. the human author, ad-hoc in Windows PowerShell (no stored script);
#   2. an OpenCode-written recomputation script with 77 named checks;
#   3. read-only agent audits that produced written reports.
# They are not interchangeable, so each claim below is tied to its own file.
# ---------------------------------------------------------------------------
def flatten(text: str) -> str:
    """Collapse markdown whitespace and emphasis so phrases can be matched."""
    return re.sub(r"\s+", " ", text.replace("**", "").replace("`", "")).strip()


def hex_after(text: str, label: str) -> str:
    idx = text.find(label)
    require(idx >= 0, f"label {label!r} not found in verification evidence")
    match = re.search(r"[0-9a-f]{64}", text[idx : idx + 400])
    require(match is not None, f"no 64-hex digest after {label!r}")
    return match.group(0)


human_doc = read_text(HUMAN_VERIFICATION)
prov_doc = read_text(PROVENANCE)
recompute_doc = read_text(RECOMPUTE)
audit_doc = read_text(PROTOCOL_AUDIT)
conformance = read_json(CONFORMANCE)

human_flat = flatten(human_doc)
prov_flat = flatten(prov_doc)
recompute_flat = flatten(recompute_doc)
audit_flat = flatten(audit_doc)

# --- attribution: who did what ------------------------------------------------
require(
    "performed independently by the human project author" in human_flat,
    "human verification report does not attribute the work to the human author",
)
require(
    "directly in Windows PowerShell" in human_flat,
    "human verification report does not record PowerShell execution",
)
require(
    "ad-hoc PowerShell, not from a stored script" in prov_flat,
    "provenance note does not distinguish ad-hoc PowerShell from a stored script",
)
require(
    "supplements but does not replace the human PowerShell verification" in recompute_flat,
    "recomputation report does not defer to the human verification",
)
require(
    "OpenCode-generated, independently implemented computational check" in recompute_flat,
    "recomputation report does not describe itself as an OpenCode-generated check",
)
require(
    "Guided vs Autonomous comparative audit" in prov_flat
    and "Random-split protocol root-cause audit" in prov_flat,
    "provenance note does not list the two agent audits",
)

# --- what the human check covered --------------------------------------------
for phrase in (
    "6,783 rows",
    "2,669 positive outcomes",
    "ROC-AUC",
    "average precision",
    "Brier score",
    "lift at the top 10%",
    "numpy.random.default_rng(42)",
    "matched the corresponding reported value within 1e-6",
    "test labels",
):
    require(phrase in human_flat, f"human verification report is missing {phrase!r}")

human_bootstrap_ci = re.search(
    r"Δ ROC-AUC \(B − A\) \| \[([0-9.]+), ([0-9.]+)\]", human_flat
)
if human_bootstrap_ci is None:
    human_bootstrap_ci = re.search(r"Δ ROC-AUC .{0,20}\[([0-9.]+), ([0-9.]+)\]", human_flat)
require(human_bootstrap_ci is not None, "independent human B−A CI not found")
human_ci_lo, human_ci_hi = (float(v) for v in human_bootstrap_ci.groups())
close(human_ci_lo, 0.026085, 5e-7, "human independent CI lower bound")
close(human_ci_hi, 0.034267, 5e-7, "human independent CI upper bound")

# --- the automated check ------------------------------------------------------
require("77 of 77 checks passed" in recompute_flat, "77/77 PASS not recorded")
require(int(checks["n_checks"]) == 77 and int(checks["n_failed"]) == 0, "77/77 mismatch")
require(
    "verify_final_test_recompute.py" in recompute_flat,
    "recomputation script name not recorded",
)

# --- the A-5 protocol discrepancy --------------------------------------------
shared = {}
for block, label in (
    ("train", "Training"),
    ("validation", "Validation"),
    ("test", "Test"),
):
    match = re.search(rf"{label}: ([\d,]+) of ([\d,]+)", audit_flat)
    require(match is not None, f"shared-row count for {block} not found in audit")
    shared[block] = {
        "shared": int(match.group(1).replace(",", "")),
        "total": int(match.group(2).replace(",", "")),
    }
require(shared["train"] == {"shared": 26036, "total": 31647}, "train overlap changed")
require(shared["validation"] == {"shared": 563, "total": 6781}, "validation overlap changed")
require(shared["test"] == {"shared": 3381, "total": 6783}, "test overlap changed")

overlap_guided = re.search(r"Guided: ([\d,]+)", audit_flat)
overlap_reference = re.search(r"reference: ([\d,]+)", audit_flat)
require(overlap_guided is not None and overlap_reference is not None, "overlap counts not found")
require(int(overlap_guided.group(1).replace(",", "")) == 1008, "guided overlap changed")
require(int(overlap_reference.group(1).replace(",", "")) == 1022, "reference overlap changed")

require(
    "Reproducibility check, not a conformance check" in audit_flat
    or "reproducibility check, not a conformance check" in audit_flat,
    "audit does not state the reproducibility/conformance distinction",
)
require(
    "return assignments reversed" in audit_flat,
    "the reversed-return-order detour is not recorded in the audit",
)
require(
    "preserved byte-for-byte as evidence" in audit_flat,
    "audit does not record that the erroneous run was preserved",
)
require("status closed, not rewritten" in flatten(read_text(MISTAKES)), "A-5 status wording changed")

erroneous_hash = hex_after(audit_doc, "The erroneous split hash was:")
erroneous_script_sha = hex_after(audit_doc, "The erroneous script SHA-256 was:")
require(
    conformance["all_checks_passed"] is True,
    "corrected split conformance checks did not all pass",
)
require(
    conformance["reference_comparison"]["all_blocks_equal"] is True,
    "corrected split does not match the authorised reference",
)
require(
    int(conformance["expected_vs_actual"]["chronological_test_overlap"]["actual"]) == 1022,
    "corrected split chronological overlap changed",
)

hitl = {
    "counts": {"human_batches": 3, "automated": int(checks["n_checks"]), "audits": 2},
    "human": {
        "executor": "Human project author",
        "environment": "Windows PowerShell, ad-hoc",
        "rerunnable_script": False,
        "ci": {"lo": human_ci_lo, "hi": human_ci_hi},
        "source": HUMAN_VERIFICATION,
        "note_source": PROVENANCE,
    },
    "automated": {
        "executor": "OpenCode audit session",
        "script": "results/verification/verify_final_test_recompute.py",
        "n_checks": int(checks["n_checks"]),
        "n_failed": int(checks["n_failed"]),
        "tolerance": float(checks["tolerance"]),
        "groups": dict(check_counts),
        "source": RECOMPUTE,
        "json": CHECKS,
    },
    "a5": {
        "erroneous_hash": erroneous_hash,
        "erroneous_script_sha256": erroneous_script_sha,
        "corrected_fingerprint": fingerprint,
        "shared": shared,
        "overlap_guided": 1008,
        "overlap_reference": 1022,
        "archive": "results/sensitivity_erroneous_v1/",
        "steps": [
            {
                "id": "agent",
                "actor": "Agent",
                "title": "Train-first split",
                "lead": "Original AI implementation",
                "body": (
                    "The script split off the training block first (test_size=13564) and then "
                    "divided the remainder into validation and test (test_size=6783). The "
                    "authorised procedure was the reverse: hold out the 6,783 test rows first."
                ),
                "facts": [
                    {"k": "Split order", "v": "train-first"},
                    {"k": "Block sizes", "v": "31,647 / 6,781 / 6,783"},
                    {"k": "Self-check", "v": "reproducible from seed"},
                ],
                "source": "results/audit/guided_random_split_protocol_discrepancy.md §3",
            },
            {
                "id": "human",
                "actor": "Human",
                "title": "Cross-run comparison",
                "lead": "Human-directed check",
                "body": (
                    "The project author independently regenerated the authorised test-first "
                    "procedure and compared its index sets with the saved guided and autonomous "
                    "sensitivity splits."
                ),
                "facts": [
                    {"k": "Procedure rebuilt", "v": "test-first"},
                    {"k": "Compared against", "v": "guided + autonomous splits"},
                    {"k": "Method", "v": "index-set comparison"},
                ],
                "source": "results/audit/guided_random_split_protocol_discrepancy.md §4",
            },
            {
                "id": "found",
                "actor": "Finding",
                "title": "Protocol discrepancy",
                "lead": "What the comparison exposed",
                "body": (
                    "Equal block sizes and internal reproducibility had not proved correct "
                    "membership. Guided matched the authorised split for none of the three "
                    "blocks; autonomous matched for all three."
                ),
                "facts": [
                    {"k": "Guided matches", "v": "FALSE · all blocks"},
                    {"k": "Autonomous matches", "v": "TRUE · all blocks"},
                    {
                        "k": "Shared rows",
                        "v": f"{shared['train']['shared']:,} / {shared['validation']['shared']:,} / {shared['test']['shared']:,}",
                    },
                ],
                "source": "results/audit/guided_random_split_protocol_discrepancy.md §4",
            },
            {
                "id": "fixed",
                "actor": "Evidence",
                "title": "Corrected split verified",
                "lead": "Regenerated, then checked",
                "body": (
                    "The corrected split was regenerated by the human author and compared with "
                    "an independently implemented reference before any model was fitted on it."
                ),
                "facts": [
                    {"k": "Fingerprint", "v": fingerprint[:16] + "…"},
                    {"k": "Matches reference", "v": "TRUE · all blocks"},
                    {"k": "Chronological overlap", "v": "1,022 rows"},
                ],
                "source": "results/sensitivity_corrected_v1/conformance_check.json",
            },
            {
                "id": "kept",
                "actor": "Evidence",
                "title": "Erroneous run retained",
                "lead": "Preserved, not overwritten",
                "body": (
                    "The incorrect outputs stay in the repository, byte-for-byte, under an "
                    "explicit invalid banner. They are historical evidence of the error and are "
                    "never cited as valid sensitivity results."
                ),
                "facts": [
                    {"k": "Archive", "v": "sensitivity_erroneous_v1/"},
                    {"k": "Inventory", "v": "SHA256SUMS.txt"},
                    {"k": "Status", "v": "invalid · never cited"},
                ],
                "source": "results/sensitivity_erroneous_v1/ARCHIVE_NOTICE.md",
            },
        ],
        "detour": (
            "The audit's own proposed repair snippet reversed sklearn's (train, test) return "
            "order. The human author caught it before execution, and the corrected script uses "
            "the authorised call order."
        ),
    },
    "limits": [
        "Reproducing metrics from saved predictions does not establish that feature engineering, preprocessing, hyperparameter selection or training were correct.",
        "No full clean-environment retraining run from the pinned lockfile has been demonstrated.",
        "The automated chronology assessment returned INCONCLUSIVE; the ordering rests on a documented human adjudication.",
        "The calibration defect is documented and measured, not corrected.",
        "There is no second independent holdout: the random split shares 1,022 rows with the chronological test block.",
    ],
    "not_claimed": [
        "a second independent holdout",
        "clean-room retraining",
        "complete test-label isolation",
        "resolved calibration",
        "causal effects",
    ],
}

# ---------------------------------------------------------------------------
# Derived, explicitly labelled arithmetic
# ---------------------------------------------------------------------------
ratios = {m: prevalence / models_out[m]["mean_predicted"] for m in MODELS}
brier_diff = models_out["A"]["brier"] - models_out["B"]["brier"]
top_k = models_out["B"]["lift_k"]

# Read the README only to verify the documented random/chronological overlap.
readme_doc = read_text(README)
overlap_match = re.search(r"overlaps the chronological test block by ([\d,]+) rows", readme_doc)
require(overlap_match is not None, "random/chronological overlap not found in README")
overlap_rows = int(overlap_match.group(1).replace(",", ""))

plan_doc = read_text(PLAN)
require("ΔROC-AUC ≥ 0.01" in plan_doc, "materiality threshold not found in plan")
require("≥ 0.05" in plan_doc, "H2 threshold not found in plan")

DATA = {
    "meta": {
        "generated_utc": utc_now(),
        "generator": "dashboard/build_dashboard.py",
        "test_label": "Frozen chronological test",
        "bootstrap_label": "Paired bootstrap · percentile 95% CI",
    },
    "test": {
        "n_test": n_test,
        "n_positive": n_positive,
        "prevalence": prevalence,
        "models": models_out,
        "bootstrap": {"n_boot": int(hyp["n_boot"]), "seed": int(hyp["seed"]), "skipped": int(hyp["resamples_skipped_single_class"])},
        "h1_classification": hyp["H1_classification"],
        "h2_threshold": float(hyp["H2_threshold"]),
        "h2_material": bool(hyp["H2_material"]),
        "hindsight_brier": hindsight_brier,
    },
    "primary": primary,
    "sensitivity": {
        **sensitivity,
        "fingerprint": fingerprint,
        "overlap_test_rows": overlap_rows,
        "role": sens_split["analysis_role"],
    },
    "calibration": calibration,
    "hyperparameters": hyper,
    "split": {
        "sizes": sizes,
        "n_rows": int(split["n_rows"]),
        "hash": split_hash,
        "train_prevalence": train_prev,
        "train_positive": 1840,
        "validation_prevalence": val_prev,
        "validation_positive": 780,
        "strategy": split["strategy"],
    },
    "subgroups": subgroups,
    "comparison": comparison,
    "hitl": hitl,
    "verification": {
        "n_checks": int(checks["n_checks"]),
        "n_failed": int(checks["n_failed"]),
        "status": checks["status"],
        "tolerance": float(checks["tolerance"]),
        "n_boot": int(checks["n_boot"]),
        "seed": int(checks["seed"]),
        "valid_resamples": int(checks["valid_resamples"]),
        "skipped_single_class": int(checks["skipped_single_class"]),
        "integrity": integrity,
        "scope": checks["scope"],
        "model_c": checks["model_c"],
        "supplements": checks["supplements"],
        "groups": dict(check_counts),
        "checks": checks_out,
        "source": CHECKS,
    },
    "export": {
        "models": export_models,
        "validation_rows": 6781,
        "tolerance": 1e-12,
        "source": EXPORT_VALIDATION,
        "manifest": EXPORT_MANIFEST,
    },
    "inference": inference,
    "mistakes": {
        "items": mistake_items,
        "updates": updates,
        "counts": dict(counts),
        "reflection_quote": "Documentation is not mitigation. A known, measured defect should have been recorded as a defect with an owner and a decision attached, at the moment it was first observed.",
    },
    "ratios": ratios,
    "brier_diff": brier_diff,
    "top_k": top_k,
    "sources": [
        TEST_METRICS,
        HYPOTHESIS,
        CALIBRATION,
        CALIBRATION_NOTE,
        HYPERPARAMS,
        VALIDATION_REPORT,
        SPLIT,
        SUBGROUP,
        SENS_METRICS,
        SENS_BOOT,
        SENS_SPLIT,
        CHECKS,
        COMPARISON,
        EXPORT_MANIFEST,
        EXPORT_VALIDATION,
        INFERENCE_SMOKE,
        EXAMPLE_CUSTOMER,
        MISTAKES,
        PLAN,
        README,
        HUMAN_VERIFICATION,
        PROVENANCE,
        RECOMPUTE,
        PROTOCOL_AUDIT,
        CONFORMANCE,
    ],
}

print(f"loaded primary test metrics: {len(metric_rows)} model rows (kept all 3: A, B, C)")
print(f"loaded calibration: {len(cal_rows)} decile rows (kept all 30: 10 bins x 3 models)")
print(f"loaded hyperparameter grid: {len(hp_rows)} rows (kept all 27: 9 x 3 models)")
print(f"loaded verification checks: {len(checks_out)} rows (kept all; 0 dropped; groups={dict(check_counts)})")
print(f"parsed mistake ledger: {len(mistake_items)} rows (kept all; H=6, A=5, S=5) + {len(updates)} later updates")
print(f"validation complete: n_test={n_test}, positives={n_positive}, prevalence={prevalence!r}")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bank Term-Deposit Prediction · Project Dashboard</title>
<style>
:root{
  --bg:#070a0f;
  --bg-2:#090d14;
  --panel:rgba(13,19,28,.88);
  --card:linear-gradient(180deg,rgba(19,27,39,.92),rgba(12,18,27,.96));
  --card-solid:#101722;
  --line:rgba(148,163,184,.14);
  --line-strong:rgba(148,163,184,.24);
  --text:#e8eef8;
  --muted:#93a2b8;
  --faint:#64748b;
  --teal:#5eead4;
  --teal-dim:rgba(94,234,212,.13);
  --blue:#7aa2ff;
  --blue-dim:rgba(122,162,255,.13);
  --violet:#c4b5fd;
  --violet-dim:rgba(196,181,253,.13);
  --amber:#f5b461;
  --amber-dim:rgba(245,180,97,.12);
  --red:#fb7185;
  --red-dim:rgba(251,113,133,.12);
  --green:#4ade80;
  --radius:18px;
  --radius-sm:12px;
  --shadow:0 24px 60px -30px rgba(0,0,0,.75);
  --sidebar:252px;
  --mono:"Cascadia Code","Cascadia Mono",Consolas,ui-monospace,monospace;
  --sans:"Segoe UI Variable Text","Segoe UI",system-ui,-apple-system,"Helvetica Neue",sans-serif;
  --display:"Segoe UI Variable Display","Segoe UI",system-ui,-apple-system,sans-serif;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;
  min-height:100vh;
  color:var(--text);
  background:
    radial-gradient(1100px 620px at 8% -12%,rgba(94,234,212,.09),transparent 60%),
    radial-gradient(900px 520px at 98% 2%,rgba(122,162,255,.10),transparent 58%),
    radial-gradient(800px 500px at 70% 100%,rgba(196,181,253,.05),transparent 62%),
    var(--bg);
  font:15px/1.58 var(--sans);
  -webkit-font-smoothing:antialiased;
}
body::before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(148,163,184,.032) 1px,transparent 1px),linear-gradient(90deg,rgba(148,163,184,.032) 1px,transparent 1px);
  background-size:56px 56px;
  mask-image:linear-gradient(180deg,rgba(0,0,0,.7),transparent 88%);
}
::selection{background:rgba(94,234,212,.25);color:#fff}
a{color:var(--teal);text-decoration:none}
a:hover{text-decoration:underline}
button{font:inherit}
code{font-family:var(--mono);font-size:.86em}
.app{display:flex;min-height:100vh;position:relative;z-index:1}

/* ---------- sidebar ---------- */
.sidebar{
  position:fixed;inset:0 auto 0 0;width:var(--sidebar);z-index:40;
  display:flex;flex-direction:column;gap:22px;
  padding:22px 16px 18px;
  background:linear-gradient(180deg,rgba(9,14,21,.98),rgba(7,11,17,.99));
  border-right:1px solid var(--line);
  transition:transform .28s cubic-bezier(.22,.61,.36,1);
}
.brand{display:flex;gap:12px;align-items:center;padding:2px 6px}
.brand-mark{
  width:38px;height:38px;flex:0 0 38px;border-radius:12px;
  display:grid;place-items:center;
  background:linear-gradient(145deg,rgba(94,234,212,.22),rgba(122,162,255,.14));
  border:1px solid rgba(94,234,212,.3);
  color:var(--teal);font-family:var(--mono);font-size:13px;font-weight:700;letter-spacing:.06em;
  box-shadow:inset 0 1px 0 rgba(255,255,255,.08);
}
.brand-text strong{display:block;font-family:var(--display);font-size:14px;letter-spacing:.01em}
.brand-text span{display:block;color:var(--faint);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;margin-top:2px}
.nav{display:flex;flex-direction:column;gap:3px}
.nav-label{font-size:10px;letter-spacing:.16em;text-transform:uppercase;color:var(--faint);padding:0 10px 8px}
.nav button{
  appearance:none;border:1px solid transparent;background:transparent;color:var(--muted);
  display:flex;align-items:center;gap:11px;width:100%;text-align:left;
  padding:9px 10px;border-radius:10px;cursor:pointer;font-size:13px;font-weight:550;
  transition:background .18s,color .18s,border-color .18s,transform .18s;
}
.nav button:hover{color:var(--text);background:rgba(148,163,184,.06)}
.nav button.active{
  color:var(--text);
  background:linear-gradient(90deg,rgba(94,234,212,.13),rgba(94,234,212,.03));
  border-color:rgba(94,234,212,.18);
  box-shadow:inset 2px 0 0 var(--teal);
}
.nav .n{font-family:var(--mono);font-size:10.5px;color:var(--faint);width:18px;flex:0 0 18px}
.nav button.active .n{color:var(--teal)}
.side-foot{margin-top:auto;display:flex;flex-direction:column;gap:12px;padding:0 6px}
.side-note{
  border:1px solid var(--line);border-radius:12px;padding:11px 12px;
  background:rgba(148,163,184,.035);color:var(--muted);font-size:11.5px;line-height:1.45;
}
.side-note strong{display:block;color:var(--text);font-size:11.5px;margin-bottom:3px}
.side-dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--teal);margin-right:6px;box-shadow:0 0 0 3px rgba(94,234,212,.12)}
.side-meta{display:flex;justify-content:space-between;color:var(--faint);font-size:10px;font-family:var(--mono);letter-spacing:.02em}
.scrim{display:none}

/* ---------- main ---------- */
.main{flex:1;min-width:0;margin-left:var(--sidebar)}
.topbar{
  position:sticky;top:0;z-index:30;
  display:flex;align-items:center;gap:14px;justify-content:space-between;
  min-height:58px;padding:10px clamp(16px,3.2vw,40px);
  background:linear-gradient(180deg,rgba(7,10,15,.94),rgba(7,10,15,.78));
  border-bottom:1px solid var(--line);
  backdrop-filter:blur(14px);
}
.crumb{display:flex;align-items:center;gap:8px;color:var(--faint);font-size:12px;min-width:0}
.crumb strong{color:var(--text);font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.crumb .sep{color:rgba(148,163,184,.35)}
.top-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.progress{position:fixed;top:0;left:0;height:2px;background:linear-gradient(90deg,var(--teal),var(--blue));z-index:50;width:0;transition:width .08s linear}
.menu-btn{display:none}
.content{max-width:1240px;margin:0 auto;padding:clamp(20px,4vw,48px) clamp(16px,3.2vw,40px) 72px}
.section{scroll-margin-top:82px;margin-bottom:clamp(42px,7vw,86px)}
.section-head{max-width:860px;margin-bottom:22px}
.eyebrow{display:flex;align-items:center;gap:9px;color:var(--teal);font-family:var(--mono);font-size:10.5px;letter-spacing:.16em;text-transform:uppercase;margin-bottom:9px}
.eyebrow::before{content:"";width:24px;height:1px;background:linear-gradient(90deg,var(--teal),transparent)}
h1,h2,h3,h4{font-family:var(--display);letter-spacing:-.025em;line-height:1.12}
h1{font-size:clamp(30px,4.5vw,54px);margin:0 0 16px;font-weight:680;max-width:14ch}
h2{font-size:clamp(23px,2.7vw,34px);margin:0 0 10px;font-weight:660}
h3{font-size:18px;margin:0 0 8px;font-weight:640}
h4{font-size:15px;margin:0 0 6px;font-weight:640}
.lede{color:var(--muted);font-size:clamp(14.5px,1.6vw,17px);max-width:74ch;margin:0}
.sub{color:var(--muted);font-size:13.5px;margin:8px 0 0;max-width:80ch}
.source{color:var(--faint);font-size:11px;line-height:1.5;margin:10px 0 0;font-family:var(--mono);word-break:break-word}
.source b{color:var(--muted);font-weight:600}
.mono{font-family:var(--mono);font-size:.92em;color:#c6d4e5}

/* ---------- hero ---------- */
.hero{
  display:grid;grid-template-columns:minmax(0,1.45fr) minmax(280px,.75fr);gap:clamp(18px,3vw,32px);
  align-items:stretch;padding:clamp(22px,4vw,40px);
  border:1px solid var(--line);border-radius:26px;position:relative;overflow:hidden;
  background:
    radial-gradient(600px 300px at 0% 0%,rgba(94,234,212,.10),transparent 60%),
    radial-gradient(500px 320px at 100% 100%,rgba(122,162,255,.12),transparent 60%),
    linear-gradient(145deg,rgba(17,25,37,.96),rgba(9,14,21,.96));
  box-shadow:var(--shadow);
}
.hero::after{content:"";position:absolute;right:-80px;top:-120px;width:320px;height:320px;border-radius:50%;border:1px solid rgba(94,234,212,.1);pointer-events:none}
.hero-copy{min-width:0;display:flex;flex-direction:column;justify-content:center}
.hero h1{max-width:16ch;background:linear-gradient(180deg,#fff,#b9c6d8);-webkit-background-clip:text;background-clip:text;color:transparent}
.hero .lede{max-width:60ch}
.hero-actions{display:flex;gap:9px;flex-wrap:wrap;margin-top:22px}
.hero-pills{display:flex;flex-wrap:wrap;gap:6px;margin-top:22px}
.hero-pills span{font-size:11px;color:var(--muted);border:1px solid var(--line);background:rgba(148,163,184,.035);padding:5px 9px;border-radius:999px}
.hero-panel{
  align-self:stretch;display:flex;flex-direction:column;justify-content:center;gap:7px;
  border:1px solid rgba(94,234,212,.22);border-radius:18px;padding:22px;
  background:linear-gradient(180deg,rgba(94,234,212,.09),rgba(10,16,23,.62));
  box-shadow:inset 0 1px 0 rgba(255,255,255,.06);
}
.hero-panel-label{font-family:var(--mono);font-size:10px;letter-spacing:.15em;text-transform:uppercase;color:var(--teal)}
.hero-delta{font-family:var(--display);font-size:clamp(42px,6vw,66px);font-weight:680;line-height:1;letter-spacing:-.04em;color:#eafffb;margin:4px 0}
.hero-delta small{font-size:.34em;color:var(--muted);font-weight:500;letter-spacing:0;margin-left:5px}
.hero-ci{font-family:var(--mono);font-size:12px;color:var(--muted)}
.hero-verdict{display:inline-flex;width:fit-content;margin-top:9px;padding:5px 9px;border-radius:999px;border:1px solid rgba(94,234,212,.3);background:rgba(94,234,212,.09);color:#c8fff5;font-size:11px;font-weight:600}
.hero-warn{font-size:11.5px;color:var(--amber);border-top:1px solid rgba(245,180,97,.18);padding-top:10px;margin-top:10px;line-height:1.45}

/* ---------- layout blocks ---------- */
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(2,minmax(0,1fr))}
.g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.g4{grid-template-columns:repeat(4,minmax(0,1fr))}
.card{
  position:relative;background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
  padding:clamp(16px,2.4vw,24px);box-shadow:0 16px 40px -32px rgba(0,0,0,.8);
  min-width:0;
}
.grid > *{min-width:0}
.card.tight{padding:16px}
.card.flat{box-shadow:none}
.card-head{display:flex;gap:12px;align-items:flex-start;justify-content:space-between;margin-bottom:14px}
.card-kicker{font-family:var(--mono);font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin-bottom:7px}
.card h3{margin-top:0}
.card p{margin:0 0 10px;color:var(--muted);font-size:13.5px}
.card p:last-child{margin-bottom:0}
.card strong{color:var(--text)}
.kpi{
  display:flex;flex-direction:column;gap:4px;min-height:142px;
  border-top:2px solid transparent;background-clip:padding-box;overflow:hidden;
}
.kpi::before{content:"";position:absolute;inset:0 0 auto;height:2px;background:linear-gradient(90deg,var(--teal),transparent 70%);opacity:.8}
.kpi.blue::before{background:linear-gradient(90deg,var(--blue),transparent 70%)}
.kpi.amber::before{background:linear-gradient(90deg,var(--amber),transparent 70%)}
.kpi.violet::before{background:linear-gradient(90deg,var(--violet),transparent 70%)}
.kpi-label{font-size:11px;color:var(--muted);letter-spacing:.03em}
.kpi-value{font-family:var(--display);font-size:clamp(25px,3.4vw,36px);font-weight:670;line-height:1.05;letter-spacing:-.03em;margin:2px 0}
.kpi-value small{font-size:.42em;color:var(--muted);font-weight:500;letter-spacing:0}
.kpi-sub{font-size:11.5px;color:var(--faint);line-height:1.4;margin-top:auto}
.kpi-ci{font-family:var(--mono);font-size:10.5px;color:var(--muted);margin-top:2px}
.badge{display:inline-flex;align-items:center;gap:5px;border:1px solid var(--line);background:rgba(148,163,184,.05);color:var(--muted);padding:3px 7px;border-radius:999px;font-size:10px;font-weight:650;letter-spacing:.02em;white-space:nowrap}
.badge.teal{border-color:rgba(94,234,212,.25);background:var(--teal-dim);color:#c9fff7}
.badge.blue{border-color:rgba(122,162,255,.25);background:var(--blue-dim);color:#d7e3ff}
.badge.violet{border-color:rgba(196,181,253,.28);background:var(--violet-dim);color:#eee7ff}
.badge.amber{border-color:rgba(245,180,97,.28);background:var(--amber-dim);color:#ffe6bd}
.badge.red{border-color:rgba(251,113,133,.28);background:var(--red-dim);color:#ffd4da}
.badge.green{border-color:rgba(74,222,128,.25);background:rgba(74,222,128,.10);color:#d3ffe0}
.dot{width:6px;height:6px;border-radius:50%;background:currentColor;display:inline-block}
.btn{
  appearance:none;display:inline-flex;align-items:center;justify-content:center;gap:7px;
  border:1px solid var(--line-strong);background:rgba(148,163,184,.06);color:var(--text);
  padding:9px 14px;border-radius:10px;font-size:12.5px;font-weight:600;cursor:pointer;
  transition:background .18s,border-color .18s,transform .08s;
}
.btn:hover{background:rgba(148,163,184,.11);border-color:rgba(148,163,184,.34)}
.btn:active{transform:translateY(1px)}
.btn.primary{border-color:rgba(94,234,212,.35);background:linear-gradient(180deg,rgba(94,234,212,.20),rgba(94,234,212,.09));color:#d8fff9}
.btn.primary:hover{background:linear-gradient(180deg,rgba(94,234,212,.27),rgba(94,234,212,.12))}
.btn.ghost{background:transparent}
.btn.sm{padding:6px 10px;border-radius:8px;font-size:11px}
.segmented{display:inline-flex;flex-wrap:wrap;gap:3px;padding:3px;border:1px solid var(--line);border-radius:10px;background:rgba(7,11,17,.55)}
.segmented button{
  appearance:none;border:0;background:transparent;color:var(--muted);padding:6px 10px;border-radius:7px;
  font-size:11.5px;font-weight:600;cursor:pointer;transition:background .16s,color .16s;
}
.segmented button:hover{color:var(--text)}
.segmented button[aria-pressed="true"],.segmented button[aria-selected="true"]{background:rgba(94,234,212,.14);color:#d8fff9;box-shadow:inset 0 0 0 1px rgba(94,234,212,.2)}
.toolbar{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-bottom:14px}
.toolbar-group{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.notice{
  display:flex;gap:11px;align-items:flex-start;border:1px solid var(--line);border-left:3px solid var(--blue);
  border-radius:12px;padding:12px 14px;background:rgba(122,162,255,.055);color:var(--muted);font-size:12.5px;line-height:1.55;
}
.notice.amber{border-left-color:var(--amber);background:rgba(245,180,97,.055)}
.notice.violet{border-left-color:var(--violet);background:rgba(196,181,253,.055)}
.notice.red{border-left-color:var(--red);background:rgba(251,113,133,.055)}
.notice.teal{border-left-color:var(--teal);background:rgba(94,234,212,.055)}
.notice strong{color:var(--text)}
.notice .ico{font-family:var(--mono);font-size:13px;flex:0 0 auto;color:var(--blue);line-height:1.4}
.notice.amber .ico{color:var(--amber)}
.notice.violet .ico{color:var(--violet)}
.notice.red .ico{color:var(--red)}
.notice.teal .ico{color:var(--teal)}
.callout{border-radius:14px;padding:16px 18px;background:linear-gradient(135deg,rgba(94,234,212,.08),rgba(122,162,255,.05));border:1px solid rgba(94,234,212,.18)}
.callout p{color:#c6d4e5}
.quote{font-family:var(--display);font-size:clamp(16px,2vw,20px);line-height:1.45;letter-spacing:-.01em;color:#e9f6f4;margin:0}
.quote::before{content:"“";color:var(--teal);font-size:1.4em;line-height:0;vertical-align:-.18em;margin-right:2px}
ul.clean{list-style:none;padding:0;margin:10px 0 0;display:flex;flex-direction:column;gap:8px}
ul.clean li{position:relative;padding-left:17px;color:var(--muted);font-size:13px;line-height:1.5}
ul.clean li::before{content:"";position:absolute;left:0;top:.55em;width:5px;height:5px;border-radius:50%;background:var(--teal);opacity:.7}
ul.clean.warn li::before{background:var(--amber)}
ul.clean.violet li::before{background:var(--violet)}
ul.clean li strong{color:var(--text)}

/* ---------- charts ---------- */
.chart{width:100%;min-width:0;position:relative}
.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart text{font-family:var(--sans);fill:var(--muted)}
.chart .axis{stroke:rgba(148,163,184,.22);stroke-width:1}
.chart .gridline{stroke:rgba(148,163,184,.10);stroke-width:1}
.chart .axis-label{font-size:10.5px;fill:var(--faint);font-family:var(--mono);letter-spacing:.03em}
.chart .tick{font-size:10px;fill:var(--faint);font-family:var(--mono)}
.chart .series-label{font-size:11px;fill:var(--text);font-weight:600}
.chart .value-label{font-size:10.5px;fill:var(--text);font-family:var(--mono);font-weight:600}
.chart .ref-label{font-size:9.5px;fill:var(--faint);font-family:var(--mono)}
.chart .hit{fill:transparent;cursor:pointer}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px;color:var(--muted);font-size:11px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:16px;height:2px;border-radius:2px;background:var(--teal);display:inline-block}
.legend i.dash{background:none;border-top:2px dashed var(--faint);height:0}
.legend i.dot{width:7px;height:7px;border-radius:50%}
.chart-tooltip{
  position:fixed;z-index:100;max-width:280px;pointer-events:none;opacity:0;transform:translateY(4px);
  transition:opacity .12s,transform .12s;background:#0b111a;border:1px solid var(--line-strong);
  border-radius:10px;padding:9px 11px;font-size:11.5px;line-height:1.45;color:var(--text);
  box-shadow:0 18px 50px -18px rgba(0,0,0,.85);
}
.chart-tooltip b{color:var(--teal);font-weight:600}
.chart-tooltip .t-mono{font-family:var(--mono);font-size:10.5px;color:var(--muted)}
.chart-tooltip.show{opacity:1;transform:translateY(0)}

/* ---------- tables ---------- */
.table-wrap{overflow:auto;max-width:100%;border:1px solid var(--line);border-radius:12px;background:rgba(7,11,17,.42)}
table{width:100%;border-collapse:collapse;font-size:12px;min-width:520px}
th,td{padding:9px 11px;text-align:left;border-bottom:1px solid rgba(148,163,184,.09);vertical-align:top}
th{position:sticky;top:0;z-index:2;background:#0c131d;color:var(--faint);font-size:9.5px;text-transform:uppercase;letter-spacing:.1em;font-weight:650;white-space:nowrap}
td{color:var(--muted)}
td.num,th.num{text-align:right;font-family:var(--mono);font-size:11px;white-space:nowrap}
tbody tr:hover td{background:rgba(148,163,184,.035)}
tbody tr:last-child td{border-bottom:0}
td strong{color:var(--text)}
.tag{display:inline-block;padding:1px 6px;border-radius:5px;font-size:9.5px;font-family:var(--mono);border:1px solid var(--line)}
.tag.c{border-color:rgba(196,181,253,.3);color:var(--violet);background:var(--violet-dim)}

/* ---------- methodology ---------- */
.split-bar{display:flex;height:54px;border-radius:12px;overflow:hidden;border:1px solid var(--line);margin:6px 0 12px}
.split-seg{position:relative;display:flex;flex-direction:column;justify-content:center;padding:0 12px;min-width:0;border-right:1px solid rgba(7,10,15,.55)}
.split-seg:last-child{border-right:0}
.split-seg.s1{background:linear-gradient(180deg,rgba(122,162,255,.20),rgba(122,162,255,.08));width:70%}
.split-seg.s2{background:linear-gradient(180deg,rgba(94,234,212,.20),rgba(94,234,212,.08));width:15%}
.split-seg.s3{background:linear-gradient(180deg,rgba(245,180,97,.22),rgba(245,180,97,.09));width:15%}
.split-seg b{font-size:11.5px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.split-seg span{font-family:var(--mono);font-size:9.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.feature-list{display:flex;flex-wrap:wrap;gap:5px;margin-top:8px}
.feature-list code{font-size:10px;color:#c6d4e5;background:rgba(148,163,184,.07);border:1px solid var(--line);padding:2px 6px;border-radius:5px}
.pill-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.pill-row .badge{font-size:10px}

/* ---------- delta / forest ---------- */
.delta-table td:first-child{white-space:nowrap;color:var(--text)}
.ci-bar{position:relative;height:8px;background:rgba(148,163,184,.10);border-radius:99px;min-width:110px;overflow:visible}
.ci-bar i{position:absolute;top:0;height:8px;border-radius:99px;background:linear-gradient(90deg,var(--teal),var(--blue))}
.ci-bar b{position:absolute;top:50%;width:8px;height:8px;border-radius:50%;background:#fff;transform:translate(-50%,-50%);box-shadow:0 0 0 2px rgba(94,234,212,.35)}

/* ---------- calibration ---------- */
.cal-stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:12px}
.cal-stat{border:1px solid var(--line);border-radius:10px;padding:9px 10px;background:rgba(148,163,184,.035)}
.cal-stat .l{font-size:9.5px;color:var(--faint);text-transform:uppercase;letter-spacing:.08em}
.cal-stat .v{font-family:var(--mono);font-size:13px;color:var(--text);margin-top:2px}
.cal-stat .v.bad{color:var(--amber)}

/* ---------- mistakes ---------- */
.timeline{position:relative;display:flex;flex-direction:column;gap:10px;padding-left:20px}
.timeline::before{content:"";position:absolute;left:6px;top:6px;bottom:6px;width:1px;background:linear-gradient(180deg,rgba(94,234,212,.35),rgba(148,163,184,.12))}
.m-card{position:relative;border:1px solid var(--line);border-radius:13px;background:rgba(13,19,28,.72);overflow:hidden}
.m-card::before{content:"";position:absolute;left:-20px;top:19px;width:9px;height:9px;border-radius:50%;background:#0b111a;border:2px solid var(--faint);z-index:2}
.m-card[data-cat="A"]::before{border-color:var(--amber)}
.m-card[data-cat="S"]::before{border-color:var(--teal)}
.m-card[data-cat="U"]::before,.m-card[data-cat="R"]::before{border-color:var(--blue)}
.m-card summary{list-style:none;cursor:pointer;padding:12px 14px;display:flex;gap:10px;align-items:flex-start}
.m-card summary::-webkit-details-marker{display:none}
.m-card summary:hover{background:rgba(148,163,184,.035)}
.m-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:4px}
.m-id{font-family:var(--mono);font-size:10px;font-weight:700;padding:2px 6px;border-radius:5px;border:1px solid var(--line);color:var(--muted)}
.m-id.h{border-color:rgba(122,162,255,.3);color:#cfe0ff;background:var(--blue-dim)}
.m-id.a{border-color:rgba(245,180,97,.3);color:#ffe3b8;background:var(--amber-dim)}
.m-id.s{border-color:rgba(94,234,212,.3);color:#c9fff7;background:var(--teal-dim)}
.m-id.u{border-color:rgba(122,162,255,.3);color:#cfe0ff;background:var(--blue-dim)}
.m-date{font-family:var(--mono);font-size:10px;color:var(--faint)}
.m-title{font-size:12.5px;color:var(--text);font-weight:600;line-height:1.45}
.m-chev{margin-left:auto;color:var(--faint);transition:transform .2s;flex:0 0 auto;font-size:11px}
.m-card[open] .m-chev{transform:rotate(90deg)}
.m-body{padding:0 14px 14px 14px;border-top:1px solid rgba(148,163,184,.08);margin-top:0;padding-top:12px}
.m-field{margin-bottom:10px}
.m-field:last-child{margin-bottom:0}
.m-field .k{font-family:var(--mono);font-size:9.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);margin-bottom:3px}
.m-field .v{font-size:12px;color:var(--muted);line-height:1.55}
.m-field .v code{font-size:10.5px;color:#c6d4e5}
.m-facts{display:flex;flex-direction:column;gap:4px;margin-top:8px}
.m-fact{display:flex;gap:8px;justify-content:space-between;border-bottom:1px dashed rgba(148,163,184,.10);padding:4px 0;font-size:10.5px;gap:14px}
.m-fact .k{color:var(--faint)}
.m-fact .v{font-family:var(--mono);color:var(--text);text-align:right;word-break:break-all}
.m-note{margin-top:9px;font-size:11px;color:var(--amber);line-height:1.5}

/* ---------- verification ---------- */
.ring-wrap{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
.ring{width:126px;height:126px;flex:0 0 126px;position:relative}
.ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring-c{fill:none;stroke:rgba(148,163,184,.12);stroke-width:8}
.ring-v{fill:none;stroke:url(#ringGrad);stroke-width:8;stroke-linecap:round;stroke-dasharray:339.292;stroke-dashoffset:339.292;animation:ringDraw 1.2s .25s cubic-bezier(.22,.61,.36,1) forwards}
.ring-txt{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center}
.ring-txt b{font-family:var(--display);font-size:24px;line-height:1;letter-spacing:-.03em}
.ring-txt span{font-size:9px;color:var(--faint);text-transform:uppercase;letter-spacing:.1em;margin-top:3px}
@keyframes ringDraw{to{stroke-dashoffset:0}}
.check-list{max-height:380px;overflow:auto;border:1px solid var(--line);border-radius:12px;background:rgba(7,11,17,.40)}
.check-row{display:flex;gap:9px;align-items:flex-start;padding:8px 11px;border-bottom:1px solid rgba(148,163,184,.07)}
.check-row:last-child{border-bottom:0}
.check-row:hover{background:rgba(148,163,184,.03)}
.check-state{font-family:var(--mono);font-size:9px;font-weight:700;color:#bfffd5;border:1px solid rgba(74,222,128,.28);background:rgba(74,222,128,.08);padding:1px 5px;border-radius:4px;flex:0 0 auto;margin-top:1px}
.check-row code{font-size:10px;color:#c6d4e5;display:block;margin-bottom:1px}
.check-row p{margin:0;font-size:10.5px;color:var(--faint);line-height:1.4}
.bar-meter{height:7px;border-radius:99px;background:rgba(148,163,184,.10);overflow:hidden;margin-top:8px}
.bar-meter i{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,var(--teal),var(--blue))}

/* ---------- inference ---------- */
.pipeline{display:flex;align-items:stretch;gap:0;flex-wrap:wrap;margin:10px 0 4px}
.pipe-step{flex:1 1 140px;min-width:130px;border:1px solid var(--line);border-radius:10px;padding:9px 10px;background:rgba(148,163,184,.035);position:relative;margin:4px 12px 4px 0}
.pipe-step:last-child{margin-right:0}
.pipe-step::after{content:"›";position:absolute;right:-10px;top:50%;transform:translateY(-50%);color:var(--faint);font-size:17px}
.pipe-step:last-child::after{display:none}
.pipe-step b{display:block;font-size:11px;color:var(--text)}
.pipe-step span{display:block;font-size:9.5px;color:var(--faint);font-family:var(--mono);margin-top:2px}
.customer-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.customer{border:1px solid var(--line);border-radius:14px;padding:15px;background:rgba(148,163,184,.028)}
.customer h4{display:flex;justify-content:space-between;gap:8px;align-items:center;margin-bottom:10px}
.customer h4 code{font-size:11px;color:var(--teal)}
.score-row{display:grid;grid-template-columns:38px 1fr 64px;gap:8px;align-items:center;margin:8px 0}
.score-row .lab{font-family:var(--mono);font-size:10px;color:var(--muted)}
.score-track{height:9px;border-radius:99px;background:rgba(148,163,184,.10);overflow:hidden}
.score-track i{display:block;height:100%;border-radius:99px}
.score-track i.a{background:linear-gradient(90deg,#4f7fdb,#7aa2ff)}
.score-track i.b{background:linear-gradient(90deg,#24a98b,#5eead4)}
.score-row .val{font-family:var(--mono);font-size:11px;color:var(--text);text-align:right}
.score-note{font-size:10px;color:var(--faint);margin-top:6px}
.feature-mini{display:flex;flex-wrap:wrap;gap:4px;margin-top:9px}
.feature-mini span{font-family:var(--mono);font-size:9.5px;color:var(--muted);border:1px solid var(--line);padding:1px 5px;border-radius:4px;background:rgba(7,11,17,.35)}
.codeblock{position:relative;border:1px solid var(--line);border-radius:12px;background:#080d14;overflow:hidden}
.codeblock header{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:7px 10px;border-bottom:1px solid rgba(148,163,184,.10);background:rgba(148,163,184,.035)}
.codeblock header span{font-family:var(--mono);font-size:9.5px;color:var(--faint);letter-spacing:.08em;text-transform:uppercase}
.codeblock pre{margin:0;padding:12px 13px;overflow:auto;color:#c9fff7;font-family:var(--mono);font-size:11px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
.copy-state{font-size:10px;color:var(--teal);font-family:var(--mono);min-height:14px}

/* ---------- footer ---------- */
.footer{border-top:1px solid var(--line);padding-top:22px;color:var(--faint);font-size:11px;display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap}
.footer strong{color:var(--muted)}
details.raw{border:1px solid var(--line);border-radius:11px;background:rgba(7,11,17,.32)}
details.raw summary{cursor:pointer;padding:9px 12px;font-size:11.5px;color:var(--muted);font-weight:600;list-style:none;display:flex;justify-content:space-between;gap:8px}
details.raw summary::-webkit-details-marker{display:none}
details.raw summary::after{content:"+";font-family:var(--mono);color:var(--faint)}
details.raw[open] summary::after{content:"−"}
details.raw .raw-body{padding:0 12px 12px;border-top:1px solid rgba(148,163,184,.08);padding-top:10px}
.kv{display:grid;grid-template-columns:minmax(100px,1fr) minmax(0,1.6fr);gap:8px;padding:4px 0;border-bottom:1px dashed rgba(148,163,184,.08);font-size:10.5px}
.kv:last-child{border-bottom:0}
.kv .k{color:var(--faint)}
.kv .v{font-family:var(--mono);color:var(--text);text-align:right;word-break:break-all}

/* ---------- human-in-the-loop verification ---------- */
.hitl-head{margin:26px 0 14px;padding-top:20px;border-top:1px solid var(--line)}
.hitl-head .eyebrow{margin-bottom:8px}
.hitl-head h3{font-size:clamp(19px,2.2vw,25px);margin:0 0 8px;font-weight:660;letter-spacing:-.02em}
.hitl-head p{color:var(--muted);font-size:13px;margin:0;max-width:82ch}
.lane-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.lane{border:1px solid var(--line);border-radius:14px;padding:14px;background:rgba(148,163,184,.028);display:flex;flex-direction:column;gap:9px;min-width:0;position:relative;overflow:hidden}
.lane::before{content:"";position:absolute;inset:0 0 auto;height:2px;background:var(--teal);opacity:.55}
.lane.auto::before{background:var(--blue)}
.lane.agent::before{background:var(--violet)}
.lane-h{display:flex;align-items:center;gap:9px;min-width:0}
.lane-ico{width:27px;height:27px;border-radius:8px;flex:0 0 27px;display:grid;place-items:center;font-family:var(--mono);font-size:11px;font-weight:700;background:var(--teal-dim);border:1px solid rgba(94,234,212,.3);color:#c9fff7}
.lane.auto .lane-ico{background:var(--blue-dim);border-color:rgba(122,162,255,.3);color:#d7e3ff}
.lane.agent .lane-ico{background:var(--violet-dim);border-color:rgba(196,181,253,.3);color:#eee7ff}
.lane-h b{font-size:12.5px;color:var(--text);font-weight:640;line-height:1.25}
.lane-h em{display:block;font-style:normal;font-family:var(--mono);font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin-top:2px}
.lane ul{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:6px}
.lane li{position:relative;padding-left:13px;font-size:11.5px;color:var(--muted);line-height:1.45}
.lane li::before{content:"";position:absolute;left:0;top:.5em;width:4px;height:4px;border-radius:50%;background:var(--teal);opacity:.65}
.lane.auto li::before{background:var(--blue)}
.lane.agent li::before{background:var(--violet)}
.lane .lane-src{margin-top:auto;font-family:var(--mono);font-size:9.5px;color:var(--faint);line-height:1.45;word-break:break-word}
.stepper{display:flex;gap:2px;overflow-x:auto;padding:2px 0 8px;scrollbar-width:thin}
.step{flex:1 1 0;min-width:142px;display:flex;flex-direction:column;gap:7px;align-items:flex-start;text-align:left;background:transparent;border:0;padding:8px 10px 10px;border-radius:11px;cursor:pointer;position:relative;transition:background .16s;appearance:none}
.step:hover{background:rgba(148,163,184,.055)}
.step:focus-visible{outline:2px solid rgba(94,234,212,.55);outline-offset:1px}
.step:not(:last-child)::after{content:"›";position:absolute;right:-3px;top:13px;color:rgba(148,163,184,.4);font-size:15px;line-height:1}
.step .node{width:26px;height:26px;border-radius:50%;display:grid;place-items:center;font-family:var(--mono);font-size:11px;font-weight:700;background:rgba(148,163,184,.09);border:1px solid var(--line-strong);color:var(--muted);transition:background .18s,color .18s,border-color .18s,box-shadow .18s}
.step[aria-selected="true"]{background:linear-gradient(180deg,rgba(94,234,212,.10),rgba(94,234,212,.015))}
.step[aria-selected="true"] .node{background:var(--teal);border-color:var(--teal);color:#04211d;box-shadow:0 0 0 4px rgba(94,234,212,.14)}
.step-label{font-size:11.5px;color:var(--muted);font-weight:600;line-height:1.32}
.step[aria-selected="true"] .step-label{color:var(--text)}
.step-actor{font-family:var(--mono);font-size:8.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint)}
.step-panel{border:1px solid var(--line);border-radius:14px;padding:16px 18px;min-height:190px;background:radial-gradient(560px 200px at 0% 0%,rgba(94,234,212,.07),transparent 62%),rgba(7,11,17,.40)}
.sp-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:7px}
.sp-title{font-family:var(--display);font-size:16.5px;font-weight:645;letter-spacing:-.015em;margin:0 0 7px;color:var(--text)}
.sp-body{font-size:12.5px;color:var(--muted);line-height:1.6;margin:0 0 13px;max-width:80ch}
.sp-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:8px}
.sp-src{margin-top:11px;font-family:var(--mono);font-size:9.5px;color:var(--faint);word-break:break-word}
.cmp-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.cmp{border:1px solid var(--line);border-radius:13px;padding:14px;background:rgba(148,163,184,.028);min-width:0}
.cmp.teal{border-color:rgba(94,234,212,.20)}
.cmp.blue{border-color:rgba(122,162,255,.20)}
.cmp h4{display:flex;align-items:center;gap:7px;margin:0 0 9px;font-size:12.5px}
.cmp dl{margin:0;display:flex;flex-direction:column;gap:7px}
.cmp dt{font-family:var(--mono);font-size:9px;letter-spacing:.09em;text-transform:uppercase;color:var(--faint)}
.cmp dd{margin:1px 0 0;font-size:11.5px;color:var(--muted);line-height:1.5}
.chip-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip{font-size:10.5px;color:var(--faint);border:1px dashed rgba(251,113,133,.28);background:rgba(251,113,133,.055);padding:3px 8px;border-radius:999px}
.chip::before{content:"×";color:var(--red);margin-right:5px;font-weight:700}

/* ---------- responsive ---------- */
@media (max-width:1080px){
  .hero{grid-template-columns:1fr}
  .g4{grid-template-columns:repeat(2,minmax(0,1fr))}
  .g3{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media (max-width:900px){
  .sidebar{transform:translateX(-100%);box-shadow:30px 0 80px rgba(0,0,0,.55)}
  body.nav-open .sidebar{transform:translateX(0)}
  body.nav-open .scrim{display:block;position:fixed;inset:0;background:rgba(3,6,10,.66);z-index:35;backdrop-filter:blur(2px)}
  .main{margin-left:0}
  .menu-btn{display:inline-flex}
  .topbar{padding-left:14px;padding-right:14px}
  h1{max-width:none}
  .hero h1{max-width:none}
}
@media (max-width:820px){
  .grid.g2,.grid.g3,.grid.g4{grid-template-columns:1fr!important}
  .source{overflow-wrap:anywhere}
  .topbar{flex-wrap:wrap}
  .top-actions{width:100%;justify-content:flex-start}
  .lane-grid{grid-template-columns:1fr}
  .cmp-grid{grid-template-columns:1fr}
}
@media (max-width:760px){
  .stepper{flex-direction:column;overflow:visible;gap:3px}
  .step{min-width:0;flex-direction:row;align-items:center;gap:10px;padding:7px 9px}
  .step:not(:last-child)::after{display:none}
  .step-actor{margin-left:auto}
}
@media (max-width:680px){
  .g2,.g3,.g4{grid-template-columns:1fr}
  .customer-grid{grid-template-columns:1fr}
  .split-seg span{display:none}
  .split-seg{padding:0 7px}
  .split-seg b{font-size:10px}
  .cal-stats{grid-template-columns:1fr}
  .hero{padding:20px}
  .hero-delta{font-size:48px}
  .content{padding-left:12px;padding-right:12px}
  .card{padding:15px}
  .kpi{min-height:126px}
}
@media (prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *,*::before,*::after{animation-duration:.001ms!important;transition-duration:.001ms!important}
}
@media print{
  .sidebar,.topbar,.scrim,.progress,.hero-actions,.toolbar,.segmented,.btn{display:none!important}
  .main{margin:0}
  body{background:#fff;color:#111}
  .card,.hero{border-color:#ccc;background:#fff;box-shadow:none}
  .section{page-break-inside:avoid}
}

</style>
</head>
<body>
<div class="progress" id="progress"></div>
<div class="app">
  <aside class="sidebar" id="sidebar" aria-label="Report navigation">
    <div class="brand">
      <div class="brand-mark">BA</div>
      <div class="brand-text">
        <strong>Term-Deposit Study</strong>
        <span>Business Analytics · 20595</span>
      </div>
    </div>
    <nav class="nav" id="nav">
      <div class="nav-label">Report</div>
      <button type="button" data-target="overview" class="active"><span class="n">01</span>Overview</button>
      <button type="button" data-target="data"><span class="n">02</span>Data &amp; method</button>
      <button type="button" data-target="performance"><span class="n">03</span>Performance</button>
      <button type="button" data-target="workflows"><span class="n">04</span>Guided vs autonomous</button>
      <button type="button" data-target="calibration"><span class="n">05</span>Calibration &amp; limits</button>
      <button type="button" data-target="mistakes"><span class="n">06</span>Mistakes &amp; reflection</button>
      <button type="button" data-target="verification"><span class="n">07</span>Verification</button>
      <button type="button" data-target="inference"><span class="n">08</span>Inference demo</button>
    </nav>
    <div class="side-foot">
      <div class="side-note">
        <strong><span class="side-dot"></span>Frozen evidence</strong>
        Every number is read from a saved artifact. Nothing is scored live in this page.
      </div>
      <div class="side-meta"><span>v1.0</span><span id="sideStamp">—</span></div>
    </div>
  </aside>
  <div class="scrim" id="scrim" aria-hidden="true"></div>

  <main class="main">
    <div class="topbar">
      <div style="display:flex;align-items:center;gap:10px;min-width:0">
        <button class="btn sm menu-btn" id="menuBtn" type="button" aria-label="Open navigation" aria-expanded="false">☰</button>
        <div class="crumb"><span>Project report</span><span class="sep">/</span><strong id="crumbSection">Overview</strong></div>
      </div>
      <div class="top-actions">
        <span class="badge teal"><span class="dot"></span>Primary: chronological test</span>
        <span class="badge green">77/77 checks</span>
      </div>
    </div>

    <div class="content">
      <!-- ================= 01 OVERVIEW ================= -->
      <section class="section" id="overview" aria-labelledby="overviewTitle">
        <div class="hero" style="margin-bottom:14px">
          <div class="hero-copy">
            <div class="eyebrow">Individual AI project · bank marketing</div>
            <h1 id="overviewTitle">Does campaign history improve term-deposit targeting?</h1>
            <p class="lede">A frozen chronological evaluation of three logistic-regression models — built, audited and independently recomputed with AI coding agents.</p>
            <div class="hero-actions">
              <button class="btn primary" type="button" data-scroll="performance">Explore the evidence</button>
              <button class="btn ghost" type="button" data-scroll="calibration">Read the caveats</button>
            </div>
            <div class="hero-pills">
              <span id="heroRows">6,783 test rows</span>
              <span id="heroPositives">2,669 positives</span>
              <span id="heroBoot">2,000 bootstrap resamples</span>
              <span>Seed 42</span>
            </div>
          </div>
          <div class="hero-panel">
            <div class="hero-panel-label">Primary result · B − A</div>
            <div class="hero-delta" id="heroDelta">+0.0302<small>ROC-AUC</small></div>
            <div class="hero-ci" id="heroCI">95% CI 0.0261 – 0.0343</div>
            <div class="hero-verdict" id="heroVerdict">H0 rejected · materially relevant</div>
            <div class="hero-warn">But the probability scale is badly off: mean predictions near 6.8% against a 39.3% observed rate.</div>
          </div>
        </div>

        <div class="grid g4" style="margin-bottom:14px">
          <div class="card kpi">
            <div class="kpi-label">ΔROC-AUC · B − A</div>
            <div class="kpi-value" id="kpiDelta">+0.0302</div>
            <div class="kpi-ci" id="kpiDeltaCI">95% CI 0.0261 – 0.0343</div>
            <div class="kpi-sub">Paired bootstrap · whole interval above zero</div>
          </div>
          <div class="card kpi blue">
            <div class="kpi-label">Model B ROC-AUC</div>
            <div class="kpi-value" id="kpiRoc">0.5858</div>
            <div class="kpi-ci" id="kpiRocSub">A 0.5556 · C 0.7674 diagnostic</div>
            <div class="kpi-sub">Ranking on the frozen chronological test</div>
          </div>
          <div class="card kpi">
            <div class="kpi-label">Lift@10% · Model B</div>
            <div class="kpi-value" id="kpiLift">1.364<small>×</small></div>
            <div class="kpi-ci" id="kpiLiftSub">364 of the top 678 are positive</div>
            <div class="kpi-sub">Useful for prioritisation, not for probability statements</div>
          </div>
          <div class="card kpi amber">
            <div class="kpi-label">Prevalence ÷ mean prediction</div>
            <div class="kpi-value" id="kpiCal">5.82<small>× low</small></div>
            <div class="kpi-ci" id="kpiCalSub">B: 0.0677 vs observed 0.3935</div>
            <div class="kpi-sub">Systematic level error, visible in every decile</div>
          </div>
        </div>

        <div class="grid g2" style="margin-bottom:14px">
          <div class="card">
            <div class="card-kicker">Research question 1 · statistical</div>
            <h3>Does previous-campaign information add predictive value?</h3>
            <p>In the Bank Marketing dataset, does adding previous-campaign information to pre-call customer characteristics improve out-of-sample prediction of term-deposit subscription, relative to customer characteristics alone?</p>
            <p style="color:var(--faint);font-size:11px">Source: <code>plan.md</code> §1 · associational prediction only; no causal effect is estimated or claimed.</p>
          </div>
          <div class="card">
            <div class="card-kicker">Research question 2 · process</div>
            <h3>Can an AI coding agent execute the analysis reliably?</h3>
            <p>Can an AI coding agent execute this analysis without data leakage, without methodological error, and with decisions that match the pre-registered plan and survive independent verification?</p>
            <p style="color:var(--faint);font-size:11px">RQ1 and RQ2 are answered separately. A good AUC does not answer RQ2, and a clean audit does not answer RQ1.</p>
          </div>
        </div>

        <div class="grid g3" style="margin-bottom:14px">
          <div class="card">
            <span class="badge teal">Finding 1 · ranking</span>
            <h3 style="margin-top:10px">A small but consistently positive gain</h3>
            <p>Model B improves ROC-AUC by <strong id="findingDelta">0.0302</strong> over Model A. The 95% interval <span class="mono" id="findingCI">[0.0261, 0.0343]</span> lies entirely above zero, so H0 is rejected and the pre-registered materiality threshold of 0.01 is met.</p>
            <p>Average precision rises by <strong id="findingAP">0.0292</strong>. The improvement is real for ranking, but small.</p>
          </div>
          <div class="card">
            <span class="badge amber">Finding 2 · calibration</span>
            <h3 style="margin-top:10px">Probabilities are not usable as probabilities</h3>
            <p>On the test block, B's mean predicted probability is <strong id="findingMeanB">0.0677</strong> against an observed prevalence of <strong id="findingPrev">0.3935</strong> — about <strong id="findingRatioB">5.82×</strong> too low. Every decile sits far below its observed rate.</p>
            <p>The Brier score is worse than a hindsight constant that is told the test prevalence. That constant is a diagnostic, not an operational baseline.</p>
          </div>
          <div class="card">
            <span class="badge violet">Finding 3 · leakage diagnostic</span>
            <h3 style="margin-top:10px">Duration inflates performance, but cannot be deployed</h3>
            <p>Model C adds <code>duration</code>, known only after the call. Its ROC-AUC is <strong id="findingC">0.7674</strong>, with C − B ΔROC-AUC <span class="mono" id="findingCB">0.1816 [0.1655, 0.1977]</span> — far above the 0.05 diagnostic threshold.</p>
            <p><strong>Model C is a NON-DEPLOYABLE leakage diagnostic, never an operational model.</strong></p>
          </div>
        </div>

        <div class="grid g2">
          <div class="card callout">
            <div class="card-kicker">Conclusion</div>
            <p class="quote" style="margin-bottom:10px">Model B is a ranking prototype with a documented probability-scale defect — not a deployment-ready risk model.</p>
            <p>Previous-campaign information improves the <em>ordering</em> of customers on a later time period. It does not make the output a reliable personal probability, and it does not establish that contacting a customer because of this score would change their behaviour.</p>
            <p style="color:var(--faint);font-size:11px">Sources: <code>README.md</code> §3 · <code>results/guided/test/test_metrics.csv</code> · <code>results/guided/test/calibration_interpretation.md</code></p>
          </div>
          <div class="card">
            <div class="card-kicker">Limitations that travel with the numbers</div>
            <ul class="clean warn">
              <li><strong>Base-rate shift.</strong> Prevalence rises 0.058 → 0.115 → 0.393 across train, validation and test. The cause is not established.</li>
              <li><strong>No recalibration.</strong> No prior correction, threshold tuning or calibration map was pre-registered or performed.</li>
              <li><strong>Temporal ambiguity.</strong> The automated chronology check was INCONCLUSIVE; ordering rests on a documented human adjudication.</li>
              <li><strong>One split, one period.</strong> No claim about future campaigns, other markets or causal effects.</li>
              <li><strong>Model C is never deployable.</strong> Its scores measure leakage, not usefulness.</li>
            </ul>
          </div>
        </div>
      </section>

      <!-- ================= 02 DATA & METHOD ================= -->
      <section class="section" id="data" aria-labelledby="dataTitle">
        <div class="section-head">
          <div class="eyebrow">02 · Data &amp; methodology</div>
          <h2 id="dataTitle">One dataset, one frozen chronological split, three information sets</h2>
          <p class="lede">The design was fixed before the test block was opened: 70/15/15 by row order, a single primary metric, and a paired bootstrap that reuses the same resample indices across models.</p>
        </div>

        <div class="grid g2" style="margin-bottom:14px">
          <div class="card">
            <div class="card-kicker">Dataset &amp; provenance</div>
            <h3>Bank Marketing · 45,211 rows</h3>
            <p>Binary target <code>y</code> (term-deposit subscription), sourced from UCI via a Hugging Face mirror. The raw file was never modified.</p>
            <div class="kv"><span class="k">Dataset SHA-256</span><span class="v">0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f</span></div>
            <div class="kv"><span class="k">Split strategy</span><span class="v" id="splitStrategy">chronological row-index split</span></div>
            <div class="kv"><span class="k">Canonical split hash</span><span class="v" id="splitHash">—</span></div>
            <p class="source">Sources: <code>README.md</code> §2 · <code>results/splits/temporal_split_v1.json</code> · <code>results/guided/test/run_manifest_test.json</code></p>
          </div>
          <div class="card">
            <div class="card-kicker">Chronological blocks</div>
            <h3>Prevalence rises at every step</h3>
            <div class="split-bar" aria-label="Train 70 percent, validation 15 percent, test 15 percent">
              <div class="split-seg s1"><b>Train · <span id="trainN">31,647</span></b><span>prevalence <span id="trainPrev">0.0581</span></span></div>
              <div class="split-seg s2"><b>Val · <span id="valN">6,781</span></b><span><span id="valPrev">0.1150</span></span></div>
              <div class="split-seg s3"><b>Test · <span id="testN">6,783</span></b><span><span id="testPrev">0.3935</span></span></div>
            </div>
            <p>The test block is roughly 6.8× more positive than training. This is a property of the chronological split, not a modelling-code defect — and it is the backdrop for the calibration failure.</p>
            <p class="source">Sources: <code>results/splits/temporal_split_v1.json</code> · <code>results/guided/validation_report.md</code> · <code>results/guided/test/test_metrics.csv</code></p>
          </div>
        </div>

        <div class="grid g3" style="margin-bottom:14px">
          <div class="card">
            <div class="card-head"><div><div class="card-kicker">Model A</div><h3>Customer characteristics</h3></div><span class="badge blue">baseline</span></div>
            <p>Static, pre-call information only.</p>
            <div class="feature-list"><code>age</code><code>balance</code><code>job</code><code>marital</code><code>education</code><code>default</code><code>housing</code><code>loan</code></div>
            <p style="margin-top:11px;font-size:12px">Selected <code>C = 0.01</code> by time-series CV on the training block.</p>
          </div>
          <div class="card">
            <div class="card-head"><div><div class="card-kicker">Model B</div><h3>A + campaign history</h3></div><span class="badge teal">primary candidate</span></div>
            <p>Adds previous-campaign information available before the next call.</p>
            <div class="feature-list"><code>previous</code><code>poutcome_derived</code></div>
            <p style="margin-top:11px;font-size:12px"><code>poutcome_derived = 'never_contacted'</code> when <code>pdays == 999</code>, else <code>poutcome</code>. Selected <code>C = 0.01</code>.</p>
          </div>
          <div class="card" style="border-color:rgba(196,181,253,.22)">
            <div class="card-head"><div><div class="card-kicker">Model C</div><h3>B + call duration</h3></div><span class="badge violet">NON-DEPLOYABLE</span></div>
            <p>Adds <code>duration</code>, recorded after the call ends and mechanically related to the outcome.</p>
            <p><strong>Leakage diagnostic only.</strong> It cannot support a pre-call decision and is never presented as an operational model.</p>
            <p style="margin-top:11px;font-size:12px">Selected <code>C = 0.001</code>.</p>
          </div>
        </div>

        <div class="grid g2" style="margin-bottom:14px">
          <div class="card">
            <div class="card-kicker">Evaluation design</div>
            <h3>Pre-specified, paired, and frozen</h3>
            <ul class="clean">
              <li><strong>Hypotheses:</strong> H0: ΔROC-AUC ≤ 0; H1: ΔROC-AUC &gt; 0. H0 is rejected only if the 95% interval excludes zero on the positive side. A CI that includes zero is insufficient evidence, not proof of no improvement.</li>
              <li><strong>Primary metric:</strong> ΔROC-AUC = AUC(B) − AUC(A) on the chronological test block.</li>
              <li><strong>Inference:</strong> paired bootstrap over test rows, 2,000 resamples, seed 42, percentile 95% intervals.</li>
              <li><strong>Materiality:</strong> ΔROC-AUC ≥ 0.01 is a project-level judgment, separate from the statistical test.</li>
              <li><strong>Secondary diagnostic:</strong> C − B ΔROC-AUC ≥ 0.05 is the pre-registered leakage threshold.</li>
              <li><strong>No test reuse:</strong> no recalibration, threshold tuning or reselection after the test block was opened.</li>
            </ul>
            <div class="pill-row"><span class="badge">ROC-AUC</span><span class="badge">Average precision</span><span class="badge">Brier</span><span class="badge">Lift@10%</span></div>
            <p class="source">Sources: <code>plan.md</code> §3, §5.7 · <code>results/guided/test/hypothesis_tests_test.json</code></p>
          </div>
          <div class="card">
            <div class="card-kicker">Hyperparameter selection · validation-side</div>
            <h3>How <code>C</code> was chosen</h3>
            <p class="sub" style="margin-top:0">Mean time-series CV ROC-AUC across the guided grid. The selected point is ringed. This is validation-stage evidence; it is not a test-set result.</p>
            <div class="chart" id="cvChart" role="img" aria-label="Cross-validation ROC-AUC by regularisation strength for Models A, B and C"></div>
            <div class="legend">
              <span><i style="background:#7aa2ff"></i>Model A</span>
              <span><i style="background:#5eead4"></i>Model B</span>
              <span><i style="background:#c4b5fd"></i>Model C</span>
              <span><i class="dash"></i>selected C (ringed)</span>
            </div>
            <p class="source">Source: <code>results/guided/hyperparameter_selection.csv</code></p>
          </div>
        </div>

        <div class="notice amber">
          <span class="ico">!</span>
          <div><strong>Chronology caveat.</strong> The automated chronology validation returned <strong>INCONCLUSIVE</strong> because one transition moved forward by a month. A documented human adjudication retained the chronological ordering with that limitation disclosed; it did not upgrade the automated verdict. The year of each record cannot be recovered from the data, so strict time ordering is an assumption rather than a proven fact. <span style="color:var(--faint)">Sources: <code>results/splits/chronology_checks.md</code> · <code>results/splits/chronology_adjudication.md</code></span></div>
        </div>
      </section>

      <!-- ================= 03 PERFORMANCE ================= -->
      <section class="section" id="performance" aria-labelledby="perfTitle">
        <div class="section-head">
          <div class="eyebrow">03 · Model performance</div>
          <h2 id="perfTitle">The primary comparison — and a secondary view that must not replace it</h2>
          <p class="lede">The default is the frozen chronological test. The sensitivity switch shows the corrected random-split result, which is optimistic by construction and is never primary evidence.</p>
        </div>

        <div class="card" style="margin-bottom:14px">
          <div class="toolbar">
            <div class="toolbar-group">
              <div class="segmented" role="group" aria-label="Evaluation dataset">
                <button type="button" id="dsPrimary" aria-pressed="true">Primary · chronological</button>
                <button type="button" id="dsSensitivity" aria-pressed="false">Sensitivity · random split</button>
              </div>
            </div>
            <div class="toolbar-group">
              <div class="segmented" role="group" aria-label="Metric">
                <button type="button" data-metric="roc_auc" aria-pressed="true">ROC-AUC</button>
                <button type="button" data-metric="pr_auc" aria-pressed="false">Avg precision</button>
                <button type="button" data-metric="brier" aria-pressed="false">Brier</button>
                <button type="button" data-metric="lift" aria-pressed="false">Lift@10%</button>
              </div>
            </div>
          </div>
          <div class="notice teal" id="datasetNote">
            <span class="ico">✓</span>
            <div><strong>Primary chronological test.</strong> Frozen rows, one evaluation, 6,783 customers. This is the evidence for the headline conclusion. <span style="color:var(--faint)">Source: <code>results/guided/test/test_metrics.csv</code></span></div>
          </div>

          <div class="grid g2" style="grid-template-columns:minmax(0,1.55fr) minmax(250px,.65fr);margin-top:14px">
            <div>
              <div class="chart" id="metricChart" role="img" aria-label="Model comparison bar chart"></div>
              <div class="legend" id="metricLegend"></div>
            </div>
            <div>
              <div class="card flat" style="padding:14px;height:100%">
                <div class="card-kicker" id="metricSummaryKicker">ROC-AUC · higher is better</div>
                <h3 id="metricSummaryTitle" style="margin-bottom:6px">B leads A by 0.0302</h3>
                <p id="metricSummaryText">Model C is higher only because it uses post-call duration. It is a NON-DEPLOYABLE leakage diagnostic.</p>
                <div id="metricSideStats" style="margin-top:10px"></div>
                <div class="pill-row" id="metricChipRow"></div>
              </div>
            </div>
          </div>
          <p class="source" id="metricSource">Source: <code>results/guided/test/test_metrics.csv</code> · bootstrap: <code>results/guided/test/hypothesis_tests_test.json</code></p>
        </div>

        <div class="grid g2" style="margin-bottom:14px;grid-template-columns:minmax(0,1.15fr) minmax(290px,.85fr)">
          <div class="card">
            <div class="card-kicker">Paired differences · 95% confidence intervals</div>
            <h3 id="deltaTitle">Primary test · both intervals sit above zero</h3>
            <div class="chart" id="deltaChart" role="img" aria-label="Forest plot of paired bootstrap differences"></div>
            <div class="legend">
              <span><i class="dot" style="background:#5eead4"></i>B − A ranking gain</span>
              <span><i class="dot" style="background:#c4b5fd"></i>C − B leakage diagnostic</span>
              <span><i class="dash"></i>zero / thresholds</span>
            </div>
            <p class="source" id="deltaSource">Source: <code>results/guided/test/hypothesis_tests_test.json</code> · 2,000 resamples · seed 42</p>
          </div>
          <div class="card">
            <div class="card-kicker">Pre-registered interpretation</div>
            <h3 id="interpTitle">H0 rejected, materially relevant</h3>
            <ul class="clean" id="interpList"></ul>
            <div class="notice amber" style="margin-top:12px">
              <span class="ico">!</span>
              <div><strong>Model C warning.</strong> C is a NON-DEPLOYABLE duration-leakage diagnostic in both datasets. Its higher scores do not mean it is a better model to use.</div>
            </div>
          </div>
        </div>

        <div class="grid g2" style="margin-bottom:14px">
          <div class="card">
            <div class="card-kicker">All metrics · exact saved values</div>
            <h3>Model comparison table</h3>
            <div class="table-wrap" style="max-height:360px">
              <table id="perfTable">
                <thead><tr><th>Model</th><th class="num">ROC-AUC</th><th class="num">Avg precision</th><th class="num">Brier</th><th class="num">Lift@10%</th><th class="num">Mean pred.</th></tr></thead>
                <tbody></tbody>
              </table>
            </div>
            <p class="source" id="perfTableSource">Source: <code>results/guided/test/test_metrics.csv</code>. Full precision in the expandable block below.</p>
          </div>
          <div class="card">
            <div class="card-kicker">Lift@10% mechanics</div>
            <h3 id="topKTitle">Top 678 customers · Model B</h3>
            <p id="topKText">The top decile contains 364 positives against 266.8 expected by chance, for a lift of 1.364×. Lift depends only on ordering.</p>
            <div class="grid g3" style="margin-top:12px;gap:8px">
              <div class="cal-stat"><div class="l">Top-k rows</div><div class="v" id="topK">678</div></div>
              <div class="cal-stat"><div class="l">Positives in top-k</div><div class="v" id="topKPos">364</div></div>
              <div class="cal-stat"><div class="l">Expected by chance</div><div class="v" id="topKExp">266.78</div></div>
            </div>
            <p style="margin-top:11px">A lift of 1.19 (A) and 1.36 (B) means the top decile contains about 19% and 36% more positives than random selection. Useful for prioritisation, not evidence of a well-scaled probability.</p>
            <p class="source">Source: <code>results/guided/test/test_metrics.csv</code> · <code>results/guided/test/calibration_interpretation.md</code> §4</p>
          </div>
        </div>

        <details class="raw">
          <summary><span>Exact values and provenance (full precision)</span></summary>
          <div class="raw-body">
            <div class="grid g2" style="gap:16px">
              <div id="exactPrimary"></div>
              <div id="exactSensitivity"></div>
            </div>
            <p class="source">Primary source: <code>results/guided/test/test_metrics.csv</code> · <code>results/guided/test/hypothesis_tests_test.json</code><br>
            Sensitivity source: <code>results/sensitivity_corrected_v1/metrics.csv</code> · <code>results/sensitivity_corrected_v1/bootstrap.json</code></p>
          </div>
        </details>

        <div class="notice amber" style="margin-top:14px" id="sensitivityNotice">
          <span class="ico">!</span>
          <div><strong>Sensitivity is secondary, not a replacement.</strong> The corrected random split mixes time periods between training and evaluation, so it is optimistic by construction. Its test block also overlaps the chronological test block by <span id="overlapRows">1,022</span> rows, so the two results are not independent. <span style="color:var(--faint)">Sources: <code>README.md</code> §4 · <code>results/sensitivity_corrected_v1/report_evaluation.md</code></span></div>
        </div>
      </section>

      <!-- ================= 04 WORKFLOWS ================= -->
      <section class="section" id="workflows" aria-labelledby="workflowTitle">
        <div class="section-head">
          <div class="eyebrow">04 · Guided vs autonomous</div>
          <h2 id="workflowTitle">Two agent runs, one frozen plan — and a precise account of what matched</h2>
          <p class="lede">The comparison below uses only the verified comparative audit already stored in this project. The separate autonomous workspace was not opened or re-read.</p>
        </div>

        <div class="grid g2" style="margin-bottom:14px">
          <div class="card">
            <div class="card-head"><div><div class="card-kicker">Guided run</div><h3>Explicit, numbered instructions</h3></div><span class="badge blue">opencode/big-pickle</span></div>
            <p>The human supervisor issued stage-by-stage instructions. This run produced the primary chronological evaluation and the frozen test artifacts used throughout this report.</p>
            <ul class="clean">
              <li>Models A and B: selected <code>C = 0.01</code>.</li>
              <li>Model C: selected <code>C = 0.001</code>.</li>
              <li>Grid: 9 regularisation values.</li>
            </ul>
          </div>
          <div class="card">
            <div class="card-head"><div><div class="card-kicker">Autonomous run</div><h3>Independent execution, same plan</h3></div><span class="badge violet">GLM-5.3-Flash Free</span></div>
            <p>An independent execution in a separate workspace, used as a cross-check. It did not share the guided workspace's fitted objects.</p>
            <ul class="clean">
              <li>Models A and B: selected <code>C = 0.01</code>.</li>
              <li>Model C: selected <code>C = 0.01</code> — not comparable with guided C.</li>
              <li>Grid: 5 regularisation values.</li>
            </ul>
          </div>
        </div>

        <div class="card" style="margin-bottom:14px">
          <div class="card-kicker">Verified comparison · validation-stage evidence</div>
          <h3>What was identical, what differed, and why</h3>
          <div class="table-wrap">
            <table>
              <thead><tr><th>Comparison</th><th>Verified result</th><th>Interpretation</th></tr></thead>
              <tbody>
                <tr><td><strong>A and B predictions</strong></td><td>Bit-identical across all 6,781 validation rows; <code>max |p_guided − p_autonomous| = 0.000e+00</code>; coefficients identical (e.g. <code>num__age = -0.05842904778480015</code>).</td><td>The fold construction, preprocessing, seed and fitted A/B models are equivalent.</td></tr>
                <tr><td><strong>Model C predictions</strong></td><td>All 6,781 rows differ; <code>max |Δp| = 0.2505</code>.</td><td>Guided selected <code>C = 0.001</code>; autonomous selected <code>C = 0.01</code>. A different penalty, not a different pipeline.</td></tr>
                <tr><td><strong>Hyperparameter grids</strong></td><td>9 values vs 5 values; overlap <code>{0.01, 0.1, 1.0, 10.0}</code>.</td><td>On every overlapping point, CV scores agree to the last displayed digit.</td></tr>
                <tr><td><strong>Model C comparability</strong></td><td>Not directly comparable across runs.</td><td><code>plan.md</code> §5.6 did not specify grid values, so neither grid is a plan violation. This is an implementation choice left open by the design.</td></tr>
              </tbody>
            </table>
          </div>
          <p class="source">Source: <code>results/verification/guided_vs_autonomous_comparison.md</code> §1, §3.2, §4.1 · <code>mistakes.md</code> H-1</p>
        </div>

        <div class="grid g3" style="margin-bottom:14px">
          <div class="card">
            <span class="badge amber">Audit finding A-1</span>
            <h3 style="margin-top:9px">Test-isolation wording was overstated</h3>
            <p>The autonomous documentation claimed the test block was “never loaded” and labels “never read”. The code loaded all 45,211 rows and aggregated the full label column. Reading labels is not the same as using them for fitting or evaluation — but the claim went further than the code.</p>
          </div>
          <div class="card">
            <span class="badge amber">Audit finding A-2</span>
            <h3 style="margin-top:9px">A check that could not fail</h3>
            <p><code>test_untouched = np.array_equal(raw_y[test], raw_y[test])</code> is a tautology. It was reported as PASS evidence. A verification check that cannot fail is not a check.</p>
          </div>
          <div class="card">
            <span class="badge amber">Audit finding A-3</span>
            <h3 style="margin-top:9px">Documentation did not match the columns passed</h3>
            <p>The documentation said categorical codes were decoded and never used as numerics. In fact, the original integer code columns were passed to the one-hot encoder for six variables; only <code>poutcome_level</code> was decoded. The fitted model is unaffected — identical coefficients prove one-hot equivalence — but the documentation was wrong.</p>
          </div>
        </div>

        <div class="callout">
          <div class="card-kicker">What the cross-run comparison establishes</div>
          <p class="quote" style="margin-bottom:8px">For Models A and B, two independent executions of the frozen specification produced the same fitted predictions. For Model C, the runs answered slightly different questions.</p>
          <p>The identical A/B result is evidence that the pipeline and seed are reproducible across runs. It is not evidence that the modelling choices are correct, and it does not extend to Model C because the selected penalty differed. The audit also shows why an independent run is valuable: it caught claims that the original documentation could not support.</p>
          <p class="source">Source: <code>results/verification/guided_vs_autonomous_comparison.md</code> · <code>mistakes.md</code> A-1, A-2, A-3</p>
        </div>
      </section>

      <!-- ================= 05 CALIBRATION ================= -->
      <section class="section" id="calibration" aria-labelledby="calTitle">
        <div class="section-head">
          <div class="eyebrow">05 · Calibration &amp; limitations</div>
          <h2 id="calTitle">The ranking is modest; the probability scale is severely wrong</h2>
          <p class="lede">Every A and B decile sits far below its observed subscription rate. These scores can order customers. They cannot answer “what is the probability this customer subscribes?”</p>
        </div>

        <div class="notice red" style="margin-bottom:14px">
          <span class="ico">!</span>
          <div><strong>Do not read these scores as personal probabilities.</strong> On the chronological test block the mean predicted probability is about 6.8% while about 39.3% of customers actually subscribed. A low score does not mean the customer will not subscribe. Ranking only.</div>
        </div>

        <div class="card" style="margin-bottom:14px">
          <div class="toolbar">
            <div class="toolbar-group">
              <div class="segmented" role="tablist" aria-label="Calibration model">
                <button type="button" role="tab" data-cal="A" aria-selected="true">Model A</button>
                <button type="button" role="tab" data-cal="B" aria-selected="false">Model B</button>
                <button type="button" role="tab" data-cal="C" aria-selected="false">Model C · diagnostic</button>
              </div>
            </div>
            <span class="badge violet">C is NON-DEPLOYABLE</span>
          </div>

          <div class="grid g2" style="grid-template-columns:minmax(0,1.5fr) minmax(240px,.6fr)">
            <div>
              <div class="chart" id="calChart" role="img" aria-label="Calibration decile chart"></div>
              <div class="legend">
                <span><i class="dot" id="calLegendDot" style="background:#5eead4"></i><span id="calLegendName">Model A</span> deciles</span>
                <span><i class="dash"></i>perfect calibration (y = x)</span>
                <span><i class="dash" style="border-color:#f5b461"></i>observed test prevalence</span>
                <span><i class="dash" style="border-color:#7aa2ff"></i>mean predicted</span>
              </div>
            </div>
            <div>
              <div class="cal-stats" id="calStats"></div>
              <div class="notice amber" style="margin-top:10px;padding:10px 11px">
                <span class="ico">!</span>
                <div style="font-size:11.5px">The vertical distance from a point to the dashed diagonal is the calibration gap at that decile.</div>
              </div>
            </div>
          </div>
          <p class="source">Source: <code>results/guided/test/test_calibration.csv</code> · interpretation: <code>results/guided/test/calibration_interpretation.md</code> · 10 equal-count deciles</p>
        </div>

        <div class="grid g2" style="margin-bottom:14px">
          <div class="card">
            <div class="card-kicker" id="calTableKicker">Model A · decile detail</div>
            <h3>Predicted vs observed, everywhere in the score range</h3>
            <div class="table-wrap" style="max-height:330px">
              <table id="calTable">
                <thead><tr><th>Decile bin</th><th class="num">n</th><th class="num">Mean predicted</th><th class="num">Observed</th><th class="num">Gap</th></tr></thead>
                <tbody></tbody>
              </table>
            </div>
            <p class="source">Source: <code>results/guided/test/test_calibration.csv</code>. Gap = observed − predicted (arithmetic on the saved columns).</p>
          </div>
          <div class="card">
            <div class="card-kicker">Why this is systematic, not noise</div>
            <h3>Every decile is displaced in the same direction</h3>
            <p>A model whose probabilities were merely noisy would scatter above and below the diagonal. Here every A and B decile sits far below its observed rate: mean predictions span roughly 0.04–0.11, while the outcome occurs in roughly a quarter to a half of each decile.</p>
            <p>The predicted level on test is close to the <strong>training</strong> prevalence (0.058) and far from the test prevalence (0.393). That description is consistent with fitted probabilities reflecting the base rate the model was trained on. It is a description, not an established cause: these artifacts do not separate base-rate shift from regularisation/shrinkage or any other mechanism.</p>
            <p class="source">Source: <code>results/guided/test/calibration_interpretation.md</code> §3</p>
          </div>
        </div>

        <div class="grid g3" style="margin-bottom:14px">
          <div class="card">
            <div class="card-kicker">Hindsight diagnostic · not a baseline</div>
            <h3>Constant predictor Brier</h3>
            <div class="kpi-value" style="font-size:31px;margin:4px 0" id="hindsightBrier">0.2387</div>
            <p>Assigning every test row the observed prevalence gives Brier <code>p(1 − p)</code> = <strong id="hindsightExact">0.2386542798098002</strong>. This uses the test labels, so it is a hindsight diagnostic only — not an operationally available forecast and not a fair prospective baseline.</p>
            <p>Both A and B have worse Brier scores (0.3427, 0.3424) than a constant that is told the answer in advance. That is the size of the probability-quality gap.</p>
            <p class="source">Sources: <code>results/guided/test/calibration_interpretation.md</code> §4 · <code>results/guided/test/test_metrics.csv</code></p>
          </div>
          <div class="card">
            <div class="card-kicker">Subgroup descriptives · no hypothesis test</div>
            <h3>The gap is not confined to one group</h3>
            <div class="table-wrap">
              <table>
                <thead><tr><th>Group</th><th class="num">n</th><th class="num">Prevalence</th><th class="num">B − A ΔAUC</th></tr></thead>
                <tbody>
                  <tr><td>Never contacted</td><td class="num" id="subNeverN">3,343</td><td class="num" id="subNeverPrev">0.3314</td><td class="num" id="subNeverDelta">0.000144</td></tr>
                  <tr><td>Previously contacted</td><td class="num" id="subPrevN">3,440</td><td class="num" id="subPrevPrev">0.4538</td><td class="num" id="subPrevDelta">0.069125</td></tr>
                </tbody>
              </table>
            </div>
            <p style="margin-top:10px">The apparent benefit is concentrated in previously contacted customers. These are descriptive numbers with no subgroup hypothesis test attached and should not be read as a confirmed moderation effect.</p>
            <p class="source">Source: <code>results/verification/subgroup_descriptives_test.md</code></p>
          </div>
          <div class="card">
            <div class="card-kicker">Deployment verdict</div>
            <h3>Ranking prototype, not risk model</h3>
            <ul class="clean warn">
              <li>Mean predicted probability is about one-sixth of the observed test rate, and every decile is similarly displaced.</li>
              <li>Brier is worse than the hindsight constant diagnostic.</li>
              <li>No recalibration, prior correction or threshold tuning was pre-registered or performed at this stage.</li>
              <li>B's advantage over A supports “B ranks better than A”, not “B is deployable”.</li>
            </ul>
            <p style="margin-top:10px">Estimating a deployable probability would require a separately designed and pre-registered calibration procedure fitted without the test block. This report does not propose or perform one.</p>
            <p class="source">Source: <code>results/guided/test/calibration_interpretation.md</code> §6</p>
          </div>
        </div>

        <div class="notice amber">
          <span class="ico">!</span>
          <div><strong>No causal claim.</strong> These are single-block descriptive results from one chronological split. They do not support causal effects and do not establish performance on future periods. The random-split sensitivity result is optimistic by construction and does not replace the chronological test.</div>
        </div>
      </section>

      <!-- ================= 06 MISTAKES ================= -->
      <section class="section" id="mistakes" aria-labelledby="mistakesTitle">
        <div class="section-head">
          <div class="eyebrow">06 · AI mistakes &amp; reflection</div>
          <h2 id="mistakesTitle">An append-only ledger: errors, detections, corrections — and what changed later</h2>
          <p class="lede">Historical entries keep the meaning they had when written. Later status updates and the calibration reflection are separate dated records. Corrections are not used to relabel an earlier mistake as valid.</p>
        </div>

        <div class="grid g4" style="margin-bottom:14px">
          <div class="card tight">
            <div class="kpi-label">Ledger entries</div>
            <div class="kpi-value" style="font-size:29px" id="mistakeTotal">18</div>
            <div class="kpi-sub">6 human design · 5 agent · 5 self-corrections</div>
          </div>
          <div class="card tight">
            <div class="kpi-label">Human design omissions</div>
            <div class="kpi-value" style="font-size:29px;color:#cfe0ff" id="mistakeH">6</div>
            <div class="kpi-sub">Plan gaps that affected comparability or reporting</div>
          </div>
          <div class="card tight">
            <div class="kpi-label">Agent mistakes</div>
            <div class="kpi-value" style="font-size:29px;color:#ffe3b8" id="mistakeA">5</div>
            <div class="kpi-sub">Including overstated isolation claims</div>
          </div>
          <div class="card tight">
            <div class="kpi-label">Successful self-corrections</div>
            <div class="kpi-value" style="font-size:29px;color:#c9fff7" id="mistakeS">5</div>
            <div class="kpi-sub">Detected and fixed within the run</div>
          </div>
        </div>

        <div class="card" style="margin-bottom:14px">
          <div class="toolbar" style="margin-bottom:10px">
            <div class="toolbar-group">
              <div class="segmented" role="group" aria-label="Filter mistake ledger">
                <button type="button" data-filter="all" aria-pressed="true">All</button>
                <button type="button" data-filter="H" aria-pressed="false">Human design · 6</button>
                <button type="button" data-filter="A" aria-pressed="false">Agent · 5</button>
                <button type="button" data-filter="S" aria-pressed="false">Self-corrections · 5</button>
                <button type="button" data-filter="U" aria-pressed="false">Later updates · 2</button>
              </div>
            </div>
            <button class="btn sm" type="button" id="expandMistakes">Expand visible</button>
          </div>
          <p class="sub" style="margin:0 0 12px">Ledger order follows <code>mistakes.md</code>. Click any card to expand the full detection and correction record.</p>
          <div class="timeline" id="mistakesList"></div>
        </div>

        <div class="grid g2">
          <div class="card">
            <span class="badge amber">A-5 · the invalid random-split run</span>
            <h3 style="margin-top:9px">Wrong procedure, preserved as evidence</h3>
            <p>The original guided random-split evaluation implemented a train-first split instead of the authorised test-first procedure. Block sizes were correct and the run was internally reproducible, but membership was wrong. It is archived unchanged under <code>results/sensitivity_erroneous_v1/</code> and is never cited as valid evidence.</p>
            <p>The corrected split was independently regenerated by the human author, verified against the authorised procedure, and only then evaluated. The correction is a later record — the original row is not rewritten.</p>
            <p class="source">Sources: <code>mistakes.md</code> A-5 · <code>results/audit/guided_random_split_protocol_discrepancy.md</code> · <code>results/sensitivity_corrected_v1/report_evaluation.md</code></p>
          </div>
          <div class="card">
            <span class="badge blue">H-7 · calibration reflection</span>
            <h3 style="margin-top:9px">Documentation is not mitigation</h3>
            <p>The warning was visible before the test block was opened: validation prevalence was roughly double training prevalence, and mean predictions were roughly half the validation rate. The gap was documented but not escalated into a stop-and-review decision.</p>
            <p class="quote" id="reflectionQuote" style="margin-top:10px">Documentation is not mitigation.</p>
            <p style="margin-top:10px">The test block then repeated the pattern at larger magnitude. The reflection adds no new metric and does not recalibrate anything; it records what should have been planned in advance.</p>
          </div>
        </div>
      </section>

      <!-- ================= 07 VERIFICATION ================= -->
      <section class="section" id="verification" aria-labelledby="verifyTitle">
        <div class="section-head">
          <div class="eyebrow">07 · Independent verification</div>
          <h2 id="verifyTitle">What was checked, and what the check does not establish</h2>
          <p class="lede">The reported numbers were checked three ways — by the human author in PowerShell, by a 77-check recomputation script, and by read-only agent audits. They support different claims, and this section keeps them apart.</p>
        </div>

        <div class="grid g2" style="margin-bottom:14px;grid-template-columns:minmax(0,.8fr) minmax(0,1.2fr)">
          <div class="card">
            <div class="ring-wrap">
              <div class="ring" aria-label="77 of 77 checks passed">
                <svg viewBox="0 0 120 120" aria-hidden="true">
                  <defs><linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#5eead4"/><stop offset="1" stop-color="#7aa2ff"/></linearGradient></defs>
                  <circle class="ring-c" cx="60" cy="60" r="54"></circle>
                  <circle class="ring-v" cx="60" cy="60" r="54"></circle>
                </svg>
                <div class="ring-txt"><b id="ringChecks">77/77</b><span>checks pass</span></div>
              </div>
              <div>
                <span class="badge green">STATUS PASS</span>
                <h3 style="margin-top:9px">Reproduced to 1e-6</h3>
                <p style="font-size:12.5px">All three models' metrics, B − A and C − B point estimates and bootstrap intervals, split metadata and SHA-256 integrity.</p>
              </div>
            </div>
            <div class="grid g2" style="margin-top:14px;gap:8px">
              <div class="cal-stat"><div class="l">Failed checks</div><div class="v" id="verifyFailed">0</div></div>
              <div class="cal-stat"><div class="l">Tolerance</div><div class="v">1e-6</div></div>
              <div class="cal-stat"><div class="l">Bootstrap resamples</div><div class="v" id="verifyBoot">2,000</div></div>
              <div class="cal-stat"><div class="l">Skipped single-class</div><div class="v" id="verifySkipped">0</div></div>
            </div>
            <p class="source">Source: <code>results/verification/final_test_recompute_checks.json</code></p>
          </div>
          <div class="card">
            <div class="card-kicker">Frozen artifact integrity</div>
            <h3><span id="integrityFiles">112/112</span> files unchanged</h3>
            <p>Before and after the recomputation, every tracked file matched its recorded SHA-256. No files were added, removed or changed during the check.</p>
            <div class="bar-meter"><i style="width:100%"></i></div>
            <div class="grid g3" style="margin-top:12px;gap:8px">
              <div class="cal-stat"><div class="l">Added</div><div class="v" id="intAdded">0</div></div>
              <div class="cal-stat"><div class="l">Removed</div><div class="v" id="intRemoved">0</div></div>
              <div class="cal-stat"><div class="l">Changed</div><div class="v" id="intChanged">0</div></div>
            </div>
            <div class="notice teal" style="margin-top:12px">
              <span class="ico">✓</span>
              <div><strong>Model export validation.</strong> The reconstructed A and B pipelines reproduce all <span id="exportRows">6,781</span> saved validation predictions with max absolute difference <code>0.000e+00</code> at tolerance <code>1e-12</code>. These are post-test reconstructions, not the original in-memory objects.</div>
            </div>
            <div class="table-wrap" style="margin-top:12px">
              <table>
                <thead><tr><th>Exported model</th><th>SHA-256</th><th class="num">Bytes</th></tr></thead>
                <tbody id="exportTable"></tbody>
              </table>
            </div>
            <p class="source">Sources: <code>results/verification/model_export_validation.md</code> · <code>models/export_manifest.json</code></p>
          </div>
        </div>

        <div class="card" style="margin-bottom:14px">
          <div class="toolbar">
            <div>
              <div class="card-kicker">Check explorer</div>
              <h3 style="margin:0">All 77 checks, grouped</h3>
            </div>
            <div class="segmented" role="group" aria-label="Filter checks" id="checkFilters">
              <button type="button" data-check="all" aria-pressed="true">All · 77</button>
            </div>
          </div>
          <div class="check-list" id="checkList"></div>
          <p class="source">Source: <code>results/verification/final_test_recompute_checks.json</code> · the script recomputes from saved predictions without importing the original metric code.</p>
        </div>

        <div class="grid g2">
          <div class="card">
            <div class="card-kicker">What the verification establishes</div>
            <h3>Reproducibility of the reported numbers</h3>
            <ul class="clean">
              <li>The frozen metrics agree with an independently written recomputation to within 1e-6.</li>
              <li>The paired-bootstrap point estimates and intervals are reproduced.</li>
              <li>Split sizes, row alignment, metadata and artifact hashes are consistent.</li>
              <li>All 77 checks pass, with zero failed checks.</li>
            </ul>
          </div>
          <div class="card">
            <div class="card-kicker">What was NOT verified</div>
            <h3>Limits of the check</h3>
            <ul class="clean warn">
              <li>Feature engineering, categorical decoding, the derived previous-campaign variable, preprocessing and training were not independently re-derived. The checks are metric-level.</li>
              <li>Labels come from the saved prediction file. Their match to the raw Parquet was checked by the human author, not re-established by the script.</li>
              <li>No clean-environment rerun from the pinned <code>uv.lock</code> was performed. Determinism evidence comes from repeated runs in the same environment.</li>
              <li>Verification establishes reproducibility, not the correctness of the modelling pipeline.</li>
            </ul>
          </div>
        </div>

        <!-- ---- 07b Human-in-the-loop ---- -->
        <div class="hitl-head">
          <div class="eyebrow">07b · Human-in-the-loop</div>
          <h3>Human-in-the-loop: Independent Verification</h3>
          <p>Three different kinds of checking ran on this project: the human author working ad-hoc in PowerShell, a 77-check recomputation script, and read-only agent audits. They are kept separate here, because they support different claims.</p>
        </div>

        <div class="lane-grid" style="margin-bottom:14px">
          <div class="lane">
            <div class="lane-h">
              <span class="lane-ico">H</span>
              <b>Human-directed<em>3 recorded check batches</em></b>
            </div>
            <ul>
              <li>6,783 test rows aligned to <code>[38428, 45211)</code>, with 2,669 positives.</li>
              <li>Test labels matched against the original Parquet file.</li>
              <li>ROC-AUC, average precision, Brier and lift@10% recomputed for A, B and C in both executions.</li>
              <li>Primary B − A paired bootstrap: 2,000 resamples, seed 42.</li>
              <li>Subgroup descriptives and corrected split membership.</li>
            </ul>
            <div class="lane-src">results/verification/final_test_independent_verification.md · subgroup_descriptives_test.md · provenance_update.md §4</div>
          </div>
          <div class="lane auto">
            <div class="lane-h">
              <span class="lane-ico">A</span>
              <b>Automated recomputation<em>77 checks · 1 script</em></b>
            </div>
            <ul>
              <li>File hashes, split metadata, row alignment and prevalence.</li>
              <li>A, B and C point metrics, with alternative implementations for ROC-AUC and Brier.</li>
              <li>Both paired bootstraps, B − A and the C − B leakage diagnostic.</li>
              <li>Tolerance 1e-6, zero failed checks, 112/112 files unchanged.</li>
            </ul>
            <div class="lane-src">results/verification/final_test_recompute_report.md · final_test_recompute_checks.json</div>
          </div>
          <div class="lane agent">
            <div class="lane-h">
              <span class="lane-ico">R</span>
              <b>Read-only agent audits<em>2 archived reports</em></b>
            </div>
            <ul>
              <li>Guided versus autonomous comparative audit.</li>
              <li>Random-split protocol root-cause audit.</li>
              <li>Wrote reports only; changed no result, prediction or metric file.</li>
            </ul>
            <div class="lane-src">results/verification/guided_vs_autonomous_comparison.md · results/audit/guided_random_split_protocol_discrepancy.md</div>
          </div>
        </div>

        <div class="card" style="margin-bottom:14px">
          <div class="card-kicker">Case study · mistake A-5</div>
          <h3 style="margin:0 0 4px">One wrong split, caught by comparison</h3>
          <p class="sub" style="margin:0 0 8px">Five steps. Select any step to see its evidence.</p>
          <div class="stepper" id="a5Steps" role="tablist" aria-label="How mistake A-5 was found and corrected"></div>
          <div class="step-panel" id="a5Panel" role="tabpanel" aria-live="polite"></div>
          <div class="notice amber" style="margin-top:12px">
            <span class="ico">!</span>
            <div><strong>A detour inside the repair.</strong> The audit's own proposed fix reversed sklearn's <code>(train, test)</code> return order. The human author caught it before it ran, and the corrected script uses the authorised call order. The mistake is recorded rather than minimised.</div>
          </div>
        </div>

        <div class="cmp-grid" style="margin-bottom:14px">
          <div class="cmp teal">
            <h4><span class="badge teal">H</span>Human-directed checking</h4>
            <dl>
              <dt>Executor</dt><dd id="humanExecutor">Human project author</dd>
              <dt>Environment</dt><dd id="humanEnv">Windows PowerShell, ad-hoc</dd>
              <dt>Re-runnable script</dt><dd>No — what survives is the written record of each check and its result.</dd>
              <dt>What it added</dt><dd>Independent reasoning, protocol comparison against a separately implemented reference, and detection of an anomaly no automated check was looking for.</dd>
              <dt>Independent B − A CI</dt><dd><code id="humanCI">[0.026085, 0.034267]</code> — matched the saved interval within 1e-6.</dd>
            </dl>
          </div>
          <div class="cmp blue">
            <h4><span class="badge blue">A</span>Automated recomputation</h4>
            <dl>
              <dt>Executor</dt><dd>OpenCode audit session</dd>
              <dt>Script</dt><dd><code>results/verification/verify_final_test_recompute.py</code></dd>
              <dt>Structure</dt><dd><span id="autoChecks">77</span> named checks across five groups, stored as JSON and re-runnable.</dd>
              <dt>What it added</dt><dd>Exhaustive, repeatable arithmetic and file-integrity assertions at a stated tolerance.</dd>
              <dt>What it shares</dt><dd>It reads the same saved probabilities and the same bootstrap convention, so near-exact agreement is expected by construction.</dd>
            </dl>
          </div>
        </div>

        <div class="notice teal" style="margin-bottom:14px">
          <span class="ico">✓</span>
          <div><strong>77/77 is not 77 manual calculations.</strong> It is the number of assertions in one script. The human verification is a separate, smaller body of PowerShell checks whose value is independence of reasoning, not volume. The model-export validation and the inference smoke test were added on 2026-11-16, after the experiment had ended: they are post-test usability checks, not original experimental evidence.</div>
        </div>

        <div class="card">
          <div class="grid g2" style="gap:18px">
            <div>
              <div class="card-kicker">What the verification does not prove</div>
              <h3>Bounded claims</h3>
              <div id="hitlLimits"></div>
            </div>
            <div>
              <div class="card-kicker">Explicitly not claimed</div>
              <h3>Outside the evidence</h3>
              <div class="chip-row" id="hitlNotClaimed" style="margin-top:12px"></div>
              <p style="margin-top:12px;font-size:12px">The primary chronological test remains the only primary evaluation. The random-split work is post-test sensitivity analysis, and the automated chronology verdict is still <strong>INCONCLUSIVE</strong> alongside its documented human adjudication.</p>
              <p class="source">Sources: <code>results/verification/provenance_update.md</code> · <code>results/splits/chronology_adjudication.md</code> · <code>results/sensitivity_corrected_v1/conformance_check.json</code></p>
            </div>
          </div>
        </div>
      </section>

      <!-- ================= 08 INFERENCE ================= -->
      <section class="section" id="inference" aria-labelledby="infTitle">
        <div class="section-head">
          <div class="eyebrow">08 · Inference demonstration</div>
          <h2 id="infTitle">Two synthetic customers, their saved scores — and the tool that produced them</h2>
          <p class="lede">This section displays a documented smoke test. It does not reimplement the model in JavaScript and does not score anything live.</p>
        </div>

        <div class="notice violet" style="margin-bottom:14px">
          <span class="ico">i</span>
          <div><strong>OpenCode is the AI coding agent, not the predictive model.</strong> The scores come from saved scikit-learn logistic-regression pipelines (<code>models/A.joblib</code>, <code>models/B.joblib</code>). OpenCode built, ran and audited the analysis; it plays no part in scoring. Model C was not exported because it is a non-deployable leakage diagnostic.</div>
        </div>

        <div class="card" style="margin-bottom:14px">
          <div class="card-kicker">Documented inference path</div>
          <div class="pipeline" aria-label="Inference pipeline">
            <div class="pipe-step"><b>Input CSV</b><span>11 required columns</span></div>
            <div class="pipe-step"><b>Validation</b><span>codes, blanks, schema</span></div>
            <div class="pipe-step"><b>Preprocessing</b><span>StandardScaler + OneHot</span></div>
            <div class="pipe-step"><b>Logistic regression</b><span>saved .joblib pipeline</span></div>
            <div class="pipe-step"><b>Probability</b><span>predict_proba</span></div>
          </div>
          <p class="sub">The CLI prints scores to the terminal only. It writes no file, reads no labels, and fits nothing.</p>
        </div>

        <div class="customer-grid" style="margin-bottom:14px" id="customerGrid"></div>

        <div class="grid g2" style="margin-bottom:14px">
          <div class="card">
            <div class="card-kicker">Reproduce the documented run</div>
            <h3>Inference command</h3>
            <div class="codeblock">
              <header><span>PowerShell · repository root</span><button class="btn sm ghost" type="button" id="copyCmd">Copy</button></header>
              <pre id="commandText">—</pre>
            </div>
            <div class="copy-state" id="copyState" aria-live="polite"></div>
            <p class="source" style="margin-top:10px">Source: <code>results/verification/inference_smoke_test.md</code> §1 · status PASS · exit code 0 · two consecutive runs byte-identical</p>
          </div>
          <div class="card">
            <div class="card-kicker">What this demonstrates — and what it does not</div>
            <h3>Plumbing, not accuracy</h3>
            <div class="grid g2" style="gap:14px">
              <div>
                <span class="badge green">Demonstrates</span>
                <ul class="clean" style="margin-top:7px">
                  <li>The saved models load and produce finite scores in [0, 1].</li>
                  <li>Two consecutive runs produce byte-identical output.</li>
                  <li>Invalid input is rejected with a clear error.</li>
                  <li><code>duration</code> and the outcome column are ignored, not consumed.</li>
                </ul>
              </div>
              <div>
                <span class="badge amber">Does not demonstrate</span>
                <ul class="clean warn" style="margin-top:7px">
                  <li>Predictive accuracy or any test-set result.</li>
                  <li>That the scores are usable probabilities for an individual.</li>
                  <li>That the reconstructed pipelines are the original in-memory objects.</li>
                  <li>That Model C is deployable — it is not.</li>
                </ul>
              </div>
            </div>
            <p class="source">Sources: <code>results/verification/inference_smoke_test.md</code> · <code>results/verification/model_export_validation.md</code></p>
          </div>
        </div>

        <div class="notice amber">
          <span class="ico">!</span>
          <div><strong>Calibration warning travels with every score.</strong> On the test block the mean predicted probability was about 6.8% while about 39.3% of customers subscribed. A displayed score of 5.37% is not “this customer has a 5.37% chance”. Use the models to rank customers relative to each other, never as a reliable personal probability.</div>
        </div>
      </section>

      <footer class="footer">
        <div>
          <strong>Business Analytics 20595 · Individual AI Project</strong><br>
          Generated by <code>dashboard/build_dashboard.py</code> on <span id="generatedUtc">—</span>. All values are read from saved artifacts.
        </div>
        <details class="raw" style="max-width:520px;min-width:240px">
          <summary><span>Source files used</span></summary>
          <div class="raw-body" id="sourcesList"></div>
        </details>
      </footer>
    </div>
  </main>
</div>
<div class="chart-tooltip" id="chartTooltip" role="status" aria-live="polite"></div>



<script>
const DATA = __DASHBOARD_DATA__;
'use strict';
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const NS = 'http://www.w3.org/2000/svg';
const MODEL_ORDER = ['A', 'B', 'C'];
const MODEL_COLORS = { A: '#7aa2ff', B: '#5eead4', C: '#c4b5fd' };
const MODEL_FILLS = { A: 'rgba(122,162,255,.72)', B: 'rgba(94,234,212,.72)', C: 'rgba(196,181,253,.72)' };
const state = { dataset: 'primary', metric: 'roc_auc', calModel: 'A', mistakeFilter: 'all', checkGroup: 'all', a5Step: 0 };

function fmtInt(n) { return Number(n).toLocaleString('en-US'); }
function f(x, n = 4) { return Number(x).toFixed(n); }
function signed(x, n = 4) { const v = Number(x); return (v > 0 ? '+' : '') + v.toFixed(n); }
function pct(x, n = 1) { return (Number(x) * 100).toFixed(n) + '%'; }
function esc(value) {
  return String(value == null ? '' : value)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function md(value) {
  return esc(value)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+?)`/g, '<code>$1</code>');
}
function setText(id, value) { const el = document.getElementById(id); if (el) el.textContent = value; }
function setHTML(id, value) { const el = document.getElementById(id); if (el) el.innerHTML = value; }
function svgEl(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const key of Object.keys(attrs)) node.setAttribute(key, String(attrs[key]));
  if (parent) parent.appendChild(node);
  return node;
}
function chartRoot(host, w, h) {
  host.innerHTML = '';
  const svg = svgEl('svg', { viewBox: `0 0 ${w} ${h}`, role: 'img', 'aria-hidden': 'false' }, host);
  svg.style.width = '100%';
  return svg;
}
const tipEl = $('#chartTooltip');
function bindTip(node, html) {
  node.style.cursor = 'pointer';
  node.addEventListener('pointerenter', (event) => {
    tipEl.innerHTML = html;
    tipEl.classList.add('show');
    moveTip(event);
  });
  node.addEventListener('pointermove', moveTip);
  node.addEventListener('pointerleave', () => tipEl.classList.remove('show'));
}
function moveTip(event) {
  const pad = 12;
  const rect = tipEl.getBoundingClientRect();
  let x = event.clientX + 14;
  let y = event.clientY + 14;
  if (x + rect.width + pad > window.innerWidth) x = event.clientX - rect.width - 14;
  if (y + rect.height + pad > window.innerHeight) y = event.clientY - rect.height - 14;
  tipEl.style.left = Math.max(pad, x) + 'px';
  tipEl.style.top = Math.max(pad, y) + 'px';
}

function currentDataset() { return state.dataset === 'primary' ? DATA.test : DATA.sensitivity; }
function datasetLabel() { return state.dataset === 'primary' ? 'Primary chronological test' : 'Secondary random-split sensitivity'; }

/* ------------------------------------------------------------------ */
/* Static values                                                       */
/* ------------------------------------------------------------------ */
function fillStatic() {
  const t = DATA.test;
  const m = t.models;
  const meta = DATA.meta;
  setText('sideStamp', meta.generated_utc.slice(0, 10));
  setText('generatedUtc', meta.generated_utc.replace('T', ' ').replace('+00:00', ' UTC'));
  setText('heroRows', fmtInt(t.n_test) + ' test rows');
  setText('heroPositives', fmtInt(t.n_positive) + ' positives');
  setText('heroBoot', fmtInt(t.bootstrap.n_boot) + ' bootstrap resamples');
  setHTML('heroDelta', `${signed(DATA.primary['B-A'].roc_auc.point, 4)}<small>ROC-AUC</small>`);
  setText('heroCI', `95% CI ${f(DATA.primary['B-A'].roc_auc.lo, 4)} – ${f(DATA.primary['B-A'].roc_auc.hi, 4)}`);
  setText('heroVerdict', t.h1_classification);

  const ba = DATA.primary['B-A'];
  setText('kpiDelta', signed(ba.roc_auc.point, 4));
  setText('kpiDeltaCI', `95% CI ${f(ba.roc_auc.lo, 4)} – ${f(ba.roc_auc.hi, 4)}`);
  setText('kpiRoc', f(m.B.roc_auc, 4));
  setText('kpiRocSub', `A ${f(m.A.roc_auc, 4)} · C ${f(m.C.roc_auc, 4)} diagnostic`);
  setHTML('kpiLift', `${f(m.B.lift, 3)}<small>×</small>`);
  setText('kpiLiftSub', `${fmtInt(m.B.lift_tp_in_top_k)} of the top ${fmtInt(m.B.lift_k)} are positive`);
  setHTML('kpiCal', `${f(DATA.ratios.B, 2)}<small>× low</small>`);
  setText('kpiCalSub', `B: ${f(m.B.mean_predicted, 4)} vs observed ${f(t.prevalence, 4)}`);

  setText('findingDelta', f(ba.roc_auc.point, 4));
  setText('findingCI', `[${f(ba.roc_auc.lo, 4)}, ${f(ba.roc_auc.hi, 4)}]`);
  setText('findingAP', f(ba.pr_auc.point, 4));
  setText('findingMeanB', f(m.B.mean_predicted, 4));
  setText('findingPrev', f(t.prevalence, 4));
  setText('findingRatioB', f(DATA.ratios.B, 2) + '×');
  setText('findingC', f(m.C.roc_auc, 4));
  setText('findingCB', `${f(DATA.primary['C-B'].roc_auc.point, 4)} [${f(DATA.primary['C-B'].roc_auc.lo, 4)}, ${f(DATA.primary['C-B'].roc_auc.hi, 4)}]`);

  setText('splitStrategy', DATA.split.strategy);
  setText('splitHash', DATA.split.hash);
  setText('trainN', fmtInt(DATA.split.sizes.train));
  setText('valN', fmtInt(DATA.split.sizes.validation));
  setText('testN', fmtInt(DATA.split.sizes.test));
  setText('trainPrev', f(DATA.split.train_prevalence, 4));
  setText('valPrev', f(DATA.split.validation_prevalence, 4));
  setText('testPrev', f(DATA.test.prevalence, 4));

  setText('hindsightBrier', f(DATA.test.hindsight_brier, 4));
  setText('hindsightExact', String(DATA.test.hindsight_brier));
  setText('subNeverN', fmtInt(DATA.subgroups.never.n));
  setText('subPrevN', fmtInt(DATA.subgroups.previous.n));
  setText('subNeverPrev', f(DATA.subgroups.never.prevalence, 4));
  setText('subPrevPrev', f(DATA.subgroups.previous.prevalence, 4));
  setText('subNeverDelta', f(DATA.subgroups.never.delta_roc_auc, 6));
  setText('subPrevDelta', f(DATA.subgroups.previous.delta_roc_auc, 6));

  setText('topK', fmtInt(m.B.lift_k));
  setText('topKPos', fmtInt(m.B.lift_tp_in_top_k));
  setText('topKExp', f(m.B.lift_expected_tp_by_chance, 2));
  setText('topKTitle', `Top ${fmtInt(m.B.lift_k)} customers · Model B`);
  setText('topKText', `The top decile contains ${fmtInt(m.B.lift_tp_in_top_k)} positives against ${f(m.B.lift_expected_tp_by_chance, 1)} expected by chance, for a lift of ${f(m.B.lift, 3)}×. Lift depends only on ordering.`);
  setText('overlapRows', fmtInt(DATA.sensitivity.overlap_test_rows));

  setText('ringChecks', `${DATA.verification.n_checks}/${DATA.verification.n_checks}`);
  setText('verifyFailed', String(DATA.verification.n_failed));
  setText('verifyBoot', fmtInt(DATA.verification.n_boot));
  setText('verifySkipped', String(DATA.verification.skipped_single_class));
  setText('integrityFiles', `${DATA.verification.integrity.n_files_before}/${DATA.verification.integrity.n_files_after}`);
  setText('intAdded', String(DATA.verification.integrity.added.length));
  setText('intRemoved', String(DATA.verification.integrity.removed.length));
  setText('intChanged', String(DATA.verification.integrity.changed.length));
  setText('exportRows', fmtInt(DATA.export.validation_rows));

  setText('mistakeTotal', String(DATA.mistakes.items.length + DATA.mistakes.updates.length));
  setText('mistakeH', String(DATA.mistakes.counts.H));
  setText('mistakeA', String(DATA.mistakes.counts.A));
  setText('mistakeS', String(DATA.mistakes.counts.S));
  setText('reflectionQuote', DATA.mistakes.reflection_quote);

  setText('commandText', DATA.inference.command);
  const sources = DATA.sources.map((s) => `<div class="kv"><span class="k">source</span><span class="v">${esc(s)}</span></div>`).join('');
  setHTML('sourcesList', sources);
}

/* ------------------------------------------------------------------ */
/* Metric bar chart                                                    */
/* ------------------------------------------------------------------ */
const METRICS = {
  roc_auc: { label: 'ROC-AUC', max: 1.0, ref: 0.5, refLabel: 'chance · 0.5', digits: 4, better: 'higher is better' },
  pr_auc: { label: 'Average precision', max: 1.0, ref: null, refLabel: 'no-skill · prevalence', digits: 4, better: 'higher is better' },
  brier: { label: 'Brier score', max: 0.5, ref: null, refLabel: 'hindsight constant', digits: 4, better: 'lower is better' },
  lift: { label: 'Lift@10%', max: 2.0, ref: 1.0, refLabel: 'chance · 1.0×', digits: 3, better: 'higher is better' }
};

function metricSummary(metric, ds) {
  const m = ds.models;
  const d = DATA[state.dataset === 'primary' ? 'primary' : 'sensitivity'];
  if (metric === 'roc_auc') {
    return {
      kicker: 'ROC-AUC · higher is better',
      title: `B leads A by ${f(d['B-A'].roc_auc.point, 4)}`,
      text: 'A and B rank customers only modestly above chance. Model C is higher only because it uses post-call duration; it is a NON-DEPLOYABLE leakage diagnostic.',
      stats: [
        ['A', f(m.A.roc_auc, 4)], ['B', f(m.B.roc_auc, 4)], ['C · diagnostic', f(m.C.roc_auc, 4)]
      ]
    };
  }
  if (metric === 'pr_auc') {
    return {
      kicker: 'Average precision · higher is better',
      title: `B precision is ${f(m.B.pr_auc, 4)}`,
      text: `The no-skill reference is the observed prevalence, ${f(ds.prevalence, 4)}. Both A and B are above it, but the gain is modest.`,
      stats: [['A', f(m.A.pr_auc, 4)], ['B', f(m.B.pr_auc, 4)], ['C · diagnostic', f(m.C.pr_auc, 4)]]
    };
  }
  if (metric === 'brier') {
    const hb = ds.prevalence * (1 - ds.prevalence);
    return {
      kicker: 'Brier score · lower is better',
      title: `A and B are worse than a hindsight constant`,
      text: `Brier depends on probability level. The hindsight constant (${f(hb, 4)}) is told the answer in advance and is a diagnostic, not an operational baseline.`,
      stats: [['A', f(m.A.brier, 4)], ['B', f(m.B.brier, 4)], ['C · diagnostic', f(m.C.brier, 4)]]
    };
  }
  return {
    kicker: 'Lift@10% · higher is better',
    title: `B finds ${f(m.B.lift, 3)}× the chance rate`,
    text: 'Lift measures how many more positives appear in the top decile than random selection would produce. It depends only on ordering, not probability level.',
    stats: [['A', f(m.A.lift, 3) + '×'], ['B', f(m.B.lift, 3) + '×'], ['C · diagnostic', f(m.C.lift, 3) + '×']]
  };
}

function renderMetricChart() {
  const ds = currentDataset();
  const def = METRICS[state.metric];
  const host = $('#metricChart');
  const w = 760, h = 370;
  const pad = { l: 52, r: 22, t: 26, b: 74 };
  const svg = chartRoot(host, w, h);
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const yMax = def.max;
  const y = (v) => pad.t + ph * (1 - v / yMax);

  for (let i = 0; i <= 4; i++) {
    const value = (yMax / 4) * i;
    const yy = y(value);
    svgEl('line', { x1: pad.l, x2: w - pad.r, y1: yy, y2: yy, class: 'gridline' }, svg);
    svgEl('text', { x: pad.l - 8, y: yy + 3.5, 'text-anchor': 'end', class: 'tick' }, svg).textContent =
      state.metric === 'lift' ? value.toFixed(1) : value.toFixed(2);
  }
  svgEl('line', { x1: pad.l, x2: w - pad.r, y1: y(0), y2: y(0), class: 'axis' }, svg);

  let ref = def.ref;
  if (state.metric === 'pr_auc') ref = ds.prevalence;
  if (state.metric === 'brier') ref = state.dataset === 'primary' ? DATA.test.hindsight_brier : ds.prevalence * (1 - ds.prevalence);
  if (ref != null) {
    const yy = y(ref);
    svgEl('line', { x1: pad.l, x2: w - pad.r, y1: yy, y2: yy, stroke: 'rgba(245,180,97,.55)', 'stroke-width': 1, 'stroke-dasharray': '4 4' }, svg);
    svgEl('text', { x: w - pad.r, y: yy - 5, 'text-anchor': 'end', class: 'ref-label' }, svg).textContent =
      `${def.refLabel} · ${ref.toFixed(state.metric === 'lift' ? 1 : 4)}`;
  }

  const band = pw / 3;
  const barW = Math.min(84, band * 0.42);
  MODEL_ORDER.forEach((model, index) => {
    const value = ds.models[model][state.metric];
    const cx = pad.l + band * index + band / 2;
    const x = cx - barW / 2;
    const yy = y(value);
    const rect = svgEl('rect', {
      x, y: yy, width: barW, height: Math.max(2, y(0) - yy), rx: 7,
      fill: MODEL_FILLS[model], stroke: MODEL_COLORS[model], 'stroke-width': model === 'C' ? 1.4 : 0.6,
      'stroke-dasharray': model === 'C' ? '5 3' : 'none'
    }, svg);
    bindTip(rect, `<b>Model ${model}</b><br>${def.label}: <span class="t-mono">${value}</span><br>${model === 'C' ? 'NON-DEPLOYABLE leakage diagnostic' : 'Operational candidate'}<br><span class="t-mono">${datasetLabel()}</span>`);

    svgEl('text', { x: cx, y: yy - 8, 'text-anchor': 'middle', class: 'value-label' }, svg).textContent =
      state.metric === 'lift' ? value.toFixed(3) : value.toFixed(4);
    svgEl('text', { x: cx, y: h - pad.b + 21, 'text-anchor': 'middle', class: 'series-label' }, svg).textContent = `Model ${model}`;
    const role = model === 'C' ? 'NON-DEPLOYABLE' : model === 'B' ? 'campaign history' : 'characteristics';
    svgEl('text', { x: cx, y: h - pad.b + 36, 'text-anchor': 'middle', class: 'tick', fill: model === 'C' ? '#c4b5fd' : '#64748b' }, svg).textContent = role;
  });

  const summary = metricSummary(state.metric, ds);
  setText('metricSummaryKicker', summary.kicker);
  setText('metricSummaryTitle', summary.title);
  setText('metricSummaryText', summary.text);
  setHTML('metricSideStats', summary.stats.map(([k, v]) =>
    `<div class="kv"><span class="k">${esc(k)}</span><span class="v">${esc(v)}</span></div>`).join(''));
  setHTML('metricChipRow', `<span class="badge ${state.dataset === 'primary' ? 'teal' : 'amber'}">${esc(datasetLabel())}</span>${state.dataset === 'sensitivity' ? '<span class="badge amber">optimistic by construction</span>' : '<span class="badge">one frozen evaluation</span>'}<span class="badge violet">C is a diagnostic</span>`);
  setHTML('metricLegend', `<span><i style="background:#7aa2ff"></i>Model A</span><span><i style="background:#5eead4"></i>Model B</span><span><i style="background:#c4b5fd"></i>Model C · NON-DEPLOYABLE</span><span><i class="dash"></i>${esc(def.refLabel)}</span>`);
}

/* ------------------------------------------------------------------ */
/* Forest plot of paired differences                                   */
/* ------------------------------------------------------------------ */
function renderDeltaChart() {
  const ds = state.dataset === 'primary' ? DATA.primary : DATA.sensitivity;
  const rows = [
    { pair: 'B-A', metric: 'roc_auc', label: 'B − A', metricLabel: 'ROC-AUC', color: MODEL_COLORS.B, diagnostic: false },
    { pair: 'B-A', metric: 'pr_auc', label: 'B − A', metricLabel: 'Avg precision', color: MODEL_COLORS.B, diagnostic: false },
    { pair: 'C-B', metric: 'roc_auc', label: 'C − B', metricLabel: 'ROC-AUC', color: MODEL_COLORS.C, diagnostic: true },
    { pair: 'C-B', metric: 'pr_auc', label: 'C − B', metricLabel: 'Avg precision', color: MODEL_COLORS.C, diagnostic: true }
  ];
  const host = $('#deltaChart');
  const w = 660, h = 320;
  const pad = { l: 104, r: 112, t: 26, b: 38 };
  const svg = chartRoot(host, w, h);
  const pw = w - pad.l - pad.r;
  const values = rows.flatMap((r) => [ds[r.pair][r.metric].lo, ds[r.pair][r.metric].hi]);
  const xMax = Math.max(0.06, Math.max(...values) * 1.16);
  const x = (v) => pad.l + (v / xMax) * pw;

  for (let i = 0; i <= 4; i++) {
    const value = (xMax / 4) * i;
    const xx = x(value);
    svgEl('line', { x1: xx, x2: xx, y1: pad.t - 6, y2: h - pad.b, class: 'gridline' }, svg);
    svgEl('text', { x: xx, y: h - pad.b + 18, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = value.toFixed(2);
  }
  svgEl('line', { x1: x(0), x2: x(0), y1: pad.t - 6, y2: h - pad.b, stroke: 'rgba(148,163,184,.55)', 'stroke-width': 1 }, svg);
  const thresholds = state.dataset === 'primary'
    ? [{ v: 0.01, label: '0.01 materiality' }, { v: 0.05, label: '0.05 leakage' }]
    : [];
  thresholds.forEach((t) => {
    svgEl('line', { x1: x(t.v), x2: x(t.v), y1: pad.t - 6, y2: h - pad.b, stroke: 'rgba(245,180,97,.38)', 'stroke-width': 1, 'stroke-dasharray': '3 4' }, svg);
    svgEl('text', { x: x(t.v) + 3, y: pad.t - 10, class: 'ref-label' }, svg).textContent = t.label;
  });

  const rowH = (h - pad.t - pad.b) / rows.length;
  rows.forEach((row, index) => {
    const entry = ds[row.pair][row.metric];
    const cy = pad.t + rowH * index + rowH / 2;
    svgEl('text', { x: pad.l - 12, y: cy - 2, 'text-anchor': 'end', class: 'series-label', fill: row.color }, svg).textContent = row.label;
    svgEl('text', { x: pad.l - 12, y: cy + 12, 'text-anchor': 'end', class: 'tick' }, svg).textContent = row.metricLabel + (row.diagnostic ? ' · diag' : '');
    svgEl('line', { x1: x(entry.lo), x2: x(entry.hi), y1: cy, y2: cy, stroke: row.color, 'stroke-width': 5, 'stroke-linecap': 'round', opacity: 0.55 }, svg);
    svgEl('line', { x1: x(entry.lo), x2: x(entry.lo), y1: cy - 6, y2: cy + 6, stroke: row.color, 'stroke-width': 1.4 }, svg);
    svgEl('line', { x1: x(entry.hi), x2: x(entry.hi), y1: cy - 6, y2: cy + 6, stroke: row.color, 'stroke-width': 1.4 }, svg);
    const dot = svgEl('circle', { cx: x(entry.point), cy, r: 5.5, fill: row.color, stroke: '#0b111a', 'stroke-width': 2 }, svg);
    const hiText = signed(entry.point, 4);
    bindTip(dot, `<b>${row.label} · ${row.metricLabel}</b><br>Point: <span class="t-mono">${entry.point}</span><br>95% CI: <span class="t-mono">[${entry.lo}, ${entry.hi}]</span><br>${row.diagnostic ? 'Model C is a NON-DEPLOYABLE leakage diagnostic.' : ''}<br><span class="t-mono">${datasetLabel()}</span>`);
    svgEl('text', { x: w - pad.r + 8, y: cy + 4, class: 'value-label', fill: row.color }, svg).textContent = hiText;
  });

  setText('deltaTitle', state.dataset === 'primary' ? 'Primary test · intervals sit above zero' : 'Sensitivity · optimistic interval, not primary evidence');
  setHTML('deltaSource', state.dataset === 'primary'
    ? 'Source: <code>results/guided/test/hypothesis_tests_test.json</code> · 2,000 resamples · seed 42'
    : 'Source: <code>results/sensitivity_corrected_v1/bootstrap.json</code> · 2,000 resamples · seed 42');
  renderInterpretation();
}

function renderInterpretation() {
  const t = DATA.test;
  const ds = state.dataset === 'primary' ? DATA.primary : DATA.sensitivity;
  const ba = ds['B-A'];
  const cb = ds['C-B'];
  let html = '';
  if (state.dataset === 'primary') {
    html += `<li><strong>Classification:</strong> ${esc(t.h1_classification)}. The B − A ROC-AUC interval [${f(ba.roc_auc.lo, 4)}, ${f(ba.roc_auc.hi, 4)}] is entirely above zero.</li>`;
    html += `<li><strong>Materiality:</strong> ΔROC-AUC ${f(ba.roc_auc.point, 4)} ≥ 0.01. This is a project-level judgment, separate from the statistical test.</li>`;
    html += `<li><strong>Average precision:</strong> ${signed(ba.pr_auc.point, 4)} [${f(ba.pr_auc.lo, 4)}, ${f(ba.pr_auc.hi, 4)}]. Supporting evidence, not the primary criterion.</li>`;
    html += `<li><strong>Leakage diagnostic:</strong> C − B ΔROC-AUC ${f(cb.roc_auc.point, 4)} [${f(cb.roc_auc.lo, 4)}, ${f(cb.roc_auc.hi, 4)}] is far above the pre-registered 0.05 threshold. C is never operational.</li>`;
    setText('interpTitle', t.h1_classification);
  } else {
    html += `<li><strong>Descriptive only:</strong> the corrected random split was not a new pre-registered hypothesis test and does not replace the chronological evaluation.</li>`;
    html += `<li><strong>Optimistic by construction:</strong> the split mixes time periods between training and evaluation. The interval is not an independent confirmation.</li>`;
    html += `<li><strong>Not independent:</strong> the random test block overlaps the chronological test block by ${fmtInt(DATA.sensitivity.overlap_test_rows)} rows.</li>`;
    html += `<li><strong>Still a diagnostic:</strong> C − B ΔROC-AUC ${f(cb.roc_auc.point, 4)} measures duration leakage, not deployment value.</li>`;
    setText('interpTitle', 'Secondary sensitivity · optimistic, not primary');
  }
  setHTML('interpList', html);
}

/* ------------------------------------------------------------------ */
/* Tables                                                              */
/* ------------------------------------------------------------------ */
function renderPerfTable() {
  const ds = currentDataset();
  const tbody = $('#perfTable tbody');
  tbody.innerHTML = MODEL_ORDER.map((model) => {
    const m = ds.models[model];
    const role = model === 'C' ? '<span class="tag c">NON-DEPLOYABLE</span>' : '';
    return `<tr>
      <td><strong>Model ${model}</strong> ${role}<br><span style="font-size:10px;color:var(--faint)">${esc(m.role)}</span></td>
      <td class="num">${f(m.roc_auc, 6)}</td>
      <td class="num">${f(m.pr_auc, 6)}</td>
      <td class="num">${f(m.brier, 6)}</td>
      <td class="num">${f(m.lift, 4)}</td>
      <td class="num">${f(m.mean_predicted, 6)}</td>
    </tr>`;
  }).join('');
  setHTML('perfTableSource', state.dataset === 'primary'
    ? 'Source: <code>results/guided/test/test_metrics.csv</code>. Full precision in the expandable block below.'
    : 'Source: <code>results/sensitivity_corrected_v1/metrics.csv</code>. Secondary and optimistic by construction.');
}

function exactBlock(ds, title, source) {
  let html = `<div class="card-kicker">${esc(title)}</div>`;
  html += `<div class="kv"><span class="k">n_test</span><span class="v">${fmtInt(ds.n_test != null ? ds.n_test : DATA.test.n_test)}</span></div>`;
  html += `<div class="kv"><span class="k">n_positive</span><span class="v">${fmtInt(ds.n_positive != null ? ds.n_positive : DATA.test.n_positive)}</span></div>`;
  html += `<div class="kv"><span class="k">prevalence</span><span class="v">${ds.prevalence != null ? ds.prevalence : DATA.test.prevalence}</span></div>`;
  MODEL_ORDER.forEach((model) => {
    const m = ds.models[model];
    html += `<div class="kv"><span class="k">${model} · ROC-AUC</span><span class="v">${m.roc_auc}</span></div>`;
    html += `<div class="kv"><span class="k">${model} · AP</span><span class="v">${m.pr_auc}</span></div>`;
    html += `<div class="kv"><span class="k">${model} · Brier</span><span class="v">${m.brier}</span></div>`;
    html += `<div class="kv"><span class="k">${model} · mean predicted</span><span class="v">${m.mean_predicted}</span></div>`;
    html += `<div class="kv"><span class="k">${model} · lift</span><span class="v">${m.lift}</span></div>`;
  });
  const pairs = ds === DATA.test ? DATA.primary : DATA.sensitivity;
  ['B-A', 'C-B'].forEach((pair) => {
    ['roc_auc', 'pr_auc'].forEach((metric) => {
      const e = pairs[pair][metric];
      html += `<div class="kv"><span class="k">${pair} · ${metric === 'roc_auc' ? 'ΔROC-AUC' : 'ΔAP'}</span><span class="v">${e.point} [${e.lo}, ${e.hi}]</span></div>`;
    });
  });
  html += `<p class="source">Source: <code>${esc(source)}</code></p>`;
  return html;
}
function renderExact() {
  setHTML('exactPrimary', exactBlock(DATA.test, 'Primary chronological test', 'results/guided/test/test_metrics.csv; results/guided/test/hypothesis_tests_test.json'));
  setHTML('exactSensitivity', exactBlock(DATA.sensitivity, 'Secondary random-split sensitivity', 'results/sensitivity_corrected_v1/metrics.csv; results/sensitivity_corrected_v1/bootstrap.json'));
}

/* ------------------------------------------------------------------ */
/* Hyperparameter cross-validation chart                                */
/* ------------------------------------------------------------------ */
function renderCVChart() {
  const host = $('#cvChart');
  if (!host) return;
  const w = 620, h = 300;
  const pad = { l: 52, r: 22, t: 22, b: 52 };
  const svg = chartRoot(host, w, h);
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const all = MODEL_ORDER.flatMap((m) => DATA.hyperparameters[m].map((r) => r.mean_cv_roc_auc));
  const lo = Math.min(...all), hi = Math.max(...all);
  const yMin = Math.max(0, Math.floor((lo - 0.01) * 50) / 50);
  const yMax = Math.ceil((hi + 0.008) * 50) / 50;
  const xMin = Math.log10(0.001), xMax = Math.log10(10);
  const x = (v) => pad.l + ((Math.log10(v) - xMin) / (xMax - xMin)) * pw;
  const y = (v) => pad.t + ph * (1 - (v - yMin) / (yMax - yMin));

  for (let i = 0; i <= 4; i++) {
    const value = yMin + ((yMax - yMin) / 4) * i;
    const yy = y(value);
    svgEl('line', { x1: pad.l, x2: w - pad.r, y1: yy, y2: yy, class: 'gridline' }, svg);
    svgEl('text', { x: pad.l - 8, y: yy + 3.5, 'text-anchor': 'end', class: 'tick' }, svg).textContent = value.toFixed(3);
  }
  [0.001, 0.01, 0.1, 1, 10].forEach((value) => {
    const xx = x(value);
    svgEl('line', { x1: xx, x2: xx, y1: pad.t, y2: h - pad.b, class: 'gridline' }, svg);
    svgEl('text', { x: xx, y: h - pad.b + 18, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = value >= 1 ? String(value) : value.toFixed(3).replace('0.00', '0.0');
  });
  svgEl('text', { x: pad.l + pw / 2, y: h - 10, 'text-anchor': 'middle', class: 'axis-label' }, svg).textContent = 'regularisation strength C (log scale)';
  svgEl('text', { x: 13, y: pad.t + ph / 2, 'text-anchor': 'middle', class: 'axis-label', transform: `rotate(-90 13 ${pad.t + ph / 2})` }, svg).textContent = 'mean CV ROC-AUC';

  MODEL_ORDER.forEach((model) => {
    const rows = DATA.hyperparameters[model];
    const points = rows.map((r) => [x(r.C), y(r.mean_cv_roc_auc)]);
    svgEl('polyline', { points: points.map((p) => p.join(',')).join(' '), fill: 'none', stroke: MODEL_COLORS[model], 'stroke-width': 1.7, opacity: 0.78 }, svg);
    rows.forEach((r) => {
      const cx = x(r.C), cy = y(r.mean_cv_roc_auc);
      const dot = svgEl('circle', { cx, cy, r: r.selected ? 6 : 3.6, fill: r.selected ? '#0b111a' : MODEL_COLORS[model], stroke: MODEL_COLORS[model], 'stroke-width': r.selected ? 2.4 : 1 }, svg);
      bindTip(dot, `<b>Model ${model}</b><br>C = <span class="t-mono">${r.C}</span><br>Mean CV ROC-AUC: <span class="t-mono">${r.mean_cv_roc_auc}</span><br>SD: <span class="t-mono">${r.std_cv_roc_auc}</span><br>Rank ${r.rank}${r.selected ? ' · SELECTED' : ''}<br><span class="t-mono">validation-side CV, not a test result</span>`);
    });
    const selected = rows.find((r) => r.selected);
    if (selected) {
      svgEl('text', { x: x(selected.C), y: y(selected.mean_cv_roc_auc) - 12, 'text-anchor': 'middle', class: 'tick', fill: MODEL_COLORS[model] }, svg).textContent = `C=${selected.C}`;
    }
  });
}

/* ------------------------------------------------------------------ */
/* Calibration                                                         */
/* ------------------------------------------------------------------ */
function renderCalibration() {
  const model = state.calModel;
  const bins = DATA.calibration[model].slice().sort((a, b) => a.mean_predicted - b.mean_predicted);
  const color = MODEL_COLORS[model];
  const host = $('#calChart');
  const w = 700, h = 390;
  const pad = { l: 62, r: 22, t: 24, b: 56 };
  const svg = chartRoot(host, w, h);
  const pw = w - pad.l - pad.r;
  const ph = h - pad.t - pad.b;
  const xMax = Math.max(0.12, Math.max(...bins.map((b) => b.mean_predicted)) * 1.16);
  const yMax = 0.75;
  const x = (v) => pad.l + (v / xMax) * pw;
  const y = (v) => pad.t + ph * (1 - v / yMax);

  for (let i = 0; i <= 3; i++) {
    const value = (yMax / 3) * i;
    const yy = y(value);
    svgEl('line', { x1: pad.l, x2: w - pad.r, y1: yy, y2: yy, class: 'gridline' }, svg);
    svgEl('text', { x: pad.l - 8, y: yy + 3.5, 'text-anchor': 'end', class: 'tick' }, svg).textContent = value.toFixed(2);
  }
  for (let i = 0; i <= 4; i++) {
    const value = (xMax / 4) * i;
    const xx = x(value);
    svgEl('line', { x1: xx, x2: xx, y1: pad.t, y2: h - pad.b, class: 'gridline' }, svg);
    svgEl('text', { x: xx, y: h - pad.b + 18, 'text-anchor': 'middle', class: 'tick' }, svg).textContent = value.toFixed(2);
  }
  svgEl('line', { x1: pad.l, x2: w - pad.r, y1: y(0), y2: y(0), class: 'axis' }, svg);
  svgEl('text', { x: pad.l + pw / 2, y: h - 12, 'text-anchor': 'middle', class: 'axis-label' }, svg).textContent = 'mean predicted probability';
  svgEl('text', { x: 14, y: pad.t + ph / 2, 'text-anchor': 'middle', class: 'axis-label', transform: `rotate(-90 14 ${pad.t + ph / 2})` }, svg).textContent = 'observed subscription rate';

  const diagEnd = Math.min(xMax, yMax);
  svgEl('line', { x1: x(0), y1: y(0), x2: x(diagEnd), y2: y(diagEnd), stroke: 'rgba(148,163,184,.5)', 'stroke-width': 1, 'stroke-dasharray': '5 5' }, svg);
  svgEl('text', { x: x(diagEnd) - 4, y: y(diagEnd) - 7, 'text-anchor': 'end', class: 'ref-label' }, svg).textContent = 'perfect calibration';

  const prevY = y(DATA.test.prevalence);
  svgEl('line', { x1: pad.l, x2: w - pad.r, y1: prevY, y2: prevY, stroke: 'rgba(245,180,97,.58)', 'stroke-width': 1, 'stroke-dasharray': '4 4' }, svg);
  svgEl('text', { x: w - pad.r, y: prevY - 5, 'text-anchor': 'end', class: 'ref-label' }, svg).textContent = `observed prevalence ${f(DATA.test.prevalence, 4)}`;

  const meanX = x(DATA.test.models[model].mean_predicted);
  svgEl('line', { x1: meanX, x2: meanX, y1: pad.t, y2: h - pad.b, stroke: 'rgba(122,162,255,.55)', 'stroke-width': 1, 'stroke-dasharray': '4 4' }, svg);
  svgEl('text', { x: meanX + 4, y: pad.t + 10, class: 'ref-label' }, svg).textContent = `mean predicted ${f(DATA.test.models[model].mean_predicted, 4)}`;

  const pts = bins.map((b) => [x(b.mean_predicted), y(b.observed_rate)]);
  svgEl('polyline', { points: pts.map((p) => p.join(',')).join(' '), fill: 'none', stroke: color, 'stroke-width': 1.8, opacity: 0.8 }, svg);
  bins.forEach((b, index) => {
    const cx = x(b.mean_predicted), cy = y(b.observed_rate);
    const dot = svgEl('circle', { cx, cy, r: 5, fill: color, stroke: '#0b111a', 'stroke-width': 1.6 }, svg);
    bindTip(dot, `<b>Model ${model} · decile ${index + 1}</b><br>Bin: <span class="t-mono">${esc(b.bin)}</span><br>n = ${b.n}<br>Mean predicted: <span class="t-mono">${b.mean_predicted}</span><br>Observed rate: <span class="t-mono">${b.observed_rate}</span><br>Gap: <span class="t-mono">${(b.observed_rate - b.mean_predicted).toFixed(4)}</span>`);
  });

  svgEl('text', { x: x(xMax * 0.53), y: y(0.66), class: 'ref-label', fill: '#f5b461' }, svg).textContent = 'every decile sits far above the diagonal';

  $('#calLegendDot').style.background = color;
  setText('calLegendName', `Model ${model}${model === 'C' ? ' · diagnostic' : ''}`);
  setText('calTableKicker', `Model ${model} · decile detail${model === 'C' ? ' · NON-DEPLOYABLE' : ''}`);

  const m = DATA.test.models[model];
  const ratio = DATA.ratios[model];
  setHTML('calStats', `
    <div class="cal-stat"><div class="l">Mean predicted</div><div class="v bad">${f(m.mean_predicted, 4)}</div></div>
    <div class="cal-stat"><div class="l">Observed prevalence</div><div class="v">${f(DATA.test.prevalence, 4)}</div></div>
    <div class="cal-stat"><div class="l">Prevalence ÷ prediction</div><div class="v bad">${f(ratio, 2)}×</div></div>
    <div class="cal-stat"><div class="l">Brier score</div><div class="v">${f(m.brier, 4)}</div></div>`);

  $('#calTable tbody').innerHTML = bins.map((b, index) => `
    <tr>
      <td>${index + 1} <span style="color:var(--faint);font-size:9.5px">${esc(b.bin)}</span></td>
      <td class="num">${fmtInt(b.n)}</td>
      <td class="num">${f(b.mean_predicted, 4)}</td>
      <td class="num">${f(b.observed_rate, 4)}</td>
      <td class="num" style="color:var(--amber)">+${f(b.observed_rate - b.mean_predicted, 4)}</td>
    </tr>`).join('');
}

/* ------------------------------------------------------------------ */
/* Mistakes ledger                                                     */
/* ------------------------------------------------------------------ */
function renderMistakes() {
  const host = $('#mistakesList');
  const all = DATA.mistakes.items.map((item) => ({ ...item, group: item.letter }))
    .concat(DATA.mistakes.updates.map((item) => ({ ...item, group: 'U' })));
  const filter = state.mistakeFilter;
  const visible = all.filter((item) => filter === 'all' || (filter === 'U' ? (item.group === 'U' || item.group === 'R') : item.group === filter));
  host.innerHTML = visible.map((item) => {
    const isUpdate = item.group === 'U' || item.group === 'R';
    const cls = item.group === 'H' ? 'h' : item.group === 'A' ? 'a' : item.group === 'S' ? 's' : 'u';
    const badge = item.group === 'H' ? 'blue' : item.group === 'A' ? 'amber' : item.group === 'S' ? 'teal' : 'blue';
    let body = '';
    if (isUpdate) {
      body = `<p style="font-size:12px;color:var(--muted);line-height:1.55">${md(item.body)}</p>
        <div class="m-facts">${item.facts.map((x) => `<div class="m-fact"><span class="k">${esc(x.k)}</span><span class="v">${esc(x.v)}</span></div>`).join('')}</div>
        <div class="m-note">${md(item.note)}</div>`;
    } else {
      body = `
        <div class="m-field"><div class="k">Mistake</div><div class="v">${md(item.mistake)}</div></div>
        <div class="m-field"><div class="k">How detected</div><div class="v">${md(item.detection)}</div></div>
        <div class="m-field"><div class="k">Correction</div><div class="v">${md(item.correction)}</div></div>
        <div class="m-field"><div class="k">Lesson learned</div><div class="v">${md(item.lesson)}</div></div>`;
    }
    return `<details class="m-card" data-cat="${item.group}">
      <summary>
        <div style="flex:1;min-width:0">
          <div class="m-head">
            <span class="m-id ${cls}">${esc(item.id)}</span>
            <span class="badge ${badge}">${esc(item.category)}</span>
            <span class="m-date">${esc(item.date)}</span>
            ${isUpdate ? '<span class="badge">later record</span>' : ''}
          </div>
          <div class="m-title">${md(item.title)}</div>
        </div>
        <span class="m-chev">›</span>
      </summary>
      <div class="m-body">${body}</div>
    </details>`;
  }).join('');
}

/* ------------------------------------------------------------------ */
/* Verification check explorer                                         */
/* ------------------------------------------------------------------ */
function renderCheckFilters() {
  const host = $('#checkFilters');
  const order = ['Integrity', 'Alignment', 'Metrics', 'Bootstrap & hypotheses', 'Model role'];
  const counts = DATA.verification.groups;
  let html = `<button type="button" data-check="all" aria-pressed="${state.checkGroup === 'all'}">All · ${DATA.verification.n_checks}</button>`;
  order.forEach((group) => {
    html += `<button type="button" data-check="${esc(group)}" aria-pressed="${state.checkGroup === group}">${esc(group)} · ${counts[group]}</button>`;
  });
  host.innerHTML = html;
}
function renderChecks() {
  const list = DATA.verification.checks.filter((c) => state.checkGroup === 'all' || c.group === state.checkGroup);
  $('#checkList').innerHTML = list.map((c) => `
    <div class="check-row">
      <span class="check-state">PASS</span>
      <div><code>${esc(c.id)}</code><p>${esc(c.description)}</p></div>
    </div>`).join('');
  renderCheckFilters();
}

/* ------------------------------------------------------------------ */
/* Human-in-the-loop verification                                      */
/* ------------------------------------------------------------------ */
const HITL_BADGE = { Agent: 'amber', Human: 'teal', Finding: 'blue', Evidence: 'violet' };

function selectA5(index) {
  const steps = DATA.hitl.a5.steps;
  const i = Math.max(0, Math.min(steps.length - 1, index));
  state.a5Step = i;
  const step = steps[i];
  $$('#a5Steps .step').forEach((btn, n) => {
    btn.setAttribute('aria-selected', String(n === i));
    btn.setAttribute('tabindex', String(n === i ? 0 : -1));
  });
  const badge = HITL_BADGE[step.actor] || 'blue';
  setHTML('a5Panel', `
    <div class="sp-head">
      <span class="badge ${badge}">${esc(step.actor)}</span>
      <span class="card-kicker" style="margin:0">${esc(step.lead)} · step ${i + 1} of ${steps.length}</span>
    </div>
    <h4 class="sp-title">${esc(step.title)}</h4>
    <p class="sp-body">${esc(step.body)}</p>
    <div class="sp-facts">${step.facts.map((fact) => `
      <div class="cal-stat"><div class="l">${esc(fact.k)}</div><div class="v">${esc(fact.v)}</div></div>`).join('')}
    </div>
    <div class="sp-src">Source: ${esc(step.source)}</div>`);
}

function renderHitl() {
  const hitl = DATA.hitl;
  $('#a5Steps').innerHTML = hitl.a5.steps.map((s, i) => `
    <button type="button" class="step" role="tab" data-step="${i}" aria-controls="a5Panel"
      aria-selected="${i === state.a5Step}" tabindex="${i === state.a5Step ? 0 : -1}">
      <span class="node">${i + 1}</span>
      <span class="step-label">${esc(s.title)}</span>
      <span class="step-actor">${esc(s.actor)}</span>
    </button>`).join('');
  setText('humanExecutor', hitl.human.executor);
  setText('humanEnv', hitl.human.environment);
  setText('humanCI', `[${hitl.human.ci.lo.toFixed(6)}, ${hitl.human.ci.hi.toFixed(6)}]`);
  setText('autoChecks', String(hitl.automated.n_checks));
  setHTML('hitlLimits', `<ul class="clean warn">${hitl.limits.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>`);
  setHTML('hitlNotClaimed', hitl.not_claimed.map((t) => `<span class="chip">${esc(t)}</span>`).join(''));
  selectA5(state.a5Step);
}

function wireA5() {
  $('#a5Steps').addEventListener('click', (event) => {
    const btn = event.target.closest('[data-step]');
    if (btn) selectA5(Number(btn.dataset.step));
  });
  $('#a5Steps').addEventListener('keydown', (event) => {
    const keys = ['ArrowRight', 'ArrowLeft', 'Home', 'End'];
    if (!keys.includes(event.key)) return;
    const total = DATA.hitl.a5.steps.length;
    let next = state.a5Step;
    if (event.key === 'ArrowRight') next = (state.a5Step + 1) % total;
    if (event.key === 'ArrowLeft') next = (state.a5Step - 1 + total) % total;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = total - 1;
    event.preventDefault();
    selectA5(next);
    const btn = document.querySelector(`#a5Steps .step[data-step="${next}"]`);
    if (btn) btn.focus();
  });
}

/* ------------------------------------------------------------------ */
/* Inference demonstration                                             */
/* ------------------------------------------------------------------ */
function renderCustomers() {
  const prevalence = DATA.test.prevalence;
  $('#customerGrid').innerHTML = DATA.inference.customers.map((c) => {
    const features = Object.keys(c.features).map((k) => `<span>${esc(k)}=${esc(c.features[k])}</span>`).join('');
    return `<div class="customer">
      <h4><code>${esc(c.id)}</code><span class="badge">synthetic row</span></h4>
      <div class="score-row"><span class="lab">A</span><span class="score-track"><i class="a" style="width:${Math.max(1.5, c.p_a * 100)}%"></i></span><span class="val">${c.pct_a.toFixed(2)}%</span></div>
      <div class="score-row"><span class="lab">B</span><span class="score-track"><i class="b" style="width:${Math.max(1.5, c.p_b * 100)}%"></i></span><span class="val">${c.pct_b.toFixed(2)}%</span></div>
      <div class="score-note">Saved scores p<sub>A</sub> = ${c.p_a}, p<sub>B</sub> = ${c.p_b}. For contrast, the test prevalence was ${pct(prevalence, 2)} — that is an observed outcome rate, not a prediction for this customer.</div>
      <div class="feature-mini">${features}</div>
    </div>`;
  }).join('');
}

/* ------------------------------------------------------------------ */
/* Navigation and interaction wiring                                   */
/* ------------------------------------------------------------------ */
function setSection(id) {
  $$('#nav button').forEach((b) => b.classList.toggle('active', b.dataset.target === id));
  const active = $('#nav button.active');
  if (active) setText('crumbSection', active.textContent.replace(/^\d+\s*/, '').trim());
}
function wireNav() {
  $$('#nav button').forEach((button) => {
    button.addEventListener('click', () => {
      setSection(button.dataset.target);
      const target = document.getElementById(button.dataset.target);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      document.body.classList.remove('nav-open');
      $('#menuBtn').setAttribute('aria-expanded', 'false');
    });
  });
  $$('[data-scroll]').forEach((button) => {
    button.addEventListener('click', () => {
      setSection(button.dataset.scroll);
      const target = document.getElementById(button.dataset.scroll);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  });
  const sections = $$('.section');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => { if (entry.isIntersecting) setSection(entry.target.id); });
    }, { rootMargin: '-22% 0px -68% 0px', threshold: 0 });
    sections.forEach((section) => observer.observe(section));
  }
  let activeId = 'overview';
  const onScroll = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    $('#progress').style.width = (max > 0 ? Math.min(100, (window.scrollY / max) * 100) : 0) + '%';
    const marker = window.scrollY + 150;
    let current = sections.length ? sections[0].id : 'overview';
    sections.forEach((section) => { if (section.offsetTop <= marker) current = section.id; });
    if (current !== activeId) { activeId = current; setSection(current); }
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}
function wireControls() {
  $('#menuBtn').addEventListener('click', () => {
    const open = document.body.classList.toggle('nav-open');
    $('#menuBtn').setAttribute('aria-expanded', String(open));
  });
  $('#scrim').addEventListener('click', () => {
    document.body.classList.remove('nav-open');
    $('#menuBtn').setAttribute('aria-expanded', 'false');
  });

  $('#dsPrimary').addEventListener('click', () => {
    state.dataset = 'primary';
    $('#dsPrimary').setAttribute('aria-pressed', 'true');
    $('#dsSensitivity').setAttribute('aria-pressed', 'false');
    setHTML('datasetNote', `<span class="ico">✓</span><div><strong>Primary chronological test.</strong> Frozen rows, one evaluation, ${fmtInt(DATA.test.n_test)} customers. This is the evidence for the headline conclusion. <span style="color:var(--faint)">Source: <code>results/guided/test/test_metrics.csv</code></span></div>`);
    $('#datasetNote').className = 'notice teal';
    renderMetricChart(); renderDeltaChart(); renderPerfTable();
  });
  $('#dsSensitivity').addEventListener('click', () => {
    state.dataset = 'sensitivity';
    $('#dsSensitivity').setAttribute('aria-pressed', 'true');
    $('#dsPrimary').setAttribute('aria-pressed', 'false');
    setHTML('datasetNote', `<span class="ico">!</span><div><strong>Secondary random-split sensitivity.</strong> Optimistic by construction; the split mixes time periods. It does not replace the chronological test and is not an independent confirmation. <span style="color:var(--faint)">Source: <code>results/sensitivity_corrected_v1/metrics.csv</code></span></div>`);
    $('#datasetNote').className = 'notice amber';
    renderMetricChart(); renderDeltaChart(); renderPerfTable();
  });
  $$('[data-metric]').forEach((button) => {
    button.addEventListener('click', () => {
      state.metric = button.dataset.metric;
      $$('[data-metric]').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
      renderMetricChart();
    });
  });
  $$('[data-cal]').forEach((button) => {
    button.addEventListener('click', () => {
      state.calModel = button.dataset.cal;
      $$('[data-cal]').forEach((b) => b.setAttribute('aria-selected', String(b === button)));
      renderCalibration();
    });
  });
  $$('[data-filter]').forEach((button) => {
    button.addEventListener('click', () => {
      state.mistakeFilter = button.dataset.filter;
      $$('[data-filter]').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
      renderMistakes();
    });
  });
  $('#expandMistakes').addEventListener('click', () => {
    const cards = $$('#mistakesList details');
    const anyClosed = cards.some((card) => !card.open);
    cards.forEach((card) => { card.open = anyClosed; });
    $('#expandMistakes').textContent = anyClosed ? 'Collapse visible' : 'Expand visible';
  });
  $('#checkFilters').addEventListener('click', (event) => {
    const button = event.target.closest('[data-check]');
    if (!button) return;
    state.checkGroup = button.dataset.check;
    renderChecks();
  });
  $('#copyCmd').addEventListener('click', async () => {
    const text = DATA.inference.command;
    let ok = false;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(text);
        ok = true;
      }
    } catch (err) { ok = false; }
    if (!ok) {
      try {
        const area = document.createElement('textarea');
        area.value = text;
        area.style.position = 'fixed';
        area.style.opacity = '0';
        document.body.appendChild(area);
        area.select();
        ok = document.execCommand('copy');
        document.body.removeChild(area);
      } catch (err) { ok = false; }
    }
    setText('copyState', ok ? 'Copied to clipboard.' : 'Copy unavailable in this view — select the command manually.');
  });
}

function init() {
  fillStatic();
  wireNav();
  wireControls();
  wireA5();
  renderMetricChart();
  renderDeltaChart();
  renderPerfTable();
  renderExact();
  renderCVChart();
  renderCalibration();
  renderMistakes();
  renderChecks();
  renderHitl();
  renderCustomers();
  setSection('overview');
}
init();

</script>
</body>
</html>
"""

def main() -> None:
    payload = json.dumps(DATA, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DASHBOARD_DATA__", payload)

    leftovers = re.findall(r"__[A-Z0-9_]+_MARKER__", html)
    require(not leftovers, f"unreplaced template markers: {sorted(set(leftovers))}")
    require("__DASHBOARD_DATA__" not in html, "dashboard data placeholder remains")
    require("<script src=" not in html.lower(), "external script reference found")
    require("fetch(" not in html, "runtime fetch() found")
    require("NaN" not in html, "literal NaN found in output")
    require("undefined" not in html, "literal undefined found in output")
    # The only permitted URL is the SVG namespace identifier (not a network request).
    network_scan = html.replace("http://www.w3.org/2000/svg", "")
    require("http://" not in network_scan and "https://" not in network_scan, "network URL found in output")
    require(html.count("<section") == 8, f"expected 8 sections, found {html.count('<section')}")
    require(html.count("</section>") == 8, "unbalanced section tags")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(html, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    print(f"wrote {OUT_FILE.relative_to(ROOT)} ({len(html.encode('utf-8')):,} bytes)")
    print(f"sha256 {digest}")
    print(f"sections {html.count('<section')} · checks displayed {DATA['verification']['n_checks']} · models A/B/C")
    print("source files were read only; no source artifact was modified")


if __name__ == "__main__":
    main()
