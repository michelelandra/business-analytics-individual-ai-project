"""Autonomous random-split sensitivity analysis (POST-TEST, authorized).

Plan: ``plan.md`` v1.2 (frozen), §5.5 secondary sensitivity analysis. This is a
NEW script written AFTER the primary chronological analysis and the authorized
one-shot test evaluation were complete. It is a secondary robustness analysis
and does NOT replace or alter the frozen primary chronological results.

What it does (exactly, nothing more):
1. Generates the pre-specified stratified random split of all 45,211 rows with
   the two-step procedure:
       step 1: train_test_split(all_idx, test_size=6783, stratify=y,
                               random_state=42)          -> rest + test
       step 2: train_test_split(rest_idx, test_size=6781,
                                stratify=y_rest,
                                random_state=42)           -> train + validation
   Original row indices are preserved and the three blocks are verified to be
   mutually disjoint and to cover all 45,211 rows.
2. Fits Models A, B, C with the FROZEN settings of the original autonomous run
   (C = 0.01 for all three; L2 logistic regression, lbfgs, max_iter=5000,
   random_state=42; original feature engineering, categorical representation
   and preprocessing) on the RANDOM TRAINING BLOCK ONLY. No search, no
   selection, no recalibration.
3. Evaluates ONCE on the RANDOM TEST BLOCK: ROC-AUC, average precision
   (PR-AUC) with its prevalence-based no-skill baseline, Brier score,
   lift@10%, and calibration summaries.
4. Computes the 2,000-resample paired bootstrap (seed 42) for B-A and C-B
   ROC-AUC and average-precision differences, with the same definitions as the
   original autonomous analysis.

Model C is a NON-DEPLOYABLE DURATION-LEAKAGE DIAGNOSTIC in every table.

All implementation choices below are POST-TEST decisions (disclosed in the
generated report). The test mode of ``autonomous_test_evaluate.py`` and every
primary artifact remain untouched.

Run with (from repo root):
    $env:PYTHONPATH = "src"
    .venv\\Scripts\\python.exe -m my_analysis.autonomous_random_split_sensitivity
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
TEMPORAL_SPLIT = ROOT / "results" / "splits" / "temporal_split_v1.json"
OUT = ROOT / "results" / "sensitivity"
SCRIPT_REL = "src/my_analysis/autonomous_random_split_sensitivity.py"

SEED = 42
N_ROWS = 45_211
TEST_SIZE = 6_783
VAL_SIZE = 6_781
TRAIN_SIZE = N_ROWS - TEST_SIZE - VAL_SIZE  # 31,647
MAX_ITER = 5_000
N_BOOT = 2_000
LIFT_FRACTION = 0.10

EXPECTED_DATASET_SHA256 = (
    "0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f"
)
EXPECTED_TEMPORAL_SPLIT_SHA256 = (
    "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c"
)

# Frozen hyperparameters of the original AUTONOMOUS run (constants; no search).
SELECTED_C: dict[str, float] = {"A": 0.01, "B": 0.01, "C": 0.01}

# Original AUTONOMOUS codebook and feature sets, verbatim from
# src/my_analysis/autonomous_train_val.py (the six base categoricals are the
# integer code columns, exactly as the original pipeline used them).
CODEBOOK: dict[str, list[str]] = {
    "job": [
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ],
    "marital": ["divorced", "married", "single", "unknown"],
    "education": [
        "primary", "secondary", "tertiary", "unknown", "basic.4y", "basic.6y",
        "basic.9y", "high.school", "illiterate", "professional.course",
        "university.degree",
    ],
    "default": ["no", "yes", "unknown"],
    "housing": ["no", "yes", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "poutcome": ["failure", "other", "success", "unknown", "nonexistent"],
}

MODELS: dict[str, dict] = {
    "A": {
        "role": "A: customer characteristics only (random-split sensitivity)",
        "numeric": ["age", "balance"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan"],
    },
    "B": {
        "role": ("B: A + previous-campaign information "
                 "(random-split sensitivity)"),
        "numeric": ["age", "balance", "previous"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan", "poutcome_level"],
    },
    "C": {
        "role": ("C: B + duration (DIAGNOSTIC - LEAKAGE, NEVER DEPLOYABLE; "
                 "random-split sensitivity)"),
        "numeric": ["age", "balance", "previous", "duration"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan", "poutcome_level"],
    },
}


# ---------------------------------------------------------------------------
# Helpers (same conventions as the original autonomous scripts)
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj: object) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rows_report(label: str, n: int, reason: str) -> None:
    """AGENTS.md rule 2: report row counts after every step."""
    print(f"[rows] {label}: n={n:,} ({reason})")


def hash_tree(base: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(base.rglob("*")):
        if path.is_file():
            rel = str(path.relative_to(base)).replace("\\", "/")
            out[rel] = sha256_file(path)
    return out


# ---------------------------------------------------------------------------
# Random split (pre-specified two-step procedure)
# ---------------------------------------------------------------------------
def build_random_split(y_all: np.ndarray) -> dict:
    all_idx = np.arange(N_ROWS)
    rows_report("dataset loaded", N_ROWS, "no filter applied; y read for "
                "stratification (legitimate for the random split)")

    # Step 1: rest (38,428) + test (6,783)
    rest_idx, test_idx = train_test_split(
        all_idx, test_size=TEST_SIZE, stratify=y_all, random_state=SEED
    )
    # Step 2: train (31,647) + validation (6,781) from the remaining rows
    y_rest = y_all[rest_idx]
    train_idx, val_idx = train_test_split(
        rest_idx, test_size=VAL_SIZE, stratify=y_rest, random_state=SEED
    )

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    rows_report("random training block", len(train_idx),
                "step 2 remainder; fitted on only")
    rows_report("random validation block", len(val_idx),
                "step 2 test part; NOT used for fitting or selection")
    rows_report("random test block", len(test_idx),
                "step 1 test part; evaluated once")

    # Verification: disjointness and full coverage
    all_concat = np.concatenate([train_idx, val_idx, test_idx])
    checks = {
        "sizes_match_protocol": (
            len(train_idx) == TRAIN_SIZE
            and len(val_idx) == VAL_SIZE
            and len(test_idx) == TEST_SIZE
        ),
        "no_duplicate_indices": len(np.unique(all_concat)) == N_ROWS,
        "full_coverage_exactly_once": bool(
            np.array_equal(np.sort(all_concat), all_idx)
        ),
        "mutually_disjoint": bool(
            len(np.intersect1d(train_idx, val_idx)) == 0
            and len(np.intersect1d(train_idx, test_idx)) == 0
            and len(np.intersect1d(val_idx, test_idx)) == 0
        ),
        "all_indices_within_range": bool(
            all_concat.min() == 0 and all_concat.max() == N_ROWS - 1
        ),
    }
    if not all(checks.values()):
        raise SystemExit(f"FATAL: random split verification failed: {checks}")

    positives = {
        "train": int(y_all[train_idx].sum()),
        "validation": int(y_all[val_idx].sum()),
        "test": int(y_all[test_idx].sum()),
    }

    # File hash (depends on serialization) + canonical fingerprint of the
    # three index arrays, defined order-insensitively for cross-run
    # comparison of the actual index sets.
    indices = {
        "train": [int(i) for i in train_idx],
        "validation": [int(i) for i in val_idx],
        "test": [int(i) for i in test_idx],
    }
    split_hash = None  # file hash recorded after writing (serialization-dependent)
    canonical_fingerprint = canonical_hash(
        {k: sorted(v) for k, v in indices.items()}
    )
    return {
        "split_name": "random_split_v1",
        "analysis_role": (
            "secondary sensitivity analysis (plan.md 5.5); clearly labelled "
            "optimistic; does NOT replace or alter the primary chronological "
            "results"
        ),
        "generated_utc": utc_now(),
        "script": SCRIPT_REL,
        "input_file": "data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet",
        "input_sha256": sha256_file(RAW),
        "n_rows": N_ROWS,
        "procedure": {
            "step_1": (
                "sklearn train_test_split(arange(45211), test_size=6783, "
                "stratify=y, random_state=42) -> rest (38,428) + test"
            ),
            "step_2": (
                "sklearn train_test_split(rest_idx, test_size=6781, "
                "stratify=y_rest, random_state=42) -> train (31,647) + "
                "validation"
            ),
            "shuffle": True,
            "random_state": SEED,
            "stratify": "y (whole-file target)",
        },
        "sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "positives": positives,
        "prevalence": {
            k: positives[k] / size
            for k, size in (
                ("train", TRAIN_SIZE), ("validation", VAL_SIZE),
                ("test", TEST_SIZE),
            )
        },
        "verification": checks,
        "indices": indices,
        "canonical_fingerprint": {
            "value": canonical_fingerprint,
            "definition": (
                "sha256 of json.dumps({'train': sorted(train), "
                "'validation': sorted(validation), 'test': sorted(test)}, "
                "sort_keys=True, separators=(',',':')) encoded as UTF-8; "
                "order-insensitive so it depends only on the index SETS"
            ),
        },
        "guided_reference_note": (
            "The independently completed Guided run reported 3,702 / 793 / "
            "794 positives and JSON hash e43308ca7a0485fdf114be43372be3c0cc0"
            "7017897ed766895de47e02f6eca3b. Treated as reference only; no "
            "Guided file was read and no procedure was changed to force a "
            "hash match."
        ),
    }


# ---------------------------------------------------------------------------
# Frozen model construction (verbatim from the original autonomous run)
# ---------------------------------------------------------------------------
def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim reproduction of autonomous_train_val.decode_categoricals."""
    data = df.copy()
    for col, names in CODEBOOK.items():
        if col == "poutcome":
            continue
        codes = data[col].to_numpy()
        if codes.min() < 0 or codes.max() >= len(names):
            raise SystemExit(f"FATAL: {col} code outside documented name list")
        data[col + "_label"] = [names[c] for c in codes]

    poc_names = CODEBOOK["poutcome"]
    poc = data["poutcome"].to_numpy()
    if poc.min() < 0 or poc.max() >= len(poc_names):
        raise SystemExit("FATAL: poutcome code outside documented name list")
    poc_label = np.array([poc_names[c] for c in poc], dtype=object)
    data["poutcome_level"] = np.where(
        data["pdays"].to_numpy() == 999, "never_contacted", poc_label
    )
    return data


def build_pipeline(model_key: str) -> Pipeline:
    spec = MODELS[model_key]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), spec["numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),
             spec["categorical"]),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        penalty="l2",
        C=SELECTED_C[model_key],
        solver="lbfgs",
        max_iter=MAX_ITER,
        random_state=SEED,
    )
    return Pipeline([("prep", preprocess), ("clf", clf)])


# ---------------------------------------------------------------------------
# Metrics (verbatim definitions from the original autonomous run)
# ---------------------------------------------------------------------------
def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    n = len(y)
    n_top = int(round(LIFT_FRACTION * n))
    order = np.lexsort((np.arange(n), -p))
    top_prec = float(y[order[:n_top]].mean())
    prevalence = float(y.mean())
    return {
        "n": int(n),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "pr_auc_no_skill_baseline": prevalence,
        "brier": float(brier_score_loss(y, p)),
        "lift_top10pct": top_prec / prevalence,
        "top_decile_precision": top_prec,
        "n_top_decile": n_top,
    }


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10
                      ) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(
        y, p, n_bins=n_bins, strategy="uniform"
    )
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=True), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    pos = np.array([y[bins == b].sum() for b in range(n_bins)])
    keep = counts > 0
    return pd.DataFrame({
        "bin": np.arange(n_bins)[keep],
        "n": counts[keep],
        "mean_predicted": prob_pred,
        "fraction_positive": prob_true,
        "positives_in_bin": pos[keep],
    })


def paired_bootstrap(y: np.ndarray, preds: dict[str, np.ndarray],
                     comparisons: list[tuple[str, str]]) -> dict:
    """Identical definition to autonomous_train_val.paired_bootstrap."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    res = {f"{hi}-{lo}": {"roc": [], "pr": []} for hi, lo in comparisons}
    skipped = 0
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        aucs = {m: roc_auc_score(yb, preds[m][idx]) for m in preds}
        aps = {m: average_precision_score(yb, preds[m][idx]) for m in preds}
        for hi, lo in comparisons:
            key = f"{hi}-{lo}"
            res[key]["roc"].append(aucs[hi] - aucs[lo])
            res[key]["pr"].append(aps[hi] - aps[lo])
    out: dict[str, dict] = {}
    for key, d in res.items():
        roc = np.asarray(d["roc"])
        pr = np.asarray(d["pr"])
        out[key] = {
            "n_valid_resamples": int(len(roc)),
            "resamples_skipped_single_class": skipped,
            "delta_roc_auc_point": float(
                roc_auc_score(y, preds[key.split("-")[0]])
                - roc_auc_score(y, preds[key.split("-")[1]])
            ),
            "delta_roc_auc_ci95": [float(np.percentile(roc, 2.5)),
                                   float(np.percentile(roc, 97.5))],
            "delta_pr_auc_point": float(
                average_precision_score(y, preds[key.split("-")[0]])
                - average_precision_score(y, preds[key.split("-")[1]])
            ),
            "delta_pr_auc_ci95": [float(np.percentile(pr, 2.5)),
                                  float(np.percentile(pr, 97.5))],
        }
    return out


def classify_h1(delta: float, ci: list[float]) -> str:
    """Same decision rule as plan.md 3.2, applied as POST-TEST reference
    context only (the pre-registered decision was made on the chronological
    test block)."""
    lo, hi = ci
    if delta >= 0.01 and lo > 0:
        return "H0 rejected, materially relevant"
    if 0 < delta < 0.01 and lo > 0:
        return "H0 rejected, below relevance threshold"
    if lo <= 0 <= hi:
        return "Insufficient evidence"
    if delta < 0 and hi < 0:
        return "H0 not rejected / direction contradicted"
    return "Unclassified - human review required"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- integrity of inputs and preservation baseline ------------------
    if sha256_file(RAW) != EXPECTED_DATASET_SHA256:
        raise SystemExit("FATAL: raw dataset hash mismatch")
    split_meta = json.loads(TEMPORAL_SPLIT.read_text(encoding="utf-8"))
    if canonical_hash(split_meta["indices"]) != split_meta["split_hash_sha256"]:
        raise SystemExit("FATAL: temporal split file integrity failure")
    if split_meta["split_hash_sha256"] != EXPECTED_TEMPORAL_SPLIT_SHA256:
        raise SystemExit("FATAL: unexpected temporal split hash")
    print("[check] raw dataset and frozen temporal split verified")

    pre_hashes = {
        "governing": {
            f: sha256_file(ROOT / f)
            for f in (
                "plan.md", "AGENTS.md", "data/README.md", "mistakes.md",
                "data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet",
                "results/splits/temporal_split_v1.json",
                "pyproject.toml", "uv.lock",
            )
        },
        "results_autonomous": hash_tree(ROOT / "results" / "autonomous"),
        "src_my_analysis": hash_tree(ROOT / "src" / "my_analysis"),
    }
    print(f"[preserve] baseline captured: {len(pre_hashes['governing'])} "
          f"governing files, {len(pre_hashes['results_autonomous'])} "
          f"autonomous result files, {len(pre_hashes['src_my_analysis'])} "
          f"source files")

    # ---- data + split ----------------------------------------------------
    df = pd.read_parquet(RAW)
    y_all = df["y"].to_numpy(dtype=np.int64)
    split_info = build_random_split(y_all)

    split_json = OUT / "random_split_v1.json"
    split_json.write_text(json.dumps(split_info, indent=2) + "\n",
                          encoding="utf-8")
    # file hash recorded after writing (depends on serialization)
    split_info["split_file_sha256"] = sha256_file(split_json)
    split_json.write_text(json.dumps(split_info, indent=2) + "\n",
                          encoding="utf-8")
    print(f"[split] random_split_v1 written; file sha256 "
          f"{split_info['split_file_sha256'][:12]}...; canonical fingerprint "
          f"{split_info['canonical_fingerprint']['value'][:12]}...")

    idx = split_info["indices"]
    train_idx = np.asarray(idx["train"], dtype=np.int64)
    val_idx = np.asarray(idx["validation"], dtype=np.int64)
    test_idx = np.asarray(idx["test"], dtype=np.int64)

    # ---- overlap with the frozen chronological blocks --------------------
    chron = {
        "train": np.asarray(split_meta["indices"]["train"], dtype=np.int64),
        "validation": np.asarray(split_meta["indices"]["validation"],
                                 dtype=np.int64),
        "test": np.asarray(split_meta["indices"]["test"], dtype=np.int64),
    }
    overlap = {}
    for rname, ridx in (("train", train_idx), ("validation", val_idx),
                        ("test", test_idx)):
        for cname, cidx in chron.items():
            overlap[f"random_{rname}_x_chronological_{cname}"] = int(
                len(np.intersect1d(ridx, cidx))
            )
    print("[split] overlap with frozen chronological blocks:", overlap)

    # ---- frozen models on the random training block ----------------------
    data = prepare_frame(df)
    rows_report("after categorical decoding + poutcome_level recode",
                len(data), "row-wise relabeling; 0 rows added or dropped")
    print("[recode] poutcome_level counts:",
          {k: int(v) for k, v in
           data["poutcome_level"].value_counts().sort_index().items()})

    X_train = data.iloc[train_idx].reset_index(drop=True)
    X_val = data.iloc[val_idx].reset_index(drop=True)
    X_test = data.iloc[test_idx].reset_index(drop=True)
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]
    rows_report("random train used for fitting", len(X_train),
                "frozen settings; no search, no selection")
    rows_report("random validation (held out, unused)", len(X_val),
                "no fitting, no selection, no recalibration")
    rows_report("random test evaluated once", len(X_test),
                "pre-registered metrics only")

    # static audit: frozen settings, no search
    audit = {
        "selected_C": dict(SELECTED_C),
        "selected_C_match_frozen": SELECTED_C == {"A": 0.01, "B": 0.01,
                                                  "C": 0.01},
        "solver": "lbfgs",
        "penalty": "l2",
        "max_iter": MAX_ITER,
        "random_state": SEED,
        "feature_sets_source": "verbatim from autonomous_train_val.py",
        "hyperparameter_origin": (
            "frozen from the original autonomous chronological analysis "
            "(results/autonomous/models/model_settings.json, C=0.01 for all "
            "models, selected there by TimeSeriesSplit CV on the "
            "chronological training block); re-used here without any new "
            "search"
        ),
    }
    print("[audit] frozen settings verified:", audit["selected_C_match_frozen"])

    preds: dict[str, np.ndarray] = {}
    metric_rows: list[dict] = []
    cal_frames: list[pd.DataFrame] = []
    warn_log: list[str] = []
    for name in ("A", "B", "C"):
        pipe = build_pipeline(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipe.fit(X_train, y_train)
        for w in caught:
            warn_log.append(f"fit {name}: {w.category.__name__}: {w.message}")
        p = pipe.predict_proba(X_test)[:, 1]
        preds[name] = p
        met = evaluate(y_test, p)
        met["model"] = name
        met["label"] = MODELS[name]["role"]
        met["selected_C"] = SELECTED_C[name]
        metric_rows.append(met)
        cal = calibration_table(y_test, p)
        cal.insert(0, "model", name)
        cal_frames.append(cal)
        n_coef = len(pipe.named_steps["clf"].coef_[0])
        print(f"[fit] model {name} fitted on random train (n={len(X_train):,}),"
              f" C={SELECTED_C[name]}, {n_coef} coefficients")
        print(f"[test-sens] model {name}: ROC-AUC={met['roc_auc']:.6f} "
              f"PR-AUC={met['pr_auc']:.6f} (baseline "
              f"{met['pr_auc_no_skill_baseline']:.6f}) Brier={met['brier']:.6f}"
              f" lift@10%={met['lift_top10pct']:.4f}")

    metrics_df = pd.DataFrame(metric_rows)
    cols = ["model", "label", "n", "positives", "prevalence", "roc_auc",
            "pr_auc", "pr_auc_no_skill_baseline", "brier", "lift_top10pct",
            "top_decile_precision", "n_top_decile", "selected_C"]
    metrics_df[cols].to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame({
        "row_index": test_idx,
        "y": y_test,
        "p_model_a": preds["A"],
        "p_model_b": preds["B"],
        "p_model_c": preds["C"],
    }).to_csv(OUT / "predictions.csv", index=False)
    pd.concat(cal_frames, ignore_index=True).to_csv(
        OUT / "calibration.csv", index=False)

    # ---- paired bootstrap (same definitions as the original analysis) ----
    boot = paired_bootstrap(y_test, preds, comparisons=[("B", "A"), ("C", "B")])
    boot_payload = {
        "evaluation_set": "random test block (random_split_v1)",
        "n": int(len(y_test)),
        "n_boot": N_BOOT,
        "seed": SEED,
        "method": (
            "paired bootstrap over random-test rows; percentile 95% CI; "
            "identical resample indices reused across models and metrics; "
            "same definitions as the original autonomous analysis"
        ),
        "comparisons": boot,
        "materiality_threshold_delta_auc": 0.01,
        "leakage_diagnostic_threshold_delta_auc_C_minus_B": 0.05,
        "decision_rule_applied_as_reference": {
            "B-A": classify_h1(boot["B-A"]["delta_roc_auc_point"],
                               boot["B-A"]["delta_roc_auc_ci95"]),
            "note": (
                "POST-TEST reference application of the plan.md 3.2 decision "
                "rule to the random split. The pre-registered decision was "
                "made on the chronological test block; this does not replace "
                "or alter it."
            ),
        },
        "model_c_note": (
            "Model C is a NON-DEPLOYABLE DURATION-LEAKAGE DIAGNOSTIC in every "
            "table."
        ),
    }
    (OUT / "bootstrap.json").write_text(
        json.dumps(boot_payload, indent=2) + "\n", encoding="utf-8")
    for key, d in boot.items():
        print(f"[boot] {key}: dROC-AUC={d['delta_roc_auc_point']:.6f} "
              f"95% CI {d['delta_roc_auc_ci95']}; "
              f"dPR-AUC={d['delta_pr_auc_point']:.6f} "
              f"95% CI {d['delta_pr_auc_ci95']}")

    # ---- split summary markdown ------------------------------------------
    _write_split_md(split_info, overlap)

    # ---- report -----------------------------------------------------------
    _write_report(split_info, metrics_df, boot, audit, overlap, warn_log)

    # ---- preservation check (post-execution) ------------------------------
    post_hashes = {
        "governing": {
            f: sha256_file(ROOT / f)
            for f in pre_hashes["governing"]
        },
        "results_autonomous": hash_tree(ROOT / "results" / "autonomous"),
        "src_my_analysis": hash_tree(ROOT / "src" / "my_analysis"),
    }
    changed = {
        section: sorted(
            k for k in pre_hashes[section]
            if pre_hashes[section][k] != post_hashes[section].get(k)
        )
        for section in pre_hashes
    }
    print("[preserve] changed existing files:",
          {k: v for k, v in changed.items() if v} or "NONE")
    if any(changed.values()):
        raise SystemExit(
            f"FATAL: existing files changed during the run: {changed}"
        )

    # ---- manifest -----------------------------------------------------------
    manifest = {
        "run": "autonomous",
        "stage": "post-test random-split sensitivity analysis (authorized)",
        "script": SCRIPT_REL,
        "script_sha256": sha256_file(Path(__file__)),
        "generated_utc": utc_now(),
        "plan_version": "1.2 (frozen)",
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "temporal_split_sha256": EXPECTED_TEMPORAL_SPLIT_SHA256,
        "seed": SEED,
        "selected_C": dict(SELECTED_C),
        "post_test_decision_disclosure": (
            "This script, its split procedure and all implementation choices "
            "were created AFTER the primary chronological analysis and the "
            "one-shot test evaluation. It is a secondary sensitivity "
            "analysis and does not replace or alter the primary results."
        ),
        "preservation": {
            "pre_execution_hashes": pre_hashes,
            "post_execution_hashes": post_hashes,
            "changed_existing_files": changed,
        },
        "artifacts": {
            p.name: sha256_file(p)
            for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name != "run_manifest_sensitivity.json"
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    (OUT / "run_manifest_sensitivity.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] artifacts written under results/sensitivity/ "
          f"({len(manifest['artifacts'])} files hashed)")
    return 0


def _write_split_md(info: dict, overlap: dict) -> None:
    pos = info["positives"]
    prev = info["prevalence"]
    lines = [
        "# random_split_v1 — autonomous random-split sensitivity analysis",
        "",
        f"Generated (UTC): {info['generated_utc']}",
        f"Script: `{info['script']}`",
        f"Dataset: `{info['input_file']}` (SHA-256 `{info['input_sha256']}`)",
        "",
        "## Role",
        "",
        "Secondary sensitivity analysis (plan.md §5.5), created AFTER the",
        "primary chronological analysis and the authorized one-shot test",
        "evaluation. It does **not** replace or alter the frozen primary",
        "chronological results.",
        "",
        "## Procedure (pre-specified, exactly as authorized)",
        "",
        "1. `train_test_split(arange(45211), test_size=6783, stratify=y,",
        "   random_state=42)` → rest (38,428) + test (6,783)",
        "2. `train_test_split(rest_idx, test_size=6781, stratify=y_rest,",
        "   random_state=42)` → train (31,647) + validation (6,781)",
        "",
        "Original row indices are preserved (indices into the raw Parquet,",
        "0-based).",
        "",
        "## Sizes and class counts",
        "",
        "| Block | n | positives | prevalence |",
        "|---|---|---|---|",
    ]
    for b in ("train", "validation", "test"):
        lines.append(
            f"| {b} | {info['sizes'][b]:,} | {pos[b]:,} | {prev[b]:.6f} |")
    lines += [
        "",
        "## Verification",
        "",
    ]
    for k, v in info["verification"].items():
        lines.append(f"- {k}: **{v}**")
    lines += [
        "",
        "## Overlap with the frozen chronological blocks",
        "",
        "| Random block \\ Chronological block | train | validation | test |",
        "|---|---|---|---|",
    ]
    for rname in ("train", "validation", "test"):
        lines.append(
            f"| {rname} "
            f"| {overlap[f'random_{rname}_x_chronological_train']:,} "
            f"| {overlap[f'random_{rname}_x_chronological_validation']:,} "
            f"| {overlap[f'random_{rname}_x_chronological_test']:,} |")
    lines += [
        "",
        "## Hashes",
        "",
        f"- Split-file SHA-256 (serialization-dependent):",
        f"  `{info['split_file_sha256']}`",
        f"- Canonical index fingerprint (order-insensitive, defined in the",
        f"  JSON): `{info['canonical_fingerprint']['value']}`",
        "",
        info["guided_reference_note"],
        "",
        "## Disclosures",
        "",
        "- This random split uses observations from **all temporal periods**",
        "  (May 2008 – November 2010) in every block; it removes the",
        "  chronological separation that defines the primary analysis.",
        "- The blocks overlap substantially with the frozen chronological",
        "  blocks (table above); the random test set is **not** an",
        "  independent confirmation of the primary chronological test.",
        "- Hyperparameters were **frozen from the original autonomous",
        "  analysis** (C=0.01 for all models); no new search was performed.",
        "- All implementation choices in this script are **post-test",
        "  decisions**.",
        "- Whole-file target prevalence (11.70%) is preserved by the",
        "  stratification, so every block has the same prevalence by",
        "  construction — unlike the primary chronological split, where",
        "  prevalence drifts between blocks.",
        "",
    ]
    (OUT / "random_split_v1.md").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")


def _write_report(info: dict, metrics_df: pd.DataFrame, boot: dict,
                  audit: dict, overlap: dict, warn_log: list[str]) -> None:
    pos = info["positives"]
    prev = info["prevalence"]
    lines = [
        "# Random-split sensitivity analysis — autonomous run (POST-TEST)",
        "",
        f"Generated (UTC): {utc_now()}",
        f"Script: `{SCRIPT_REL}` (SHA-256 `{sha256_file(Path(__file__))}`)",
        "Plan: `plan.md` v1.2 (frozen), §5.5 secondary sensitivity analysis.",
        "",
        "## Scope and status",
        "",
        "This is a **secondary robustness analysis**, authorized after the",
        "primary chronological analysis and the authorized one-shot test",
        "evaluation were complete. It does **not** replace or alter the",
        "frozen primary chronological results, and it is **not** an",
        "independent confirmation of the primary test.",
        "",
        "## Disclosures (required)",
        "",
        "1. The random split uses observations from **all temporal periods**",
        "   in every block; the chronological separation that defines the",
        "   primary analysis is removed.",
        "2. The random blocks overlap substantially with the frozen",
        "   chronological blocks (see the split summary",
        "   `random_split_v1.md`).",
        "3. **This random split is not an independent confirmation of the",
        "   primary test.**",
        "4. Average precision and the Brier score are affected by the very",
        "   different outcome prevalence regime: stratification fixes every",
        "   block's prevalence at the whole-file 11.70% (train "
        f"{pos['train']:,}/{info['sizes']['train']:,} = "
        f"{prev['train']:.4f}), whereas the primary chronological blocks",
        "   drift strongly (5.81% train vs 11.50% validation and higher in",
        "   the test period). AP and Brier values here are therefore **not",
        "   comparable** with the primary chronological values.",
        "5. Hyperparameters were **frozen using the original analysis**:",
        f"   C = {audit['selected_C']} (recorded in",
        "   `results/autonomous/models/model_settings.json`); no feature",
        "   change, no new CV, no recalibration.",
        "6. **All additional implementation choices are post-test",
        "   decisions**, including this script, the split, the report",
        "   structure and the reference application of the decision rule.",
        "7. No Guided-run file, workspace or result was accessed, copied or",
        "   compared; the Guided-reported split hash was treated as",
        "   reference only.",
        "",
        "## Split",
        "",
        f"- Sizes: train {info['sizes']['train']:,} / validation "
        f"{info['sizes']['validation']:,} / test {info['sizes']['test']:,}",
        f"- Positives: train {pos['train']:,} / validation "
        f"{pos['validation']:,} / test {pos['test']:,}",
        f"- Verification: {info['verification']}",
        f"- Split-file SHA-256: `{info['split_file_sha256']}`",
        f"- Canonical index fingerprint: "
        f"`{info['canonical_fingerprint']['value']}`",
        "",
        "## Frozen model settings (from the original autonomous analysis)",
        "",
        f"- L2 logistic regression, solver `{audit['solver']}`, "
        f"`max_iter={audit['max_iter']}`, `random_state={audit['random_state']}`",
        f"- Selected C: {audit['selected_C']} — frozen, no new search",
        "- Original autonomous feature engineering, categorical",
        "  representation (integer code columns + derived `poutcome_level`)",
        "  and preprocessing (StandardScaler / OneHotEncoder", 
        "  `handle_unknown='ignore'`), fitted on the random training block",
        "  only.",
        "",
        "## Test-block metrics (random test, evaluated once)",
        "",
        "Model C is a **NON-DEPLOYABLE DURATION-LEAKAGE DIAGNOSTIC** in",
        "every table.",
        "",
        "| Model | ROC-AUC | AP (PR-AUC) | No-skill AP baseline | Brier | Lift @10% |",
        "|---|---|---|---|---|---|",
    ]
    for r in metrics_df.to_dict("records"):
        lines.append(
            f"| {r['model']} | {r['roc_auc']:.6f} | {r['pr_auc']:.6f} | "
            f"{r['pr_auc_no_skill_baseline']:.6f} | {r['brier']:.6f} | "
            f"{r['lift_top10pct']:.4f} |")
    lines += [
        "",
        "Baseline = prevalence of the random test block itself.",
        "",
        "## Paired bootstrap (2,000 resamples, seed 42)",
        "",
        "| Comparison | ΔROC-AUC (point) | 95% CI | ΔAP (point) | 95% CI |",
        "|---|---|---|---|---|",
    ]
    for key in ("B-A", "C-B"):
        d = boot[key]
        lines.append(
            f"| {key} | {d['delta_roc_auc_point']:.6f} "
            f"| [{d['delta_roc_auc_ci95'][0]:.6f}, "
            f"{d['delta_roc_auc_ci95'][1]:.6f}] "
            f"| {d['delta_pr_auc_point']:.6f} "
            f"| [{d['delta_pr_auc_ci95'][0]:.6f}, "
            f"{d['delta_pr_auc_ci95'][1]:.6f}] |")
    lines += [
        "",
        "Identical resample indices reused across models and metrics, as in",
        "the original autonomous analysis. The decision rule of plan.md",
        "§3.2 applied to this random split (reference context only):",
        f"**{boot.get('decision_rule_applied_as_reference', {}).get('B-A', 'n/a')}**",
        "— this does not replace or alter the primary chronological",
        "decision.",
        "",
        "## Interpretation limits",
        "",
        "- No causal claim is made anywhere (associational prediction only,",
        "  plan §1–§2).",
        "- Because stratification equalizes prevalence, the ROC-AUC values",
        "  here are optimistic relative to a deployment setting with",
        "  drifting prevalence; the comparison with the primary",
        "  chronological result is a robustness contrast, not a",
        "  replication.",
        f"- Warnings raised during fitting: "
        f"{len(warn_log)} (see below).",
        "",
    ]
    if warn_log:
        lines.append("### Warnings")
        lines.append("")
        for w in warn_log:
            lines.append(f"- {w}")
        lines.append("")
    (OUT / "report.md").write_text("\n".join(lines) + "\n",
                                   encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
