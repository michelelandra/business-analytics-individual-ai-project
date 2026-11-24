"""Corrected guided random-split sensitivity evaluation (secondary, post-test).

This is PHASE 4 of the controlled correction: the corrected Guided
random-split sensitivity EVALUATION. It loads the human-verified corrected
index arrays from::

    results/sensitivity_corrected_v1/random_split_v1.json

and does NOT generate, regenerate or reorder any split.

Guarantees
----------
* The corrected split fingerprint, sizes, class counts, coverage and
  disjointness are verified BEFORE any model is constructed or fitted.
* Models A, B and C use the original Guided feature sets, codebook decoding,
  ``poutcome_derived`` construction, preprocessing and metric definitions.
* Hyperparameters are frozen constants: A ``C=0.01``, B ``C=0.01``,
  C ``C=0.001``; L2, ``lbfgs``, ``max_iter=1000``, ``random_state=42``.
* Models are fitted on the corrected RANDOM TRAINING BLOCK ONLY.
* The random validation block is never loaded into an ``X_val``/``y_val``
  object, so it cannot be used for fitting, tuning, calibration or selection.
* Evaluation is on the corrected RANDOM TEST BLOCK.
* Model C is a NON-DEPLOYABLE DURATION-LEAKAGE DIAGNOSTIC.
* The split-only artifacts and all primary/erroneous/archive artifacts are
  hashed before and after; any unexpected modification is fatal.
* Outputs are written exclusively under
  ``results/sensitivity_corrected_v1/`` using the ``*_evaluation`` names.

This is a POST-TEST SECONDARY SENSITIVITY ANALYSIS. It does not replace the
primary chronological test.

Run with (from the repository root)::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m my_analysis.guided_random_split_sensitivity_evaluate

This script performs no LLM calls.
"""

from __future__ import annotations

import ast
import hashlib
import json
import platform
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
TEMPORAL_SPLIT = ROOT / "results" / "splits" / "temporal_split_v1.json"
OUT = ROOT / "results" / "sensitivity_corrected_v1"
SPLIT_JSON = OUT / "random_split_v1.json"
ERRONEOUS_DIR = ROOT / "results" / "sensitivity_erroneous_v1"
SCRIPT_REL = "src/my_analysis/guided_random_split_sensitivity_evaluate.py"
SCRIPT_PATH = ROOT / SCRIPT_REL

EXPECTED_DATASET_SHA256 = (
    "0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f"
)
EXPECTED_TEMPORAL_SPLIT_SHA256 = (
    "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c"
)
EXPECTED_SPLIT_FINGERPRINT = (
    "6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0"
)

SEED = 42
MAX_ITER = 1000
LIFT_FRACTION = 0.10
N_BOOT = 2000

N_TOTAL = 45_211
EXPECTED_SIZES = {"train": 31_647, "validation": 6_781, "test": 6_783}
EXPECTED_POSITIVES = {"train": 3_702, "validation": 793, "test": 794}
EXPECTED_NEGATIVES = {"train": 27_945, "validation": 5_988, "test": 5_989}

CHRONO_TRAIN = (0, 31_647)
CHRONO_VAL = (31_647, 38_428)
CHRONO_TEST = (38_428, 45_211)

# Frozen guided hyperparameters. Constants, not search results.
SELECTED_C: dict[str, float] = {"A": 0.01, "B": 0.01, "C": 0.001}

# ---------------------------------------------------------------------------
# Codebook, copied verbatim from src/my_analysis/guided_train_validate.py.
# 1:1 relabel only; uses no target information.
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
    "poutcome": {
        0: "failure", 1: "other", 2: "success", 3: "unknown", 4: "nonexistent",
    },
}

NUMERIC_A = ["age", "balance"]
CATEGORICAL_A = ["job", "marital", "education", "default", "housing", "loan"]
NUMERIC_B = NUMERIC_A + ["previous"]
CATEGORICAL_B = CATEGORICAL_A + ["poutcome_derived"]
NUMERIC_C = NUMERIC_B + ["duration"]
CATEGORICAL_C = list(CATEGORICAL_B)

MODELS = {
    "A": {
        "role": "customer characteristics only",
        "numeric": NUMERIC_A,
        "categorical": CATEGORICAL_A,
    },
    "B": {
        "role": "A + previous-campaign information",
        "numeric": NUMERIC_B,
        "categorical": CATEGORICAL_B,
    },
    "C": {
        "role": "diagnostic - leakage (B + duration); NOT deployable",
        "numeric": NUMERIC_C,
        "categorical": CATEGORICAL_C,
    },
}

EXPECTED_FEATURES = {
    "A": (NUMERIC_A, CATEGORICAL_A),
    "B": (NUMERIC_B, CATEGORICAL_B),
    "C": (NUMERIC_C, CATEGORICAL_C),
}

FEATURE_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "previous", "poutcome", "pdays", "duration",
]

PREPROC_SPEC = {
    "numeric": {
        "transformer": "StandardScaler",
        "params": {"with_mean": True, "with_std": True},
    },
    "categorical": {
        "transformer": "OneHotEncoder",
        "params": {
            "handle_unknown": "ignore",
            "sparse_output": True,
            "dtype": "float64",
        },
    },
    "remainder": "drop",
    "fitted_on": "corrected random training block only",
}

MODEL_SPEC = {
    "estimator": "LogisticRegression",
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": MAX_ITER,
    "class_weight": None,
    "random_state": SEED,
    "C": dict(SELECTED_C),
    "selection": "frozen from the primary guided run; no re-selection",
}

EVAL_OUTPUT_FILES = [
    "metrics.csv",
    "predictions.csv",
    "calibration.csv",
    "bootstrap.json",
    "report_evaluation.md",
    "run_manifest_evaluation.json",
    "console_output_evaluation.txt",
    "execution_log_evaluation.md",
]
EVAL_OUTPUT_SET = {
    f"results/sensitivity_corrected_v1/{name}" for name in EVAL_OUTPUT_FILES
}
ATTEMPTS_DIR_REL = "results/sensitivity_corrected_v1/evaluation_attempts"

EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "data/raw", ATTEMPTS_DIR_REL,
}
EXCLUDE_FILES = EVAL_OUTPUT_SET


# ---------------------------------------------------------------------------
# Hashing / I/O helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_hash(obj: object) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, separators=(",", ":")))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Tee:
    """Mirror stdout to a file without losing console output."""

    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data: str) -> int:
        self.stdout.write(data)
        self.file.write(data)
        return len(data)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()

    def close(self) -> None:
        self.flush()
        self.file.close()
        sys.stdout = self.stdout


def iter_tracked_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDE_FILES:
            continue
        if any(rel == d or rel.startswith(d + "/") for d in EXCLUDE_DIRS):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def snapshot_files() -> dict[str, str]:
    return {
        p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in iter_tracked_files()
    }


def compare_snapshots(before: dict[str, str], after: dict[str, str]) -> dict:
    before_keys = set(before)
    after_keys = set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(k for k in (before_keys & after_keys) if before[k] != after[k])
    return {
        "n_files_before": len(before),
        "n_files_after": len(after),
        "added_files": added,
        "removed_files": removed,
        "changed_files": changed,
        "unexpected_modification": bool(added or removed or changed),
    }


def archive_prior_evaluation_attempts() -> dict:
    """Preserve any prior evaluation outputs before this attempt writes."""
    existing = [OUT / name for name in EVAL_OUTPUT_FILES if (OUT / name).exists()]
    if not existing:
        return {"archived": False, "attempt": None, "files": [], "path": None}
    attempts_root = OUT / "evaluation_attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    n = 1
    while (attempts_root / f"attempt_{n}").exists():
        n += 1
    dest = attempts_root / f"attempt_{n}"
    dest.mkdir()
    moved = []
    for path in existing:
        shutil.move(str(path), str(dest / path.name))
        moved.append(path.name)
    return {
        "archived": True,
        "attempt": n,
        "files": moved,
        "path": dest.relative_to(ROOT).as_posix(),
    }


# ---------------------------------------------------------------------------
# Data loading / preprocessing (frozen guided definitions)
# ---------------------------------------------------------------------------
def verify_dataset() -> str:
    dataset_sha = sha256_file(RAW)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise SystemExit(f"FATAL: dataset hash mismatch: {dataset_sha}")
    return dataset_sha


def load_full_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    raw = pd.read_parquet(RAW, columns=FEATURE_COLUMNS + ["y"])
    if len(raw) != N_TOTAL:
        raise SystemExit(f"FATAL: expected {N_TOTAL} rows, found {len(raw)}")
    y = raw["y"].to_numpy(dtype=np.int64)
    X = raw[FEATURE_COLUMNS].copy()
    return X, y


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col, mapping in CODEBOOK.items():
        data[col] = data[col].map(mapping)
        if data[col].isna().any():
            raise SystemExit(f"FATAL: unmapped code in {col}")
    data["poutcome_derived"] = np.where(
        data["pdays"].to_numpy() == 999, "never_contacted", data["poutcome"]
    )
    return data


def build_pipeline(model_key: str) -> Pipeline:
    spec = MODELS[model_key]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), spec["numeric"]),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float64,
                ),
                spec["categorical"],
            ),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        penalty="l2",
        C=SELECTED_C[model_key],
        solver="lbfgs",
        max_iter=MAX_ITER,
        class_weight=None,
        random_state=SEED,
    )
    return Pipeline([("prep", preprocess), ("clf", clf)])


# ---------------------------------------------------------------------------
# Corrected split loading and verification
# ---------------------------------------------------------------------------
def load_and_verify_corrected_split(
    y: np.ndarray, dataset_sha: str, temporal_sha: str
) -> tuple[dict, dict[str, np.ndarray], dict]:
    if not SPLIT_JSON.exists():
        raise SystemExit(f"FATAL: corrected split not found: {SPLIT_JSON}")
    doc = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))

    if doc.get("split_name") != "random_split_v1_corrected":
        raise SystemExit("FATAL: unexpected split_name in corrected split")
    if doc.get("model_results_generated") is not False:
        raise SystemExit("FATAL: split artifact is not split-only")
    if doc.get("status") != "split-only; no models constructed, fitted or evaluated":
        raise SystemExit("FATAL: unexpected split status")
    if doc.get("input_sha256") != dataset_sha:
        raise SystemExit("FATAL: split input hash does not match dataset")
    if doc.get("canonical_fingerprint_sha256") != EXPECTED_SPLIT_FINGERPRINT:
        raise SystemExit("FATAL: split fingerprint does not match verified value")
    if doc.get("chronological_reference", {}).get("sha256") != temporal_sha:
        raise SystemExit("FATAL: chronological split hash mismatch in split doc")

    split_script = ROOT / doc["script"]
    if not split_script.exists():
        raise SystemExit(f"FATAL: split-producing script missing: {split_script}")
    if sha256_file(split_script) != doc.get("script_sha256"):
        raise SystemExit("FATAL: split-producing script hash mismatch")

    indices: dict[str, np.ndarray] = {}
    for key in ("train", "validation", "test"):
        arr = np.asarray(doc["indices"][key], dtype=np.int64)
        if len(arr) != len(set(int(i) for i in arr)):
            raise SystemExit(f"FATAL: duplicate indices in {key}")
        if not np.array_equal(arr, np.sort(arr)):
            raise SystemExit(f"FATAL: {key} indices are not sorted ascending")
        indices[key] = arr

    sizes = {k: int(len(v)) for k, v in indices.items()}
    if sizes != EXPECTED_SIZES:
        raise SystemExit(f"FATAL: split sizes {sizes} != {EXPECTED_SIZES}")

    coverage = np.sort(np.concatenate([indices[k] for k in ("train", "validation", "test")]))
    if len(coverage) != N_TOTAL or not np.array_equal(coverage, np.arange(N_TOTAL)):
        raise SystemExit("FATAL: corrected split does not cover all rows exactly once")

    pairwise_disjoint = (
        set(int(i) for i in indices["train"]).isdisjoint(
            set(int(i) for i in indices["validation"])
        )
        and set(int(i) for i in indices["train"]).isdisjoint(
            set(int(i) for i in indices["test"])
        )
        and set(int(i) for i in indices["validation"]).isdisjoint(
            set(int(i) for i in indices["test"])
        )
    )
    if not pairwise_disjoint:
        raise SystemExit("FATAL: corrected split blocks are not pairwise disjoint")

    class_counts: dict[str, dict] = {}
    for key in ("train", "validation", "test"):
        idx = indices[key]
        positive = int((y[idx] == 1).sum())
        negative = int((y[idx] == 0).sum())
        class_counts[key] = {
            "n": int(len(idx)),
            "negative": negative,
            "positive": positive,
            "prevalence": float(y[idx].mean()),
        }
        if positive != EXPECTED_POSITIVES[key]:
            raise SystemExit(
                f"FATAL: {key} positives {positive} != {EXPECTED_POSITIVES[key]}"
            )
        if negative != EXPECTED_NEGATIVES[key]:
            raise SystemExit(
                f"FATAL: {key} negatives {negative} != {EXPECTED_NEGATIVES[key]}"
            )
        recorded = doc["class_counts"][key]
        if (
            recorded["n"] != class_counts[key]["n"]
            or recorded["positive"] != positive
            or recorded["negative"] != negative
        ):
            raise SystemExit(f"FATAL: recorded class counts differ for {key}")

    payload = {
        key: [int(i) for i in indices[key]]
        for key in ("train", "validation", "test")
    }
    fingerprint = sha256_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    if fingerprint != EXPECTED_SPLIT_FINGERPRINT:
        raise SystemExit(
            f"FATAL: recomputed fingerprint {fingerprint} != {EXPECTED_SPLIT_FINGERPRINT}"
        )
    if fingerprint != doc.get("reference_fingerprint_sha256"):
        raise SystemExit("FATAL: fingerprint does not match reference fingerprint")

    chrono_test_overlap = int(
        len(set(int(i) for i in indices["test"]) & set(range(*CHRONO_TEST)))
    )
    if chrono_test_overlap != 1_022:
        raise SystemExit(
            f"FATAL: chronological-test overlap {chrono_test_overlap} != 1022"
        )

    verification = {
        "split_path": SPLIT_JSON.relative_to(ROOT).as_posix(),
        "split_artifact_sha256": sha256_file(SPLIT_JSON),
        "canonical_fingerprint_sha256": fingerprint,
        "expected_fingerprint_sha256": EXPECTED_SPLIT_FINGERPRINT,
        "sizes": sizes,
        "class_counts": class_counts,
        "complete_coverage": True,
        "pairwise_disjoint": True,
        "duplicate_indices": False,
        "chronological_test_overlap": chrono_test_overlap,
        "split_only_status_verified": True,
        "split_script_sha256": doc.get("script_sha256"),
        "split_script_hash_verified": True,
        "input_sha256": doc.get("input_sha256"),
        "input_hash_verified": True,
    }
    return doc, indices, verification


# ---------------------------------------------------------------------------
# Metrics (verbatim guided definitions)
# ---------------------------------------------------------------------------
def lift_at_fraction(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    n = len(y)
    k = max(1, int(np.floor(frac * n)))
    order = np.argsort(-p, kind="stable")
    top = order[:k]
    tp = int(y[top].sum())
    prevalence = float(y.mean())
    precision = tp / k
    return {
        "k": k,
        "k_fraction": k / n,
        "tp_in_top_k": tp,
        "precision_at_k": precision,
        "recall_at_k": tp / float(y.sum()) if y.sum() else float("nan"),
        "expected_tp_by_chance": k * prevalence,
        "lift": precision / prevalence if prevalence > 0 else float("nan"),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, q: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"p": p, "y": y})
    frame["bin"] = pd.qcut(frame["p"], q=q, duplicates="drop")
    out = (
        frame.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    out["bin"] = out["bin"].astype(str)
    return out


def calibration_summary(cal: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict:
    gap = (cal["mean_predicted"] - cal["observed_rate"]).abs()
    weights = cal["n"] / cal["n"].sum()
    return {
        "n_bins": int(len(cal)),
        "mean_predicted": float(p.mean()),
        "observed_rate": float(y.mean()),
        "mean_absolute_calibration_gap_unweighted": float(gap.mean()),
        "mean_absolute_calibration_gap_weighted": float((gap * weights).sum()),
        "max_absolute_calibration_gap": float(gap.max()),
        "max_gap_bin": str(cal.loc[gap.idxmax(), "bin"]),
    }


def paired_bootstrap(
    y: np.ndarray, p: dict[str, np.ndarray], n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    d_ba_auc: list[float] = []
    d_cb_auc: list[float] = []
    d_ba_ap: list[float] = []
    d_cb_ap: list[float] = []
    skipped = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            skipped += 1
            continue
        auc = {m: roc_auc_score(yb, p[m][idx]) for m in ("A", "B", "C")}
        ap = {m: average_precision_score(yb, p[m][idx]) for m in ("A", "B", "C")}
        d_ba_auc.append(auc["B"] - auc["A"])
        d_cb_auc.append(auc["C"] - auc["B"])
        d_ba_ap.append(ap["B"] - ap["A"])
        d_cb_ap.append(ap["C"] - ap["B"])

    def summarize(point: float, values: list[float]) -> dict:
        arr = np.asarray(values)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {
            "point": float(point),
            "ci95": [float(lo), float(hi)],
            "n_valid_resamples": int(len(arr)),
        }

    point_ba_auc = roc_auc_score(y, p["B"]) - roc_auc_score(y, p["A"])
    point_cb_auc = roc_auc_score(y, p["C"]) - roc_auc_score(y, p["B"])
    point_ba_ap = average_precision_score(y, p["B"]) - average_precision_score(y, p["A"])
    point_cb_ap = average_precision_score(y, p["C"]) - average_precision_score(y, p["B"])
    return {
        "method": (
            "paired bootstrap over corrected random-test rows; percentile 95% CI; "
            "identical resample indices reused across models and metrics"
        ),
        "n_boot": n_boot,
        "seed": seed,
        "resamples_skipped_single_class": int(skipped),
        "B-A": {
            "delta_roc_auc": summarize(point_ba_auc, d_ba_auc),
            "delta_pr_auc": summarize(point_ba_ap, d_ba_ap),
        },
        "C-B": {
            "delta_roc_auc": summarize(point_cb_auc, d_cb_auc),
            "delta_pr_auc": summarize(point_cb_ap, d_cb_ap),
        },
        "interpretation": (
            "Descriptive sensitivity interval only. This is not a new "
            "pre-registered hypothesis test and does not replace the primary "
            "chronological evaluation."
        ),
        "note": (
            "Model C is a leakage diagnostic only and is never an operational model."
        ),
    }


def static_audit() -> dict:
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {"GridSearchCV", "TimeSeriesSplit", "RandomizedSearchCV", "C_GRID"}
    used_forbidden = sorted(forbidden & (names | attrs))
    features_ok = all(
        (list(MODELS[k]["numeric"]), list(MODELS[k]["categorical"])) == expected
        for k, expected in EXPECTED_FEATURES.items()
    )
    c_ok = SELECTED_C == {"A": 0.01, "B": 0.01, "C": 0.001}
    validation_objects = {"X_val", "y_val", "X_validation", "y_validation"}
    return {
        "no_hyperparameter_search": not used_forbidden,
        "forbidden_names_found": used_forbidden,
        "feature_sets_match_frozen_lists": bool(features_ok),
        "selected_C_match_frozen_values": bool(c_ok),
        "selected_C": dict(SELECTED_C),
        "max_iter": MAX_ITER,
        "seed": SEED,
        "no_split_regeneration": "train_test_split" not in src,
        "no_validation_objects_defined": not bool(validation_objects & names),
        "note": (
            "Pipelines use fixed constants. The split is loaded from the verified "
            "corrected JSON; no split is regenerated and no validation object is "
            "constructed."
        ),
    }


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------
def build_report(
    split_doc: dict,
    split_verification: dict,
    metric_rows: list[dict],
    calibration_summaries: list[dict],
    boot: dict,
    comparison_rows: list[dict],
    chrono_boot: dict,
    model_calls: list[dict],
) -> str:
    cc = split_doc["class_counts"]
    metric_table = "\n".join(
        f"| {r['model']} | {r['role']} | {r['roc_auc']:.6f} | "
        f"{r['pr_auc']:.6f} | {r['pr_auc_no_skill']:.6f} | {r['brier']:.6f} | "
        f"{r['lift']:.4f} | {r['selected_C']} |"
        for r in metric_rows
    )
    cal_table = "\n".join(
        f"| {r['model']} | {r['n_bins']} | {r['mean_predicted']:.6f} | "
        f"{r['observed_rate']:.6f} | "
        f"{r['mean_absolute_calibration_gap_weighted']:.6f} | "
        f"{r['max_absolute_calibration_gap']:.6f} |"
        for r in calibration_summaries
    )
    ba = boot["B-A"]
    cb = boot["C-B"]
    comparison_table = "\n".join(
        f"| {r['model']} | {r['chronological_roc_auc']:.6f} | "
        f"{r['random_roc_auc']:.6f} | "
        f"{r['delta_roc_auc_random_minus_chronological']:+.6f} | "
        f"{r['chronological_pr_auc']:.6f} | {r['random_pr_auc']:.6f} | "
        f"{r['delta_pr_auc_random_minus_chronological']:+.6f} | "
        f"{r['chronological_brier']:.6f} | {r['random_brier']:.6f} | "
        f"{r['delta_brier_random_minus_chronological']:+.6f} |"
        for r in comparison_rows
    )
    return f"""# Corrected guided random-split sensitivity evaluation — report

Script: `{SCRIPT_REL}`
Generated (UTC): {utc_now()}
Plan reference: `plan.md` §5.5 (secondary sensitivity split)

**Status: POST-TEST SECONDARY SENSITIVITY ANALYSIS.** This evaluation uses the
human-verified corrected random split. It is **not** a new independent
confirmation, **not** a replacement for the primary chronological test, and
**not** a basis for changing the primary conclusion. The random split is
**optimistic** because it mixes temporal information between training and
evaluation.

## 1. Verified corrected split

| Item | Value |
|---|---|
| Split artifact | `{split_verification['split_path']}` |
| Canonical fingerprint | `{split_verification['canonical_fingerprint_sha256']}` |
| Sizes | train {cc['train']['n']:,} / validation {cc['validation']['n']:,} / test {cc['test']['n']:,} |
| Positive counts | train {cc['train']['positive']:,} / validation {cc['validation']['positive']:,} / test {cc['test']['positive']:,} |
| Test prevalence | {cc['test']['prevalence']:.6f} |
| Complete coverage | {split_verification['complete_coverage']} |
| Pairwise disjoint | {split_verification['pairwise_disjoint']} |
| Random-test vs chronological-test overlap | {split_verification['chronological_test_overlap']:,} |
| Split-only status verified | {split_verification['split_only_status_verified']} |

The fingerprint, sizes and class counts were verified before any model was
constructed or fitted. The index arrays were loaded directly from the split
JSON; no split was regenerated.

## 2. Frozen features, preprocessing and hyperparameters

Feature sets, codebook relabelling, `poutcome_derived` construction and the
`StandardScaler` + `OneHotEncoder(handle_unknown="ignore")` preprocessing are
unchanged from the guided primary analysis. Preprocessing is fitted on the
corrected random training block only.

Frozen guided hyperparameters (no re-selection, no CV):

| Model | Role | C | penalty | solver | max_iter | random_state |
|---|---|---|---|---|---|---|
| A | customer characteristics only | 0.01 | l2 | lbfgs | 1000 | 42 |
| B | A + previous-campaign information | 0.01 | l2 | lbfgs | 1000 | 42 |
| C | **diagnostic — leakage; NOT deployable** | 0.001 | l2 | lbfgs | 1000 | 42 |

The random validation block ({cc['validation']['n']:,} rows) was verified but
**never loaded into a fitting object** and was not used for hyperparameter
selection, calibration, model selection or refitting.

## 3. Corrected random-test metrics

Average-precision no-skill baseline is the **corrected random-test prevalence**
({cc['test']['prevalence']:.6f}), not the chronological-test prevalence and not
the full-dataset prevalence.

| Model | Role | ROC-AUC | Average precision | AP no-skill | Brier | Lift@10% | C |
|---|---|---|---|---|---|---|---|
{metric_table}

Model C is a **non-deployable duration-leakage diagnostic**. Its higher scores
measure how much of the outcome is encoded in call `duration`, not a deployable
gain.

## 4. Calibration summaries

| Model | Bins | Mean predicted | Observed rate | Weighted mean abs gap | Max abs gap |
|---|---|---|---|---|---|
{cal_table}

The full ten-bin tables are in `calibration.csv`.

## 5. Paired bootstrap (2,000 resamples, seed 42)

| Comparison | Metric | Point estimate | 95% percentile CI | Valid resamples |
|---|---|---|---|---|
| B − A | ΔROC-AUC | {ba['delta_roc_auc']['point']:.6f} | [{ba['delta_roc_auc']['ci95'][0]:.6f}, {ba['delta_roc_auc']['ci95'][1]:.6f}] | {ba['delta_roc_auc']['n_valid_resamples']:,} |
| B − A | ΔAP | {ba['delta_pr_auc']['point']:.6f} | [{ba['delta_pr_auc']['ci95'][0]:.6f}, {ba['delta_pr_auc']['ci95'][1]:.6f}] | {ba['delta_pr_auc']['n_valid_resamples']:,} |
| C − B | ΔROC-AUC | {cb['delta_roc_auc']['point']:.6f} | [{cb['delta_roc_auc']['ci95'][0]:.6f}, {cb['delta_roc_auc']['ci95'][1]:.6f}] | {cb['delta_roc_auc']['n_valid_resamples']:,} |
| C − B | ΔAP | {cb['delta_pr_auc']['point']:.6f} | [{cb['delta_pr_auc']['ci95'][0]:.6f}, {cb['delta_pr_auc']['ci95'][1]:.6f}] | {cb['delta_pr_auc']['n_valid_resamples']:,} |

These intervals are **descriptive sensitivity intervals**, not a new
pre-registered hypothesis test. They do not override the primary chronological
bootstrap result.

## 6. Descriptive comparison with the primary chronological test

**Descriptive only.** The two evaluations use different test rows and different
prevalences; the differences below are not a test of a pre-registered
hypothesis and must not be read as a causal or confirmatory result.

| Model | Chrono ROC-AUC | Random ROC-AUC | Δ (random − chrono) | Chrono AP | Random AP | Δ AP | Chrono Brier | Random Brier | Δ Brier |
|---|---|---|---|---|---|---|---|---|---|
{comparison_table}

Primary chronological B − A (for reference, from
`results/guided/test/hypothesis_tests_test.json`): ΔROC-AUC
{chrono_boot['B-A']['delta_roc_auc']['point']:.6f}
[{chrono_boot['B-A']['delta_roc_auc']['ci95'][0]:.6f},
{chrono_boot['B-A']['delta_roc_auc']['ci95'][1]:.6f}];
ΔAP {chrono_boot['B-A']['delta_pr_auc']['point']:.6f}
[{chrono_boot['B-A']['delta_pr_auc']['ci95'][0]:.6f},
{chrono_boot['B-A']['delta_pr_auc']['ci95'][1]:.6f}].

The random split generally produces higher apparent performance than the
chronological split. That is expected: random splitting lets the model learn
from rows interleaved in time with the evaluation rows, whereas the
chronological split asks for performance on a later, unseen period.

## 7. Warnings and limitations

- **Optimistic estimate.** Random splitting mixes temporal information; the
  random-split metrics are not an estimate of future-period performance.
- **Not independent.** The random test rows overlap all chronological blocks,
  including the primary chronological test block.
- **Different prevalence.** Random-test prevalence is
  {cc['test']['prevalence']:.6f} versus 0.393484 chronological; AP and Brier are
  prevalence-sensitive.
- **Model C is not deployable.** It uses `duration`, which is unknown before a
  call ends.
- **Secondary.** This report must not replace the primary chronological test.

## 8. Artifacts

- `metrics.csv`
- `predictions.csv`
- `calibration.csv`
- `bootstrap.json`
- `run_manifest_evaluation.json`
- `console_output_evaluation.txt`
- `execution_log_evaluation.md`

The human-verified split-only artifacts were not modified.
"""


def build_execution_log(
    split_verification: dict,
    metric_rows: list[dict],
    boot: dict,
    model_calls: list[dict],
    preservation: dict,
    attempts: dict,
) -> str:
    metrics = "\n".join(
        f"| {r['model']} | {r['roc_auc']:.6f} | {r['pr_auc']:.6f} | "
        f"{r['brier']:.6f} | {r['lift']:.4f} |"
        for r in metric_rows
    )
    warnings_text = "\n".join(
        f"- Model {r['model']}: {r['warnings'] if r['warnings'] else 'none'}"
        for r in model_calls
    )
    ba = boot["B-A"]
    cb = boot["C-B"]
    return f"""# Corrected guided random-split sensitivity evaluation — execution log

## {utc_now()} — corrected evaluation

### Scope and authorization

PHASE 4 of the controlled correction, explicitly authorized by the human
project author after independent verification that the corrected Guided index
sets match both the authorized procedure and the existing Autonomous split.

This is a **post-test secondary sensitivity analysis**. It does not replace the
primary chronological test. No subgroup analysis was started.

### Inputs verified before model construction

- Corrected split: `{split_verification['split_path']}`
- Canonical fingerprint: `{split_verification['canonical_fingerprint_sha256']}`
  (expected `{split_verification['expected_fingerprint_sha256']}`)
- Sizes: train {split_verification['sizes']['train']:,} / validation
  {split_verification['sizes']['validation']:,} / test
  {split_verification['sizes']['test']:,}
- Positive counts: train
  {split_verification['class_counts']['train']['positive']:,} / validation
  {split_verification['class_counts']['validation']['positive']:,} / test
  {split_verification['class_counts']['test']['positive']:,}
- Complete coverage: {split_verification['complete_coverage']}
- Pairwise disjoint: {split_verification['pairwise_disjoint']}
- Random-test vs chronological-test overlap:
  {split_verification['chronological_test_overlap']:,}
- Split-only status verified: {split_verification['split_only_status_verified']}
- Split-producing script hash verified:
  {split_verification['split_script_hash_verified']}
- Dataset hash verified: {split_verification['input_hash_verified']}

### Command

```
$env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m my_analysis.guided_random_split_sensitivity_evaluate
```

### Fit and evaluation

- Fitted on the corrected random training block only.
- Validation block used for fitting, tuning or calibration: **no**.
- Evaluated on the corrected random test block.
- No split was regenerated; the index arrays were loaded directly.

| Model | ROC-AUC | AP | Brier | Lift@10% |
|---|---|---|---|---|
{metrics}

### Paired bootstrap

- Resamples: {boot['n_boot']}, seed {boot['seed']}
- Skipped single-class resamples: {boot['resamples_skipped_single_class']}
- B−A ΔROC-AUC: {ba['delta_roc_auc']['point']:.6f}
  [{ba['delta_roc_auc']['ci95'][0]:.6f}, {ba['delta_roc_auc']['ci95'][1]:.6f}]
- B−A ΔAP: {ba['delta_pr_auc']['point']:.6f}
  [{ba['delta_pr_auc']['ci95'][0]:.6f}, {ba['delta_pr_auc']['ci95'][1]:.6f}]
- C−B ΔROC-AUC: {cb['delta_roc_auc']['point']:.6f}
  [{cb['delta_roc_auc']['ci95'][0]:.6f}, {cb['delta_roc_auc']['ci95'][1]:.6f}]
- C−B ΔAP: {cb['delta_pr_auc']['point']:.6f}
  [{cb['delta_pr_auc']['ci95'][0]:.6f}, {cb['delta_pr_auc']['ci95'][1]:.6f}]

### Warnings

{warnings_text}

### Integrity

- Tracked files hashed before: {preservation['n_files_before']:,}
- Tracked files hashed after: {preservation['n_files_after']:,}
- Added: {preservation['added_files']}
- Removed: {preservation['removed_files']}
- Changed: {preservation['changed_files']}
- Unexpected modification: {preservation['unexpected_modification']}

The baseline includes the human-verified split-only artifacts, the original
erroneous run, the error archive, all primary artifacts, the frozen plan and
the dataset (verified separately by SHA-256). Only this evaluation's own eight
output files and the `evaluation_attempts/` subtree are excluded.

### Attempts and retries

Prior evaluation outputs archived before this attempt:
{attempts}

### Status

- Evaluation succeeded: see manifest.
- Model C: NON-DEPLOYABLE duration-leakage diagnostic.
- Subgroup analysis: **not started**.
- Final report: **not written**.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    attempts = archive_prior_evaluation_attempts()
    tee = Tee(OUT / "console_output_evaluation.txt")
    sys.stdout = tee
    warnings.simplefilter("always")

    try:
        print("[eval] corrected guided random-split sensitivity evaluation")
        print(f"[eval] started_utc={utc_now()}")
        print(f"[eval] prior_attempts_archived={attempts['archived']}")

        before_snapshot = snapshot_files()
        print(f"[integrity] baseline files hashed: {len(before_snapshot):,}")

        dataset_sha = verify_dataset()
        temporal_meta = json.loads(TEMPORAL_SPLIT.read_text(encoding="utf-8"))
        if canonical_hash(temporal_meta["indices"]) != temporal_meta["split_hash_sha256"]:
            raise SystemExit("FATAL: chronological split indices do not match hash")
        if temporal_meta["split_hash_sha256"] != EXPECTED_TEMPORAL_SPLIT_SHA256:
            raise SystemExit("FATAL: unexpected chronological split hash")
        print(f"[verify] dataset sha256={dataset_sha}")
        print(f"[verify] chronological split sha256={EXPECTED_TEMPORAL_SPLIT_SHA256}")

        X_raw, y = load_full_dataset()
        print(f"[rows] total={len(y):,}; no filtering, merging or dropping applied")
        print(
            f"[target] full prevalence={y.mean():.6f} "
            f"(positives={int(y.sum()):,}/{len(y):,})"
        )

        split_doc, split, split_verification = load_and_verify_corrected_split(
            y, dataset_sha, EXPECTED_TEMPORAL_SPLIT_SHA256
        )
        print(f"[split] artifact={split_verification['split_path']}")
        print(
            f"[split] fingerprint={split_verification['canonical_fingerprint_sha256']}"
        )
        for key in ("train", "validation", "test"):
            c = split_verification["class_counts"][key]
            print(
                f"[split] {key}: n={c['n']:,} negative={c['negative']:,} "
                f"positive={c['positive']:,} prevalence={c['prevalence']:.6f}"
            )
        print(
            f"[split] coverage={split_verification['complete_coverage']} "
            f"disjoint={split_verification['pairwise_disjoint']} "
            f"chrono_test_overlap={split_verification['chronological_test_overlap']:,}"
        )

        data = prepare_frame(X_raw)
        n_never = int((data["poutcome_derived"] == "never_contacted").sum())
        n_unknown = int((data["poutcome_derived"] == "unknown").sum())
        if n_never != 36_954 or n_unknown != 5:
            raise SystemExit(
                f"FATAL: derived poutcome counts unexpected: "
                f"never_contacted={n_never}, unknown={n_unknown}"
            )
        print(
            f"[derived] poutcome: never_contacted={n_never:,}, unknown={n_unknown}"
        )

        train_idx = split["train"]
        test_idx = split["test"]
        if not np.array_equal(train_idx, np.sort(train_idx)):
            raise SystemExit("FATAL: training indices changed order")
        if not np.array_equal(test_idx, np.sort(test_idx)):
            raise SystemExit("FATAL: test indices changed order")
        if len(set(int(i) for i in train_idx) & set(int(i) for i in test_idx)):
            raise SystemExit("FATAL: train and test indices overlap")

        X_train = data.iloc[train_idx].reset_index(drop=True)
        X_test = data.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(X_train) != EXPECTED_SIZES["train"] or len(X_test) != EXPECTED_SIZES["test"]:
            raise SystemExit("FATAL: unexpected fit/evaluation row count")
        print(
            f"[fit] corrected random training block only: n={len(X_train):,}; "
            f"validation block n={split_verification['sizes']['validation']:,} "
            f"recorded but NOT loaded for fitting"
        )

        predictions: dict[str, np.ndarray] = {}
        metric_rows: list[dict] = []
        calibration_frames: list[pd.DataFrame] = []
        calibration_summaries: list[dict] = []
        model_calls: list[dict] = []

        for name in ("A", "B", "C"):
            pipeline = build_pipeline(name)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipeline.fit(X_train, y_train)
            p = pipeline.predict_proba(X_test)[:, 1]
            predictions[name] = p

            lift = lift_at_fraction(y_test, p, LIFT_FRACTION)
            metric_rows.append(
                {
                    "model": name,
                    "role": MODELS[name]["role"],
                    "split": "corrected_random",
                    "n_test": int(len(y_test)),
                    "n_positive": int(y_test.sum()),
                    "prevalence": float(y_test.mean()),
                    "roc_auc": float(roc_auc_score(y_test, p)),
                    "pr_auc": float(average_precision_score(y_test, p)),
                    "pr_auc_no_skill": float(y_test.mean()),
                    "brier": float(brier_score_loss(y_test, p)),
                    "mean_predicted": float(p.mean()),
                    "lift_k": lift["k"],
                    "lift_k_fraction": lift["k_fraction"],
                    "lift_tp_in_top_k": lift["tp_in_top_k"],
                    "lift_precision_at_k": lift["precision_at_k"],
                    "lift_recall_at_k": lift["recall_at_k"],
                    "lift_expected_tp_by_chance": lift["expected_tp_by_chance"],
                    "lift": lift["lift"],
                    "selected_C": SELECTED_C[name],
                }
            )
            cal = calibration_table(y_test, p)
            cal.insert(0, "model", name)
            calibration_frames.append(cal)
            summary = calibration_summary(cal, y_test, p)
            summary["model"] = name
            calibration_summaries.append(summary)

            model_calls.append(
                {
                    "model": name,
                    "estimator": "LogisticRegression",
                    "C": SELECTED_C[name],
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": MAX_ITER,
                    "random_state": SEED,
                    "n_train": int(len(X_train)),
                    "n_test": int(len(y_test)),
                    "warnings": [
                        f"{w.category.__name__}: {w.message}" for w in caught
                    ],
                }
            )
            print(
                f"[model {name}] C={SELECTED_C[name]} "
                f"roc_auc={metric_rows[-1]['roc_auc']:.6f} "
                f"ap={metric_rows[-1]['pr_auc']:.6f} "
                f"brier={metric_rows[-1]['brier']:.6f} "
                f"lift@10%={metric_rows[-1]['lift']:.4f}"
            )

        metrics_df = pd.DataFrame(metric_rows)
        metrics_df.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(calibration_frames, ignore_index=True).to_csv(
            OUT / "calibration.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "row_index": test_idx,
                "y": y_test,
                "p_A": predictions["A"],
                "p_B": predictions["B"],
                "p_C": predictions["C"],
            }
        ).to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

        boot = paired_bootstrap(y_test, predictions)
        (OUT / "bootstrap.json").write_text(
            json.dumps(boot, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[bootstrap] n_boot={N_BOOT} seed={SEED} "
            f"skipped={boot['resamples_skipped_single_class']} "
            f"B-A dAUC={boot['B-A']['delta_roc_auc']['point']:.6f} "
            f"CI={boot['B-A']['delta_roc_auc']['ci95']}"
        )

        # Descriptive comparison with the primary chronological result (read-only).
        chrono_metrics = pd.read_csv(
            ROOT / "results" / "guided" / "test" / "test_metrics.csv"
        )
        chrono_boot = json.loads(
            (ROOT / "results" / "guided" / "test" / "hypothesis_tests_test.json").read_text(
                encoding="utf-8"
            )
        )
        comparison_rows: list[dict] = []
        for name in ("A", "B", "C"):
            r = next(row for row in metric_rows if row["model"] == name)
            c = chrono_metrics[chrono_metrics["model"] == name].iloc[0]
            comparison_rows.append(
                {
                    "model": name,
                    "chronological_roc_auc": float(c["roc_auc"]),
                    "random_roc_auc": r["roc_auc"],
                    "delta_roc_auc_random_minus_chronological": r["roc_auc"]
                    - float(c["roc_auc"]),
                    "chronological_pr_auc": float(c["pr_auc"]),
                    "random_pr_auc": r["pr_auc"],
                    "delta_pr_auc_random_minus_chronological": r["pr_auc"]
                    - float(c["pr_auc"]),
                    "chronological_brier": float(c["brier"]),
                    "random_brier": r["brier"],
                    "delta_brier_random_minus_chronological": r["brier"]
                    - float(c["brier"]),
                }
            )

        report = build_report(
            split_doc,
            split_verification,
            metric_rows,
            calibration_summaries,
            boot,
            comparison_rows,
            chrono_boot,
            model_calls,
        )
        (OUT / "report_evaluation.md").write_text(report, encoding="utf-8")

        after_snapshot = snapshot_files()
        preservation = compare_snapshots(before_snapshot, after_snapshot)
        print(
            f"[integrity] files before={preservation['n_files_before']:,} "
            f"after={preservation['n_files_after']:,} "
            f"unexpected_modification={preservation['unexpected_modification']}"
        )
        if preservation["unexpected_modification"]:
            print(
                "[integrity] changed="
                f"{preservation['changed_files']} added={preservation['added_files']} "
                f"removed={preservation['removed_files']}"
            )
            raise SystemExit("FATAL: existing artifacts changed during evaluation")

        (OUT / "execution_log_evaluation.md").write_text(
            build_execution_log(
                split_verification,
                metric_rows,
                boot,
                model_calls,
                preservation,
                attempts,
            ),
            encoding="utf-8",
        )

        print("[eval] complete; artifacts written to results/sensitivity_corrected_v1/")
        tee.flush()

        artifact_hashes = {
            p.name: sha256_file(p)
            for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name != "run_manifest_evaluation.json"
        }
        manifest = {
            "run_mode": "guided",
            "stage": "corrected random-split sensitivity evaluation (secondary, post-test)",
            "status": "evaluation complete; secondary post-test sensitivity analysis",
            "authorization": (
                "Phase 4 explicitly authorized by the human project author on "
                "2026-11-14 after independent verification of the corrected "
                "Guided index sets against the authorized procedure and the "
                "Autonomous split."
            ),
            "script": SCRIPT_REL,
            "script_sha256": sha256_file(SCRIPT_PATH),
            "generated_utc": utc_now(),
            "plan_reference": "plan.md section 5.5 (secondary sensitivity split)",
            "analysis_role": (
                "post-test secondary sensitivity analysis; not an independent "
                "confirmation and not a replacement for the primary chronological "
                "evaluation"
            ),
            "corrected_split": {
                **split_verification,
                "supersedes_erroneous_split_sha256": split_doc["supersedes"][
                    "erroneous_split_sha256"
                ],
                "erroneous_archive": ERRONEOUS_DIR.relative_to(ROOT).as_posix(),
                "human_verified": True,
            },
            "dataset_sha256": dataset_sha,
            "chronological_split_sha256": EXPECTED_TEMPORAL_SPLIT_SHA256,
            "feature_sets": {
                name: {
                    "role": spec["role"],
                    "numeric": spec["numeric"],
                    "categorical": spec["categorical"],
                }
                for name, spec in MODELS.items()
            },
            "preprocessing": PREPROC_SPEC,
            "preprocessing_spec_sha256": canonical_hash(PREPROC_SPEC),
            "model_spec": MODEL_SPEC,
            "model_spec_sha256": canonical_hash(MODEL_SPEC),
            "selected_C": dict(SELECTED_C),
            "max_iter": MAX_ITER,
            "seed": SEED,
            "n_boot": N_BOOT,
            "lift_fraction": LIFT_FRACTION,
            "fitted_on": "corrected random training block only",
            "random_validation_used_for_fitting_or_tuning": False,
            "random_validation_loaded_into_fitting_object": False,
            "random_test_evaluated": True,
            "random_test_labels_used_for": "evaluation only",
            "random_test_labels_used_for_fitting_or_tuning": False,
            "split_regenerated": False,
            "model_calls": model_calls,
            "metrics": metric_rows,
            "calibration_summaries": calibration_summaries,
            "bootstrap": boot,
            "descriptive_chronological_comparison": comparison_rows,
            "static_audit": static_audit(),
            "preservation": {
                **preservation,
                "baseline_note": (
                    "Baseline includes the human-verified split-only artifacts, "
                    "the original erroneous run, the error archive, all primary "
                    "artifacts, the frozen plan and the dataset (verified by "
                    "SHA-256). Only this evaluation's eight output files and the "
                    "evaluation_attempts/ subtree are excluded."
                ),
            },
            "attempts": attempts,
            "artifacts": artifact_hashes,
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
        }
        (OUT / "run_manifest_evaluation.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        tee.close()
        return 0
    except BaseException:
        tee.close()
        raise


if __name__ == "__main__":
    sys.exit(main())
