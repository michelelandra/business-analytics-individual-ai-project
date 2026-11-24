"""Guided random-split sensitivity analysis (secondary, post-test).

Scope
-----
This script implements the SECONDARY sensitivity analysis described in
``plan.md`` section 5.5 ("stratified 70/15/15 random split with a fixed seed
(42), same preprocessing discipline"). It is NOT a new independent
confirmation and does NOT replace the primary chronological evaluation.

Guarantees
----------
* The random split is generated ONCE with ``train_test_split``, stratified on
  ``y``, seed 42, with exact sizes 31,647 / 6,781 / 6,783 over all 45,211 rows.
  Full indices, class counts, generation details and a canonical split hash are
  recorded. Complete coverage, pairwise disjointness and reproducibility are
  verified before any model is fitted.
* The original guided feature sets and preprocessing are reused verbatim.
* The original guided hyperparameters are frozen constants:
  A ``C=0.01``, B ``C=0.01``, C ``C=0.001``; ``penalty="l2"``,
  ``solver="lbfgs"``, ``max_iter=1000``, ``random_state=42``.
  There is no grid search, no cross-validation and no re-selection.
* Models are fitted on the RANDOM TRAINING BLOCK ONLY. The random validation
  block is recorded but never used for fitting, tuning, calibration, model
  selection or refitting.
* Model C is a NON-DEPLOYABLE LEAKAGE DIAGNOSTIC and is labelled as such.
* All outputs are written exclusively under ``results/sensitivity/``.
* Existing files are hashed before and after the run; unexpected modification
  is reported and treated as a fatal integrity failure.

Run with (from the repository root)::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m my_analysis.guided_random_split_sensitivity

This script performs no LLM calls.
"""

from __future__ import annotations

import ast
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
SCRIPT_REL = "src/my_analysis/guided_random_split_sensitivity.py"
SCRIPT_PATH = ROOT / SCRIPT_REL

EXPECTED_DATASET_SHA256 = (
    "0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f"
)
EXPECTED_TEMPORAL_SPLIT_SHA256 = (
    "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c"
)

SEED = 42
MAX_ITER = 1000
LIFT_FRACTION = 0.10
N_BOOT = 2000

N_TOTAL = 45_211
N_TRAIN = 31_647
N_VAL = 6_781
N_TEST = 6_783

# Frozen guided hyperparameters. Constants, not search results.
SELECTED_C: dict[str, float] = {"A": 0.01, "B": 0.01, "C": 0.001}

# Chronological boundaries from the frozen primary split.
CHRONO_TRAIN = (0, 31_647)
CHRONO_VAL = (31_647, 38_428)
CHRONO_TEST = (38_428, 45_211)

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
    "fitted_on": "random training block only",
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

# Files/directories excluded from the preservation baseline. ``results/sensitivity``
# is this analysis's own output directory; the raw data is verified by hash.
EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "results/sensitivity", "data/raw",
}


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


# ---------------------------------------------------------------------------
# Preservation baseline
# ---------------------------------------------------------------------------
def iter_tracked_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
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
    changed = sorted(
        k for k in (before_keys & after_keys) if before[k] != after[k]
    )
    return {
        "n_files_before": len(before),
        "n_files_after": len(after),
        "added_files": added,
        "removed_files": removed,
        "changed_files": changed,
        "unexpected_modification": bool(added or removed or changed),
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
    """Load all 45,211 feature rows and labels (authorised for this sensitivity run)."""
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
# Random split generation and verification
# ---------------------------------------------------------------------------
def generate_random_split(y: np.ndarray) -> dict[str, np.ndarray]:
    """Deterministic two-step stratified split with exact block sizes.

    Step 1 removes the training block (31,647 rows), leaving 13,564 rows.
    Step 2 divides those 13,564 rows into validation (6,781) and test (6,783).
    Indices are sorted ascending after generation; membership is unchanged.
    """
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=N_VAL + N_TEST,
        stratify=y,
        random_state=SEED,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=N_TEST,
        stratify=y[temp_idx],
        random_state=SEED,
    )
    return {
        "train": np.sort(train_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def verify_random_split(
    split: dict[str, np.ndarray], y: np.ndarray
) -> dict:
    train = split["train"]
    val = split["validation"]
    test = split["test"]

    sizes = {"train": len(train), "validation": len(val), "test": len(test)}
    expected = {"train": N_TRAIN, "validation": N_VAL, "test": N_TEST}
    if sizes != expected:
        raise SystemExit(f"FATAL: random split sizes {sizes} != {expected}")

    coverage = np.sort(np.concatenate([train, val, test]))
    complete_coverage = bool(
        len(coverage) == N_TOTAL and np.array_equal(coverage, np.arange(N_TOTAL))
    )
    disjoint = bool(
        set(train).isdisjoint(set(val))
        and set(train).isdisjoint(set(test))
        and set(val).isdisjoint(set(test))
    )
    if not complete_coverage or not disjoint:
        raise SystemExit("FATAL: random split coverage/disjointness failed")

    regenerated = generate_random_split(y)
    reproducible = all(
        np.array_equal(split[k], regenerated[k]) for k in ("train", "validation", "test")
    )
    if not reproducible:
        raise SystemExit("FATAL: random split is not reproducible from seed 42")

    class_counts = {
        k: {
            "n": int(len(v)),
            "negative": int((y[v] == 0).sum()),
            "positive": int((y[v] == 1).sum()),
            "prevalence": float(y[v].mean()),
        }
        for k, v in split.items()
    }

    return {
        "sizes": sizes,
        "class_counts": class_counts,
        "complete_coverage": complete_coverage,
        "pairwise_disjoint": disjoint,
        "reproducible_from_seed": reproducible,
        "coverage_n": int(len(coverage)),
        "coverage_min_index": int(coverage.min()),
        "coverage_max_index": int(coverage.max()),
    }


def overlap_with_chronological(split: dict[str, np.ndarray]) -> dict:
    chrono = {
        "chronological_train": np.arange(*CHRONO_TRAIN),
        "chronological_validation": np.arange(*CHRONO_VAL),
        "chronological_test": np.arange(*CHRONO_TEST),
    }
    out: dict[str, dict[str, int]] = {}
    for rname, ridx in split.items():
        rset = set(ridx.tolist())
        out[f"random_{rname}"] = {
            cname: int(len(rset & set(cidx.tolist())))
            for cname, cidx in chrono.items()
        }
    return out


# ---------------------------------------------------------------------------
# Metrics
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
            "paired bootstrap over random-test rows; percentile 95% CI; "
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
    forbidden = {"GridSearchCV", "TimeSeriesSplit", "C_GRID"}
    used_forbidden = sorted(forbidden & (names | attrs))
    features_ok = all(
        (list(MODELS[k]["numeric"]), list(MODELS[k]["categorical"])) == expected
        for k, expected in EXPECTED_FEATURES.items()
    )
    c_ok = SELECTED_C == {"A": 0.01, "B": 0.01, "C": 0.001}
    return {
        "no_hyperparameter_search": not used_forbidden,
        "forbidden_names_found": used_forbidden,
        "feature_sets_match_frozen_lists": bool(features_ok),
        "selected_C_match_frozen_values": bool(c_ok),
        "selected_C": dict(SELECTED_C),
        "max_iter": MAX_ITER,
        "seed": SEED,
        "note": (
            "Pipelines are constructed with fixed constants. No GridSearchCV, "
            "TimeSeriesSplit or C grid is used."
        ),
    }


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------
def build_split_document(
    split: dict[str, np.ndarray],
    verification: dict,
    overlap: dict,
    dataset_sha: str,
    split_sha: str,
    temporal_meta: dict,
) -> dict:
    return {
        "split_name": "random_split_v1",
        "analysis_role": (
            "secondary post-test sensitivity analysis; optimistic, not an "
            "independent confirmation and not a replacement for the primary "
            "chronological evaluation"
        ),
        "plan_reference": "plan.md section 5.5 (secondary sensitivity split)",
        "created_utc": utc_now(),
        "script": SCRIPT_REL,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "input_file": RAW.relative_to(ROOT).as_posix(),
        "input_sha256": dataset_sha,
        "n_rows": N_TOTAL,
        "strategy": "stratified random 70/15/15 split on y",
        "generation": {
            "library": "sklearn.model_selection.train_test_split",
            "scikit_learn_version": sklearn.__version__,
            "seed": SEED,
            "steps": [
                {
                    "step": 1,
                    "call": "train_test_split(all_indices, test_size=13564, stratify=y, random_state=42)",
                    "result_sizes": {"train": N_TRAIN, "remaining": N_VAL + N_TEST},
                },
                {
                    "step": 2,
                    "call": "train_test_split(remaining_indices, test_size=6783, stratify=y[remaining], random_state=42)",
                    "result_sizes": {"validation": N_VAL, "test": N_TEST},
                },
            ],
            "index_order": "sorted ascending after generation; membership unchanged",
            "stratification_uses_labels": True,
            "stratification_note": (
                "Labels are used only to assign split membership. They are not "
                "used for feature construction, preprocessing, hyperparameter "
                "selection or calibration."
            ),
        },
        "sizes": verification["sizes"],
        "class_counts": verification["class_counts"],
        "prevalence": {
            k: verification["class_counts"][k]["prevalence"]
            for k in ("train", "validation", "test")
        },
        "verification": {
            "complete_coverage": verification["complete_coverage"],
            "pairwise_disjoint": verification["pairwise_disjoint"],
            "reproducible_from_seed": verification["reproducible_from_seed"],
            "coverage_n": verification["coverage_n"],
            "coverage_min_index": verification["coverage_min_index"],
            "coverage_max_index": verification["coverage_max_index"],
        },
        "chronological_reference": {
            "path": "results/splits/temporal_split_v1.json",
            "sha256": EXPECTED_TEMPORAL_SPLIT_SHA256,
            "sizes": temporal_meta["sizes"],
            "boundaries": temporal_meta["boundaries"],
        },
        "overlap_with_chronological_blocks": overlap,
        "indices": {k: v.tolist() for k, v in split.items()},
        "split_hash_sha256": split_sha,
        "hash_definition": (
            "SHA-256 of json.dumps(indices, sort_keys=True, separators=(',', ':'))"
        ),
    }


def split_markdown(doc: dict) -> str:
    cc = doc["class_counts"]
    ov = doc["overlap_with_chronological_blocks"]
    lines = [
        "# Random split v1 — guided sensitivity analysis",
        "",
        f"Created (UTC): {doc['created_utc']}",
        f"Script: `{doc['script']}`",
        f"Dataset SHA-256: `{doc['input_sha256']}`",
        f"Split SHA-256: `{doc['split_hash_sha256']}`",
        "",
        "**Role.** Secondary post-test sensitivity analysis. The split is",
        "optimistic because random splitting mixes temporal information. It is",
        "**not** an independent confirmation and **not** a replacement for the",
        "primary chronological evaluation.",
        "",
        "## Generation",
        "",
        f"- Method: `{doc['generation']['library']}`, seed "
        f"{doc['generation']['seed']}, stratified on `y`.",
        f"- Step 1: {doc['generation']['steps'][0]['call']}",
        f"- Step 2: {doc['generation']['steps'][1]['call']}",
        f"- Indices: {doc['generation']['index_order']}.",
        "- Labels are used only to assign split membership.",
        "",
        "## Sizes and class counts",
        "",
        "| Block | Rows | Negative | Positive | Prevalence |",
        "|---|---|---|---|---|",
    ]
    for k in ("train", "validation", "test"):
        c = cc[k]
        lines.append(
            f"| {k} | {c['n']:,} | {c['negative']:,} | {c['positive']:,} | "
            f"{c['prevalence']:.6f} |"
        )
    lines += [
        "",
        "## Verification",
        "",
        f"- Complete coverage of all {doc['n_rows']:,} rows: "
        f"{doc['verification']['complete_coverage']}",
        f"- Pairwise disjoint blocks: {doc['verification']['pairwise_disjoint']}",
        f"- Reproducible from seed {doc['generation']['seed']}: "
        f"{doc['verification']['reproducible_from_seed']}",
        f"- Covered index range: [{doc['verification']['coverage_min_index']}, "
        f"{doc['verification']['coverage_max_index']}]",
        "",
        "## Overlap with the primary chronological blocks",
        "",
        "| Random block | Chronological train [0, 31,647) | "
        "Chronological validation [31,647, 38,428) | "
        "Chronological test [38,428, 45,211) |",
        "|---|---|---|---|",
    ]
    for rname in ("random_train", "random_validation", "random_test"):
        row = ov[rname]
        lines.append(
            f"| {rname} | {row['chronological_train']:,} | "
            f"{row['chronological_validation']:,} | "
            f"{row['chronological_test']:,} |"
        )
    lines += [
        "",
        "The overlap is the mechanism of the sensitivity analysis: every random",
        "block draws rows from all chronological periods, so temporal",
        "information is mixed between training and evaluation.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tee = Tee(OUT / "console_output.txt")
    sys.stdout = tee
    warnings.simplefilter("always")

    try:
        print("[sensitivity] guided random-split sensitivity analysis")
        print(f"[sensitivity] started_utc={utc_now()}")

        # ------------------------------------------------------------------
        # 0. Preservation baseline
        # ------------------------------------------------------------------
        before_snapshot = snapshot_files()
        print(f"[integrity] baseline files hashed: {len(before_snapshot)}")

        # ------------------------------------------------------------------
        # 1. Verify dataset and frozen chronological split
        # ------------------------------------------------------------------
        dataset_sha = verify_dataset()
        temporal_meta = json.loads(TEMPORAL_SPLIT.read_text(encoding="utf-8"))
        if canonical_hash(temporal_meta["indices"]) != temporal_meta["split_hash_sha256"]:
            raise SystemExit("FATAL: chronological split indices do not match hash")
        if temporal_meta["split_hash_sha256"] != EXPECTED_TEMPORAL_SPLIT_SHA256:
            raise SystemExit("FATAL: unexpected chronological split hash")
        print(f"[verify] dataset sha256={dataset_sha}")
        print(f"[verify] chronological split sha256={EXPECTED_TEMPORAL_SPLIT_SHA256}")

        # ------------------------------------------------------------------
        # 2. Load full data and generate the random split
        # ------------------------------------------------------------------
        X_raw, y = load_full_dataset()
        print(f"[rows] total={len(y):,}; no filtering, merging or dropping applied")
        print(
            f"[target] full prevalence={y.mean():.6f} "
            f"(positives={int(y.sum()):,}/{len(y):,})"
        )

        split = generate_random_split(y)
        verification = verify_random_split(split, y)
        split_sha = canonical_hash({k: v.tolist() for k, v in split.items()})
        overlap = overlap_with_chronological(split)

        print(
            f"[split] sizes: train={len(split['train']):,} "
            f"validation={len(split['validation']):,} "
            f"test={len(split['test']):,}"
        )
        for k in ("train", "validation", "test"):
            c = verification["class_counts"][k]
            print(
                f"[split] {k}: negative={c['negative']:,} "
                f"positive={c['positive']:,} prevalence={c['prevalence']:.6f}"
            )
        print(
            f"[split] coverage={verification['complete_coverage']} "
            f"disjoint={verification['pairwise_disjoint']} "
            f"reproducible={verification['reproducible_from_seed']}"
        )
        print(f"[split] split_hash_sha256={split_sha}")
        for rname, row in overlap.items():
            print(
                f"[overlap] {rname}: chronological_train={row['chronological_train']:,} "
                f"chronological_validation={row['chronological_validation']:,} "
                f"chronological_test={row['chronological_test']:,}"
            )

        split_doc = build_split_document(
            split, verification, overlap, dataset_sha, split_sha, temporal_meta
        )
        (OUT / "random_split_v1.json").write_text(
            json.dumps(split_doc, indent=2) + "\n", encoding="utf-8"
        )
        (OUT / "random_split_v1.md").write_text(
            split_markdown(split_doc) + "\n", encoding="utf-8"
        )

        # ------------------------------------------------------------------
        # 3. Prepare features (frozen guided definitions)
        # ------------------------------------------------------------------
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

        X_train = data.iloc[split["train"]].reset_index(drop=True)
        X_test = data.iloc[split["test"]].reset_index(drop=True)
        y_train = y[split["train"]]
        y_test = y[split["test"]]

        print(
            f"[fit] random training block only: n={len(X_train):,}; "
            f"random validation block recorded but NOT used "
            f"(n={len(split['validation']):,})"
        )

        # ------------------------------------------------------------------
        # 4. Fit A, B, C and evaluate the random test block once
        # ------------------------------------------------------------------
        predictions: dict[str, np.ndarray] = {}
        metric_rows: list[dict] = []
        calibration_frames: list[pd.DataFrame] = []
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
                    "split": "random",
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
                f"[model {name}] C={SELECTED_C[name]} roc_auc={metric_rows[-1]['roc_auc']:.6f} "
                f"ap={metric_rows[-1]['pr_auc']:.6f} brier={metric_rows[-1]['brier']:.6f} "
                f"lift@10%={metric_rows[-1]['lift']:.4f}"
            )

        metrics_df = pd.DataFrame(metric_rows)
        metrics_df.to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
        pd.concat(calibration_frames, ignore_index=True).to_csv(
            OUT / "calibration.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(
            {
                "row_index": split["test"],
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

        # ------------------------------------------------------------------
        # 5. Descriptive comparison with the chronological result
        # ------------------------------------------------------------------
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
                    "chronological_lift": float(c["lift"]),
                    "random_lift": r["lift"],
                    "delta_lift_random_minus_chronological": r["lift"]
                    - float(c["lift"]),
                }
            )
        comparison_df = pd.DataFrame(comparison_rows)

        # ------------------------------------------------------------------
        # 6. Human-readable report
        # ------------------------------------------------------------------
        report = build_report(
            split_doc,
            metric_rows,
            boot,
            comparison_rows,
            chrono_boot,
            temporal_meta,
            model_calls,
        )
        (OUT / "report.md").write_text(report, encoding="utf-8")

        # ------------------------------------------------------------------
        # 7. Logs
        # ------------------------------------------------------------------
        (OUT / "prompt_log.md").write_text(build_prompt_log(), encoding="utf-8")
        (OUT / "execution_log.md").write_text(
            build_execution_log(
                split_doc, metric_rows, boot, model_calls, before_snapshot
            ),
            encoding="utf-8",
        )

        # ------------------------------------------------------------------
        # 8. Preservation check
        # ------------------------------------------------------------------
        after_snapshot = snapshot_files()
        integrity = compare_snapshots(before_snapshot, after_snapshot)
        print(
            f"[integrity] files before={integrity['n_files_before']} "
            f"after={integrity['n_files_after']} "
            f"unexpected_modification={integrity['unexpected_modification']}"
        )
        if integrity["unexpected_modification"]:
            print(
                "[integrity] changed="
                f"{integrity['changed_files']} added={integrity['added_files']} "
                f"removed={integrity['removed_files']}"
            )
            raise SystemExit(
                "FATAL: existing artifacts changed during the sensitivity run"
            )

        # ------------------------------------------------------------------
        # 9. Manifest (written last; console hash included)
        # ------------------------------------------------------------------
        print("[sensitivity] complete; artifacts written to results/sensitivity/")
        tee.flush()
        artifact_hashes = {
            p.name: sha256_file(p)
            for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name != "run_manifest_sensitivity.json"
        }
        manifest = {
            "run_mode": "guided",
            "stage": "random-split sensitivity analysis (secondary, post-test)",
            "script": SCRIPT_REL,
            "script_sha256": sha256_file(SCRIPT_PATH),
            "generated_utc": utc_now(),
            "plan_reference": "plan.md section 5.5 (secondary sensitivity split)",
            "analysis_role": (
                "secondary post-test sensitivity analysis; not an independent "
                "confirmation and not a replacement for the primary chronological "
                "evaluation"
            ),
            "dataset_sha256": dataset_sha,
            "chronological_split_sha256": EXPECTED_TEMPORAL_SPLIT_SHA256,
            "random_split_sha256": split_sha,
            "random_split_sizes": {
                "train": N_TRAIN,
                "validation": N_VAL,
                "test": N_TEST,
            },
            "random_split_class_counts": split_doc["class_counts"],
            "random_split_verification": split_doc["verification"],
            "overlap_with_chronological_blocks": split_doc[
                "overlap_with_chronological_blocks"
            ],
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
            "fitted_on": "random training block only",
            "random_validation_used_for_fitting_or_tuning": False,
            "random_test_evaluated": True,
            "random_test_labels_used_for": "evaluation only",
            "random_test_labels_used_for_fitting_or_tuning": False,
            "model_calls": model_calls,
            "static_audit": static_audit(),
            "preservation": {
                **integrity,
                "baseline_note": (
                    "Baseline excludes .git, .venv, __pycache__, data/raw and "
                    "results/sensitivity. The raw dataset is verified by its "
                    "recorded SHA-256."
                ),
            },
            "artifacts": artifact_hashes,
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
        }
        (OUT / "run_manifest_sensitivity.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        tee.close()
        return 0
    except BaseException:
        tee.close()
        raise


# ---------------------------------------------------------------------------
# Report / log text
# ---------------------------------------------------------------------------
def build_report(
    split_doc: dict,
    metric_rows: list[dict],
    boot: dict,
    comparison_rows: list[dict],
    chrono_boot: dict,
    temporal_meta: dict,
    model_calls: list[dict],
) -> str:
    cc = split_doc["class_counts"]
    ov = split_doc["overlap_with_chronological_blocks"]
    rnd_test_prev = cc["test"]["prevalence"]
    chrono_test_prev = 2_669 / 6_783

    metric_table = "\n".join(
        f"| {r['model']} | {r['role']} | {r['roc_auc']:.6f} | "
        f"{r['pr_auc']:.6f} | {r['pr_auc_no_skill']:.6f} | {r['brier']:.6f} | "
        f"{r['lift']:.4f} | {r['selected_C']} |"
        for r in metric_rows
    )
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
    overlap_table = "\n".join(
        f"| {rname} | {row['chronological_train']:,} | "
        f"{row['chronological_validation']:,} | {row['chronological_test']:,} |"
        for rname, row in ov.items()
    )

    ba = boot["B-A"]
    cb = boot["C-B"]

    return f"""# Guided random-split sensitivity analysis — report

Script: `{SCRIPT_REL}`
Generated (UTC): {split_doc['created_utc']}
Plan reference: `plan.md` §5.5 (secondary sensitivity split)

**Status: secondary post-test sensitivity analysis.** This is not a new
independent confirmation, not a replacement for the primary chronological
evaluation, and not a basis for changing the primary conclusion. The random
split is **optimistic** because it mixes temporal information between training
and evaluation.

## 1. Split

| Item | Value |
|---|---|
| Method | stratified random 70/15/15, `train_test_split`, seed {SEED} |
| Dataset SHA-256 | `{split_doc['input_sha256']}` |
| Split SHA-256 | `{split_doc['split_hash_sha256']}` |
| Sizes | train {cc['train']['n']:,} / validation {cc['validation']['n']:,} / test {cc['test']['n']:,} |
| Test positives | {cc['test']['positive']:,} / {cc['test']['n']:,} |
| Test prevalence | {cc['test']['prevalence']:.6f} |
| Complete coverage | {split_doc['verification']['complete_coverage']} |
| Pairwise disjoint | {split_doc['verification']['pairwise_disjoint']} |
| Reproducible from seed | {split_doc['verification']['reproducible_from_seed']} |

Full indices and generation details are in `results/sensitivity/random_split_v1.json`;
a human-readable summary is in `results/sensitivity/random_split_v1.md`.

**Stratification disclosure.** The target labels are used **only** to assign
split membership. They are not used for feature construction, preprocessing,
hyperparameter selection, calibration or model selection.

## 2. Overlap with the primary chronological blocks

| Random block | Chronological train [0, 31,647) | Chronological validation [31,647, 38,428) | Chronological test [38,428, 45,211) |
|---|---|---|---|
{overlap_table}

Every random block draws rows from all chronological periods. This overlap is
the mechanism that makes the random-split estimate optimistic: rows that are
temporally adjacent can appear in both the random training and random test
blocks.

## 3. Frozen features, preprocessing and hyperparameters

Feature sets, codebook relabelling, `poutcome_derived` construction and the
`StandardScaler` + `OneHotEncoder(handle_unknown="ignore")` preprocessing are
unchanged from the guided primary analysis. Preprocessing is fitted on the
random training block only.

Frozen guided hyperparameters (no re-selection, no CV):

| Model | Role | C | penalty | solver | max_iter | random_state |
|---|---|---|---|---|---|---|
| A | customer characteristics only | 0.01 | l2 | lbfgs | 1000 | 42 |
| B | A + previous-campaign information | 0.01 | l2 | lbfgs | 1000 | 42 |
| C | **diagnostic — leakage; NOT deployable** | 0.001 | l2 | lbfgs | 1000 | 42 |

The random validation block (6,781 rows) was recorded but **not used** for
hyperparameter selection, calibration fitting, model selection or refitting.

## 4. Random-test metrics

Average-precision no-skill baseline is the **random-test prevalence**
({rnd_test_prev:.6f}), not the chronological-test prevalence and not the
full-dataset prevalence.

| Model | Role | ROC-AUC | Average precision | AP no-skill | Brier | Lift@10% | C |
|---|---|---|---|---|---|---|---|
{metric_table}

Model C is a **non-deployable leakage diagnostic**. Its higher scores measure
how much of the outcome is encoded in call `duration`, not a deployable gain.

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

## 6. Descriptive comparison with the chronological result

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

## 7. Prevalence sensitivity of AP and Brier

| Block | Prevalence | Positives / rows |
|---|---|---|
| Random train | {cc['train']['prevalence']:.6f} | {cc['train']['positive']:,} / {cc['train']['n']:,} |
| Random validation | {cc['validation']['prevalence']:.6f} | {cc['validation']['positive']:,} / {cc['validation']['n']:,} |
| Random test | {cc['test']['prevalence']:.6f} | {cc['test']['positive']:,} / {cc['test']['n']:,} |
| Chronological test | {chrono_test_prev:.6f} | 2,669 / 6,783 |

The random test prevalence ({rnd_test_prev:.6f}) is much lower than the
chronological test prevalence ({chrono_test_prev:.6f}). **Average precision and
Brier score are prevalence-sensitive.** AP has a no-skill baseline equal to the
evaluated set's prevalence, and Brier score depends on the base rate. Therefore
random-test AP and Brier are not directly comparable to chronological-test AP
and Brier, and the comparison table above is descriptive only.

## 8. Methodological decisions made after seeing primary test results

All of the following were fixed by the approved sensitivity protocol and
implemented without using the observed primary test results to choose a
feature, hyperparameter, split seed or model specification:

1. Random split treated as the pre-registered secondary sensitivity analysis
   (`plan.md` §5.5), not as a replacement for the primary analysis.
2. Two-step stratified `train_test_split` with seed 42 and exact sizes
   31,647 / 6,781 / 6,783.
3. Indices sorted ascending after generation; membership unchanged.
4. Original guided feature sets and preprocessing reused verbatim.
5. Original guided `C` values frozen (A 0.01, B 0.01, C 0.001); no re-selection
   or cross-validation.
6. Random validation block recorded but not used for fitting, tuning,
   calibration or model selection.
7. Same metrics, lift definition and 2,000-resample paired bootstrap (seed 42)
   as the primary evaluation.
8. Model C retained only as a labelled non-deployable leakage diagnostic.
9. No recalibration, threshold tuning or post-hoc feature change.
10. Comparison with the chronological result reported descriptively only.

## 9. Warnings and limitations

- **Optimistic estimate.** Random splitting mixes temporal information; the
  random-split metrics are not an estimate of future-period performance.
- **Not independent.** The random test rows overlap all chronological blocks,
  including the primary chronological test block.
- **Different prevalence.** Random-test prevalence is
  {rnd_test_prev:.6f} versus {chrono_test_prev:.6f} chronological; AP and Brier
  are prevalence-sensitive.
- **Stratification uses labels.** Labels assign split membership only; this is
  disclosed and does not affect feature construction or tuning.
- **Frozen C.** The random split was not used to re-select regularisation; the
  primary selected values were reused.
- **Model C is not deployable.** `duration` is post-contact information.
- **Single execution.** The analysis was executed once; no repeated random
  seeds were used, so split-to-split variability is not characterised.
- **No causal claims.** Nothing here supports a causal reading.

## 10. Files created

All files are under `results/sensitivity/`:

| File | Contents |
|---|---|
| `random_split_v1.json` | full indices, class counts, generation details, hash, overlap |
| `random_split_v1.md` | human-readable split summary |
| `metrics.csv` | one row per model, random-test metrics |
| `predictions.csv` | `row_index`, `y`, `p_A`, `p_B`, `p_C` for the 6,783 random-test rows |
| `calibration.csv` | decile calibration per model |
| `bootstrap.json` | 2,000-resample paired bootstrap results |
| `report.md` | this report |
| `run_manifest_sensitivity.json` | hashes, config, preservation check, environment |
| `console_output.txt` | captured console output |
| `prompt_log.md` | prompt and model-call log |
| `execution_log.md` | execution record |

Every number above comes from one of these files, produced by
`{SCRIPT_REL}`.
"""


def build_prompt_log() -> str:
    return """# Guided random-split sensitivity analysis — prompt log

Purpose: record the authorising prompt and the model calls for the guided
random-split sensitivity analysis (`AGENTS.md` rule 5, `plan.md` §7).

## Agent / session

- Agent/model: `opencode/big-pickle` ("Big Pickle"), OpenCode session.
- Workspace: `C:\\Users\\alber\\OneDrive\\Documenti\\my-analysis` (guided only).
- The autonomous workspace `my-analysis-autonomous` was **not** accessed.

## Authorising prompt (verbatim)

```
I approve the proposed RANDOM-SPLIT SENSITIVITY ANALYSIS protocol, with the methodological clarifications below.

You are authorized to IMPLEMENT AND EXECUTE the GUIDED sensitivity analysis only, in the original `my-analysis` workspace.

Follow the protocol you just proposed:

1. Generate one deterministic stratified random split across all 45,211 rows using `train_test_split`, seed 42, with exact sizes:
   - Training: 31,647
   - Validation: 6,781
   - Test: 6,783

2. Record the full split indices, class counts, split-generation details, dataset hash and a reproducible split hash. Verify complete coverage, disjointness and reproducibility.

3. Keep the original GUIDED feature sets and preprocessing exactly unchanged.

4. Freeze the original GUIDED hyperparameters:
   - Model A: C=0.01
   - Model B: C=0.01
   - Model C: C=0.001
   Preserve the original solver, regularization, max_iter and random_state settings.

5. Fit on the RANDOM TRAINING BLOCK ONLY. Do not use the random validation block for hyperparameter selection, calibration fitting, model selection or refitting.

6. Evaluate the random test block with the same metrics as the primary analysis: ROC-AUC, average precision, its prevalence baseline, Brier, lift@10%, calibration summary and 2,000 paired bootstrap comparisons with seed 42.

7. Include Model C only as a clearly labelled NON-DEPLOYABLE LEAKAGE DIAGNOSTIC.

8. Use a NEW script:
   `src/my_analysis/guided_random_split_sensitivity.py`

9. Save all results exclusively under:
   `results/sensitivity/`

   Include:
   - random_split_v1.json
   - random_split_v1.md
   - metrics.csv
   - predictions.csv
   - calibration.csv
   - bootstrap.json
   - report.md
   - run_manifest_sensitivity.json
   - console_output.txt
   - prompt_log.md
   - execution_log.md

10. Do not access or modify the autonomous workspace. The autonomous execution will be authorized separately.

11. Treat this as a SECONDARY POST-TEST SENSITIVITY ANALYSIS, not a new independent confirmation or replacement of the original chronological evaluation.

12. Explicitly disclose:
   - Overlap between the random blocks and primary chronological blocks.
   - That stratification uses the target labels solely to assign split membership.
   - Temporal information mixing caused by random splitting.
   - The different outcome prevalence of random vs chronological test.
   - That AP and Brier are prevalence-sensitive.
   - All post-test methodological decisions.

13. Do not use the previously observed primary test results to choose any sensitivity-analysis feature, hyperparameter, split seed or model specification.

14. Preserve every existing primary analysis artifact. Hash existing files before and after execution; report any unexpected modification.

15. If an integrity check fails, stop rather than silently changing the protocol.

Execute the guided sensitivity analysis once, verify its outputs and report:
- Execution status.
- Random split hash and class counts.
- Test metrics for Models A, B and C.
- B-minus-A bootstrap confidence interval.
- Comparison with the original chronological result, clearly marked descriptive only.
- Warnings, methodological limitations and files created.

Then STOP. Do not begin the autonomous sensitivity analysis.
```

## Model calls

No LLM calls are made by `src/my_analysis/guided_random_split_sensitivity.py`.
The machine-learning model calls are three frozen `LogisticRegression` fits
(A, B, C) on the random training block. Their exact configuration, training
size, test size and captured warnings are recorded in
`results/sensitivity/run_manifest_sensitivity.json` under `model_calls`, and in
`results/sensitivity/execution_log.md`.
"""


def build_execution_log(
    split_doc: dict,
    metric_rows: list[dict],
    boot: dict,
    model_calls: list[dict],
    before_snapshot: dict[str, str],
) -> str:
    cc = split_doc["class_counts"]
    model_lines = "\n".join(
        f"| {m['model']} | {m['estimator']} | {m['C']} | {m['penalty']} | "
        f"{m['solver']} | {m['max_iter']} | {m['random_state']} | "
        f"{m['n_train']:,} | {m['n_test']:,} |"
        for m in model_calls
    )
    metric_lines = "\n".join(
        f"| {r['model']} | {r['roc_auc']:.6f} | {r['pr_auc']:.6f} | "
        f"{r['pr_auc_no_skill']:.6f} | {r['brier']:.6f} | {r['lift']:.4f} |"
        for r in metric_rows
    )
    warning_lines = "\n".join(
        f"- model {m['model']}: {w}" for m in model_calls for w in m["warnings"]
    ) or "- none"

    return f"""# Guided random-split sensitivity analysis — execution log

Append-only execution record. This file belongs to the guided workspace only.

## {split_doc['created_utc']} — one-shot sensitivity execution

### Scope

Secondary post-test sensitivity analysis using a stratified random split. The
primary chronological evaluation and every existing artifact were preserved.
The autonomous workspace was not accessed.

### Approvals in force

The authorising prompt is recorded verbatim in
`results/sensitivity/prompt_log.md`. It authorises the guided sensitivity
analysis only and requires the analysis to stop before the autonomous run.

### Files read

- `data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet` (verified by SHA-256)
- `results/splits/temporal_split_v1.json` (verified by SHA-256)
- `results/guided/test/test_metrics.csv` (descriptive comparison only)
- `results/guided/test/hypothesis_tests_test.json` (descriptive comparison only)

### Command

```
$env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m my_analysis.guided_random_split_sensitivity
```

### Split

- Method: `train_test_split`, stratified on `y`, seed {SEED}.
- Sizes: train {cc['train']['n']:,} / validation {cc['validation']['n']:,} / test {cc['test']['n']:,}.
- Class counts: train {cc['train']['negative']:,}/{cc['train']['positive']:,};
  validation {cc['validation']['negative']:,}/{cc['validation']['positive']:,};
  test {cc['test']['negative']:,}/{cc['test']['positive']:,} (negative/positive).
- Test prevalence: {cc['test']['prevalence']:.6f}.
- Split SHA-256: `{split_doc['split_hash_sha256']}`.
- Complete coverage: {split_doc['verification']['complete_coverage']};
  pairwise disjoint: {split_doc['verification']['pairwise_disjoint']};
  reproducible from seed: {split_doc['verification']['reproducible_from_seed']}.

### Model calls (frozen settings)

| Model | Estimator | C | penalty | solver | max_iter | random_state | n train | n test |
|---|---|---|---|---|---|---|---|---|
{model_lines}

Models were fitted on the random training block only. The random validation
block was recorded but not used for fitting, tuning, calibration, model
selection or refitting.

### Results

| Model | ROC-AUC | Average precision | AP no-skill | Brier | Lift@10% |
|---|---|---|---|---|---|
{metric_lines}

B − A ΔROC-AUC: point {boot['B-A']['delta_roc_auc']['point']:.6f}, 95% CI
[{boot['B-A']['delta_roc_auc']['ci95'][0]:.6f},
{boot['B-A']['delta_roc_auc']['ci95'][1]:.6f}].
B − A ΔAP: point {boot['B-A']['delta_pr_auc']['point']:.6f}, 95% CI
[{boot['B-A']['delta_pr_auc']['ci95'][0]:.6f},
{boot['B-A']['delta_pr_auc']['ci95'][1]:.6f}].
Bootstrap: {boot['n_boot']:,} paired resamples, seed {boot['seed']}, skipped
single-class resamples {boot['resamples_skipped_single_class']}.

### Warnings

{warning_lines}

### Preservation

Baseline files hashed before execution: {len(before_snapshot):,}. The run
compared every tracked file after execution; `run_manifest_sensitivity.json`
records the result. Any unexpected modification would have aborted the run.
"""


if __name__ == "__main__":
    sys.exit(main())
