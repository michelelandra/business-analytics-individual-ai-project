"""Guided final evaluation: preflight reproduction + explicitly gated test mode.

Plan: ``plan.md`` v1.2 (frozen). This script is a NEW file; the original
training/validation script and all its artifacts are left untouched.

Two modes
---------
``--mode preflight`` (default, safe)
    Re-fits Models A, B and C with the FROZEN settings recorded by the original
    guided run and checks that the regenerated validation predictions match
    ``results/guided/validation_predictions.csv`` to <= 1e-12.

    Test LABELS are never read. The Parquet file has 46 row groups of 1,000
    rows; the training block ``[0, 31,647)`` lies entirely inside row groups
    0-31, so the ``y`` column is read only from those groups. Validation
    FEATURES need row group 38 (rows 38,000-38,999), which physically contains
    572 test feature rows; they are sliced away before use and never
    inspected. No test label is loaded in preflight.

``--mode test`` (GATED, NOT EXECUTED in the current stage)
    Requires ``--confirm-test-evaluation <token>``. Runs the preflight check
    first (fail-closed), then opens the frozen test block ONCE, computes the
    pre-registered metrics and the 2,000-resample paired bootstrap, and writes
    ``results/guided/test/``. It performs no hyperparameter search and changes
    no setting based on any outcome.

Run preflight with (from repo root):
    uv run python -m my_analysis.guided_test_evaluate --mode preflight

The test mode is intentionally not run here.
"""

from __future__ import annotations

import argparse
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
import pyarrow as pa
import pyarrow.parquet as pq
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
SPLIT_FILE = ROOT / "results" / "splits" / "temporal_split_v1.json"
SAVED_VAL_PRED = ROOT / "results" / "guided" / "validation_predictions.csv"
PREFLIGHT_OUT = ROOT / "results" / "guided" / "preflight"
TEST_OUT = ROOT / "results" / "guided" / "test"
SCRIPT_REL = "src/my_analysis/guided_test_evaluate.py"

EXPECTED_DATASET_SHA256 = (
    "0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f"
)
EXPECTED_SPLIT_SHA256 = (
    "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c"
)

SEED = 42
MAX_ITER = 1000
LIFT_FRACTION = 0.10
N_BOOT = 2000
TOL = 1e-12
TEST_TOKEN = "GUIDED_TEST_AUTHORISED_BY_OWNER"

# Frozen selected hyperparameters from the original guided run.
# These are CONSTANTS. No grid search and no re-selection happens here.
SELECTED_C: dict[str, float] = {"A": 0.01, "B": 0.01, "C": 0.001}

# Frozen block boundaries (plan.md 5.5 / temporal_split_v1.json).
TRAIN_END = 31_647
VAL_END = 38_428
TEST_END = 45_211
ROW_GROUP_SIZE = 1_000

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


def read_row_groups(columns: list[str], group_indices: list[int]) -> pd.DataFrame:
    """Read only the requested Parquet row groups and concatenate them."""
    pf = pq.ParquetFile(RAW)
    tables = [pf.read_row_group(i, columns=columns) for i in group_indices]
    return pa.concat_tables(tables).to_pandas()


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------
def verify_inputs() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, str, str]:
    dataset_sha = sha256_file(RAW)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        raise SystemExit(f"FATAL: dataset hash mismatch: {dataset_sha}")

    split_meta = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if canonical_hash(split_meta["indices"]) != split_meta["split_hash_sha256"]:
        raise SystemExit("FATAL: split indices do not match recorded hash")
    if split_meta["split_hash_sha256"] != EXPECTED_SPLIT_SHA256:
        raise SystemExit("FATAL: unexpected split hash")

    idx_train = np.asarray(split_meta["indices"]["train"], dtype=np.int64)
    idx_val = np.asarray(split_meta["indices"]["validation"], dtype=np.int64)
    idx_test = np.asarray(split_meta["indices"]["test"], dtype=np.int64)

    if (len(idx_train), len(idx_val), len(idx_test)) != (TRAIN_END, 6_781, 6_783):
        raise SystemExit("FATAL: unexpected split sizes")
    if not (
        set(idx_train).isdisjoint(idx_test)
        and set(idx_val).isdisjoint(idx_test)
        and set(idx_train).isdisjoint(idx_val)
    ):
        raise SystemExit("FATAL: split blocks overlap")
    if not idx_train.max() < idx_val.min() < idx_test.min():
        raise SystemExit("FATAL: split blocks are not ordered")

    return split_meta, idx_train, idx_val, idx_test, dataset_sha, split_meta["split_hash_sha256"]


def load_train_validation() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict]:
    """Load train features/labels and validation features without test labels.

    y is read ONLY from row groups 0-31, i.e. rows [0, 32,000), which is
    entirely before the test block starts at 38,428. The 353 validation
    labels in [31,647, 32,000) are loaded but immediately discarded by the
    slice to the training block; no test label is read.
    """
    y_groups = list(range(0, (TRAIN_END + ROW_GROUP_SIZE - 1) // ROW_GROUP_SIZE))
    if (max(y_groups) + 1) * ROW_GROUP_SIZE > VAL_END:
        raise SystemExit(
            "FATAL: y row-group policy would read rows at or beyond the "
            "validation/test boundary; refusing to run"
        )

    feature_groups = list(range(0, (VAL_END + ROW_GROUP_SIZE - 1) // ROW_GROUP_SIZE))
    X_all = read_row_groups(FEATURE_COLUMNS, feature_groups)
    if len(X_all) < VAL_END:
        raise SystemExit("FATAL: feature row groups did not cover validation")

    y_part = read_row_groups(["y"], y_groups)["y"].to_numpy(dtype=np.int64)
    if len(y_part) < TRAIN_END:
        raise SystemExit("FATAL: label row groups did not cover training")
    if len(y_part) > VAL_END:
        raise SystemExit("FATAL: label read crossed into the test block")

    y_train = y_part[:TRAIN_END]

    X_all = X_all.iloc[:VAL_END].reset_index(drop=True)
    X_train = X_all.iloc[:TRAIN_END].reset_index(drop=True)
    X_val = X_all.iloc[TRAIN_END:VAL_END].reset_index(drop=True)

    policy = {
        "parquet_num_row_groups": int(pq.ParquetFile(RAW).metadata.num_row_groups),
        "row_group_size": ROW_GROUP_SIZE,
        "feature_row_groups_read": [int(feature_groups[0]), int(feature_groups[-1])],
        "feature_rows_physically_read": int((max(feature_groups) + 1) * ROW_GROUP_SIZE),
        "label_row_groups_read": [int(y_groups[0]), int(y_groups[-1])],
        "label_rows_physically_read": int(len(y_part)),
        "label_rows_used": int(TRAIN_END),
        "label_max_row_index_read": int(len(y_part) - 1),
        "test_block_start": VAL_END,
        "test_labels_read": False,
        "note": (
            "Validation FEATURES require row group 38 (rows 38,000-38,999), "
            "which physically contains 572 test feature rows. They are sliced "
            "away before use. No test LABEL is read: the y column is read only "
            "from row groups 0-31 (rows 0-31,999), entirely before 38,428."
        ),
    }
    return X_train, X_val, y_train, policy


# ---------------------------------------------------------------------------
# Preprocessing / model construction (frozen)
# ---------------------------------------------------------------------------
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


def static_audit() -> dict:
    """Verify that this script performs no search and uses the frozen features."""
    src = Path(__file__).read_text(encoding="utf-8")
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
            "The script constructs pipelines with fixed constants. It imports "
            "no GridSearchCV/TimeSeriesSplit and performs no model selection."
        ),
    }


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def run_preflight() -> dict:
    PREFLIGHT_OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("always")

    split_meta, idx_train, idx_val, idx_test, dataset_sha, split_sha = verify_inputs()
    X_train, X_val, y_train, policy = load_train_validation()
    data = prepare_frame(pd.concat([X_train, X_val], ignore_index=True))
    X_train_p = data.iloc[: len(X_train)].reset_index(drop=True)
    X_val_p = data.iloc[len(X_train):].reset_index(drop=True)

    saved = pd.read_csv(SAVED_VAL_PRED)
    expected_rows = np.arange(TRAIN_END, VAL_END)
    if not np.array_equal(saved["row_index"].to_numpy(), expected_rows):
        raise SystemExit("FATAL: saved validation predictions do not match frozen validation rows")

    results: dict[str, dict] = {}
    for name in ("A", "B", "C"):
        pipeline = build_pipeline(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train_p, y_train)
        p = pipeline.predict_proba(X_val_p)[:, 1]
        saved_p = saved[f"p_{name}"].to_numpy()
        diff = np.abs(p - saved_p)
        max_diff = float(diff.max())
        results[name] = {
            "selected_C": SELECTED_C[name],
            "n_train": int(len(X_train_p)),
            "n_validation": int(len(p)),
            "max_abs_prediction_diff": max_diff,
            "n_rows_diff_gt_tol": int((diff > TOL).sum()),
            "pass": bool(max_diff <= TOL),
            "warnings": [f"{w.category.__name__}: {w.message}" for w in caught],
        }
        print(
            f"[preflight] model {name}: C={SELECTED_C[name]} "
            f"max|diff|={max_diff:.3e} pass={max_diff <= TOL}"
        )

    audit = static_audit()
    passed = all(r["pass"] for r in results.values())
    report = {
        "mode": "preflight",
        "stage": "preparation and reproduction only",
        "test_evaluation_executed": False,
        "test_labels_read": False,
        "generated_utc": utc_now(),
        "script": SCRIPT_REL,
        "script_sha256": sha256_file(Path(__file__)),
        "plan_version": "1.2 (frozen)",
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "split_plan_version": split_meta.get("plan_version"),
        "tolerance": TOL,
        "overall_pass": bool(passed),
        "models": results,
        "row_group_policy": policy,
        "static_audit": audit,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
    }

    (PREFLIGHT_OUT / "preflight_checks.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (PREFLIGHT_OUT / "preflight_report.md").write_text(
        _preflight_markdown(report), encoding="utf-8"
    )

    if not passed:
        raise SystemExit("FATAL: preflight reproduction failed; test block not opened")

    print("[preflight] ALL CHECKS PASSED. Test mode was NOT executed.")
    return report


def _preflight_markdown(report: dict) -> str:
    m = report["models"]
    lines = [
        "# Guided preflight report — reproduction of validation predictions",
        "",
        f"Generated (UTC): {report['generated_utc']}",
        f"Script: `{report['script']}` (SHA-256 `{report['script_sha256']}`)",
        f"Plan: `plan.md` v{report['plan_version']}",
        f"Dataset SHA-256: `{report['dataset_sha256']}`",
        f"Split SHA-256: `{report['split_sha256']}`",
        "",
        "**Scope.** Preparation and reproduction only. The test block was not",
        "opened. No test metric, test prediction or test outcome was computed.",
        "",
        "## Reproduction of the saved validation predictions",
        "",
        f"Tolerance: max absolute difference <= {report['tolerance']:.0e}.",
        "",
        "| Model | Frozen C | n train | n validation | max abs diff | rows > tol | Result |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("A", "B", "C"):
        r = m[name]
        lines.append(
            f"| {name} | {r['selected_C']} | {r['n_train']:,} | "
            f"{r['n_validation']:,} | {r['max_abs_prediction_diff']:.3e} | "
            f"{r['n_rows_diff_gt_tol']} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        f"**Overall: {'PASS' if report['overall_pass'] else 'FAIL'}.**",
        "",
        "The models were re-fitted from scratch on the original training block",
        "with the frozen selected `C` values and the original preprocessing,",
        "solver and seed. They were then scored on the frozen validation block",
        "and compared with `results/guided/validation_predictions.csv`.",
        "",
        "## What the preflight read (and did not read)",
        "",
        f"- Parquet row groups: {report['row_group_policy']['parquet_num_row_groups']} "
        f"of {report['row_group_policy']['row_group_size']:,} rows.",
        f"- Feature row groups read: "
        f"{report['row_group_policy']['feature_row_groups_read']} "
        f"({report['row_group_policy']['feature_rows_physically_read']:,} rows).",
        f"- Label row groups read: "
        f"{report['row_group_policy']['label_row_groups_read']} "
        f"({report['row_group_policy']['label_rows_physically_read']:,} rows; "
        f"{report['row_group_policy']['label_rows_used']:,} used for training).",
        f"- Highest row index read from the label column: "
        f"{report['row_group_policy']['label_max_row_index_read']:,} "
        f"(test block starts at {report['row_group_policy']['test_block_start']:,}).",
        "",
        "**Technical limitation, stated accurately.** Validation features require",
        "row group 38 (rows 38,000–38,999), which physically contains 572 test",
        "*feature* rows. Parquet cannot read a partial row group, so those rows",
        "are loaded and immediately sliced away; they are never inspected or used.",
        "No test *label* is read: the `y` column is read only from row groups",
        "0–31 (rows 0–31,999), entirely before the test block.",
        "",
        "## Static audit (no new search, features or settings)",
        "",
        f"- No hyperparameter search: {report['static_audit']['no_hyperparameter_search']}",
        f"- Forbidden names found: {report['static_audit']['forbidden_names_found']}",
        f"- Feature sets match the frozen lists: "
        f"{report['static_audit']['feature_sets_match_frozen_lists']}",
        f"- Selected C matches the frozen values: "
        f"{report['static_audit']['selected_C_match_frozen_values']}",
        f"- Selected C: {report['static_audit']['selected_C']}",
        f"- `max_iter` = {report['static_audit']['max_iter']}, seed = "
        f"{report['static_audit']['seed']}",
        "",
        "## Environment",
        "",
    ]
    for k, v in report["environment"].items():
        lines.append(f"- {k}: {v}")
    lines += [
        "",
        "## Gated test mode",
        "",
        "`--mode test` exists but was **not executed**. It requires an explicit",
        "authorisation token and re-runs this preflight check (fail-closed)",
        "before opening the test block once.",
        "",
        "## Warnings captured during the refits",
        "",
    ]
    any_warn = False
    for name in ("A", "B", "C"):
        for w in m[name]["warnings"]:
            any_warn = True
            lines.append(f"- model {name}: {w}")
    if not any_warn:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gated test evaluation (implemented, NOT executed in the current stage)
# ---------------------------------------------------------------------------
def load_test_block() -> tuple[pd.DataFrame, np.ndarray]:
    """Read the frozen test block. Only called by the gated test mode."""
    first_group = VAL_END // ROW_GROUP_SIZE
    last_group = (TEST_END + ROW_GROUP_SIZE - 1) // ROW_GROUP_SIZE - 1
    groups = list(range(first_group, last_group + 1))

    X_all = read_row_groups(FEATURE_COLUMNS, groups)
    y_all = read_row_groups(["y"], groups)["y"].to_numpy(dtype=np.int64)
    offset = VAL_END - first_group * ROW_GROUP_SIZE
    X_test = X_all.iloc[offset:].reset_index(drop=True)
    y_test = y_all[offset:]
    if len(X_test) != 6_783 or len(y_test) != 6_783:
        raise SystemExit("FATAL: test block size mismatch")
    return X_test, y_test


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
            "paired bootstrap over test rows; percentile 95% CI; identical "
            "resample indices reused across models and metrics"
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
    }


def classify_h1(delta: float, ci: list[float]) -> str:
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


def run_test_evaluation() -> int:
    """GATED. Not executed in the current stage."""
    TEST_OUT.mkdir(parents=True, exist_ok=True)

    # Fail-closed: never open the test block unless reproduction passes.
    run_preflight()

    split_meta, idx_train, idx_val, idx_test, dataset_sha, split_sha = verify_inputs()
    X_train, X_val, y_train, _ = load_train_validation()
    data = prepare_frame(pd.concat([X_train, X_val], ignore_index=True))
    X_train_p = data.iloc[: len(X_train)].reset_index(drop=True)

    X_test, y_test = load_test_block()
    X_test_p = prepare_frame(X_test)

    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict] = []
    calibration_frames: list[pd.DataFrame] = []
    for name in ("A", "B", "C"):
        pipeline = build_pipeline(name)
        pipeline.fit(X_train_p, y_train)
        p = pipeline.predict_proba(X_test_p)[:, 1]
        predictions[name] = p

        lift = lift_at_fraction(y_test, p, LIFT_FRACTION)
        metric_rows.append(
            {
                "model": name,
                "role": MODELS[name]["role"],
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

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(TEST_OUT / "test_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        TEST_OUT / "test_calibration.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "row_index": idx_test,
            "y": y_test,
            "p_A": predictions["A"],
            "p_B": predictions["B"],
            "p_C": predictions["C"],
        }
    ).to_csv(TEST_OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")

    boot = paired_bootstrap(y_test, predictions)
    h1 = classify_h1(
        boot["B-A"]["delta_roc_auc"]["point"],
        boot["B-A"]["delta_roc_auc"]["ci95"],
    )
    boot["H1_classification"] = h1
    boot["H2_delta_roc_auc_CB"] = boot["C-B"]["delta_roc_auc"]["point"]
    boot["H2_threshold"] = 0.05
    boot["H2_material"] = bool(boot["C-B"]["delta_roc_auc"]["point"] >= 0.05)
    boot["note"] = (
        "Model C is a leakage diagnostic only and is never an operational model."
    )
    (TEST_OUT / "hypothesis_tests_test.json").write_text(
        json.dumps(boot, indent=2), encoding="utf-8"
    )

    manifest = {
        "run_mode": "guided",
        "stage": "final test evaluation (one-shot)",
        "script": SCRIPT_REL,
        "script_sha256": sha256_file(Path(__file__)),
        "generated_utc": utc_now(),
        "plan_version": "1.2 (frozen)",
        "dataset_sha256": dataset_sha,
        "split_sha256": split_sha,
        "selected_C": dict(SELECTED_C),
        "max_iter": MAX_ITER,
        "seed": SEED,
        "test_rows_used": int(len(y_test)),
        "test_evaluated": True,
        "test_labels_read": True,
        "test_labels_used_for": "evaluation only",
        "test_labels_used_for_fitting_or_tuning": False,
        "disclosure": (
            "Test labels were read only at this gated stage, after the preflight "
            "reproduction passed. The earlier autonomous run's claims that test "
            "labels were never read were incorrect; see the comparison audit. "
            "No test label influenced fitting, feature choice or C selection."
        ),
        "artifacts": {},
    }
    for path in sorted(TEST_OUT.glob("*")):
        if path.is_file() and path.name != "run_manifest_test.json":
            manifest["artifacts"][path.name] = sha256_file(path)
    (TEST_OUT / "run_manifest_test.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[test] one-shot evaluation complete; artifacts written to results/guided/test/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["preflight", "test"],
        default="preflight",
        help="preflight (default, safe) or test (gated; not run in this stage)",
    )
    parser.add_argument(
        "--confirm-test-evaluation",
        default=None,
        help="explicit authorisation token required by --mode test",
    )
    args = parser.parse_args(argv)

    if args.mode == "preflight":
        report = run_preflight()
        return 0 if report["overall_pass"] else 1

    if args.confirm_test_evaluation != TEST_TOKEN:
        print(
            "REFUSED: --mode test requires --confirm-test-evaluation with the "
            "authorisation token. No test data was opened.",
            file=sys.stderr,
        )
        return 2
    return run_test_evaluation()


if __name__ == "__main__":
    sys.exit(main())
