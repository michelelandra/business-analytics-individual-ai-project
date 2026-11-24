"""Autonomous final evaluation: preflight reproduction + explicitly gated test mode.

Plan: ``plan.md`` v1.2 (frozen). This is a NEW file. The original autonomous
training/validation script and every artifact it produced are left untouched.

Two modes
---------
``--mode preflight`` (default, safe)
    Re-fits Models A, B and C with the FROZEN settings recorded by the original
    autonomous run (``results/autonomous/models/model_settings.json``) and checks
    that the regenerated validation predictions match the saved files
    ``results/autonomous/predictions/validation_predictions_{a,b,c}.csv``
    to <= 1e-12.

    Test LABELS are never read. The Parquet file has 46 row groups of 1,000
    rows; the training block ``[0, 31,647)`` lies entirely inside row groups
    0-31, so the ``y`` column is read only from those groups (rows 0-31,999;
    highest label row index read = 31,999, before the test block starts at
    38,428). Validation FEATURES need row group 38 (rows 38,000-38,999), which
    physically contains 572 test feature rows; they are sliced away before use
    and never inspected. Validation targets are taken from the saved
    prediction artifacts, not from the Parquet ``y`` column.

``--mode test`` (GATED, NOT EXECUTED in the current stage)
    Requires ``--confirm-test-evaluation <token>``. Runs the preflight check
    first (fail-closed), then opens the frozen test block ONCE, computes the
    pre-registered metrics and the 2,000-resample paired bootstrap, and writes
    ``results/autonomous/test/``. It performs no hyperparameter search and
    changes no setting based on any outcome.

Run preflight with (from repo root):
    $env:PYTHONPATH = "src"
    .venv\\Scripts\\python.exe -m my_analysis.autonomous_test_evaluate --mode preflight

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
from sklearn.calibration import calibration_curve
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
PRED_DIR = ROOT / "results" / "autonomous" / "predictions"
SETTINGS_FILE = ROOT / "results" / "autonomous" / "models" / "model_settings.json"
PREFLIGHT_OUT = ROOT / "results" / "autonomous" / "preflight"
TEST_OUT = ROOT / "results" / "autonomous" / "test"
SCRIPT_REL = "src/my_analysis/autonomous_test_evaluate.py"

EXPECTED_DATASET_SHA256 = (
    "0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f"
)
EXPECTED_SPLIT_SHA256 = (
    "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c"
)

SEED = 42
MAX_ITER = 5000
N_BOOT = 2000
LIFT_FRACTION = 0.10
TOL = 1e-12
TEST_TOKEN = "AUTONOMOUS_TEST_AUTHORISED_BY_OWNER"

# Frozen selected hyperparameters from the original autonomous run.
# These are CONSTANTS. No grid search and no re-selection happens here.
SELECTED_C: dict[str, float] = {"A": 0.01, "B": 0.01, "C": 0.01}

# Frozen block boundaries (plan.md 5.5 / temporal_split_v1.json).
TRAIN_END = 31_647
VAL_END = 38_428
TEST_END = 45_211
ROW_GROUP_SIZE = 1_000

# HF ClassLabel name lists, copied verbatim from
# src/my_analysis/autonomous_train_val.py. 1:1 relabel only; uses no target.
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

# Feature sets copied verbatim from the original autonomous run. Note that the
# six base categoricals are the INTEGER code columns (not the *_label strings),
# exactly as the original pipeline passed them to OneHotEncoder.
MODELS: dict[str, dict] = {
    "A": {
        "role": "primary comparison baseline",
        "numeric": ["age", "balance"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan"],
    },
    "B": {
        "role": "primary model under test (H1)",
        "numeric": ["age", "balance", "previous"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan", "poutcome_level"],
    },
    "C": {
        "role": "DIAGNOSTIC - LEAKAGE, never deployable",
        "numeric": ["age", "balance", "previous", "duration"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan", "poutcome_level"],
    },
}

EXPECTED_FEATURES = {
    "A": (["age", "balance"],
          ["job", "marital", "education", "default", "housing", "loan"]),
    "B": (["age", "balance", "previous"],
          ["job", "marital", "education", "default", "housing", "loan",
           "poutcome_level"]),
    "C": (["age", "balance", "previous", "duration"],
          ["job", "marital", "education", "default", "housing", "loan",
           "poutcome_level"]),
}

# Every column needed to rebuild the frozen feature frames plus the poutcome
# recode. The target column is read separately under the row-group policy.
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
    if split_meta["input_sha256"] != EXPECTED_DATASET_SHA256:
        raise SystemExit("FATAL: split file was built from a different dataset")

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
    label_groups = list(
        range(0, (TRAIN_END + ROW_GROUP_SIZE - 1) // ROW_GROUP_SIZE)
    )
    if (max(label_groups) + 1) * ROW_GROUP_SIZE > VAL_END:
        raise SystemExit(
            "FATAL: y row-group policy would read rows at or beyond the "
            "validation/test boundary; refusing to run"
        )

    feature_groups = list(
        range(0, (VAL_END + ROW_GROUP_SIZE - 1) // ROW_GROUP_SIZE)
    )
    X_all = read_row_groups(FEATURE_COLUMNS, feature_groups)
    if len(X_all) < VAL_END:
        raise SystemExit("FATAL: feature row groups did not cover validation")

    y_part = read_row_groups(["y"], label_groups)["y"].to_numpy(dtype=np.int64)
    if len(y_part) < TRAIN_END:
        raise SystemExit("FATAL: label row groups did not cover training")
    if len(y_part) > VAL_END:
        raise SystemExit("FATAL: label read crossed into the test block")

    y_train = y_part[:TRAIN_END]

    X_all = X_all.iloc[:VAL_END].reset_index(drop=True)
    X_train = X_all.iloc[:TRAIN_END].reset_index(drop=True)
    X_val = X_all.iloc[TRAIN_END:VAL_END].reset_index(drop=True)

    feature_rows_read = int((max(feature_groups) + 1) * ROW_GROUP_SIZE)
    test_feature_rows_loaded = int(max(0, feature_rows_read - VAL_END))
    policy = {
        "parquet_num_row_groups": int(pq.ParquetFile(RAW).metadata.num_row_groups),
        "row_group_size": ROW_GROUP_SIZE,
        "feature_row_groups_read": [int(feature_groups[0]), int(feature_groups[-1])],
        "feature_rows_physically_read": feature_rows_read,
        "test_feature_rows_physically_loaded": test_feature_rows_loaded,
        "test_feature_rows_used": 0,
        "label_row_groups_read": [int(label_groups[0]), int(label_groups[-1])],
        "label_rows_physically_read": int(len(y_part)),
        "label_rows_used": int(TRAIN_END),
        "label_max_row_index_read": int(len(y_part) - 1),
        "test_block_start": VAL_END,
        "test_labels_read": False,
        "validation_targets_source": (
            "results/autonomous/predictions/validation_predictions_{a,b,c}.csv"
        ),
        "note": (
            "Validation FEATURES require row group 38 (rows 38,000-38,999), "
            "which physically contains 572 test feature rows (rows "
            "38,428-38,999). Parquet cannot read a partial row group, so those "
            "rows are loaded and immediately sliced away; they are never "
            "inspected or used. No test LABEL is read: the y column is read "
            "only from row groups 0-31 (rows 0-31,999), entirely before the "
            "test block start at 38,428."
        ),
    }
    return X_train, X_val, y_train, policy


# ---------------------------------------------------------------------------
# Preprocessing / model construction (frozen)
# ---------------------------------------------------------------------------
def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce autonomous_train_val.decode_categoricals exactly.

    The six base categoricals keep their integer codes (that is what the
    original MODELS lists reference); the *_label strings are created for
    fidelity but are not used as predictors. ``poutcome_level`` is the derived
    five-level string column that Models B and C use.
    """
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
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
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
        random_state=SEED,
    )
    return Pipeline([("prep", preprocess), ("clf", clf)])


def static_audit() -> dict:
    """Verify that this script performs no search and uses the frozen settings."""
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
    c_ok = SELECTED_C == {"A": 0.01, "B": 0.01, "C": 0.01}
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


def verify_saved_predictions() -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    saved: dict[str, pd.DataFrame] = {}
    expected_rows = np.arange(TRAIN_END, VAL_END)
    for name in ("A", "B", "C"):
        path = PRED_DIR / f"validation_predictions_{name.lower()}.csv"
        frame = pd.read_csv(path)
        if not np.array_equal(frame["row_index"].to_numpy(), expected_rows):
            raise SystemExit(
                f"FATAL: saved validation predictions for {name} do not match "
                "the frozen validation rows"
            )
        saved[name] = frame
    y_ref = saved["A"]["y"].to_numpy()
    for name in ("B", "C"):
        if not np.array_equal(saved[name]["y"].to_numpy(), y_ref):
            raise SystemExit(
                f"FATAL: validation targets in artifact {name} disagree with A"
            )
    return saved, y_ref


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def run_preflight() -> dict:
    PREFLIGHT_OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("always")

    split_meta, idx_train, idx_val, idx_test, dataset_sha, split_sha = verify_inputs()
    settings = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    for name in ("A", "B", "C"):
        recorded = settings["models"][name]
        if float(recorded["selected_C"]) != SELECTED_C[name]:
            raise SystemExit(
                f"FATAL: SELECTED_C[{name}] != recorded selected_C"
            )
        if list(recorded["numeric_features"]) != MODELS[name]["numeric"]:
            raise SystemExit(f"FATAL: numeric features differ for {name}")
        if list(recorded["categorical_features"]) != MODELS[name]["categorical"]:
            raise SystemExit(f"FATAL: categorical features differ for {name}")
        if int(recorded["max_iter"]) != MAX_ITER:
            raise SystemExit(f"FATAL: max_iter differs for {name}")
        if recorded["solver"] != "lbfgs" or recorded["penalty"] != "l2":
            raise SystemExit(f"FATAL: solver/penalty differ for {name}")

    X_train, X_val, y_train, policy = load_train_validation()
    data = prepare_frame(pd.concat([X_train, X_val], ignore_index=True))
    X_train_p = data.iloc[: len(X_train)].reset_index(drop=True)
    X_val_p = data.iloc[len(X_train):].reset_index(drop=True)

    saved, y_val_ref = verify_saved_predictions()

    results: dict[str, dict] = {}
    for name in ("A", "B", "C"):
        pipeline = build_pipeline(name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            pipeline.fit(X_train_p, y_train)
        p = pipeline.predict_proba(X_val_p)[:, 1]
        saved_p = saved[name][f"p_model_{name.lower()}"].to_numpy()
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
        "frozen_settings_source": {
            "file": "results/autonomous/models/model_settings.json",
            "sha256": sha256_file(SETTINGS_FILE),
            "selected_C": {
                m: settings["models"][m]["selected_C"] for m in ("A", "B", "C")
            },
            "max_iter": settings["models"]["A"]["max_iter"],
            "solver": settings["models"]["A"]["solver"],
            "penalty": settings["models"]["A"]["penalty"],
        },
        "saved_prediction_artifacts": {
            m: {
                "file": f"results/autonomous/predictions/validation_predictions_{m.lower()}.csv",
                "sha256": sha256_file(
                    PRED_DIR / f"validation_predictions_{m.lower()}.csv"
                ),
            }
            for m in ("A", "B", "C")
        },
        "validation_target_prevalence": float(y_val_ref.mean()),
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
    p = report["row_group_policy"]
    lines = [
        "# Autonomous preflight report — reproduction of validation predictions",
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
        "with the frozen selected `C` values and the original feature sets,",
        "categorical handling, preprocessing, solver and seed. They were then",
        "scored on the frozen validation block and compared with the saved",
        "prediction artifacts.",
        "",
        "Frozen source of truth: `results/autonomous/models/model_settings.json`",
        f"(SHA-256 `{report['frozen_settings_source']['sha256']}`), which records",
        f"selected C {report['frozen_settings_source']['selected_C']},",
        f"`max_iter={report['frozen_settings_source']['max_iter']}`,",
        f"`solver={report['frozen_settings_source']['solver']}`,",
        f"`penalty={report['frozen_settings_source']['penalty']}`.",
        "",
        "## What the preflight read (and did not read)",
        "",
        f"- Parquet row groups: {p['parquet_num_row_groups']} of "
        f"{p['row_group_size']:,} rows.",
        f"- Feature row groups read: {p['feature_row_groups_read']} "
        f"({p['feature_rows_physically_read']:,} rows).",
        f"- Label row groups read: {p['label_row_groups_read']} "
        f"({p['label_rows_physically_read']:,} rows; "
        f"{p['label_rows_used']:,} used for training).",
        f"- Highest row index read from the label column: "
        f"{p['label_max_row_index_read']:,} "
        f"(test block starts at {p['test_block_start']:,}).",
        f"- Validation targets: taken from "
        f"`{p['validation_targets_source']}`, not from the Parquet label column.",
        "",
        "**Data-isolation limitation, stated accurately.** Validation features",
        f"require row group 38 (rows 38,000-38,999), which physically contains",
        f"{p['test_feature_rows_physically_loaded']} test *feature* rows (rows",
        "38,428-38,999). Parquet cannot read a partial row group, so those rows",
        "were loaded and immediately sliced away; they were never inspected or",
        f"used ({p['test_feature_rows_used']} used). No test *label* was read: the",
        "`y` column was read only from row groups 0-31 (rows 0-31,999), entirely",
        "before the test block.",
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


def evaluate(y: np.ndarray, p: np.ndarray) -> dict:
    n = len(y)
    n_top = int(round(0.10 * n))
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


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
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


def paired_bootstrap(
    y: np.ndarray, preds: dict[str, np.ndarray],
    comparisons: list[tuple[str, str]],
) -> dict:
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
    report = run_preflight()
    if not report["overall_pass"]:
        raise SystemExit("FATAL: preflight reproduction failed; test block not opened")

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

        met = evaluate(y_test, p)
        met["model"] = name
        met["label"] = MODELS[name]["role"]
        met["selected_C"] = SELECTED_C[name]
        metric_rows.append(met)

        cal = calibration_table(y_test, p)
        cal.insert(0, "model", name)
        calibration_frames.append(cal)

    metrics_df = pd.DataFrame(metric_rows)
    cols = ["model", "label", "n", "positives", "prevalence", "roc_auc",
            "pr_auc", "pr_auc_no_skill_baseline", "brier", "lift_top10pct",
            "top_decile_precision", "n_top_decile", "selected_C"]
    metrics_df[cols].to_csv(TEST_OUT / "test_metrics.csv", index=False,
                            encoding="utf-8-sig")
    pd.concat(calibration_frames, ignore_index=True).to_csv(
        TEST_OUT / "test_calibration.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({
        "row_index": idx_test,
        "y": y_test,
        "p_model_a": predictions["A"],
        "p_model_b": predictions["B"],
        "p_model_c": predictions["C"],
    }).to_csv(TEST_OUT / "test_predictions.csv", index=False,
              encoding="utf-8-sig")

    boot = paired_bootstrap(
        y_test, predictions, comparisons=[("B", "A"), ("C", "B")]
    )
    h1 = classify_h1(
        boot["B-A"]["delta_roc_auc_point"],
        boot["B-A"]["delta_roc_auc_ci95"],
    )
    payload = {
        "evaluation_set": "test (frozen split [38428, 45211))",
        "n": int(len(y_test)),
        "n_boot": N_BOOT,
        "seed": SEED,
        "method": (
            "paired bootstrap over test rows; percentile 95% CI; identical "
            "resample indices reused across models and metrics"
        ),
        "comparisons": boot,
        "materiality_threshold_delta_auc": 0.01,
        "leakage_diagnostic_threshold_delta_auc_C_minus_B": 0.05,
        "H1_classification": h1,
        "note": "Model C is a leakage diagnostic only and is never operational.",
    }
    (TEST_OUT / "hypothesis_tests_test.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    manifest = {
        "run": "autonomous",
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
            "Test labels are read only at this gated stage, after the preflight "
            "reproduction passed. No test label influenced fitting, feature "
            "choice or C selection."
        ),
        "artifacts": {},
    }
    for path in sorted(TEST_OUT.glob("*")):
        if path.is_file() and path.name != "run_manifest_test.json":
            manifest["artifacts"][path.name] = sha256_file(path)
    (TEST_OUT / "run_manifest_test.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("[test] one-shot evaluation complete; artifacts written to results/autonomous/test/")
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
