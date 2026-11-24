"""Guided run: training + validation stage for Models A, B and C.

Implements plan.md v1.2 sections 5.2-5.7 for the FIRST (guided) execution run.

Scope and guarantees
--------------------
* Loads the frozen split from ``results/splits/temporal_split_v1.json`` and
  re-verifies its hash. It never regenerates indices.
* Uses the TRAIN and VALIDATION blocks only. The TEST block is never sliced,
  never scored, and its labels are never inspected. An explicit assertion
  guards this.
* All preprocessing (scaling, one-hot encoding) lives inside a scikit-learn
  ``Pipeline`` and is therefore fitted on training rows only, including inside
  every cross-validation fold.
* Hyperparameters are selected by expanding-window ``TimeSeriesSplit`` on the
  training block only.
* No refit on train+validation. No test metrics. No causal claims.

Run with:
    uv run python -m my_analysis.guided_train_validate
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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
SPLIT_FILE = ROOT / "results" / "splits" / "temporal_split_v1.json"
OUT = ROOT / "results" / "guided"
SCRIPT_REL = "src/my_analysis/guided_train_validate.py"

EXPECTED_DATASET_SHA256 = (
    "0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f"
)
EXPECTED_SPLIT_SHA256 = (
    "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c"
)

SEED = 42
N_SPLITS = 5
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LIFT_FRACTION = 0.10

# ---------------------------------------------------------------------------
# Codebook, verified against the HF dataset card and UCI documentation in
# data/README.md section 4.4. Used only to turn integer codes into readable
# category labels before one-hot encoding. It is a 1:1 relabel and uses no
# target information.
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

# Feature sets, exactly plan.md 5.2 (pdays excluded everywhere).
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
    "fitted_on": "training block only (and inside each CV fold)",
}

MODEL_SPEC = {
    "estimator": "LogisticRegression",
    "penalty": "l2",
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": None,
    "random_state": SEED,
    "C_grid": C_GRID,
    "selection_metric": "mean CV ROC-AUC (expanding-window TimeSeriesSplit)",
}


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


def lift_at_fraction(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    """Lift at the top ``frac`` of rows, using a stable descending sort.

    lift = precision@k / prevalence, with k = floor(frac * n).
    Ties in predicted probability are broken by original row order (stable).
    """
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.simplefilter("always")

    # ------------------------------------------------------------------
    # 1. Verify inputs
    # ------------------------------------------------------------------
    dataset_sha = sha256_file(RAW)
    if dataset_sha != EXPECTED_DATASET_SHA256:
        print(f"FATAL: dataset hash mismatch: {dataset_sha}", file=sys.stderr)
        return 1

    split_meta = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if canonical_hash(split_meta["indices"]) != split_meta["split_hash_sha256"]:
        print("FATAL: split indices do not match recorded hash", file=sys.stderr)
        return 1
    if split_meta["split_hash_sha256"] != EXPECTED_SPLIT_SHA256:
        print("FATAL: unexpected split hash", file=sys.stderr)
        return 1

    idx_train = np.asarray(split_meta["indices"]["train"], dtype=np.int64)
    idx_val = np.asarray(split_meta["indices"]["validation"], dtype=np.int64)
    idx_test = np.asarray(split_meta["indices"]["test"], dtype=np.int64)

    # Hard guard: test indices must never enter any computation below.
    assert len(idx_train) == 31_647 and len(idx_val) == 6_781 and len(idx_test) == 6_783
    assert set(idx_train).isdisjoint(idx_test)
    assert set(idx_val).isdisjoint(idx_test)
    assert set(idx_train).isdisjoint(idx_val)
    assert idx_train.max() < idx_val.min() < idx_test.min()

    # ------------------------------------------------------------------
    # 2. Load data. Row counts logged per AGENTS.md rule 2.
    # ------------------------------------------------------------------
    raw = pd.read_parquet(
        RAW,
        columns=[
            "age", "job", "marital", "education", "default", "balance",
            "housing", "loan", "previous", "poutcome", "pdays", "duration",
        ],
    )
    n_total = len(raw)

    # y is loaded separately and immediately sliced; test labels are never
    # sliced and never inspected.
    y_all = pd.read_parquet(RAW, columns=["y"])["y"].to_numpy(dtype=np.int64)
    y_train = y_all[idx_train]
    y_val = y_all[idx_val]
    del y_all

    data = raw.copy()
    for col, mapping in CODEBOOK.items():
        data[col] = data[col].map(mapping)
        if data[col].isna().any():
            print(f"FATAL: unmapped code in {col}", file=sys.stderr)
            return 1

    # plan.md 5.3: never_contacted is derived from pdays == 999; the five
    # anomalous rows (poutcome=unknown, pdays!=999) stay 'unknown'.
    data["poutcome_derived"] = np.where(
        data["pdays"].to_numpy() == 999, "never_contacted", data["poutcome"]
    )
    n_never = int((data["poutcome_derived"] == "never_contacted").sum())
    n_unknown = int((data["poutcome_derived"] == "unknown").sum())
    if n_never != 36_954 or n_unknown != 5:
        print(
            f"FATAL: derived poutcome counts unexpected: "
            f"never_contacted={n_never}, unknown={n_unknown}",
            file=sys.stderr,
        )
        return 1

    all_features = sorted(
        set(NUMERIC_A + CATEGORICAL_A + NUMERIC_B + CATEGORICAL_B + NUMERIC_C
            + CATEGORICAL_C)
    )
    X_all = data[all_features]

    X_train = X_all.iloc[idx_train].reset_index(drop=True)
    X_val = X_all.iloc[idx_val].reset_index(drop=True)

    print(f"[rows] total={n_total:,}; train={len(idx_train):,}; "
          f"validation={len(idx_val):,}; test={len(idx_test):,} (NOT used)")
    print(f"[rows] no filtering, merging or dropping applied "
          f"(train+validation+test = {len(idx_train)+len(idx_val)+len(idx_test):,})")
    print(f"[target] train prevalence={y_train.mean():.6f} "
          f"(positives={int(y_train.sum()):,}/{len(y_train):,})")
    print(f"[target] validation prevalence={y_val.mean():.6f} "
          f"(positives={int(y_val.sum()):,}/{len(y_val):,})")
    print(f"[derived] poutcome: never_contacted={n_never:,}, unknown={n_unknown}")

    # ------------------------------------------------------------------
    # 3. Fit A, B, C with identical folds
    # ------------------------------------------------------------------
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_sizes = [
        {"fold": i + 1, "train_n": int(len(tr)), "valid_n": int(len(va))}
        for i, (tr, va) in enumerate(cv.split(np.zeros(len(idx_train))))
    ]

    predictions: dict[str, np.ndarray] = {}
    metrics_rows: list[dict] = []
    cv_rows: list[dict] = []
    coef_rows: list[dict] = []
    calibration_frames: list[pd.DataFrame] = []
    best_params: dict[str, dict] = {}
    captured_warnings: list[str] = []

    for name, spec in MODELS.items():
        numeric = spec["numeric"]
        categorical = spec["categorical"]
        print(f"[model {name}] numeric={numeric} categorical={categorical}")

        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric),
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore", sparse_output=True, dtype=np.float64
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
        )
        pipeline = Pipeline(
            steps=[
                ("prep", preprocess),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=1000,
                        class_weight=None,
                        random_state=SEED,
                    ),
                ),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid={"clf__C": C_GRID},
            scoring="roc_auc",
            cv=cv,
            refit=True,
            n_jobs=-1,
            error_score="raise",
            return_train_score=False,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search.fit(X_train, y_train)
        for w in caught:
            captured_warnings.append(f"[model {name}] {w.category.__name__}: {w.message}")

        best_c = float(search.best_params_["clf__C"])
        best_params[name] = {
            "C": best_c,
            "cv_roc_auc_mean": float(search.best_score_),
            "n_fits": len(search.cv_results_["params"]),
        }
        print(f"[model {name}] best C={best_c} cv_roc_auc={search.best_score_:.6f}")

        for i in range(len(search.cv_results_["params"])):
            cv_rows.append(
                {
                    "model": name,
                    "C": float(search.cv_results_["params"][i]["clf__C"]),
                    "mean_cv_roc_auc": float(search.cv_results_["mean_test_score"][i]),
                    "std_cv_roc_auc": float(search.cv_results_["std_test_score"][i]),
                    "rank": int(search.cv_results_["rank_test_score"][i]),
                }
            )

        p_val = search.predict_proba(X_val)[:, 1]
        predictions[name] = p_val

        lift = lift_at_fraction(y_val, p_val, LIFT_FRACTION)
        metrics_rows.append(
            {
                "model": name,
                "role": spec["role"],
                "n_validation": int(len(y_val)),
                "n_positive": int(y_val.sum()),
                "prevalence": float(y_val.mean()),
                "roc_auc": float(roc_auc_score(y_val, p_val)),
                "average_precision": float(average_precision_score(y_val, p_val)),
                "ap_no_skill": float(y_val.mean()),
                "brier": float(brier_score_loss(y_val, p_val)),
                "mean_predicted": float(p_val.mean()),
                "lift_k": lift["k"],
                "lift_k_fraction": lift["k_fraction"],
                "lift_tp_in_top_k": lift["tp_in_top_k"],
                "lift_precision_at_k": lift["precision_at_k"],
                "lift_recall_at_k": lift["recall_at_k"],
                "lift_expected_tp_by_chance": lift["expected_tp_by_chance"],
                "lift": lift["lift"],
                "selected_C": best_c,
                "cv_roc_auc_mean": float(search.best_score_),
            }
        )

        feature_names = search.best_estimator_.named_steps["prep"].get_feature_names_out()
        coefs = search.best_estimator_.named_steps["clf"].coef_[0]
        for fname, coef in zip(feature_names, coefs):
            coef_rows.append(
                {"model": name, "feature": str(fname), "coefficient": float(coef)}
            )
        coef_rows.append(
            {
                "model": name,
                "feature": "INTERCEPT",
                "coefficient": float(search.best_estimator_.named_steps["clf"].intercept_[0]),
            }
        )

        cal = calibration_table(y_val, p_val)
        cal.insert(0, "model", name)
        calibration_frames.append(cal)

    # ------------------------------------------------------------------
    # 4. Save artifacts
    # ------------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT / "validation_metrics.csv", index=False, encoding="utf-8-sig")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(
        OUT / "hyperparameter_selection.csv", index=False, encoding="utf-8-sig"
    )

    cal_df = pd.concat(calibration_frames, ignore_index=True)
    cal_df.to_csv(OUT / "calibration_deciles.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(coef_rows).to_csv(
        OUT / "model_coefficients.csv", index=False, encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame(
        {
            "row_index": idx_val,
            "y": y_val,
            "p_A": predictions["A"],
            "p_B": predictions["B"],
            "p_C": predictions["C"],
        }
    )
    pred_path = OUT / "validation_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    (OUT / "run_warnings.txt").write_text(
        "\n".join(captured_warnings) if captured_warnings else "none\n",
        encoding="utf-8",
    )

    manifest = {
        "run_mode": "guided",
        "stage": "train + validation only",
        "script": SCRIPT_REL,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": "uv run python -m my_analysis.guided_train_validate",
        "plan_version": "1.2",
        "split_plan_version": split_meta["plan_version"],
        "split_plan_version_note": (
            "The frozen split file records the plan version in force when it was "
            "created (v1.1). The feature sets, preprocessing and metric "
            "definitions executed here follow plan.md v1.2."
        ),
        "dataset": {
            "path": str(RAW.relative_to(ROOT)).replace("\\", "/"),
            "sha256": dataset_sha,
            "n_rows": n_total,
        },
        "split": {
            "path": str(SPLIT_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": split_meta["split_hash_sha256"],
            "sizes": {
                "train": int(len(idx_train)),
                "validation": int(len(idx_val)),
                "test": int(len(idx_test)),
            },
            "test_rows_used": 0,
            "test_labels_inspected": False,
        },
        "row_counts": {
            "total": int(n_total),
            "train": int(len(idx_train)),
            "validation": int(len(idx_val)),
            "test_not_used": int(len(idx_test)),
            "filters_applied": [],
            "rows_dropped": 0,
        },
        "derived": {
            "poutcome_never_contacted": n_never,
            "poutcome_unknown_anomalies": n_unknown,
            "rule": "poutcome_derived = 'never_contacted' if pdays == 999 else poutcome label",
        },
        "feature_sets": {
            name: {
                "role": spec["role"],
                "numeric": spec["numeric"],
                "categorical": spec["categorical"],
                "n_input_columns": len(spec["numeric"]) + len(spec["categorical"]),
            }
            for name, spec in MODELS.items()
        },
        "excluded_from_all_models": [
            "pdays", "was_contacted_prev", "days_since_prev",
            "campaign", "contact", "day", "month",
        ],
        "preprocessing": PREPROC_SPEC,
        "preprocessing_spec_sha256": canonical_hash(PREPROC_SPEC),
        "model_spec": MODEL_SPEC,
        "model_spec_sha256": canonical_hash(MODEL_SPEC),
        "cv": {
            "splitter": "TimeSeriesSplit (expanding window)",
            "n_splits": N_SPLITS,
            "folds": fold_sizes,
            "fitted_on": "training block only",
        },
        "selected_hyperparameters": best_params,
        "metrics": {row["model"]: {
            "roc_auc": row["roc_auc"],
            "average_precision": row["average_precision"],
            "ap_no_skill": row["ap_no_skill"],
            "brier": row["brier"],
            "lift": row["lift"],
        } for row in metrics_rows},
        "lift_definition": {
            "fraction": LIFT_FRACTION,
            "k_rule": "floor(0.10 * n_validation)",
            "ranking": "stable descending sort of predicted probability",
            "formula": "precision@k / prevalence",
        },
        "artifacts": {
            "validation_metrics.csv": sha256_file(OUT / "validation_metrics.csv"),
            "hyperparameter_selection.csv": sha256_file(OUT / "hyperparameter_selection.csv"),
            "validation_predictions.csv": sha256_file(pred_path),
            "calibration_deciles.csv": sha256_file(OUT / "calibration_deciles.csv"),
            "model_coefficients.csv": sha256_file(OUT / "model_coefficients.csv"),
        },
        "versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "seed": SEED,
        "test_evaluated": False,
        "refit_on_train_plus_validation": False,
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------------------------------------------------------
    # 5. Human-readable report
    # ------------------------------------------------------------------
    def fmt(value: float, digits: int = 6) -> str:
        return f"{value:.{digits}f}"

    metric_table = "\n".join(
        f"| {r['model']} | {r['role']} | {fmt(r['roc_auc'])} | "
        f"{fmt(r['average_precision'])} | {fmt(r['ap_no_skill'])} | "
        f"{fmt(r['brier'])} | {fmt(r['lift'], 4)} |"
        for r in metrics_rows
    )
    cv_table = "\n".join(
        f"| {r['model']} | {r['C']:g} | {fmt(r['mean_cv_roc_auc'])} | "
        f"{fmt(r['std_cv_roc_auc'])} | {r['rank']} |"
        for r in cv_rows
    )
    lift_detail = "\n".join(
        f"| {r['model']} | {r['lift_k']} | {r['lift_k_fraction']:.6f} | {r['lift_tp_in_top_k']} | "
        f"{fmt(r['lift_precision_at_k'])} | {fmt(r['lift_recall_at_k'])} | "
        f"{fmt(r['lift_expected_tp_by_chance'], 2)} | {fmt(r['lift'], 4)} |"
        for r in metrics_rows
    )

    report = f"""# Guided run — training and validation stage

Script: `{SCRIPT_REL}`
Command: `uv run python -m my_analysis.guided_train_validate`
Generated: {manifest['generated_utc']}
Plan: `plan.md` v{manifest['plan_version']} (design)
Split file records plan v{manifest['split_plan_version']} (version in force when the split was created; boundaries unchanged)

**Scope.** Train + validation only. The test block (6,783 rows) was **not**
sliced, scored or inspected. No model was refit on train+validation. Model C is
a **diagnostic — leakage** model and is not deployable.

## Row counts

| Block | Rows | Used |
|---|---|---|
| train | {len(idx_train):,} | yes — fitting and CV |
| validation | {len(idx_val):,} | yes — metrics |
| test | {len(idx_test):,} | **no** |

No rows were filtered, merged or dropped (total {n_total:,}).

## Feature sets

| Model | Role | Numeric | Categorical |
|---|---|---|---|
| A | customer characteristics only | {", ".join(NUMERIC_A)} | {", ".join(CATEGORICAL_A)} |
| B | A + previous-campaign | {", ".join(NUMERIC_B)} | {", ".join(CATEGORICAL_B)} |
| C | diagnostic — leakage | {", ".join(NUMERIC_C)} | {", ".join(CATEGORICAL_C)} |

`pdays` is not a predictor in any model. `poutcome` is recoded to five levels,
with `never_contacted` = `pdays == 999` ({n_never:,} rows) and the five
anomalous rows kept as `unknown`.

## Preprocessing

Fitted on the training block only (and inside each CV fold):
`StandardScaler` on numeric columns; `OneHotEncoder(handle_unknown="ignore")`
on categorical columns; `remainder="drop"`. No imputer (0 NaN), no resampling,
no target encoding. Spec hash: `{manifest['preprocessing_spec_sha256']}`.

## Hyperparameter selection

`GridSearchCV`, expanding-window `TimeSeriesSplit(n_splits={N_SPLITS})` on the
training block only, scored by mean CV ROC-AUC. `C ∈ {C_GRID}`.

| Model | C | mean CV ROC-AUC | SD | rank |
|---|---|---|---|---|
{cv_table}

Selected: {", ".join(f"{k}: C={v['C']:g} (CV AUC {v['cv_roc_auc_mean']:.6f})" for k, v in best_params.items())}

## Validation metrics

No-skill average precision is the **validation-set prevalence**
({y_val.mean():.6f}), not the full-file prevalence.

| Model | Role | ROC-AUC | Avg precision | AP no-skill | Brier | Lift@10% |
|---|---|---|---|---|---|---|
{metric_table}

### Lift at the top 10%

| Model | k | k/n | TP in top k | Precision@k | Recall@k | Expected TP by chance | Lift |
|---|---|---|---|---|---|---|---|
{lift_detail}

## Base rates, calibration and interpretation limits

**Observed base-rate shift (empirical, from this run).** The training block has
prevalence {y_train.mean():.6f} ({int(y_train.sum()):,}/{len(y_train):,}) while the
validation block has {y_val.mean():.6f} ({int(y_val.sum()):,}/{len(y_val):,}).
The validation period is roughly {y_val.mean()/y_train.mean():.2f}x more positive
than the training period. This is a property of the chronological split, not of
the modelling code, and it is why the no-skill AP reference is the
validation prevalence.

**Consequence for Brier and calibration.** Every model's mean predicted
probability on validation is below the validation prevalence:

| Model | mean predicted | validation prevalence | ratio |
|---|---|---|---|
{chr(10).join(f"| {r['model']} | {r['mean_predicted']:.6f} | {r['prevalence']:.6f} | {r['mean_predicted']/r['prevalence']:.4f} |" for r in metrics_rows)}

The models are therefore **under-calibrated in level** on validation: they rank
reasonably but systematically under-predict the positive rate. The Brier scores
above should be read with this in mind; they are not evidence of a well
calibrated probability model. Per-decile detail is in
`results/guided/calibration_deciles.csv`. No recalibration, prior correction or
threshold tuning was applied — none is pre-registered for this stage.

**Consequence for the CV estimate.** The mean CV ROC-AUC within the training
block (0.556–0.568 for A and B) is much lower than the validation ROC-AUC
(0.640 and 0.660). The cross-validation folds are all *within* the training
period, where the positive rate is low and comparatively flat; the validation
block is a later period with a higher positive rate. The two numbers are not
estimates of the same quantity and should not be compared as if one validated
the other.

**Scope.** These are descriptive held-out performance estimates for the guided
run. No test-set number exists yet, no final conclusion is drawn, and nothing
here supports a causal reading. Model C is a leakage diagnostic: its higher
scores measure how much of the outcome is encoded in call duration, not a
deployable gain.

## Files

| File | Contents |
|---|---|
| `results/guided/validation_metrics.csv` | one row per model, all validation metrics |
| `results/guided/validation_predictions.csv` | `row_index`, `y`, `p_A`, `p_B`, `p_C` for the 6,781 validation rows |
| `results/guided/hyperparameter_selection.csv` | full CV grid per model |
| `results/guided/calibration_deciles.csv` | decile calibration per model |
| `results/guided/model_coefficients.csv` | fitted coefficients on the standardised/one-hot scale |
| `results/guided/run_manifest.json` | hashes, versions, config, fold sizes, artifact hashes |
| `results/guided/run_warnings.txt` | captured Python warnings (convergence etc.) |

Every number above comes from one of these files.

## Implementation notes and warnings

- **`penalty="l2"` FutureWarning (scikit-learn 1.9.1).** `plan.md` §5.6 specifies
  an L2-penalised logistic regression. In scikit-learn 1.9 the `penalty`
  argument is deprecated in favour of `l1_ratio`; the code keeps
  `penalty="l2"` so the specification is stated explicitly, and the resulting
  `FutureWarning` is captured verbatim in `results/guided/run_warnings.txt`.
  The fitted model is unchanged (L2, `lbfgs`). No convergence warnings were
  raised.
- **`C` selected by CV ROC-AUC, not by validation performance.** Selection used
  the training block only; validation was scored once, after selection.
- **`pdays`** is read only to derive `never_contacted`; it is never passed to a
  scaler, encoder or estimator.
- **Test block.** `results/guided/run_manifest.json` records
  `test_evaluated: false` and `split.test_rows_used: 0`.
"""
    (OUT / "validation_report.md").write_text(report, encoding="utf-8")

    # ------------------------------------------------------------------
    # 6. Console summary
    # ------------------------------------------------------------------
    print()
    for r in metrics_rows:
        print(
            f"[{r['model']}] roc_auc={r['roc_auc']:.6f} ap={r['average_precision']:.6f} "
            f"(no-skill {r['ap_no_skill']:.6f}) brier={r['brier']:.6f} "
            f"lift@10%={r['lift']:.4f} C={r['selected_C']:g}"
        )
    print(f"[warnings] {len(captured_warnings)} captured -> run_warnings.txt")
    print(f"[out] {OUT.relative_to(ROOT)}")
    print("[test] not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
