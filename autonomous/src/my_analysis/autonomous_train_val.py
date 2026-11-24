"""Autonomous training/validation analysis — Bank Marketing, plan.md v1.2.

Implements the pre-registered Models A, B, C on the FROZEN temporal split
(results/splits/temporal_split_v1.json), fitting on the training block only and
evaluating on the validation block only. The test block is never loaded and its
labels are never read; test-set evaluation is out of scope for this run.

Feature sets (plan.md §5.2, v1.2):
    A: age, job, marital, education, default, balance, housing, loan
    B: A + previous (numeric) + poutcome (5 levels, derived never_contacted)
    C: B + duration (LEAKAGE DIAGNOSTIC ONLY — never deployable)

`campaign`, `contact`, `day`, `month` are excluded everywhere. `pdays` is never
a predictor; it only constructs the `never_contacted` level of `poutcome`.

Run with (from repo root):
    $env:PYTHONPATH = "src"
    .venv\\Scripts\\python.exe -m my_analysis.autonomous_train_val
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
import pyarrow
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
SPLIT_JSON = ROOT / "results" / "splits" / "temporal_split_v1.json"
OUT = ROOT / "results" / "autonomous"
SCRIPT_REL = "src/my_analysis/autonomous_train_val.py"

SEED = 42
N_BOOT = 2000
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
N_CV_FOLDS = 5
MAX_ITER = 5000

EXPECTED_RAW_SHA256 = (
    "0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f"
)
EXPECTED_SPLIT_SHA256 = (
    "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c"
)

# HF ClassLabel name lists, exactly as verified in data/README.md §4.3-§4.4.
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

MODELS: dict[str, dict[str, list[str]]] = {
    "A": {
        "numeric": ["age", "balance"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan"],
    },
    "B": {
        "numeric": ["age", "balance", "previous"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan", "poutcome_level"],
    },
    "C": {
        "numeric": ["age", "balance", "previous", "duration"],
        "categorical": ["job", "marital", "education", "default", "housing",
                        "loan", "poutcome_level"],
    },
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def rows_report(label: str, n: int, reason: str) -> None:
    """AGENTS.md rule 2: report row counts after every filter/recode."""
    print(f"[rows] {label}: n={n:,} ({reason})")


def load_and_verify_split() -> dict:
    payload = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    indices = payload["indices"]
    canonical = json.dumps(indices, sort_keys=True, separators=(",", ":"))
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if recomputed != payload["split_hash_sha256"]:
        raise SystemExit("FATAL: split file internal hash mismatch")
    if recomputed != EXPECTED_SPLIT_SHA256:
        raise SystemExit("FATAL: split file hash != pre-registered hash")
    if payload["input_sha256"] != EXPECTED_RAW_SHA256:
        raise SystemExit("FATAL: split file was built from a different dataset")
    sizes = {k: len(v) for k, v in indices.items()}
    if sizes != {"train": 31647, "validation": 6781, "test": 6783}:
        raise SystemExit(f"FATAL: unexpected split sizes {sizes}")
    print(f"[split] temporal_split_v1 verified (sha256 {recomputed[:12]}...)")
    return payload


def decode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Decode the 6 label-encoded predictors via the verified HF name lists.

    Deterministic, row-wise, uses no target and no fitted statistic.
    """
    for col, names in CODEBOOK.items():
        if col == "poutcome":
            continue  # handled with the derived level below
        codes = df[col].to_numpy()
        if codes.min() < 0 or codes.max() >= len(names):
            raise SystemExit(f"FATAL: {col} code outside documented name list")
        df[col + "_label"] = [names[c] for c in codes]
    poc_names = CODEBOOK["poutcome"]
    poc = df["poutcome"].to_numpy()
    if poc.min() < 0 or poc.max() >= len(poc_names):
        raise SystemExit("FATAL: poutcome code outside documented name list")
    poc_label = np.array([poc_names[c] for c in poc], dtype=object)
    # plan.md §5.3: never_contacted iff pdays == 999 (sentinel); the 5 rows with
    # poutcome=unknown and pdays != 999 stay "unknown" (preserved, not merged).
    df["poutcome_level"] = np.where(
        df["pdays"].to_numpy() == 999, "never_contacted", poc_label
    )
    return df


def make_pipeline(model_key: str, C: float) -> Pipeline:
    spec = MODELS[model_key]
    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), spec["numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), spec["categorical"]),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        penalty="l2", C=C, solver="lbfgs", max_iter=MAX_ITER,
        random_state=SEED,
    )
    return Pipeline([("prep", ct), ("clf", clf)])


def select_C(X: pd.DataFrame, y: np.ndarray, model_key: str,
             folds) -> tuple[float, pd.DataFrame]:
    """Time-series CV of the L2 strength, training block only (plan §5.6)."""
    records = []
    for C in C_GRID:
        fold_scores = []
        for k, (tr, va) in enumerate(folds):
            pipe = make_pipeline(model_key, C)
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                pipe.fit(X.iloc[tr], y[tr])
                msg = "; ".join(str(w.message) for w in wlist)
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            fold_scores.append(roc_auc_score(y[va], p))
            if msg:
                print(f"[warn] C={C} fold={k} model={model_key}: {msg}")
        records.append({"model": model_key, "C": C,
                        **{f"fold{k + 1}": s for k, s in
                           enumerate(fold_scores)},
                        "mean_roc_auc": float(np.mean(fold_scores))})
    table = pd.DataFrame(records)
    best = table.loc[np.argmax(table["mean_roc_auc"].to_numpy())]
    return float(best["C"]), table


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


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10
                      ) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(
        y, p, n_bins=n_bins, strategy="uniform"
    )
    # bin membership recomputed for counts (uniform-width bins on [0, 1])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.minimum(np.digitize(p, edges[1:-1], right=True), n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    used = np.bincount(bins, weights=y, minlength=n_bins)
    pos = np.array([y[bins == b].sum() for b in range(n_bins)])
    keep = counts > 0
    out = pd.DataFrame({
        "bin": np.arange(n_bins)[keep],
        "n": counts[keep],
        "mean_predicted": prob_pred,
        "fraction_positive": prob_true,
        "positives_in_bin": pos[keep],
    })
    return out


def paired_bootstrap(y: np.ndarray, preds: dict[str, np.ndarray],
                     comparisons: list[tuple[str, str]]) -> dict:
    """2,000 paired bootstrap resamples, seed 42, percentile 95% CI."""
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


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics").mkdir(exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)

    warnings_file = OUT / "logs" / "warnings_and_errors.log"
    log_lines: list[str] = []

    # ---- 1. Integrity checks (before anything else) -------------------
    raw_sha = sha256_file(RAW)
    if raw_sha != EXPECTED_RAW_SHA256:
        raise SystemExit("FATAL: raw dataset hash mismatch")
    print(f"[hash] raw dataset sha256 verified: {raw_sha[:12]}...")
    split = load_and_verify_split()

    # ---- 2. Load data; never touch the test block ---------------------
    df = pd.read_parquet(RAW)
    rows_report("loaded full parquet", len(df), "no filter applied")
    if df.shape != (45211, 17):
        raise SystemExit(f"FATAL: unexpected shape {df.shape}")
    if df.isna().sum().sum() != 0:
        raise SystemExit("FATAL: unexpected NaN cells")
    if not set(df["y"].unique()) <= {0, 1}:
        raise SystemExit("FATAL: target codes not in {0, 1}")

    train_idx = np.array(split["indices"]["train"])
    val_idx = np.array(split["indices"]["validation"])
    test_present = len(split["indices"]["test"])
    rows_report("train slice", len(train_idx), "frozen split [0, 31647)")
    rows_report("validation slice", len(val_idx), "frozen split [31647, 38428)")
    print(f"[rows] test block: {test_present:,} indices exist in the frozen "
          f"split file but the block is NOT loaded and its labels are NEVER "
          f"read in this run (test-set evaluation out of scope)")

    df = decode_categoricals(df)
    rows_report("after categorical decoding + poutcome_level recode",
                len(df), "row-wise relabeling; 0 rows added or dropped")

    lv_counts = df["poutcome_level"].value_counts().to_dict()
    print(f"[recode] poutcome_level counts (full file): "
          f"{ {k: int(v) for k, v in sorted(lv_counts.items())} }")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    y_train = train_df["y"].to_numpy(dtype=int)
    y_val = val_df["y"].to_numpy(dtype=int)
    rows_report("train used for fitting", len(train_df),
                "labels present; preprocessing and CV fit here only")
    rows_report("validation used for evaluation", len(val_df),
                "no fitting, no tuning on validation")

    # ---- 3. Data-quality checks on the loaded splits only -------------
    anomalies_all = int(
        ((df["poutcome"] == 3) & (df["pdays"] != 999)).sum()
    )
    anomalies_loaded = int(
        ((pd.concat([train_df, val_df])["poutcome"] == 3)
         & (pd.concat([train_df, val_df])["pdays"] != 999)).sum()
    )
    sentinel_prev0 = bool(
        (((train_df["pdays"] == 999) == (train_df["previous"] == 0)).all())
        and (((val_df["pdays"] == 999) == (val_df["previous"] == 0)).all())
    )
    print(f"[check] anomaly rows (poutcome=unknown & pdays!=999): "
          f"full file={anomalies_all}, train+validation={anomalies_loaded} "
          f"(preserved as 'unknown', not merged/imputed/dropped)")
    print(f"[check] pdays==999 iff previous==0 in train and validation: "
          f"{sentinel_prev0}")
    if not sentinel_prev0:
        raise SystemExit("FATAL: sentinel consistency check failed")
    assert "duration" in MODELS["C"]["numeric"], "model C must add duration"
    for m, spec in MODELS.items():
        feats = spec["numeric"] + [c.replace("_level", "") for c
                                   in spec["categorical"]]
        assert "duration" not in feats or m == "C"
        assert "pdays" not in feats and "campaign" not in feats
        assert "contact" not in feats and "day" not in feats
        assert "month" not in feats
    print("[check] feature lists verified: duration only in C; pdays, "
          "campaign, contact, day, month in no model")

    # ---- 4. Hyperparameter selection: TimeSeriesCV on train only ------
    ts = TimeSeriesSplit(n_splits=N_CV_FOLDS)
    folds = list(ts.split(train_df))
    print(f"[cv] TimeSeriesSplit(n_splits={N_CV_FOLDS}) on the training block "
          f"only; identical folds for A/B/C")
    cv_tables = []
    chosen: dict[str, float] = {}
    fitted: dict[str, Pipeline] = {}
    for m in ("A", "B", "C"):
        C, table = select_C(train_df, y_train, m, folds)
        chosen[m] = C
        cv_tables.append(table)
        print(f"[cv] model {m}: mean CV ROC-AUC by C = "
              f"{ {float(r.C): round(r.mean_roc_auc, 6) for r in table.itertuples()} }"
              f" -> selected C={C}")
    pd.concat(cv_tables).to_csv(OUT / "models" / "cv_selection.csv",
                                index=False)

    # ---- 5. Final fit on the full training block ----------------------
    for m in ("A", "B", "C"):
        pipe = make_pipeline(m, chosen[m])
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            pipe.fit(train_df, y_train)
            for w in wlist:
                log_lines.append(f"final fit {m}: {w.category.__name__}: "
                                 f"{w.message}\n")
        fitted[m] = pipe
        feat_names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pipe.named_steps["clf"].coef_[0]
        pd.DataFrame({"feature": feat_names, "coefficient": coefs}).to_csv(
            OUT / "models" / f"coefficients_{m.lower()}.csv", index=False)
        print(f"[fit] model {m} fitted on train (n={len(train_df):,}), "
              f"C={chosen[m]}, {len(coefs)} coefficients saved")

    # ---- 6. Validation evaluation (one-shot) ---------------------------
    preds: dict[str, np.ndarray] = {}
    metrics_rows = []
    for m in ("A", "B", "C"):
        p = fitted[m].predict_proba(val_df)[:, 1]
        preds[m] = p
        met = evaluate(y_val, p)
        met["model"] = m
        met["label"] = ("A: customer characteristics only"
                        if m == "A" else
                        "B: A + previous-campaign information"
                        if m == "B" else
                        "C: B + duration (DIAGNOSTIC - LEAKAGE)")
        metrics_rows.append(met)
        pd.DataFrame({
            "row_index": val_idx,
            "y": y_val,
            "p_model_" + m.lower(): p,
        }).to_csv(OUT / "predictions" / f"validation_predictions_{m.lower()}.csv",
                  index=False)
        calibration_table(y_val, p).to_csv(
            OUT / "metrics" / f"validation_calibration_{m.lower()}.csv",
            index=False)
        print(f"[val] model {m}: ROC-AUC={met['roc_auc']:.6f} "
              f"PR-AUC={met['pr_auc']:.6f} (baseline {met['prevalence']:.6f}) "
              f"Brier={met['brier']:.6f} lift@10%={met['lift_top10pct']:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)
    cols = ["model", "label", "n", "positives", "prevalence", "roc_auc",
            "pr_auc", "pr_auc_no_skill_baseline", "brier", "lift_top10pct",
            "top_decile_precision", "n_top_decile"]
    metrics_df[cols].to_csv(OUT / "metrics" / "validation_metrics.csv",
                            index=False)

    # ---- 7. Paired bootstrap hypothesis statistics ---------------------
    boot = paired_bootstrap(y_val, preds, comparisons=[("B", "A"), ("C", "B")])
    payload = {
        "evaluation_set": "validation (frozen split [31647, 38428))",
        "n": int(len(y_val)),
        "n_boot": N_BOOT,
        "seed": SEED,
        "method": "paired bootstrap over validation rows; percentile 95% CI; "
                  "identical resample indices reused across models and metrics",
        "test_set_note": "plan.md 3.1-3.2 decision procedure applies to the "
                         "TEST set and is NOT executed in this run; all values "
                         "here are validation evidence only",
        "comparisons": boot,
        "materiality_threshold_delta_auc": 0.01,
        "leakage_diagnostic_threshold_delta_auc_C_minus_B": 0.05,
    }
    (OUT / "metrics" / "hypothesis_tests_validation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for key, d in boot.items():
        lo, hi = d["delta_roc_auc_ci95"]
        print(f"[boot] {key}: dROC-AUC={d['delta_roc_auc_point']:.6f} "
              f"95% CI [{lo:.6f}, {hi:.6f}]; "
              f"dPR-AUC={d['delta_pr_auc_point']:.6f} "
              f"95% CI {d['delta_pr_auc_ci95']}")

    # ---- 8. Train-set prevalence context (descriptive only) ------------
    train_prev = float(y_train.mean())
    print(f"[context] positive prevalence: train={train_prev:.6f} "
          f"({int(y_train.sum()):,}/{len(y_train):,}), "
          f"validation={float(y_val.mean()):.6f} "
          f"({int(y_val.sum()):,}/{len(y_val):,})")

    # ---- 9. Model settings + manifest ----------------------------------
    settings = {
        "run": "autonomous",
        "script": SCRIPT_REL,
        "agent_model": "GLM-5.3-Flash Free (opencode), OpenCode session",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "plan_version": "1.2 (frozen)",
        "split_file": "results/splits/temporal_split_v1.json",
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "raw_dataset_sha256": raw_sha,
        "rows_filtered_merged_dropped": 0,
        "seed": SEED,
        "random_elements": {
            "logistic_regression_random_state": SEED,
            "bootstrap_rng": "numpy.random.default_rng(42)",
            "TimeSeriesSplit": "deterministic, no randomness",
        },
        "models": {
            m: {
                "numeric_features": MODELS[m]["numeric"],
                "categorical_features": MODELS[m]["categorical"],
                "selected_C": chosen[m],
                "C_grid": C_GRID,
                "selection": f"mean ROC-AUC over {N_CV_FOLDS}-fold "
                             f"expanding-window TimeSeriesSplit on train; "
                             f"ties -> first grid value (smaller C)",
                "solver": "lbfgs",
                "max_iter": MAX_ITER,
                "penalty": "l2",
                "role": ("primary comparison baseline" if m == "A" else
                         "primary model under test (H1)" if m == "B" else
                         "DIAGNOSTIC - LEAKAGE, never deployable"),
            }
            for m in ("A", "B", "C")
        },
        "preprocessing": {
            "numeric": "StandardScaler (mean/SD from train only)",
            "categorical": "OneHotEncoder(handle_unknown='ignore'), levels "
                           "learned from train only, no drop-first",
            "fitted_on": "training block only, inside sklearn pipelines",
        },
        "poutcome_derivation": "pdays==999 -> never_contacted; else documented "
                               "file label; 5 anomalous rows kept as 'unknown'",
        "train_prevalence_descriptive": train_prev,
        "validation_prevalence": float(y_val.mean()),
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
            "pyarrow": pyarrow.__version__,
            "platform": platform.platform(),
        },
    }
    (OUT / "models" / "model_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    # ---- 10. Internal consistency: read predictions back ----------------
    for m in ("A", "B", "C"):
        back = pd.read_csv(
            OUT / "predictions" / f"validation_predictions_{m.lower()}.csv")
        auc_back = roc_auc_score(back["y"].to_numpy(),
                                 back[f"p_model_{m.lower()}"].to_numpy())
        diff = abs(auc_back - roc_auc_score(y_val, preds[m]))
        assert diff < 1e-12, f"prediction file round-trip mismatch for {m}"
    print("[check] prediction files round-trip: metrics reproduce exactly")

    # ---- 11. Hash every artifact ----------------------------------------
    artifacts = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json":
            rel = str(path.relative_to(OUT)).replace("\\", "/")
            artifacts[rel] = sha256_file(path)
    manifest = {
        "run": "autonomous",
        "generated_utc": settings["generated_utc"],
        "agent_model": settings["agent_model"],
        "script": SCRIPT_REL,
        "raw_dataset_sha256": raw_sha,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "test_set_evaluated": False,
        "test_labels_read": False,
        "artifacts_sha256": artifacts,
        "notes": "manifest excludes its own hash; verification artifacts "
                 "written later by verify_autonomous.py are hashed there",
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] artifacts written under results/autonomous/ "
          f"({len(artifacts)} files hashed)")

    if log_lines:
        warnings_file.write_text("".join(log_lines), encoding="utf-8")
        print(f"[warn] {len(log_lines)} warning entries -> {warnings_file}")
    else:
        warnings_file.write_text("(no warnings recorded)\n", encoding="utf-8")
        print("[warn] no warnings during final fits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
