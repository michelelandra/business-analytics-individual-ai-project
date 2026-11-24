"""Export the frozen Guided Models A and B as fitted scikit-learn pipelines.

PHASE A usability artifact. These are POST-TEST RECONSTRUCTED objects: the
original Guided run never serialised its models, so the pipelines written by
this script are rebuilt from the frozen specification after the experiment had
already finished. This is not a new experiment, not model selection, not an
independent replication, and not a change to the original analysis.

What this script does
---------------------
1. Verifies the dataset SHA-256 and the frozen chronological split against the
   recorded values (via the original ``verify_inputs``).
2. Reuses the ORIGINAL Guided implementation for feature decoding, derived
   columns, preprocessing and the logistic-regression specification
   (``prepare_frame`` and ``build_pipeline`` imported from
   ``my_analysis.guided_test_evaluate``; no definition is rewritten here).
3. Fits Models A and B ONLY on the original 31,647 chronological training rows
   with the frozen settings (A: C=0.01, B: C=0.01; L2, lbfgs, max_iter=1000,
   random_state=42). No search, no calibration, no feature selection.
4. Scores the original validation block and compares against
   ``results/guided/validation_predictions.csv``. Both models must agree to
   <= 1e-12 maximum absolute difference, otherwise nothing is exported.
5. Only after both checks pass, serialises the complete fitted pipelines
   (preprocessing + estimator) to ``models/A.joblib`` and ``models/B.joblib``,
   and writes ``models/export_manifest.json`` and
   ``results/verification/model_export_validation.md``.

What this script does NOT do
----------------------------
* It never reads chronological TEST labels and never touches the test block.
  ``load_train_validation`` reads ``y`` only from row groups 0-31 (< 32,000),
  which lie entirely before the test block that starts at row 38,428.
* It does not compute any final-test metric and does not rerun the frozen test.
* It does not train Model C (the non-deployable leakage diagnostic).
* It does not run hyperparameter search, calibration or feature selection.
* It does not modify any existing file, model, manifest or prediction.

Run from the repository root::

    .venv\\Scripts\\python.exe src\\my_analysis\\export_frozen_guided_pipelines.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from my_analysis.guided_test_evaluate import (  # noqa: E402
    MAX_ITER,
    MODELS,
    SEED,
    SELECTED_C,
    TRAIN_END,
    build_pipeline,
    load_train_validation,
    prepare_frame,
    verify_inputs,
)

MODELS_DIR = ROOT / "models"
VAL_PREDS = ROOT / "results" / "guided" / "validation_predictions.csv"
MANIFEST = MODELS_DIR / "export_manifest.json"
REPORT = ROOT / "results" / "verification" / "model_export_validation.md"
TOL = 1e-12
EXPECTED_TRAIN = 31_647
EXPECTED_VAL = 6_781


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    started = datetime.now(timezone.utc)
    print("Frozen Guided pipeline export (post-test usability artifact)")
    print(f"Started (UTC): {started.isoformat(timespec='seconds')}")

    # ------------------------------------------------------------- preconditions
    if TRAIN_END != EXPECTED_TRAIN:
        print(f"FATAL: TRAIN_END is {TRAIN_END}, expected {EXPECTED_TRAIN}")
        return 1
    if SEED != 42 or MAX_ITER != 1000:
        print(f"FATAL: SEED={SEED} MAX_ITER={MAX_ITER}; expected 42 / 1000")
        return 1
    for name in ("A", "B"):
        if SELECTED_C[name] != 0.01:
            print(f"FATAL: frozen C for {name} is {SELECTED_C[name]}, expected 0.01")
            return 1
    for path in (MODELS_DIR / "A.joblib", MODELS_DIR / "B.joblib", MANIFEST, REPORT):
        if path.exists():
            print(f"FATAL: {path.name} already exists; refusing to overwrite.")
            return 1

    # ------------------------------------------------------------------- inputs
    split_meta, idx_train, idx_val, idx_test, dataset_sha, split_sha = verify_inputs()
    X_train, X_val, y_train, _meta = load_train_validation()

    if len(X_train) != EXPECTED_TRAIN or len(X_val) != EXPECTED_VAL:
        print(f"FATAL: got {len(X_train)} train / {len(X_val)} validation rows")
        return 1
    if len(idx_train) != EXPECTED_TRAIN or len(idx_val) != EXPECTED_VAL:
        print("FATAL: frozen split index counts do not match the block sizes")
        return 1
    del idx_test  # never used: the test block is out of scope for this script

    print(f"Dataset SHA-256: {dataset_sha}")
    print(f"Split canonical hash: {split_sha}")
    print(f"Training rows: {len(X_train)}; validation rows: {len(X_val)}")
    print(f"Train row range: [{int(idx_train[0])}, {int(idx_train[-1])}]")

    # -------------------------------------------------------------- fit A and B
    X_train_p = prepare_frame(X_train)
    X_val_p = prepare_frame(X_val)

    val_ref = pd.read_csv(VAL_PREDS).sort_values("row_index").reset_index(drop=True)
    if not np.array_equal(val_ref["row_index"].to_numpy(), np.sort(np.asarray(idx_val))):
        print("FATAL: validation prediction rows do not match the frozen split indices")
        return 1

    fitted: dict[str, object] = {}
    max_diff: dict[str, float] = {}

    for name in ("A", "B"):
        pipe = build_pipeline(name)
        pipe.fit(X_train_p, y_train)
        p_val = pipe.predict_proba(X_val_p)[:, 1]

        ref = val_ref[f"p_{name}"].to_numpy(dtype=float)
        diff = float(np.max(np.abs(p_val - ref)))
        max_diff[name] = diff
        fitted[name] = pipe

        print(
            f"Model {name}: C={SELECTED_C[name]:g}, n={len(p_val)}, "
            f"max |reconstructed - saved| = {diff:.3e}"
        )

    # ------------------------------------------------------------- export gate
    failed = [m for m, d in max_diff.items() if not (d <= TOL)]
    if failed:
        print(f"FATAL: validation reproduction failed for {failed}; nothing exported.")
        return 1

    # ------------------------------------------------------------------ export
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_info: dict[str, dict] = {}
    for name in ("A", "B"):
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(fitted[name], path)
        artifact_info[f"models/{name}.joblib"] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    # ---------------------------------------------------------------- manifest
    model_spec = {
        "estimator": "LogisticRegression",
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": MAX_ITER,
        "class_weight": None,
        "random_state": SEED,
        "C": {"A": SELECTED_C["A"], "B": SELECTED_C["B"]},
        "source": "my_analysis.guided_test_evaluate.build_pipeline (imported, not rewritten)",
    }
    pipeline_spec = {
        "steps": ["prep", "clf"],
        "numeric": {"transformer": "StandardScaler", "with_mean": True, "with_std": True},
        "categorical": {
            "transformer": "OneHotEncoder",
            "handle_unknown": "ignore",
            "sparse_output": True,
            "dtype": "float64",
        },
        "remainder": "drop",
        "derived_column": "poutcome_derived = 'never_contacted' when pdays == 999 else poutcome",
    }
    feature_sets = {
        k: {
            "role": MODELS[k]["role"],
            "numeric": MODELS[k]["numeric"],
            "categorical": MODELS[k]["categorical"],
        }
        for k in ("A", "B")
    }
    train_idx_list = [int(i) for i in idx_train]
    idx_hash = hashlib.sha256(
        json.dumps(train_idx_list, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    manifest = {
        "phase": "Phase A - frozen Guided pipeline export",
        "generated_utc": started.isoformat(timespec="seconds"),
        "status": "exported after validation reproduction passed",
        "purpose": (
            "Post-test usability artifact: make the already evaluated Guided Models A and B "
            "loadable for inference without retraining. Not a new experiment."
        ),
        "post_test_reconstruction": True,
        "originally_serialized_models": False,
        "test_labels_read": False,
        "test_metrics_computed": False,
        "model_c_trained": False,
        "hyperparameter_search_performed": False,
        "existing_files_modified": False,
        "training_block": (
            "chronological rows [0, 31647): frozen train block only. "
            "The validation block was used for verification, never for fitting."
        ),
        "dataset": {
            "path": "data/raw/train-00000-of-00001-f26fa3adaf6dfdf2.parquet",
            "sha256": dataset_sha,
        },
        "split": {
            "path": "results/splits/temporal_split_v1.json",
            "canonical_sha256": split_sha,
            "sizes": split_meta["sizes"],
        },
        "training_indices": {
            "n": len(train_idx_list),
            "first": train_idx_list[0],
            "last": train_idx_list[-1],
            "contiguous_from_zero": train_idx_list == list(range(EXPECTED_TRAIN)),
            "sha256": idx_hash,
            "hash_method": "sha256 of json.dumps(indices, separators=(',',':')) as UTF-8",
        },
        "model_spec": model_spec,
        "pipeline_spec": pipeline_spec,
        "feature_sets": feature_sets,
        "selected_C": {"A": SELECTED_C["A"], "B": SELECTED_C["B"]},
        "validation_check": {
            "predictions_file": "results/guided/validation_predictions.csv",
            "predictions_sha256": sha256_file(VAL_PREDS),
            "n_rows": int(len(val_ref)),
            "tolerance": TOL,
            "max_abs_diff": max_diff,
            "passed": True,
        },
        "artifacts": artifact_info,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
            "platform": platform.platform(),
        },
        "notes": [
            "Pipelines were reconstructed from the frozen settings because the original run did not serialise models.",
            "Agreement is with the saved validation predictions, not with the test block.",
            "Model C is a non-deployable leakage diagnostic and was not trained.",
            "Loading a joblib file requires a compatible scikit-learn version; see versions.",
            "No inference interface or input-schema helper is included in this phase.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ report
    report = f"""# Model export validation (post-test usability artifact)

**Status: PASS - Models A and B exported.** Generated
{started.isoformat(timespec='seconds')} by
`src/my_analysis/export_frozen_guided_pipelines.py`.

## What this is, and what it is not

The original Guided run did **not** serialise its fitted models. This phase
reconstructs Models A and B from the frozen specification and exports them as
complete scikit-learn pipelines, so a reviewer can obtain a prediction without
retraining.

These are **post-test reconstructed artifacts, not the objects that produced
the original results**. They were built after the experiment ended. No original
metric, prediction, manifest or script was changed, and the frozen results
remain the only evidence for the reported findings.

## Procedure

1. Dataset SHA-256 and the frozen chronological split were verified against the
   recorded values (`verify_inputs`).
2. Features were decoded and prepared with the original Guided implementation
   (`prepare_frame`), and pipelines were built with the original factory
   (`build_pipeline`). Both are imported from
   `src/my_analysis/guided_test_evaluate.py`; no preprocessing or estimator
   specification was rewritten.
3. Models A and B were fitted on the original **31,647 training rows only**
   (rows `[0, 31,647)`). The validation block was not used for fitting.
4. Predictions on the original 6,781 validation rows were compared with
   `results/guided/validation_predictions.csv`.
5. Only after both comparisons passed at tolerance 1e-12 were the pipelines
   written to `models/A.joblib` and `models/B.joblib`.

## Numerical checks

| Model | Frozen C | Validation rows | Max abs difference | Tolerance | Result |
|---|---:|---:|---:|---:|---|
| A | {SELECTED_C['A']:g} | {len(val_ref)} | {max_diff['A']:.3e} | {TOL:g} | PASS |
| B | {SELECTED_C['B']:g} | {len(val_ref)} | {max_diff['B']:.3e} | {TOL:g} | PASS |

Saved predictions file SHA-256:
`{sha256_file(VAL_PREDS)}`.

## Files

| File | SHA-256 | Bytes |
|---|---|---|
| `models/A.joblib` | `{artifact_info['models/A.joblib']['sha256']}` | {artifact_info['models/A.joblib']['bytes']} |
| `models/B.joblib` | `{artifact_info['models/B.joblib']['sha256']}` | {artifact_info['models/B.joblib']['bytes']} |

## What was NOT done

- No chronological test label was read; the test block was not touched.
- No final-test metric was computed and the frozen test evaluation was not rerun.
- Model C was not trained (non-deployable leakage diagnostic).
- No hyperparameter search, calibration, feature selection or model redesign.
- No existing file, prediction, metric or manifest was modified.

## Limitations

- These pipelines are reconstructions. They reproduce the saved validation
  predictions to the stated tolerance, but they are not the original in-memory
  objects, and no test-block comparison was performed.
- Loading them requires a compatible scikit-learn version (recorded in
  `models/export_manifest.json`: scikit-learn {sklearn.__version__}); a different
  version may warn or change behaviour.
- The exported models inherit the project's calibration defect: on the
  chronological test block the mean predicted probability is about 0.068 while
  the observed subscription rate is 0.393. They are usable for ranking, not for
  probability statements about an individual customer.
- The export contains no inference interface and no input-schema helper. The
  pipelines still expect the dataset's integer codebook plus the
  `poutcome_derived` rule, so scoring a new customer requires a later phase.
"""
    REPORT.write_text(report, encoding="utf-8")

    print()
    print("Validation reproduction PASSED for A and B.")
    for name in ("A", "B"):
        info = artifact_info[f"models/{name}.joblib"]
        print(f"Wrote models/{name}.joblib  sha256={info['sha256']}  bytes={info['bytes']}")
    print(f"Wrote {MANIFEST.relative_to(ROOT).as_posix()}")
    print(f"Wrote {REPORT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
