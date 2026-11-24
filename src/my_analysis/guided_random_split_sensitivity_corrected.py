"""Corrected guided random-split sensitivity analysis — SPLIT-ONLY mode.

This script exists because the original guided sensitivity run implemented the
wrong two-step split order. The original run is preserved as evidence under
``results/sensitivity_erroneous_v1/`` and is NOT valid for the authorized
sensitivity analysis.

Authorized split procedure (test-first)
---------------------------------------
::

    remaining_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=6783,
        stratify=y,
        random_state=42
    )

    train_idx, val_idx = train_test_split(
        remaining_idx,
        test_size=6781,
        stratify=y[remaining_idx],
        random_state=42
    )

``train_test_split`` returns ``(train, test)`` in that order.

Current scope
-------------
Only ``--mode split-only`` is implemented. It generates the corrected split,
verifies it against an independently coded authorized reference, and writes
split-stage artifacts. It does NOT import, construct, fit or evaluate any
model, and it does NOT compute predictions, metrics or bootstrap results.

Model fitting/evaluation requires a separate authorization after the human
project author verifies the exact corrected index sets.

Run with (from the repository root)::

    $env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m my_analysis.guided_random_split_sensitivity_corrected --mode split-only

This script performs no LLM calls.
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
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
TEMPORAL_SPLIT = ROOT / "results" / "splits" / "temporal_split_v1.json"
OUT = ROOT / "results" / "sensitivity_corrected_v1"
ERRONEOUS_DIR = ROOT / "results" / "sensitivity_erroneous_v1"
ERRONEOUS_SPLIT = ERRONEOUS_DIR / "random_split_v1.json"
AUDIT_DOC = ROOT / "results" / "audit" / "guided_random_split_protocol_discrepancy.md"
SCRIPT_REL = "src/my_analysis/guided_random_split_sensitivity_corrected.py"
SCRIPT_PATH = ROOT / SCRIPT_REL

EXPECTED_DATASET_SHA256 = (
    "0cd36af11a3097b2a069dd171a2957c955665d2ad7b12973f124d69d444bf45f"
)
EXPECTED_TEMPORAL_SPLIT_SHA256 = (
    "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c"
)
ERRONEOUS_SPLIT_SHA256 = (
    "e43308ca7a0485fdf114be43372be3c0cc07017897ed766895de47e02f6eca3b"
)
ERRONEOUS_SCRIPT_SHA256 = (
    "3806a2b0eeedbfcd480005f754bf345417fc62a228b9d4a91fc9063d0485cc54"
)

SEED = 42
N_TOTAL = 45_211
N_TRAIN = 31_647
N_VAL = 6_781
N_TEST = 6_783

EXPECTED_POSITIVES = {"train": 3_702, "validation": 793, "test": 794}
EXPECTED_NEGATIVES = {"train": 27_945, "validation": 5_988, "test": 5_989}
EXPECTED_CHRONO_TEST_OVERLAP = 1_022

CHRONO_TRAIN = (0, 31_647)
CHRONO_VAL = (31_647, 38_428)
CHRONO_TEST = (38_428, 45_211)

# ---------------------------------------------------------------------------
# Frozen guided feature sets and hyperparameters.
#
# Preserved here so a later, separately authorized evaluation stage can reuse
# them. They are NOT used by --mode split-only.
# ---------------------------------------------------------------------------
NUMERIC_A = ["age", "balance"]
CATEGORICAL_A = ["job", "marital", "education", "default", "housing", "loan"]
NUMERIC_B = NUMERIC_A + ["previous"]
CATEGORICAL_B = CATEGORICAL_A + ["poutcome_derived"]
NUMERIC_C = NUMERIC_B + ["duration"]
CATEGORICAL_C = list(CATEGORICAL_B)

FROZEN_FEATURE_SETS = {
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

FROZEN_HYPERPARAMETERS = {
    "A": {
        "estimator": "LogisticRegression",
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
        "random_state": 42,
    },
    "B": {
        "estimator": "LogisticRegression",
        "C": 0.01,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
        "random_state": 42,
    },
    "C": {
        "estimator": "LogisticRegression",
        "C": 0.001,
        "penalty": "l2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
        "random_state": 42,
    },
}

PREPROCESSING_SPEC = {
    "numeric": {"transformer": "StandardScaler"},
    "categorical": {
        "transformer": "OneHotEncoder",
        "params": {"handle_unknown": "ignore", "sparse_output": True, "dtype": "float64"},
    },
    "remainder": "drop",
    "fitted_on": "training block only (for a later authorized evaluation)",
}

# Exclusions for the preservation baseline. The corrected output directory is
# this run's own output; the raw data is verified by SHA-256.
EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "data/raw", "results/sensitivity_corrected_v1",
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
    changed = sorted(k for k in (before_keys & after_keys) if before[k] != after[k])
    return {
        "n_files_before": len(before),
        "n_files_after": len(after),
        "added_files": added,
        "removed_files": removed,
        "changed_files": changed,
        "unexpected_modification": bool(added or removed or changed),
    }


# ---------------------------------------------------------------------------
# Authorized split implementation
# ---------------------------------------------------------------------------
def generate_corrected_split(y: np.ndarray) -> dict[str, np.ndarray]:
    """Generate the authorized test-first split with the exact calls."""
    remaining_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=6783,
        stratify=y,
        random_state=42,
    )
    train_idx, val_idx = train_test_split(
        remaining_idx,
        test_size=6781,
        stratify=y[remaining_idx],
        random_state=42,
    )
    return {
        "train": np.sort(train_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def authorized_reference_split(y: np.ndarray) -> dict[str, np.ndarray]:
    """Separately coded reference implementation of the authorized procedure.

    This function is intentionally written independently of
    ``generate_corrected_split`` so the conformance check is not a tautology.
    It uses the same authorized calls and literals.
    """
    all_indices = np.arange(len(y), dtype=np.int64)
    remaining, held_out_test = train_test_split(
        all_indices,
        test_size=6783,
        stratify=y,
        random_state=42,
    )
    reference_train, reference_val = train_test_split(
        remaining,
        test_size=6781,
        stratify=y[remaining],
        random_state=42,
    )
    return {
        "train": np.sort(reference_train),
        "validation": np.sort(reference_val),
        "test": np.sort(held_out_test),
    }


def canonical_fingerprint(split: dict[str, np.ndarray]) -> tuple[str, dict]:
    """Order-insensitive fingerprint of the three index sets."""
    payload = {
        key: sorted(int(i) for i in np.asarray(values, dtype=np.int64))
        for key, values in split.items()
    }
    definition = (
        "SHA-256 of json.dumps({block: sorted(index list)}, sort_keys=True, "
        "separators=(',', ':')); independent of within-block order and dict order"
    )
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"))), payload


def overlap_with_chronological(split: dict[str, np.ndarray]) -> dict:
    chrono = {
        "chronological_train": np.arange(*CHRONO_TRAIN),
        "chronological_validation": np.arange(*CHRONO_VAL),
        "chronological_test": np.arange(*CHRONO_TEST),
    }
    out: dict[str, dict[str, int]] = {}
    for rname, ridx in split.items():
        rset = set(int(i) for i in ridx)
        out[f"random_{rname}"] = {
            cname: int(len(rset & set(int(i) for i in cidx)))
            for cname, cidx in chrono.items()
        }
    return out


def overlap_with_erroneous(split: dict[str, np.ndarray]) -> dict:
    if not ERRONEOUS_SPLIT.exists():
        return {"available": False, "note": "erroneous split archive not found"}
    meta = json.loads(ERRONEOUS_SPLIT.read_text(encoding="utf-8"))
    err = {k: np.asarray(v, dtype=np.int64) for k, v in meta["indices"].items()}
    out = {
        "available": True,
        "path": ERRONEOUS_SPLIT.relative_to(ROOT).as_posix(),
        "erroneous_split_sha256": meta.get("split_hash_sha256"),
        "shared_rows": {},
    }
    for k in ("train", "validation", "test"):
        out["shared_rows"][k] = int(len(set(int(i) for i in split[k]) & set(int(i) for i in err[k])))
    return out


def verify_split(split: dict[str, np.ndarray], y: np.ndarray) -> dict:
    train = split["train"]
    val = split["validation"]
    test = split["test"]

    sizes = {"train": len(train), "validation": len(val), "test": len(test)}
    expected_sizes = {"train": N_TRAIN, "validation": N_VAL, "test": N_TEST}

    coverage = np.sort(np.concatenate([train, val, test]))
    complete_coverage = bool(
        len(coverage) == N_TOTAL and np.array_equal(coverage, np.arange(N_TOTAL))
    )
    pairwise_disjoint = bool(
        set(int(i) for i in train).isdisjoint(set(int(i) for i in val))
        and set(int(i) for i in train).isdisjoint(set(int(i) for i in test))
        and set(int(i) for i in val).isdisjoint(set(int(i) for i in test))
    )

    class_counts = {}
    for k, idx in split.items():
        positives = int((y[idx] == 1).sum())
        negatives = int((y[idx] == 0).sum())
        class_counts[k] = {
            "n": int(len(idx)),
            "negative": negatives,
            "positive": positives,
            "prevalence": float(y[idx].mean()),
        }

    positive_counts_ok = all(
        class_counts[k]["positive"] == EXPECTED_POSITIVES[k]
        for k in ("train", "validation", "test")
    )
    negative_counts_ok = all(
        class_counts[k]["negative"] == EXPECTED_NEGATIVES[k]
        for k in ("train", "validation", "test")
    )

    chrono_test_set = set(range(*CHRONO_TEST))
    chrono_test_overlap = int(len(set(int(i) for i in test) & chrono_test_set))
    chrono_test_overlap_ok = chrono_test_overlap == EXPECTED_CHRONO_TEST_OVERLAP

    reference = authorized_reference_split(y)
    reference_equal = {
        k: bool(np.array_equal(split[k], reference[k]))
        for k in ("train", "validation", "test")
    }
    reference_all_equal = all(reference_equal.values())

    fingerprint, fingerprint_payload = canonical_fingerprint(split)
    reference_fingerprint, _ = canonical_fingerprint(reference)
    fingerprint_equal = fingerprint == reference_fingerprint

    regenerated = generate_corrected_split(y)
    internally_reproducible = all(
        np.array_equal(split[k], regenerated[k])
        for k in ("train", "validation", "test")
    )

    checks = {
        "sizes_exact": sizes == expected_sizes,
        "complete_coverage": complete_coverage,
        "pairwise_disjoint": pairwise_disjoint,
        "positive_counts_exact": positive_counts_ok,
        "negative_counts_exact": negative_counts_ok,
        "chronological_test_overlap_exact": chrono_test_overlap_ok,
        "matches_authorized_reference": reference_all_equal,
        "reference_fingerprint_matches": fingerprint_equal,
        "internally_reproducible": internally_reproducible,
    }
    return {
        "sizes": sizes,
        "expected_sizes": expected_sizes,
        "class_counts": class_counts,
        "expected_positives": EXPECTED_POSITIVES,
        "expected_negatives": EXPECTED_NEGATIVES,
        "complete_coverage": complete_coverage,
        "pairwise_disjoint": pairwise_disjoint,
        "coverage_n": int(len(coverage)),
        "coverage_min_index": int(coverage.min()),
        "coverage_max_index": int(coverage.max()),
        "chronological_test_overlap": chrono_test_overlap,
        "expected_chronological_test_overlap": EXPECTED_CHRONO_TEST_OVERLAP,
        "reference_equal": reference_equal,
        "reference_all_equal": reference_all_equal,
        "fingerprint_sha256": fingerprint,
        "reference_fingerprint_sha256": reference_fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def static_no_model_audit() -> dict:
    """Confirm the split-only script contains no model construction/evaluation.

    The check parses the abstract syntax tree and looks for actual calls and
    imports. A plain string scan is not usable here: the audit's own list of
    forbidden names would match itself. Attempt 1 of this script used a string
    scan and reported a false failure for exactly that reason; this AST check
    replaces it.
    """
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))

    forbidden_call_names = {
        "LogisticRegression",
        "Pipeline",
        "ColumnTransformer",
        "GridSearchCV",
        "RandomizedSearchCV",
        "TimeSeriesSplit",
        "fit",
        "fit_transform",
        "predict",
        "predict_proba",
        "predict_log_proba",
        "roc_auc_score",
        "average_precision_score",
        "brier_score_loss",
        "accuracy_score",
        "f1_score",
        "precision_score",
        "recall_score",
        "confusion_matrix",
    }
    forbidden_import_names = {
        "LogisticRegression",
        "Pipeline",
        "ColumnTransformer",
        "GridSearchCV",
        "RandomizedSearchCV",
        "TimeSeriesSplit",
        "roc_auc_score",
        "average_precision_score",
        "brier_score_loss",
    }

    calls_found: list[str] = []
    imports_found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                name = None
            if name in forbidden_call_names:
                calls_found.append(f"{name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in forbidden_import_names:
                    imports_found.append(
                        f"{node.module}.{alias.name} (line {node.lineno})"
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                leaf = alias.name.rsplit(".", 1)[-1]
                if leaf in forbidden_import_names:
                    imports_found.append(f"{alias.name} (line {node.lineno})")

    clean = not calls_found and not imports_found
    return {
        "method": "ast.parse; scans actual Call and Import/ImportFrom nodes",
        "forbidden_call_names_checked": sorted(forbidden_call_names),
        "forbidden_import_names_checked": sorted(forbidden_import_names),
        "forbidden_calls_found": calls_found,
        "forbidden_imports_found": imports_found,
        "no_model_construction_or_evaluation_code": clean,
        "mode": "split-only",
        "attempt_1_note": (
            "Attempt 1 used a substring scan over the script source and "
            "reported a false failure because the forbidden-token list itself "
            "contained the searched strings. This AST-based check replaces it."
        ),
        "note": (
            "The script preserves frozen feature/hyperparameter constants for a "
            "later authorized stage, but contains no model construction, fitting, "
            "prediction, metric or bootstrap code."
        ),
    }


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------
def build_split_document(
    split: dict[str, np.ndarray],
    verification: dict,
    chrono_overlap: dict,
    erroneous_overlap: dict,
    dataset_sha: str,
    temporal_meta: dict,
) -> dict:
    return {
        "split_name": "random_split_v1_corrected",
        "status": "split-only; no models constructed, fitted or evaluated",
        "analysis_role": (
            "secondary post-test sensitivity analysis; optimistic, not an "
            "independent confirmation and not a replacement for the primary "
            "chronological evaluation"
        ),
        "supersedes": {
            "erroneous_split_name": "random_split_v1",
            "erroneous_split_sha256": ERRONEOUS_SPLIT_SHA256,
            "erroneous_script_sha256": ERRONEOUS_SCRIPT_SHA256,
            "archive": ERRONEOUS_DIR.relative_to(ROOT).as_posix(),
            "audit_document": AUDIT_DOC.relative_to(ROOT).as_posix(),
            "reason": (
                "The erroneous run implemented a train-first split "
                "(test_size=13564 then 6783) instead of the authorized "
                "test-first split (test_size=6783 then 6781)."
            ),
        },
        "authorized_procedure": {
            "order": "test-first",
            "steps": [
                {
                    "step": 1,
                    "call": (
                        "remaining_idx, test_idx = train_test_split("
                        "np.arange(len(y)), test_size=6783, stratify=y, "
                        "random_state=42)"
                    ),
                    "returns": {"remaining_idx": 38_428, "test_idx": 6_783},
                },
                {
                    "step": 2,
                    "call": (
                        "train_idx, val_idx = train_test_split("
                        "remaining_idx, test_size=6781, "
                        "stratify=y[remaining_idx], random_state=42)"
                    ),
                    "returns": {"train_idx": 31_647, "val_idx": 6_781},
                },
            ],
            "return_order_note": (
                "sklearn train_test_split returns (train, test); the first "
                "assignment therefore places the 38,428-row remainder first "
                "and the 6,783-row held-out block second."
            ),
        },
        "implemented_procedure": {
            "order": "test-first (authorized)",
            "function": "generate_corrected_split",
            "steps": [
                {
                    "step": 1,
                    "call": (
                        "remaining_idx, test_idx = train_test_split("
                        "np.arange(len(y)), test_size=6783, stratify=y, "
                        "random_state=42)"
                    ),
                },
                {
                    "step": 2,
                    "call": (
                        "train_idx, val_idx = train_test_split("
                        "remaining_idx, test_size=6781, "
                        "stratify=y[remaining_idx], random_state=42)"
                    ),
                },
            ],
        },
        "generation": {
            "library": "sklearn.model_selection.train_test_split",
            "scikit_learn_version": sklearn.__version__,
            "seed": SEED,
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
            "checks": verification["checks"],
            "all_checks_passed": verification["all_checks_passed"],
            "complete_coverage": verification["complete_coverage"],
            "pairwise_disjoint": verification["pairwise_disjoint"],
            "matches_authorized_reference": verification["reference_all_equal"],
            "reference_equal_by_block": verification["reference_equal"],
            "internally_reproducible": verification["checks"]["internally_reproducible"],
            "coverage_n": verification["coverage_n"],
            "coverage_min_index": verification["coverage_min_index"],
            "coverage_max_index": verification["coverage_max_index"],
            "chronological_test_overlap": verification["chronological_test_overlap"],
            "expected_chronological_test_overlap": verification["expected_chronological_test_overlap"],
        },
        "chronological_reference": {
            "path": "results/splits/temporal_split_v1.json",
            "sha256": EXPECTED_TEMPORAL_SPLIT_SHA256,
            "sizes": temporal_meta["sizes"],
            "boundaries": temporal_meta["boundaries"],
        },
        "overlap_with_chronological_blocks": chrono_overlap,
        "overlap_with_erroneous_v1": erroneous_overlap,
        "canonical_fingerprint_sha256": verification["fingerprint_sha256"],
        "canonical_fingerprint_definition": (
            "SHA-256 of json.dumps({block: sorted(index list)}, sort_keys=True, "
            "separators=(',', ':')); order-insensitive within blocks and across "
            "dictionary keys"
        ),
        "reference_fingerprint_sha256": verification["reference_fingerprint_sha256"],
        "indices": {
            k: [int(i) for i in v]
            for k, v in split.items()
        },
        "model_results_generated": False,
        "human_verification_required": True,
        "autonomous_comparison": (
            "Not performed by the agent. The human project author will compare "
            "the corrected guided index sets against the existing autonomous "
            "index sets."
        ),
        "input_file": RAW.relative_to(ROOT).as_posix(),
        "input_sha256": dataset_sha,
        "n_rows": N_TOTAL,
        "script": SCRIPT_REL,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "created_utc": utc_now(),
    }


def split_markdown(doc: dict) -> str:
    cc = doc["class_counts"]
    ov = doc["overlap_with_chronological_blocks"]
    err = doc["overlap_with_erroneous_v1"]
    checks = doc["verification"]["checks"]
    lines = [
        "# Corrected random split v1 — SPLIT-ONLY",
        "",
        "**SPLIT-ONLY — NO MODEL RESULTS. Model fitting/evaluation requires separate authorization.**",
        "",
        f"Created (UTC): {doc['created_utc']}",
        f"Script: `{doc['script']}`",
        f"Dataset SHA-256: `{doc['input_sha256']}`",
        f"Canonical index fingerprint: `{doc['canonical_fingerprint_sha256']}`",
        "",
        "**Role.** Secondary post-test sensitivity analysis. The split is",
        "optimistic because random splitting mixes temporal information. It is",
        "not an independent confirmation and not a replacement for the primary",
        "chronological evaluation.",
        "",
        "## Authorized procedure (test-first)",
        "",
        "```python",
        "remaining_idx, test_idx = train_test_split(",
        "    np.arange(len(y)), test_size=6783, stratify=y, random_state=42",
        ")",
        "train_idx, val_idx = train_test_split(",
        "    remaining_idx, test_size=6781, stratify=y[remaining_idx], random_state=42",
        ")",
        "```",
        "",
        "`train_test_split` returns `(train, test)` in that order.",
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
        "## Conformance checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for name, value in checks.items():
        lines.append(f"| `{name}` | {'PASS' if value else 'FAIL'} |")
    lines += [
        "",
        f"**All checks passed: {doc['verification']['all_checks_passed']}.**",
        "",
        "## Overlap with the primary chronological blocks",
        "",
        "| Random block | Chrono train | Chrono validation | Chrono test |",
        "|---|---:|---:|---:|",
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
        f"Random-test overlap with chronological test: "
        f"**{doc['verification']['chronological_test_overlap']:,}** "
        f"(expected {doc['verification']['expected_chronological_test_overlap']:,}).",
        "",
        "## Comparison with the erroneous run (evidence only)",
        "",
    ]
    if err.get("available"):
        lines += [
            f"Source: `{err['path']}` (split hash `{err['erroneous_split_sha256']}`).",
            "",
            "| Block | Shared rows with erroneous split |",
            "|---|---:|",
        ]
        for k in ("train", "validation", "test"):
            lines.append(f"| {k} | {err['shared_rows'][k]:,} |")
        lines += [
            "",
            "These shared-row counts are evidence of the discrepancy. They are",
            "not used for the corrected analysis.",
        ]
    else:
        lines.append("- Erroneous split archive not available.")
    lines += [
        "",
        "## Status",
        "",
        "- Models constructed: **none**",
        "- Models fitted: **none**",
        "- Predictions generated: **none**",
        "- Metrics computed: **none**",
        "- Bootstrap executed: **none**",
        "- Human verification of exact index sets: **required before model fitting**",
        "",
    ]
    return "\n".join(lines)


def build_conformance_report(
    doc: dict, verification: dict, static_audit: dict
) -> dict:
    return {
        "mode": "split-only",
        "status": "corrected split generated and verified; no model results",
        "authorized_procedure": doc["authorized_procedure"],
        "implemented_procedure": doc["implemented_procedure"],
        "checks": verification["checks"],
        "all_checks_passed": verification["all_checks_passed"],
        "expected_vs_actual": {
            "sizes": {
                "expected": verification["expected_sizes"],
                "actual": verification["sizes"],
            },
            "positive_counts": {
                "expected": verification["expected_positives"],
                "actual": {k: v["positive"] for k, v in verification["class_counts"].items()},
            },
            "negative_counts": {
                "expected": verification["expected_negatives"],
                "actual": {k: v["negative"] for k, v in verification["class_counts"].items()},
            },
            "chronological_test_overlap": {
                "expected": verification["expected_chronological_test_overlap"],
                "actual": verification["chronological_test_overlap"],
            },
        },
        "reference_comparison": {
            "reference_function": "authorized_reference_split",
            "exact_index_set_equality": verification["reference_equal"],
            "all_blocks_equal": verification["reference_all_equal"],
            "corrected_fingerprint_sha256": verification["fingerprint_sha256"],
            "reference_fingerprint_sha256": verification["reference_fingerprint_sha256"],
        },
        "overlap_with_erroneous_v1": doc["overlap_with_erroneous_v1"],
        "static_no_model_audit": static_audit,
        "model_results_generated": False,
        "human_verification_required": True,
        "autonomous_comparison": (
            "Not performed by the agent. The human project author will compare "
            "the corrected guided index sets against the existing autonomous "
            "index sets."
        ),
    }


def build_prompt_log() -> str:
    return """# Corrected guided random-split sensitivity — prompt log

## Agent / session

- Agent/model: `opencode/big-pickle` ("Big Pickle"), OpenCode session.
- Workspace: `C:\\Users\\alber\\OneDrive\\Documenti\\my-analysis` (guided only).
- Autonomous workspace: not accessed.

## Authorizing prompt (verbatim)

```
I have reviewed your read-only root-cause audit and approve a CONTROLLED PARTIAL CORRECTION.

IMPORTANT: Your proposed correction snippet contained an additional error. sklearn's train_test_split returns (train/remainder, test/held-out), in that order. You reversed the return assignments in your audit.

The EXACT approved split implementation is:

```python
remaining_idx, test_idx = train_test_split(
    np.arange(len(y)),
    test_size=6783,
    stratify=y,
    random_state=42
)

train_idx, val_idx = train_test_split(
    remaining_idx,
    test_size=6781,
    stratify=y[remaining_idx],
    random_state=42
)
```

The final blocks must contain 31,647 training, 6,781 validation and 6,783 test rows.

YOU ARE AUTHORIZED TO PERFORM PHASES 0, 1 AND 2 ONLY.

PHASE 0 — PRESERVE THE ORIGINAL ERROR

1. Preserve the entire original `results/sensitivity/` directory, byte-for-byte, under:
   `results/sensitivity_erroneous_v1/`

2. Also preserve the exact original erroneous script:
   `src/my_analysis/guided_random_split_sensitivity.py`

3. Create a SHA-256 inventory verifying that every archived file matches its original.

4. Do not delete, overwrite or alter the original erroneous run.

5. Label the archived analysis clearly:
   "INVALID FOR THE AUTHORIZED SENSITIVITY ANALYSIS — PRESERVED AS AGENT ERROR EVIDENCE."

PHASE 1 — DOCUMENT THE ERROR

Create:
`results/audit/guided_random_split_protocol_discrepancy.md`

Document:
- The approved test-first procedure.
- The implemented train-first procedure.
- Exact source lines responsible.
- The independent human discovery and index-set comparisons.
- Why the agent's reproducibility checks failed to detect nonconformance.
- The additional reversed-assignment bug in your proposed correction snippet.
- All affected sensitivity outputs.
- Confirmation that the primary chronological analysis is unchanged.

Append a clearly labelled evidence-based entry to `mistakes.md`, without deleting or changing earlier entries.

PHASE 2 — CREATE THE CORRECTED SPLIT ONLY

1. Create a NEW script:
   `src/my_analysis/guided_random_split_sensitivity_corrected.py`

2. Save corrected outputs exclusively under:
   `results/sensitivity_corrected_v1/`

3. Implement an explicit `--mode split-only` that generates and verifies the corrected split WITHOUT constructing, fitting or evaluating any model.

4. Use the exact two-step train_test_split calls shown above, preserving their correct return-value order.

5. Check independently that:
   - Every index belongs to exactly one block.
   - All 45,211 observations are covered.
   - The block sizes are exact.
   - Class counts are 3,702 training positives, 793 validation positives and 794 test positives.
   - The random test overlaps the chronological test in exactly 1,022 rows.
   - A separately implemented reference procedure produces the same three index sets.

6. Save the corrected index arrays in `random_split_v1.json` with a canonical order-insensitive fingerprint, together with a split-only conformance report and execution log.

7. Preserve the original Guided features and hyperparameters in the corrected script for a possible later evaluation, but DO NOT execute any training or evaluation mode.

8. Do not copy or access any Autonomous files. The human project author will independently compare the corrected Guided index sets against the existing Autonomous index sets.

9. Do not modify any original primary-analysis scripts, predictions, metrics, manifests, data or frozen plan. Preserve and verify existing hashes.

10. Make all failures, corrections and execution attempts transparent in the logs.

CRITICAL STOP CONDITION:

After generating the corrected split, STOP.

Do NOT:
- Fit Models A, B or C.
- Generate any new model predictions.
- Calculate new performance metrics.
- Run bootstrap.
- Execute the full sensitivity analysis.
- Overwrite the erroneous sensitivity results.
- Start subgroup analysis.

In your final response report:
- Archive location and hash verification.
- Root-cause documentation path.
- Corrected split output paths.
- Exact split sizes and positive counts.
- Canonical index fingerprint.
- Results of the split-only conformance checks.
- Any errors, warnings or unexpected changes.

Then wait for explicit authorization before model fitting.
```

## Model calls

No model calls were made. `--mode split-only` constructs no estimator, fits no
model, generates no predictions, computes no metrics and runs no bootstrap.
The frozen feature sets and hyperparameters are preserved as constants only.
"""


def build_execution_log(
    doc: dict,
    verification: dict,
    static_audit: dict,
    preservation: dict,
    attempt_note: str,
) -> str:
    cc = doc["class_counts"]
    checks = verification["checks"]
    check_lines = "\n".join(
        f"| `{name}` | {'PASS' if value else 'FAIL'} |" for name, value in checks.items()
    )
    return f"""# Corrected guided random-split sensitivity — execution log (split-only)

## {doc['created_utc']} — corrected split generation

### Scope

Phase 2 only: generate and verify the corrected split. No model was
constructed, fitted or evaluated; no predictions, metrics or bootstrap were
produced. The original erroneous run is preserved at
`results/sensitivity_erroneous_v1/` and was not modified.

### Approvals in force

Phases 0, 1 and 2 of the controlled partial correction. Model fitting and
evaluation are **not** authorized. The authorizing prompt is recorded verbatim
in `results/sensitivity_corrected_v1/prompt_log.md`.

### Authorized procedure

```python
remaining_idx, test_idx = train_test_split(
    np.arange(len(y)), test_size=6783, stratify=y, random_state=42
)
train_idx, val_idx = train_test_split(
    remaining_idx, test_size=6781, stratify=y[remaining_idx], random_state=42
)
```

`train_test_split` returns `(train, test)` in that order.

### Implemented procedure

`generate_corrected_split` uses the exact authorized calls and return order.
`authorized_reference_split` is a separately coded reference implementation.

### Command

```
$env:PYTHONPATH="src"; .venv\\Scripts\\python.exe -m my_analysis.guided_random_split_sensitivity_corrected --mode split-only
```

### Split

- Sizes: train {cc['train']['n']:,} / validation {cc['validation']['n']:,} / test {cc['test']['n']:,}.
- Positive counts: train {cc['train']['positive']:,}; validation {cc['validation']['positive']:,}; test {cc['test']['positive']:,}.
- Negative counts: train {cc['train']['negative']:,}; validation {cc['validation']['negative']:,}; test {cc['test']['negative']:,}.
- Canonical fingerprint: `{verification['fingerprint_sha256']}`.
- Chronological-test overlap: {verification['chronological_test_overlap']:,} (expected {verification['expected_chronological_test_overlap']:,}).

### Conformance checks

| Check | Result |
|---|---|
{check_lines}

**All checks passed: {verification['all_checks_passed']}.**

### Static no-model audit

- Method: {static_audit['method']}
- Forbidden calls found: {static_audit['forbidden_calls_found']}
- Forbidden imports found: {static_audit['forbidden_imports_found']}
- No model construction or evaluation code: {static_audit['no_model_construction_or_evaluation_code']}

Attempt 1 of this script used a substring scan over its own source and reported
a false failure: the forbidden-token list contained the strings being searched
for. The AST-based check above replaces it. The attempt-1 console output is
preserved in this directory as `console_output_attempt1.txt`.

### Preservation

- Tracked files hashed before: {preservation['n_files_before']:,}
- Tracked files hashed after: {preservation['n_files_after']:,}
- Added: {preservation['added_files']}
- Removed: {preservation['removed_files']}
- Changed: {preservation['changed_files']}
- Unexpected modification: {preservation['unexpected_modification']}

The baseline includes the original erroneous run, the archive and the audit
document. It excludes only this run's own output directory, `.git`, `.venv`,
`__pycache__` and `data/raw` (the raw data is verified by SHA-256).

### Errors, corrections and attempts

{attempt_note}

### Status

- Models constructed: **none**
- Models fitted: **none**
- Predictions generated: **none**
- Metrics computed: **none**
- Bootstrap executed: **none**
- Human verification of exact index sets: **required before model fitting**
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["split-only"],
        help="split-only: generate and verify the corrected split; no model work",
    )
    args = parser.parse_args(argv)
    if args.mode != "split-only":
        raise SystemExit("FATAL: only --mode split-only is authorized")

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "console_output.txt").exists():
        attempt = 1
        while (OUT / f"console_output_attempt{attempt}.txt").exists():
            attempt += 1
        (OUT / "console_output.txt").replace(
            OUT / f"console_output_attempt{attempt}.txt"
        )
    tee = Tee(OUT / "console_output.txt")
    sys.stdout = tee
    warnings.simplefilter("always")

    try:
        print("[corrected] guided random-split sensitivity — SPLIT-ONLY")
        print(f"[corrected] started_utc={utc_now()}")
        print("[corrected] model construction/fitting/evaluation: DISABLED")

        # Preservation baseline (before writing split artifacts).
        before_snapshot = snapshot_files()
        print(f"[integrity] baseline files hashed: {len(before_snapshot):,}")

        # Inputs.
        dataset_sha = sha256_file(RAW)
        if dataset_sha != EXPECTED_DATASET_SHA256:
            raise SystemExit(f"FATAL: dataset hash mismatch: {dataset_sha}")
        temporal_meta = json.loads(TEMPORAL_SPLIT.read_text(encoding="utf-8"))
        if canonical_hash(temporal_meta["indices"]) != temporal_meta["split_hash_sha256"]:
            raise SystemExit("FATAL: chronological split indices do not match hash")
        if temporal_meta["split_hash_sha256"] != EXPECTED_TEMPORAL_SPLIT_SHA256:
            raise SystemExit("FATAL: unexpected chronological split hash")
        print(f"[verify] dataset sha256={dataset_sha}")
        print(f"[verify] chronological split sha256={EXPECTED_TEMPORAL_SPLIT_SHA256}")

        # Load labels only. No feature matrix and no estimator is constructed.
        y = pd.read_parquet(RAW, columns=["y"])["y"].to_numpy(dtype=np.int64)
        if len(y) != N_TOTAL:
            raise SystemExit(f"FATAL: expected {N_TOTAL} rows, found {len(y)}")
        print(f"[rows] total={len(y):,}; no filtering, merging or dropping applied")
        print(
            f"[target] full prevalence={y.mean():.6f} "
            f"(positives={int(y.sum()):,}/{len(y):,})"
        )

        # Corrected split.
        split = generate_corrected_split(y)
        verification = verify_split(split, y)
        chrono_overlap = overlap_with_chronological(split)
        erroneous_overlap = overlap_with_erroneous(split)

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
            f"matches_authorized_reference={verification['reference_all_equal']}"
        )
        print(f"[split] canonical_fingerprint={verification['fingerprint_sha256']}")
        print(
            f"[split] chronological_test_overlap="
            f"{verification['chronological_test_overlap']:,} "
            f"(expected {verification['expected_chronological_test_overlap']:,})"
        )
        for rname, row in chrono_overlap.items():
            print(
                f"[overlap] {rname}: chrono_train={row['chronological_train']:,} "
                f"chrono_validation={row['chronological_validation']:,} "
                f"chrono_test={row['chronological_test']:,}"
            )
        if erroneous_overlap.get("available"):
            print(
                f"[overlap] shared with erroneous v1: "
                f"train={erroneous_overlap['shared_rows']['train']:,} "
                f"validation={erroneous_overlap['shared_rows']['validation']:,} "
                f"test={erroneous_overlap['shared_rows']['test']:,}"
            )

        static_audit = static_no_model_audit()
        print(
            f"[audit] no_model_construction_or_evaluation_code="
            f"{static_audit['no_model_construction_or_evaluation_code']} "
            f"calls={static_audit['forbidden_calls_found']} "
            f"imports={static_audit['forbidden_imports_found']}"
        )

        doc = build_split_document(
            split, verification, chrono_overlap, erroneous_overlap,
            dataset_sha, temporal_meta,
        )
        (OUT / "random_split_v1.json").write_text(
            json.dumps(doc, indent=2) + "\n", encoding="utf-8"
        )
        (OUT / "random_split_v1.md").write_text(
            split_markdown(doc) + "\n", encoding="utf-8"
        )
        conformance = build_conformance_report(doc, verification, static_audit)
        (OUT / "conformance_check.json").write_text(
            json.dumps(conformance, indent=2) + "\n", encoding="utf-8"
        )

        attempt_note = (
            "Attempt 1 (2026-11-14T10:25:12Z) generated the same split and "
            "passed every split-conformance check, but its static no-model "
            "audit used a substring scan over the script source. The scan "
            "matched its own list of forbidden token strings and therefore "
            "reported a false failure. The console output of attempt 1 is "
            "preserved as `console_output_attempt1.txt`. The audit was replaced "
            "with an AST-based check of actual Call and Import nodes, and the "
            "script was rerun. No model was constructed, fitted or evaluated "
            "in either attempt. The original erroneous guided sensitivity run "
            "was archived before this run; see `results/sensitivity_erroneous_v1/`."
        )
        (OUT / "prompt_log.md").write_text(build_prompt_log(), encoding="utf-8")

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
            raise SystemExit("FATAL: existing artifacts changed during split-only run")

        (OUT / "execution_log.md").write_text(
            build_execution_log(doc, verification, static_audit, preservation, attempt_note),
            encoding="utf-8",
        )

        print("[corrected] split-only complete; no model results generated")
        tee.flush()

        artifact_hashes = {
            p.name: sha256_file(p)
            for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name != "run_manifest_sensitivity.json"
        }
        manifest = {
            "run_mode": "guided",
            "stage": "corrected random-split sensitivity — split-only",
            "script": SCRIPT_REL,
            "script_sha256": sha256_file(SCRIPT_PATH),
            "generated_utc": utc_now(),
            "plan_reference": "plan.md section 5.5 (secondary sensitivity split)",
            "analysis_role": doc["analysis_role"],
            "status": "split-only; no models constructed, fitted or evaluated",
            "supersedes": doc["supersedes"],
            "authorized_procedure": doc["authorized_procedure"],
            "implemented_procedure": doc["implemented_procedure"],
            "dataset_sha256": dataset_sha,
            "chronological_split_sha256": EXPECTED_TEMPORAL_SPLIT_SHA256,
            "corrected_split_fingerprint_sha256": verification["fingerprint_sha256"],
            "reference_fingerprint_sha256": verification["reference_fingerprint_sha256"],
            "sizes": verification["sizes"],
            "class_counts": verification["class_counts"],
            "conformance_checks": verification["checks"],
            "all_conformance_checks_passed": verification["all_checks_passed"],
            "overlap_with_chronological_blocks": chrono_overlap,
            "overlap_with_erroneous_v1": erroneous_overlap,
            "frozen_feature_sets": FROZEN_FEATURE_SETS,
            "frozen_hyperparameters": FROZEN_HYPERPARAMETERS,
            "preprocessing_spec": PREPROCESSING_SPEC,
            "model_results_generated": False,
            "model_construction_executed": False,
            "model_fitting_executed": False,
            "predictions_generated": False,
            "metrics_computed": False,
            "bootstrap_executed": False,
            "evaluation_authorized": False,
            "human_verification_required": True,
            "autonomous_comparison": doc["autonomous_comparison"],
            "static_no_model_audit": static_audit,
            "preservation": {
                **preservation,
                "baseline_note": (
                    "Baseline includes the original erroneous run, the archive, "
                    "the audit document and all primary artifacts. It excludes "
                    "only results/sensitivity_corrected_v1, .git, .venv, "
                    "__pycache__ and data/raw; the raw dataset is verified by "
                    "SHA-256."
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
        return 0 if verification["all_checks_passed"] else 1
    except BaseException:
        tee.close()
        raise


if __name__ == "__main__":
    sys.exit(main())
