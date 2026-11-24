"""Minimal inference interface for the frozen Guided Models A and B.

POST-TEST USABILITY ADDITION
----------------------------
This script is not part of the original experiment. It scores NEW customers
with the models exported in ``models/A.joblib`` and ``models/B.joblib``, which
were themselves reconstructed after the experiment had finished (see
``models/export_manifest.json`` and
``results/verification/model_export_validation.md``).

What it does
------------
* Reads a local CSV with one or more new customers.
* Validates the columns, numeric types, integer category codes and missing
  values, and explains anything it rejects.
* Reuses the ORIGINAL Guided decoding and derived-column rule
  (``prepare_frame`` imported from ``my_analysis.guided_test_evaluate``), so the
  feature contract cannot drift from the training schema.
* Loads the saved pipelines and prints per-customer Model A and Model B scores.
* Verifies the loaded files against their recorded SHA-256 before scoring.

What it never does
------------------
* It never fits, refits, tunes or recalibrates anything.
* It never loads or mentions Model C, which is a non-deployable leakage
  diagnostic.
* It never reads the true outcome. A ``y`` column is ignored with a warning.
* It never requires ``duration``; it is ignored if present, because it is not
  available before a call and is excluded from Models A and B.
* It never writes, saves or overwrites a file. Output goes to the terminal.

Usage
-----
::

    .venv\\Scripts\\python.exe -m my_analysis.predict --input models/example_customer.csv

Exit codes: ``0`` scored, ``2`` invalid input or unusable model artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from my_analysis.guided_test_evaluate import (  # noqa: E402
    CODEBOOK,
    MODELS,
    prepare_frame,
)

MODEL_PATHS = {
    "A": ROOT / "models" / "A.joblib",
    "B": ROOT / "models" / "B.joblib",
}
EXPORT_MANIFEST = ROOT / "models" / "export_manifest.json"

NUMERIC_REQUIRED = ("age", "balance", "previous", "pdays")
CATEGORICAL_REQUIRED = ("job", "marital", "education", "default", "housing", "loan", "poutcome")
REQUIRED_COLUMNS = NUMERIC_REQUIRED + CATEGORICAL_REQUIRED

# Columns that may legitimately appear in a CSV but must not influence scoring.
IGNORED_COLUMNS = {"duration", "y", "deposit", "row_index", "id", "customer_id"}
MAX_ERRORS_SHOWN = 20


class InputError(Exception):
    """Raised for user-facing input problems; never a traceback for the user."""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def describe_codebook(col: str) -> str:
    mapping = CODEBOOK[col]
    return ", ".join(f"{code}={label}" for code, label in sorted(mapping.items()))


def read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise InputError(f"input file not found: {path}")
    if path.is_dir():
        raise InputError(f"input path is a directory, expected a CSV file: {path}")

    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    except pd.errors.EmptyDataError as exc:
        raise InputError(f"input file is empty: {path}") from exc
    except Exception as exc:  # pragma: no cover - pandas raises many types
        raise InputError(f"could not read {path} as CSV: {exc}") from exc

    if frame.shape[0] == 0:
        raise InputError(f"input file has no data rows: {path}")
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def validate(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a typed frame plus a list of non-fatal warnings."""
    problems: list[str] = []
    warnings_out: list[str] = []

    present = set(frame.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing:
        problems.append(
            "missing required column(s): "
            + ", ".join(missing)
            + " (see models/input_schema.md for the full schema)"
        )

    duplicated = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
    if duplicated:
        problems.append("duplicated column name(s): " + ", ".join(duplicated))

    ignored_present = sorted(present & IGNORED_COLUMNS)
    if "y" in ignored_present or "deposit" in ignored_present:
        warnings_out.append(
            "the outcome column is present but ignored: scoring never uses the true result"
        )
    if "duration" in ignored_present:
        warnings_out.append(
            "'duration' is present but ignored: it is unavailable before a call and is "
            "not a feature of Models A or B"
        )
    extra = sorted(present - set(REQUIRED_COLUMNS) - IGNORED_COLUMNS)
    if extra:
        warnings_out.append("unused column(s) ignored: " + ", ".join(extra))

    if problems:
        raise InputError(problems)

    clean = pd.DataFrame(index=frame.index)

    for col in NUMERIC_REQUIRED:
        raw = frame[col].astype(str).str.strip()
        blank = raw == ""
        converted = pd.to_numeric(raw.where(~blank), errors="coerce")
        bad = converted.isna() & ~blank
        for idx in list(frame.index[blank])[:MAX_ERRORS_SHOWN]:
            problems.append(f"row {idx + 2}: '{col}' is missing (blank)")
        for idx in list(frame.index[bad])[:MAX_ERRORS_SHOWN]:
            problems.append(
                f"row {idx + 2}: '{col}' must be a number, got {raw.loc[idx]!r}"
            )
        clean[col] = converted.astype(float)

    for col in CATEGORICAL_REQUIRED:
        raw = frame[col].astype(str).str.strip()
        valid = CODEBOOK[col]
        for idx in range(len(frame)):
            value = raw.iloc[idx]
            if value == "":
                problems.append(f"row {idx + 2}: '{col}' is missing (blank)")
                continue
            if not re.fullmatch(r"[+-]?\d+", value):
                problems.append(
                    f"row {idx + 2}: '{col}' must be an integer code, got {value!r}. "
                    f"Valid codes: {describe_codebook(col)}"
                )
                continue
            code = int(value)
            if code not in valid:
                problems.append(
                    f"row {idx + 2}: '{col}' code {code} is unknown. "
                    f"Valid codes: {describe_codebook(col)}"
                )
        clean[col] = pd.to_numeric(raw, errors="coerce")

    if problems:
        raise InputError(problems)
    return clean, warnings_out


def load_models() -> tuple[dict[str, object], dict]:
    if not EXPORT_MANIFEST.exists():
        raise InputError(f"export manifest not found: {EXPORT_MANIFEST.relative_to(ROOT)}")
    try:
        manifest = json.loads(EXPORT_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(f"export manifest is not valid JSON: {exc}") from exc

    recorded = manifest.get("artifacts", {})
    problems: list[str] = []

    for key, path in MODEL_PATHS.items():
        if not path.exists():
            problems.append(f"missing model file: {path.relative_to(ROOT).as_posix()}")
            continue
        rel = path.relative_to(ROOT).as_posix()
        expected = recorded.get(rel, {}).get("sha256")
        if expected is None:
            problems.append(f"{rel} is not recorded in models/export_manifest.json")
            continue
        actual = sha256_file(path)
        if actual != expected:
            problems.append(
                f"{rel} does not match the recorded hash "
                f"(expected {expected[:12]}..., found {actual[:12]}...); "
                "refusing to score with an altered model"
            )
    if problems:
        raise InputError(problems)

    import joblib  # imported late so the integrity checks run first

    models: dict[str, object] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for key, path in MODEL_PATHS.items():
            models[key] = joblib.load(path)
        version_warnings = [
            str(w.message)
            for w in caught
            if w.category.__name__ == "InconsistentVersionWarning"
            or "unpickle" in str(w.message).lower()
        ]
    if version_warnings:
        raise InputError(
            [
                "the saved pipelines were built with a different scikit-learn version than "
                "this environment "
                f"(installed {np.__version__} numpy; see models/export_manifest.json for "
                "the recorded versions). Refusing to score, because the numbers may not be "
                "trustworthy.",
                *version_warnings,
            ]
        )
    return models, manifest


def format_table(rows: list[dict], label_width: int) -> str:
    header = (
        f"{'row':>4}  {'customer':<{label_width}}  {'p_A':>10}  {'p_B':>10}  "
        f"{'A %':>7}  {'B %':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['row']:>4}  {row['label']:<{label_width}}  "
            f"{row['p_A']:.6f}  {row['p_B']:.6f}  "
            f"{row['p_A'] * 100:>6.2f}%  {row['p_B'] * 100:>6.2f}%"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score new customers with the frozen Guided Models A and B (no fitting)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV file with one or more new customers (see models/input_schema.md)",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="optional column used only to label rows in the output (e.g. customer_id)",
    )
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    try:
        frame = read_input(path)
        clean, notices = validate(frame)
        models, manifest = load_models()

        prepared = prepare_frame(clean)
        scores: dict[str, np.ndarray] = {}
        for key, pipe in models.items():
            features = MODELS[key]["numeric"] + MODELS[key]["categorical"]
            scores[key] = np.asarray(pipe.predict_proba(prepared[features])[:, 1], dtype=float)

        for key, values in scores.items():
            if not np.all(np.isfinite(values)) or values.min() < 0.0 or values.max() > 1.0:
                raise InputError(
                    [f"model {key} returned a value outside [0, 1]; refusing to report it"]
                )
    except InputError as exc:
        print("ERROR: the input could not be scored.", file=sys.stderr)
        messages = exc.args[0] if isinstance(exc.args[0], list) else [str(exc)]
        for message in messages:
            print(f"  - {message}", file=sys.stderr)
        print(
            "\nNothing was written. See models/input_schema.md for the required schema.",
            file=sys.stderr,
        )
        return 2
    except SystemExit as exc:  # raised by prepare_frame on an unmapped code
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ output
    label_col = args.label_column
    if label_col is None:
        for candidate in ("customer_id", "id"):
            if candidate in frame.columns:
                label_col = candidate
                break
    if label_col is not None and label_col not in frame.columns:
        print(f"ERROR: label column {label_col!r} is not in the input file.", file=sys.stderr)
        return 2

    rows = []
    for position in range(len(clean)):
        label = str(frame[label_col].iloc[position]) if label_col else f"#{position + 1}"
        rows.append(
            {
                "row": int(position) + 2,  # +2 = header line + 1-based
                "label": label,
                "p_A": float(scores["A"][position]),
                "p_B": float(scores["B"][position]),
            }
        )
    label_width = max(len(r["label"]) for r in rows)
    label_width = max(label_width, len("customer"))

    print("Frozen Guided Models A and B - inference (post-test usability artifact)")
    print(f"Input file: {path}")
    print(f"Rows scored: {len(rows)}")
    print(f"Model A: {MODELS['A']['role']}; numeric {MODELS['A']['numeric']}, "
          f"categorical {MODELS['A']['categorical']}")
    print(f"Model B: {MODELS['B']['role']}; numeric {MODELS['B']['numeric']}, "
          f"categorical {MODELS['B']['categorical']}")
    print(f"Frozen C: A={manifest['selected_C']['A']}, B={manifest['selected_C']['B']} "
          f"(fixed; nothing was fitted, tuned or recalibrated)")
    if label_col:
        print(f"Row labels come from column {label_col!r}.")
    for notice in notices:
        print(f"Note: {notice}")
    print()
    print(format_table(rows, label_width))
    print()
    print("Calibration warning")
    print("  These scores are not trustworthy individual probabilities. On the chronological")
    print("  test block the mean predicted probability is about 6.8% while about 39.3% of")
    print("  customers actually subscribed (results/guided/test/test_metrics.csv).")
    print("  Models A and B are useful for RANKING customers relative to each other, not")
    print("  for statements such as 'this customer has a 5% chance'. A low score does not")
    print("  mean the customer will not subscribe.")
    print()
    print("Model C was not loaded: it is a non-deployable leakage diagnostic.")
    print("Nothing was fitted, and no file was written or modified by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
