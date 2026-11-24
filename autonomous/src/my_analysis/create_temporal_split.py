"""Create and verify the frozen temporal split for plan.md v1.1.

Boundaries are the exact row-index boundaries pre-registered in plan.md 5.5:
    train      [0,      31,647)
    validation [31,647, 38,428)
    test       [38,428, 45,211)

The script does NOT read any column values (no target, no features). It reads
only the Parquet metadata to confirm the row count and hashes the file bytes.

Run with:  uv run python -m my_analysis.create_temporal_split
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
OUT = ROOT / "results" / "splits"
SCRIPT_REL = "src/my_analysis/create_temporal_split.py"

N_ROWS = 45211
TRAIN_END = 31647          # int(0.70 * 45211)
VAL_END = 38428            # TRAIN_END + int(0.15 * 45211)
EXPECTED_SIZES = {
    "train": TRAIN_END,                 # 31,647
    "validation": VAL_END - TRAIN_END,  # 6,781
    "test": N_ROWS - VAL_END,           # 6,783
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        print(f"FATAL: input not found: {RAW}", file=sys.stderr)
        return 1

    pf = pq.ParquetFile(RAW)
    n_rows = int(pf.metadata.num_rows)
    if n_rows != N_ROWS:
        print(f"FATAL: expected {N_ROWS} rows, found {n_rows}", file=sys.stderr)
        return 1

    indices = {
        "train": list(range(0, TRAIN_END)),
        "validation": list(range(TRAIN_END, VAL_END)),
        "test": list(range(VAL_END, N_ROWS)),
    }
    sizes = {k: len(v) for k, v in indices.items()}

    # ---- Verification -------------------------------------------------
    all_idx = indices["train"] + indices["validation"] + indices["test"]
    no_duplicates = len(set(all_idx)) == N_ROWS
    full_coverage = sorted(all_idx) == list(range(N_ROWS))
    strictly_increasing = all(
        all(a < b for a, b in zip(v, v[1:])) for v in indices.values()
    )
    ordered_blocks = (
        max(indices["train"]) < min(indices["validation"])
        and max(indices["validation"]) < min(indices["test"])
    )
    sizes_ok = sizes == EXPECTED_SIZES

    checks = {
        "sizes_match_plan": sizes_ok,
        "no_duplicate_indices": no_duplicates,
        "full_coverage_exactly_once": full_coverage,
        "each_index_list_strictly_increasing": strictly_increasing,
        "block_order_train_lt_validation_lt_test": ordered_blocks,
    }
    if not all(checks.values()):
        print(f"FATAL: split verification failed: {checks}", file=sys.stderr)
        return 1

    # Canonical hash of the exact index assignment.
    canonical = json.dumps(indices, sort_keys=True, separators=(",", ":"))
    split_hash = sha256_bytes(canonical.encode("utf-8"))

    payload = {
        "split_name": "temporal_split_v1",
        "plan_version": "1.1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": SCRIPT_REL,
        "input_file": str(RAW.relative_to(ROOT)).replace("\\", "/"),
        "input_sha256": sha256_file(RAW),
        "n_rows": n_rows,
        "strategy": "chronological row-index split, primary per plan.md 5.5",
        "fractions_planned": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "boundaries": {
            "train": [0, TRAIN_END],
            "validation": [TRAIN_END, VAL_END],
            "test": [VAL_END, N_ROWS],
        },
        "sizes": sizes,
        "fractions_actual": {
            k: round(v / N_ROWS, 6) for k, v in sizes.items()
        },
        "indices": indices,
        "split_hash_sha256": split_hash,
        "split_hash_method": "sha256 of json.dumps(indices, sort_keys=True, separators=(',',':')) encoded as UTF-8",
        "verification": checks,
        "target_values_read": False,
        "models_fitted": False,
        "chronology_adjudication": "results/splits/chronology_adjudication.md",
        "chronology_verdict_before_adjudication": "INCONCLUSIVE",
        "human_decision": "accept chronological ordering for the primary split",
        "python": sys.version.split()[0],
        "pyarrow": pa.__version__,
        "platform": platform.platform(),
    }
    (OUT / "temporal_split_v1.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    print(f"[rows] {n_rows:,}")
    for k in ("train", "validation", "test"):
        lo, hi = payload["boundaries"][k]
        print(f"[{k}] [{lo:,}, {hi:,})  n={sizes[k]:,}  "
              f"({payload['fractions_actual'][k]:.4%})")
    print(f"[checks] {checks}")
    print(f"[split_hash] {split_hash}")
    print(f"[out] {(OUT / 'temporal_split_v1.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
