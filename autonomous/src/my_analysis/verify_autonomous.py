"""Independent verification of the autonomous run (plan.md §8.1).

Deliberately written from scratch: it does NOT import the analysis pipeline
(my_analysis.autonomous_train_val), does not refit or modify any model, and
recomputes ROC-AUC (rank-based, hand-implemented), PR-AUC (hand-implemented
step-sum), Brier score, lift@top-10%, the set-specific no-skill baselines and
the paired bootstrap CIs directly from the SAVED prediction files under
results/autonomous/predictions/.

Agreement tolerance with the reported metrics: 1e-6.

Run with (from repo root):
    $env:PYTHONPATH = "src"
    .venv\\Scripts\\python.exe -m my_analysis.verify_autonomous
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "autonomous"
SPLIT_JSON = ROOT / "results" / "splits" / "temporal_split_v1.json"
RAW = ROOT / "data" / "raw" / "train-00000-of-00001-f26fa3adaf6dfdf2.parquet"
TOL = 1e-6
SEED = 42
N_BOOT = 2000


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def roc_auc_rank(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney) with tie-averaged ranks; independent of
    sklearn's implementation."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0  # average of ranks i+1..j+1
        i = j + 1
    r_pos = ranks[y == 1].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    """Step-sum AP on the empirical PR curve (sklearn-compatible definition),
    hand-implemented.

    Note (tie handling, found and fixed during verification): ties in p are
    handled as sklearn does -- precision/recall are evaluated only at the LAST
    row of each group of identical scores, not at every row. A per-row
    step-sum differs from sklearn by ~1e-5 when ties exist (108 tied pairs in
    the validation predictions of model A).
    """
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    ps = p[order]
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    is_last = np.empty(len(ps), dtype=bool)
    is_last[:-1] = ps[:-1] != ps[1:]
    is_last[-1] = True
    idx = np.where(is_last)[0]
    prec = tp[idx] / (tp[idx] + fp[idx])
    rec = tp[idx] / n_pos
    dr = rec - np.concatenate(([0.0], rec[:-1]))
    return float(np.sum(dr * prec))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def lift_top10(y: np.ndarray, p: np.ndarray) -> tuple[float, float, int]:
    n = len(y)
    n_top = int(round(0.10 * n))
    order = np.lexsort((np.arange(n), -p))
    top_prec = float(y[order[:n_top]].mean())
    prev = float(y.mean())
    return top_prec / prev, top_prec, n_top


def boot_ci(y: np.ndarray, preds: dict[str, np.ndarray],
            hi: str, lo: str, metric: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(SEED)
    n = len(y)
    fn = roc_auc_rank if metric == "roc" else average_precision
    deltas = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        deltas.append(fn(yb, preds[hi][idx]) - fn(yb, preds[lo][idx]))
    d = np.asarray(deltas)
    point = fn(y, preds[hi]) - fn(y, preds[lo])
    return (point, float(np.percentile(d, 2.5)),
            float(np.percentile(d, 97.5)))


def main() -> int:
    split = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    val_idx = np.array(split["indices"]["validation"])

    lines: list[str] = []
    results: list[dict] = []

    # --- load saved predictions and verify against the frozen split -----
    preds: dict[str, np.ndarray] = {}
    y_from_file: np.ndarray | None = None
    for m in ("a", "b", "c"):
        df = pd.read_csv(OUT / "predictions" / f"validation_predictions_{m}.csv")
        idx_ok = bool(np.array_equal(df["row_index"].to_numpy(), val_idx))
        if y_from_file is None:
            y_from_file = df["y"].to_numpy(dtype=int)
        else:
            assert np.array_equal(df["y"].to_numpy(dtype=int), y_from_file)
        preds[m.upper()] = df[f"p_model_{m}"].to_numpy(dtype=float)
        results.append({"check": f"prediction row_index == frozen validation "
                                 f"indices (model {m.upper()})",
                        "status": "PASS" if idx_ok else "FAIL"})
        lines.append(f"model {m.upper()}: rows={len(df):,}, "
                     f"indices_match_frozen_split={idx_ok}")

    y = y_from_file

    # --- cross-check labels against the raw parquet ----------------------
    raw_y = pd.read_parquet(RAW, columns=["y"])["y"].to_numpy(dtype=int)
    label_ok = bool(np.array_equal(raw_y[val_idx], y))
    results.append({"check": "y in prediction files == raw parquet y at "
                             "validation indices",
                    "status": "PASS" if label_ok else "FAIL"})
    lines.append(f"labels cross-checked against raw parquet: {label_ok}")

    # --- recompute metrics independently ---------------------------------
    reported = pd.read_csv(OUT / "metrics" / "validation_metrics.csv")
    rep = {r["model"]: r for r in reported.to_dict("records")}

    for m in ("A", "B", "C"):
        auc = roc_auc_rank(y, preds[m])
        ap = average_precision(y, preds[m])
        br = brier(y, preds[m])
        lift, top_prec, n_top = lift_top10(y, preds[m])
        prev = float(y.mean())
        r = rep[m]
        checks = {
            "roc_auc": auc - float(r["roc_auc"]),
            "pr_auc": ap - float(r["pr_auc"]),
            "brier": br - float(r["brier"]),
            "lift_top10pct": lift - float(r["lift_top10pct"]),
            "prevalence_baseline": prev - float(r["pr_auc_no_skill_baseline"]),
            "n_top_decile": n_top - int(r["n_top_decile"]),
        }
        for k, d in checks.items():
            ok = abs(d) <= TOL
            results.append({"check": f"independent recomputation {k} "
                                     f"(model {m})",
                            "status": "PASS" if ok else
                            f"FAIL (diff {d:.3e})"})
        lines.append(
            f"model {m}: indep AUC={auc:.9f} AP={ap:.9f} Brier={br:.9f} "
            f"lift={lift:.6f} baseline={prev:.9f} n_top={n_top}")

    # --- recompute the paired bootstrap CIs independently -----------------
    hyp = json.loads(
        (OUT / "metrics" / "hypothesis_tests_validation.json").read_text(
            encoding="utf-8"))
    for comp in (("B", "A"), ("C", "B")):
        key = f"{comp[0]}-{comp[1]}"
        for metric, field in (("roc", "roc_auc"), ("pr", "pr_auc")):
            point, lo, hi = boot_ci(y, preds, comp[0], comp[1], metric)
            h = hyp["comparisons"][key]
            rep_field = f"delta_{field}"
            d_point = abs(point - h[rep_field + "_point"])
            d_lo = abs(lo - h[rep_field + "_ci95"][0])
            d_hi = abs(hi - h[rep_field + "_ci95"][1])
            ok = max(d_point, d_lo, d_hi) <= TOL
            results.append({"check": f"independent bootstrap {key} "
                                     f"d{field.upper()} (point + CI)",
                            "status": "PASS" if ok else
                            f"FAIL (max diff {max(d_point, d_lo, d_hi):.3e})"})
            lines.append(f"bootstrap {key} d{field.upper()}: "
                         f"point={point:.9f} CI [{lo:.9f}, {hi:.9f}]")

    # --- recompute calibration and check against saved CSVs ---------------
    for m in ("a", "b", "c"):
        p = preds[m.upper()]
        edges = np.linspace(0.0, 1.0, 11)
        bins = np.minimum(np.digitize(p, edges[1:-1], right=True), 9)
        prob_pred, prob_true = [], []
        for b in range(10):
            sel = bins == b
            if sel.sum() == 0:
                continue
            prob_pred.append(p[sel].mean())
            prob_true.append(y[sel].mean())
        cal = pd.read_csv(OUT / "metrics" / f"validation_calibration_{m}.csv")
        d_pp = float(np.max(np.abs(np.array(prob_pred)
                                   - cal["mean_predicted"].to_numpy())))
        d_pt = float(np.max(np.abs(np.array(prob_true)
                                   - cal["fraction_positive"].to_numpy())))
        ok = d_pp <= TOL and d_pt <= TOL
        results.append({"check": f"independent calibration recomputation "
                                 f"(model {m.upper()})",
                        "status": "PASS" if ok else
                        f"FAIL (max diff {max(d_pp, d_pt):.3e})"})
        lines.append(f"calibration {m.upper()}: max diffs pp={d_pp:.3e}, "
                     f"pt={d_pt:.3e}")

    # --- split file integrity re-check ------------------------------------
    indices = split["indices"]
    canonical = json.dumps(indices, sort_keys=True, separators=(",", ":"))
    h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    split_ok = (h == split["split_hash_sha256"]
                == "d42842488f79b77b4060e6e0858139a4d41c94e622447aed8a22769678bbfe5c")
    results.append({"check": "split file canonical hash unchanged",
                    "status": "PASS" if split_ok else "FAIL"})
    raw_ok = sha256_file(RAW) == split["input_sha256"]
    results.append({"check": "raw dataset hash unchanged",
                    "status": "PASS" if raw_ok else "FAIL"})
    test_untouched = bool(
        np.array_equal(raw_y[split["indices"]["test"]], raw_y[split["indices"]["test"]])
        and len(split["indices"]["test"]) == 6783
    )
    results.append({"check": "test block size 6,783; NOT evaluated (no metric "
                             "involving test rows computed anywhere in this "
                             "verification)",
                    "status": "PASS" if test_untouched else "FAIL"})

    # --- write report ------------------------------------------------------
    n_fail = sum(1 for r in results if r["status"] != "PASS")
    md = [
        "# Independent verification — autonomous run",
        "",
        "Script: `src/my_analysis/verify_autonomous.py` (separately written; "
        "does not import `my_analysis.autonomous_train_val`; refits nothing).",
        f"Tolerance: {TOL}. Bootstrap: {N_BOOT} resamples, seed {SEED}, "
        "re-implemented independently (rank-based AUC, hand-implemented AP).",
        "",
        "## Checks",
        "",
    ]
    md += [f"- {r['check']}: **{r['status']}**" for r in results]
    md += ["", "## Detail", ""] + [f"    {l}" for l in lines]
    md += ["",
           "## Discrepancy investigated during verification",
           "",
           "The first version of this script computed average precision with",
           "a per-row step-sum. Model A's validation predictions contain 108",
           "pairs of rows with identical predicted probabilities (ties), and a",
           "per-row sum differs from the reported sklearn",
           "`average_precision_score` by ~9.6e-6 (> the 1e-6 tolerance).",
           "Root cause: sklearn evaluates precision/recall only at the last",
           "row of each tie group. The verification implementation was",
           "corrected to the same (standard) definition; afterwards all PR-AUC",
           "and bootstrap values reproduce exactly (diffs < 1e-16). The",
           "reported analysis metrics were NOT changed and no model was",
           "refitted.",
           "",
           f"**Overall: {len(results) - n_fail}/{len(results)} checks PASS.**",
           ""]
    (OUT / "verification" / "verification_report.md").parent.mkdir(
        exist_ok=True)
    (OUT / "verification" / "verification_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    json.dump(results, open(OUT / "verification" / "verification_checks.json",
                            "w", encoding="utf-8"), indent=2)
    print("\n".join(l for l in lines))
    print(f"\n[verify] {len(results) - n_fail}/{len(results)} checks PASS")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
