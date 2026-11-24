# WARNING — the results in this directory are INVALID

> **The Guided random-split sensitivity run stored in this directory is INVALID
> for the authorised sensitivity analysis.** It used the wrong split protocol:
> train-first instead of the authorised **test-first** procedure. Block sizes
> were correct and the split was internally reproducible, so the error cannot
> be seen from the files here.

Do not cite, quote or reuse any number from this directory.

## What went wrong

The authorised procedure holds out the test block first (6,783 rows), then the
validation block (6,781 rows) from the remainder. The implementation here split
off the 13,564 held-out rows first and then took 6,783 from them, so the index
membership was wrong even though the block sizes matched. The error was found by
independent human regeneration, not by internal checks.

Root-cause audit: `results/audit/guided_random_split_protocol_discrepancy.md`.
Log entry: `mistakes.md`, entry **A-5**.

## Where to look instead

| Purpose | Location |
|---|---|
| Preserved evidence of the error (byte-for-byte) | `results/sensitivity_erroneous_v1/` — see `ARCHIVE_NOTICE.md`, `archive_manifest.json`, `SHA256SUMS.txt` |
| Valid corrected sensitivity analysis | `results/sensitivity_corrected_v1/` — see `report_evaluation.md` and `run_manifest_evaluation.json` |
| Primary result (authoritative) | `results/guided/test/` — chronological test evaluation |

## Status of the files here

The files in this directory are **preserved unchanged** as evidence and have not
been edited or deleted. Only this README was added, on 2026-11-16, so that the
directory can no longer be mistaken for a valid analysis. The authoritative
corrected run is a post-test secondary sensitivity analysis: it is optimistic by
construction and does not replace the primary chronological test.
