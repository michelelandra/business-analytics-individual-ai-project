# Execution log — autonomous random-split sensitivity analysis (post-test)

Date: 2026-11-14. Agent: GLM-5.3-Flash Free (opencode) in OpenCode.
All times local; UTC timestamps are recorded in the JSON artifacts.

## Chronology

1. **Read / context.** Read `AGENTS.md`, `plan.md` v1.2 (§5.5 secondary
   random split, §5.7 metrics), the original autonomous scripts
   (`autonomous_train_val.py`, `verify_autonomous.py`,
   `autonomous_test_evaluate.py`), and
   `results/autonomous/models/model_settings.json` (verified frozen settings:
   A/B/C all C=0.01, L2, lbfgs, max_iter=5000, random_state=42). No
   Guided-run material accessed.

2. **Script written.** Created
   `src/my_analysis/autonomous_random_split_sensitivity.py` (new, dedicated).
   It reproduces verbatim the original AUTONOMOUS implementations:
   `prepare_frame` (= `decode_categoricals`), `build_pipeline`, `evaluate`,
   `calibration_table`, `paired_bootstrap` — with hyperparameters frozen as
   constants (C=0.01 for all models; no grid, no CV, no selection) and the
   two-step stratified random split.

3. **Two cosmetic edits BEFORE the first execution** (recorded for accurate
   chronology; no post-result code change occurred):
   a. removed a dead `split_hash = ... if exists else None` lookup that
      referenced the not-yet-written output file;
   b. simplified the `prevalence` dict comprehension (identical semantics).
   After these edits the script was final; its SHA-256 at execution time is
   recorded in `run_manifest_sensitivity.json` →
   `e55c1711cfbfc5fce44d2844bd15beb4b2fd881d7f8af86741204c93ba382078`.

4. **Single execution (no retries, no reruns).** Command:
   `$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m my_analysis.autonomous_random_split_sensitivity 2>&1 | Tee-Object -FilePath results/sensitivity/console_output.txt`
   Result: **succeeded end-to-end on the first attempt, zero errors.** Full
   console log: `console_output.txt`.

5. **In-run integrity checks (all passed):**
   - raw dataset SHA-256 and frozen temporal split canonical hash verified;
   - pre-execution hash baseline captured for 8 governing files, 33 files
     under `results/autonomous/` (recursive) and 13 source files;
   - random split verified: sizes match protocol; no duplicate indices; full
     coverage exactly once; blocks mutually disjoint; indices within range;
   - frozen-settings static audit: True;
   - post-execution preservation check: **changed existing files = NONE**
     (governing, `results/autonomous/**`, `src/my_analysis/**`).

6. **Post-execution verification (read-only, temp script outside the
   workspace):**
   - split **determinism**: re-running the two-step split in-memory
     reproduces the canonical fingerprint exactly
     (`6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`);
   - class counts match the Guided-run reference values exactly
     (3,702 / 793 / 794 positives) — reported as an observed coincidence of
     the same pre-specified procedure; the actual index-array comparison is
     reserved to the project author (per protocol item 3).

## Known disclosure: self-referential hash inside `random_split_v1.json`

The field `split_file_sha256` inside `random_split_v1.json`
(`12177a2662ccbe6ca66c726a0877e5501bcd71f6098a07a6237769e9d1c20513`) records
the SHA-256 of the split file **as first written, before that same field was
appended** — a file cannot contain its own final hash. The authoritative
hash of the **final on-disk** `random_split_v1.json` is the one recorded in
`run_manifest_sensitivity.json` →
`385e31cf8323b2a510cdfc7263b90587f75cea04a3409d19a2380e7509f73e03`
(re-verified after execution). The serialization-independent reference for
cross-run comparison is the **canonical index fingerprint**
`6975172cf5e2688d8e0c2e887e5242257a984ff7e262b590ce550498a210dfa0`
(definition inside the JSON). No file was rewritten to reconcile this; it is
documented instead.

## Warnings raised during fitting

- sklearn 1.9 `FutureWarning` on the deprecated `penalty=` argument (3
  occurrences, one per model; `penalty="l2"` is the applied and default
  penalty). Harmless; no effect on results. Listed in `report.md` and in
  `console_output.txt`.

## Errors, failed attempts, corrections, reruns

- Execution errors: **none.**
- Failed attempts: **none** (single successful run).
- Corrections after results were seen: **none** (the two cosmetic edits in
  step 3 preceded the only execution).
- Reruns: **none.**

## Hashes of the files written outside the script (this log and prompt log)

Recorded post-hoc for completeness; the script-hashed artifacts are listed in
`run_manifest_sensitivity.json`. `report.md` was edited once after the run to
add the self-referential-hash clarification (see the disclosure above); its
final hash is listed here:

- `results/sensitivity/console_output.txt` —
  `46abc27d4ab4c5eaba64413e8074b3ff3c4f5a3f4e8e1f806e8c38ce9a9e0d6f`
- `results/sensitivity/prompt_log.md` —
  `033c42b75c9710f0d9e043508d60fe439792d27d25bf530006c608c9190decc1`
- `results/sensitivity/execution_log.md` — self-referential; the hash of
  this file's final content is reported in the conversation response and
  cannot be embedded in itself (`e5ab5ff62dfdab5669a2be6fb183a0ddbe0a4833eacb3025591ae0068792fe85`
  for the content up to and including this line)
- `results/sensitivity/report.md` (final, after the hash clarification edit) —
  `4df3e64c422479e1f4a5d320f6b0ca4d647c1c228a86c7e615a9b19258b7fde5`
- `results/sensitivity/random_split_v1.md` —
  `4eb4924d02ed8d0e8f58b6a327973defc07b82fb7eaf160441d818e5a9727f3a`

The two hash clarifications (self-referential split-file hash; this log's
own hash) are documentation-only edits made after the single successful
execution; they touch no metric, prediction, split index or model output.


## End state

Authorized random-split sensitivity analysis complete. No subgroup analysis
started; the primary chronological report and all primary artifacts remain
untouched. Stopping here as instructed.
