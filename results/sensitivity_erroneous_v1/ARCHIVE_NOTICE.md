# INVALID FOR THE AUTHORIZED SENSITIVITY ANALYSIS — PRESERVED AS AGENT ERROR EVIDENCE

This directory is a byte-for-byte archive of the guided random-split sensitivity
run that was produced on 2026-11-14 and later found to implement the wrong
split order.

The authorized procedure separates the TEST block first:

    remaining_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=6783, stratify=y, random_state=42
    )
    train_idx, val_idx = train_test_split(
        remaining_idx, test_size=6781, stratify=y[remaining_idx], random_state=42
    )

The archived run instead separated the TRAINING block first, using
`test_size=13564` in the first call and `test_size=6783` in the second call.

Do not use the archived split, predictions, metrics, calibration, bootstrap or
report for the authorized sensitivity analysis. They are preserved only as
evidence of the agent error and its consequences.

See `results/audit/guided_random_split_protocol_discrepancy.md`.
