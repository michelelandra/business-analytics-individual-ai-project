# Calibration interpretation — chronological final test

**Documentation note, not a new analysis.** Created 2026-11-16. No model was
refitted, recalibrated or re-evaluated, and no metric was changed. Every number
below is copied from an existing results file, with one explicitly labelled
arithmetic diagnostic.

**Model C is a NON-DEPLOYABLE duration-leakage diagnostic** (see §7).

---

## 1. What the observed outcome rate does across the three blocks

| Block | Rows | Positives | Prevalence | Source |
|---|---:|---:|---:|---|
| Training | 31,647 | 1,840 | 0.058141 | `results/guided/validation_report.md` §"Base rates" |
| Validation | 6,781 | 780 | 0.115027 | same |
| Chronological test | 6,783 | 2,669 | 0.39348370927318294 | `results/guided/test/test_metrics.csv` |

The positive rate rises at every step. The test block's prevalence is roughly
6.8 times the training block's (arithmetic on the two figures above). This is a
property of the chronological split, not of the modelling code
(`results/guided/validation_report.md`).

## 2. What the models predict on the chronological test

| Model | Mean predicted probability | Observed prevalence | Ratio (prevalence ÷ mean predicted) |
|---|---:|---:|---:|
| A | 0.06884271817958906 | 0.39348370927318294 | ≈ 5.72 |
| B | 0.06766018052577288 | 0.39348370927318294 | ≈ 5.82 |
| C (diagnostic) | 0.06297960693204888 | 0.39348370927318294 | ≈ 6.25 |

Source: `results/guided/test/test_metrics.csv` (`mean_predicted`, `prevalence`).
Ratios are arithmetic on those two saved columns.

The same level problem was already visible on validation, before the test block
was opened: mean predicted 0.054843 (A) and 0.052808 (B) against a validation
prevalence of 0.115027, i.e. roughly 2.1× and 2.2× too low
(`results/guided/validation_report.md`, §"Consequence for Brier and
calibration"). The test block magnifies an already-documented gap; it does not
introduce a new kind of error.

## 3. Why this is substantial underprediction, not noise

The decile table `results/guided/test/test_calibration.csv` shows the gap in
every part of the score range:

| Model | Lowest decile: predicted → observed | Highest decile: predicted → observed |
|---|---|---|
| A | 0.0404 → 0.2887 | 0.1055 → 0.4698 |
| B | 0.0381 → 0.2518 | 0.1069 → 0.5376 |
| C | 0.0205 → 0.0324 | 0.2497 → 0.6834 |

A model whose probabilities were merely noisy would scatter above and below the
diagonal. Here every A and B decile sits far below its observed rate: the decile
mean predictions span roughly 0.04–0.11 while the outcome occurs in roughly a
quarter to a half of each decile. That is a systematic level error, not sampling
noise.

The predicted level on test (≈0.068–0.069) is close to the **training** prevalence
(0.058) and far from the test prevalence (0.393). That description is consistent
with the fitted probabilities reflecting the base rate the model was trained on.
It is a description, not an established cause: these artifacts do not separate
base-rate shift from regularisation/shrinkage or from any other mechanism.

## 4. What the discrimination and ranking metrics say

| Model | ROC-AUC | Average precision | No-skill AP (prevalence) | Lift@10% | Brier |
|---|---:|---:|---:|---:|---:|
| A | 0.5556035709881709 | 0.437868541560834 | 0.39348370927318294 | 1.1919846682825095 | 0.34269133584152267 |
| B | 0.5858003804279422 | 0.4670448372049348 | 0.39348370927318294 | 1.3644101234428725 | 0.34242745574504724 |
| C | 0.767392793580775 | 0.6193567633789513 | 0.39348370927318294 | 1.735499689983654 | 0.3380019092016863 |

Source: `results/guided/test/test_metrics.csv`.

- **ROC-AUC** is only modestly above chance for A and B. The frozen paired
  bootstrap gives B − A ΔROC-AUC = 0.03019680943977132, 95% CI
  [0.02608477195825771, 0.03426655663048662]
  (`results/guided/test/hypothesis_tests_test.json`): a real improvement in
  ranking, but a small one.
- **Average precision** is above the test-specific no-skill baseline of 0.3935
  for both models (0.4379 for A, 0.4670 for B), again a modest gain.
- **Lift@10%** of 1.19 (A) and 1.36 (B) means the top decile contains about 19%
  and 36% more positives than random selection. Useful for prioritisation,
  not evidence of a well-scaled probability.
- **Brier** is 0.3427 (A) and 0.3424 (B). The difference between them is
  0.000264 — negligible relative to the level error described next.

### Hindsight constant-predictor diagnostic (not an operational baseline)

A constant predictor that assigns every test row the observed test prevalence
(0.39348370927318294) would obtain

```
Brier = p(1 − p) = 0.2386542798098002
```

This figure is **arithmetic on the observed test prevalence and therefore uses
the test labels**. It is a **hindsight diagnostic only**. It is not an
operationally available forecast (no real system knows the future base rate) and
it is **not a fair prospective baseline**. Its only purpose here is to show the
size of the probability-quality gap: both A and B are worse on Brier than a
constant that is told the answer in advance.

## 5. Why better discrimination does not imply good calibration

ROC-AUC, average precision and lift depend only on the **ordering** of predicted
scores. Any strictly increasing transformation of the scores leaves them
unchanged. Brier score depends on the **level** of the probabilities. A model can
therefore rank customers correctly and still assign probabilities that are wrong
by a factor of five — which is what the tables above show.

The practical consequence: A and B can be used to answer "who is more likely to
subscribe?" but not "what is the probability this customer subscribes?". The
second question is the one a business case usually needs, and on this test block
the models do not answer it.

## 6. Why Model B is not ready for operational deployment

- Its mean predicted probability (0.0677) is about one-sixth of the observed
  test rate (0.3935), and every decile is similarly displaced
  (`results/guided/test/test_calibration.csv`).
- Its Brier score is worse than the hindsight constant diagnostic in §4.
- No recalibration, prior correction or threshold tuning was pre-registered or
  performed at this stage (`results/guided/validation_report.md`;
  `plan.md` §5.7). The saved probabilities are therefore the raw model output,
  not decision-ready risk estimates.
- Its advantage over A is a ranking improvement of 0.0302 ROC-AUC. That supports
  "B ranks better than A", not "B is deployable".

Model B should be described as a **ranking prototype with a documented
probability-scale defect**. Estimating a deployable probability would require a
separately designed and pre-registered calibration procedure fitted without the
test block; this note does not propose or perform one.

The subgroup descriptives point the same way: mean predicted probabilities are
0.070077 (never contacted) and 0.065311 (previously contacted) against group
prevalences of 0.331439 and 0.453779
(`results/verification/subgroup_descriptives_test.md`). The underprediction is
not confined to one subgroup.

## 7. Why Model C is a non-deployable leakage diagnostic

Model C adds `duration`, the length of the marketing call, which is only known
after the call has happened. It therefore cannot be used to decide whom to call.
It exists only to quantify how much of the outcome is encoded in call timing:
C − B ΔROC-AUC = 0.1815924131528327, 95% CI
[0.16547503780013015, 0.19765453993900986], against a pre-registered materiality
threshold of 0.05 (`plan.md` §3.3;
`results/guided/test/hypothesis_tests_test.json`).

Model C's much better Brier score (0.3380) is part of the same diagnostic, not
evidence that leakage makes a model better in practice. **It must never be
presented as an operational model.**

## 8. Limitations of this note

- The cause of the level gap is **not** established. Base-rate shift,
  regularisation and other mechanisms are not separated by these artifacts, and
  no recalibration experiment was run.
- The Brier comparison in §4 uses a hindsight constant; it is a diagnostic, not a
  prospective baseline, and should not be quoted as one.
- These are single-block descriptive results from one chronological split. They
  do not support causal claims and do not establish performance on future
  periods.
- The random-split sensitivity result is optimistic by construction and does not
  replace the chronological test.

## 9. Sources

- `results/guided/test/test_metrics.csv`
- `results/guided/test/test_calibration.csv`
- `results/guided/test/hypothesis_tests_test.json`
- `results/guided/validation_report.md`
- `results/verification/subgroup_descriptives_test.md`
- `results/verification/final_test_recompute_report.md`
- `plan.md` §3.3, §5.7
- `mistakes.md` (H-5, H-7)
