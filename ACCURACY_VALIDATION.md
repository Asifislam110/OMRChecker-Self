# Reliable OMR accuracy validation

This workflow tests one deliberately narrow release claim: among answers routed
to automatic acceptance, at least 99.9% are correct; answers above the selected
risk threshold are sent to review. It does not claim that every answer is
automatically accepted.

No production benchmark, drift baseline, or production window is included in
this repository. The supplied regression fixture is synthetic and is only for
testing validation behavior. It is not accuracy evidence.

## Required benchmark evidence

Use answer-level CSV, JSON, JSONL, or NDJSON records. The default field contract
is:

- `answer_id`: globally unique evaluated answer ID.
- `group_id`: leakage group. Choose the largest shared source that could leak
  information, such as student, physical sheet, capture session, or original
  image family. All augmentations and rescans of one source must share a group.
- `risk`: model risk in `[0, 1]`; lower values are safer.
- `is_correct`: `1/0` or `true/false`, determined against independent human
  ground truth.
- `ground_truth_source`: must be `human` by default.
- `capture_mode`: at least `scanner` or `mobile`.
- Additional demographic, device, paper, lighting, site, and form-version
  fields should be supplied where subgroup release gates are required.

For organization-designed sheets, include stable `organization_id` and
`sheet_design_id` (or equally explicit opaque fields). Gate every production
organization/design that is in release scope; an aggregate pass must not hide
an underrepresented or failing design. Keep each physical sheet, rescan, and
related capture in one leakage group even when designs or organizations match.
Missing organization/design sample counts are `insufficient_data`, not evidence
that aggregate scanner/mobile performance transfers.

Predictions must be frozen before labels are joined. Do not manually remove
difficult groups or records after the split manifest has been created.

## Immutable group splits

Create the manifest once from the complete benchmark:

```powershell
python scripts/create_validation_split_manifest.py `
  benchmark_answers.jsonl validation_split_manifest.json `
  --group-id-field group_id
```

The manifest assigns whole groups to train, calibration, and final test splits
using a deterministic SHA-256 ordering. It contains a hash over its canonical
contents. Commit or otherwise lock the manifest, identify it by
`manifest_sha256`, and never regenerate it to improve a result.

Validation rejects modified manifests, overlapping groups, unknown groups, and
omitted manifest groups. It also locks the record count within every group so
removing difficult answers while retaining the group is rejected as
record-level cherry-picking. A mutable `split` column in the benchmark is
ignored; the manifest is the only split authority.

## Release gate

Run:

```powershell
python scripts/validate_reliable_omr_accuracy.py `
  benchmark_answers.jsonl validation_split_manifest.json `
  --output accuracy-report.json `
  --subgroup-field device_model `
  --subgroup-field organization_id `
  --subgroup-field sheet_design_id `
  --required-subgroup capture_mode=scanner `
  --required-subgroup capture_mode=mobile
```

Scanner and mobile gates are enabled by default. Every observed value of every
configured subgroup field is gated. `--required-subgroup FIELD=VALUE` also
requires a named value even when it is absent, in which case the result is
`insufficient_data`.

The procedure is intentionally ordered:

1. Validate human labels, unique answer IDs, the manifest hash, complete group
   coverage, and disjoint group assignments.
2. Consider tied risk thresholds on the calibration split only.
3. Select the maximum-coverage threshold for which overall and subgroup
   calibration gates pass.
4. If no calibration threshold passes, leave the final test untouched and
   report `fail` or `insufficient_data`.
5. Otherwise apply the frozen threshold exactly once to the final test. Answers
   with `risk <= threshold` are auto-accepted; all others require review.
6. Report final overall and subgroup gates plus risk-coverage points. Tied risk
   values are accepted together.

The JSON report records that test records were not used for threshold selection
and includes the manifest hash, routing coverage, review count, observed errors,
point accuracy, confidence bounds, subgroup results, and risk-coverage results.
Exit code `0` means every final gate passed, `2` means a gate failed or evidence
was insufficient, and `1` means invalid input.

### Statistical method

Each gate uses a one-sided exact Clopper-Pearson lower confidence bound for a
binomial correctness probability. The default policy requires the lower 95%
bound, not the point estimate, to be at least `0.999`. It also requires at least
3,000 auto-accepted answers overall and in each gated subgroup.

With zero observed errors, the exact lower bound is
`alpha ** (1 / n)`. At 95% one-sided confidence and a 99.9% target, at least
2,995 accepted answers are required even before a larger configured minimum is
applied. Therefore zero errors in a small sample cannot pass. The report labels
that case `insufficient_data` instead of treating 100% observed accuracy as
proof.

The default is intentionally conservative. Policy owners may change target,
confidence, or minimum counts through CLI options, but a report with weaker
settings does not establish the original 99.9% claim.

## Production drift monitoring

Build a baseline only from an approved real reference window:

```powershell
python scripts/build_omr_drift_baseline.py `
  approved_reference_window.jsonl omr-drift-baseline.json
```

The command does not generate or impute records. By default it captures:

- Numeric distributions: `confidence`, `risk`, and `quality_score`.
- Categorical distributions: `quality_status`, `status`, `capture_mode`,
  `review_outcome`, and `correction_outcome`.
- Review rate, correction rate where an outcome is known, and coverage of
  accepted answers that later received human ground truth.

Use normalized `quality_score` and a stable `quality_status` vocabulary when
exporting service telemetry. Recommended correction outcomes are `corrected`,
`confirmed_correct`, `no_correction`, and `pending`. Keep `pending` records in
the window; they are excluded from the correction-rate denominator but remain
visible in the categorical distribution.

Monitor a real production window:

```powershell
python scripts/monitor_omr_production_drift.py `
  omr-drift-baseline.json production-window.jsonl `
  --output drift-report.json
```

Numeric drift uses population stability index (PSI) against immutable baseline
quantile bins. Categorical drift uses total variation distance and separately
alerts on unseen categories. Defaults are PSI warning/alert at `0.10/0.25` and
categorical warning/alert at `0.10/0.20`. The report also checks minimum window,
field, correction-outcome, and human-ground-truth sample counts. It exits
nonzero for warnings, alerts, insufficient data, or invalid input so an
orchestrator can notify operators.

Drift and outcome records should be append-only, privacy-reviewed, and joined
to corrections without exposing answer ground truth to the model at inference
time. Rebuild a baseline only after an explicitly approved release; do not
silence an alert by automatically rebasing.

## Limitations and real-data blocker

- Correctness can be measured only where independent human ground truth exists.
  Confidence, risk, agreement, and a lack of corrections are not substitutes.
- Zero observed errors is not automatically sufficient. Sample size and the
  one-sided confidence bound determine whether the target is established.
- Group safety depends on selecting a group ID that actually contains every
  related image, rescan, augmentation, student, or acquisition session.
- Subgroup gates can reveal measured disparities only for fields that are
  collected lawfully and represented with enough accepted, human-labelled
  answers.
- Distribution drift is an alerting signal, not proof of a causal accuracy
  change. Operational investigation and labelled audits remain necessary.
- A real release decision is currently blocked until the project supplies a
  locked, representative benchmark with human answer-level ground truth and
  enough calibration/test samples per scanner, mobile, and other required
  subgroup. Production monitoring is likewise blocked until a real approved
  baseline and subsequent real production windows are available.
