# OMR Data and Modeling

This repository does not contain representative physical-sheet captures. No
scanner/mobile accuracy has been measured, and the tooling below does not
manufacture samples or benchmark numbers.

## Capture plan

Collect real, manually verified captures from both `scanner` and guided
`mobile` workflows. Vary physical sheets, print batches, capture
sessions, devices, paper/marking tools, lighting, blur, rotation, and expected
failure cases. Keep the original image bytes immutable after the audit.

Use stable opaque identifiers. One `physical_sheet_id` identifies one printed
piece of paper, even when that paper is captured repeatedly. A
`capture_session_id` identifies one collection session on one device and in one
capture mode. A `print_batch_id` identifies sheets produced together.

Files in `examples/labeling/` are synthetic, format-only placeholders. Their
rows, paths, labels, and hash-looking values are not physical observations and
must never be cited as production accuracy evidence.

## Two-reviewer annotation and adjudication

Freeze the image and processor-result bytes before labeling. Two different
reviewers independently label every bubble without seeing or copying the
machine answer. Record stable `reviewer_1_id`/`reviewer_2_id` values and each
reviewer's label. When they agree, the shared label is `verified` and
`adjudicator_id` is blank. When they disagree, a third person who is different
from both reviewers sets the final label and the status is `adjudicated`.
Machine status, answer, confidence, and risk are never human truth.

The allowed bubble labels are `empty`, `filled`, `partial`, `crossed`, `erased`,
and `multiple`. `human_annotations.template.csv` shows the exact columns. The
exporter rejects missing/unknown question-option keys, duplicate annotations,
invalid labels, incomplete reviewer/adjudicator state, out-of-image processor
geometry, and metadata disagreement.

Use `capture_id` for one image, `physical_sheet_id` for one printed piece of
paper, `capture_session_id` for one device/mode collection session, `device_id`
for the acquisition device, and `print_batch_id` for sheets printed together.
Every bubble crop inherits all five identifiers plus `capture_mode`; do not
invent crop-level split or group IDs.

Generate crops from a rectified image, the matching processor JSON, complete
human annotations, and capture metadata:

```powershell
python scripts/export_bubble_crops.py `
  rectified.png processor-result.json human-annotations.csv `
  capture-metadata.json exported-bubbles
```

The command writes deterministic 64x64 grayscale PNGs and
`exported-bubbles/bubble_crops.csv`. Each row includes canonical source bounds,
the machine label only as a comparison field, and SHA-256 values for the crop,
rectified image, processor result, annotations, and metadata.

Generate an `omr-capture-v1` feature row separately:

```powershell
python scripts/export_capture_features.py `
  original-capture.png processor-result.json capture-metadata.json `
  capture-verification.json capture-row.csv
```

Capture verification follows the same independent two-reviewer/third-reviewer
rule for `is_error`. The exporter copies only `confidence_diagnostics.features`
into the risk columns. It never derives `is_error` from machine answers, status,
confidence, or risk.

## Capture CSV contract (`omr-capture-v1`)

One row is one physical image capture and one observed OMR outcome. The input
CSV must include:

- `schema_version`: exactly `omr-capture-v1`
- `capture_id`: globally unique stable identifier
- `image_path`: absolute or relative to the CSV
- `physical_sheet_id`, `capture_session_id`, `device_id`, `print_batch_id`
- `capture_mode`: exactly `scanner` or `mobile`
- `is_error`: exactly `0` or `1`, where `1` means the OMR output disagrees
  with verified ground truth for that capture
- `label_status`: `verified` or `adjudicated`; draft labels are rejected
- every feature in `src.reliable_omr.calibration.RISK_FEATURES`

`capture_sha256` is optional on raw input. The preparation command computes it;
if supplied, it must match. Extra columns such as `pen_type`, `paper_stock`, or
`lighting_condition` are retained and can be used for subgroup reporting.
Feature values must be finite numbers.

The audit rejects missing images, duplicate capture IDs, duplicate image bytes,
invalid labels/modes, inconsistent physical-sheet print batches, and capture
sessions that span devices or modes.

## Prepare and split

```powershell
python scripts/prepare_modeling_dataset.py captures.csv prepared
```

The command writes:

- `captures.normalized.csv`: stable row/column order and verified image hashes
- `splits.json`: versioned deterministic train/calibration/test assignments
- `audit.json`: overall and per-split label/capture-mode balance
- `classifier_readiness.json`: a blocked/ready gate, never fabricated metrics

`splits.json` records SHA-256 hashes for the source CSV, normalized CSV, capture
content set, every split ID set, and the complete split assignment. The splitter
forms connected components across `physical_sheet_id`, `capture_session_id`,
`device_id`, and `print_batch_id`; a value from any of those fields can occur in
only one split. This conservative rule may require collecting from several
independent devices, sessions, and print batches before three splits are
possible. Re-running with the same input, seed, and fractions is deterministic.

Structural readiness only means each split can support fitting/evaluation and
both capture modes are present. It does not prove that the collection is
representative.

## Train and evaluate the existing risk calibrator

The prepared path calls the existing `train_serialized_calibrator`; it does not
implement a second calibrator.

```powershell
python scripts/train_risk_calibrator.py `
  prepared/captures.normalized.csv risk.json `
  --split-manifest prepared/splits.json `
  --model-version physical-v1 --backend numpy --calibration auto

python scripts/evaluate_risk_calibrator.py `
  risk.json prepared/captures.normalized.csv `
  --split-manifest prepared/splits.json --split test
```

Training re-verifies image bytes, normalized-CSV provenance, split hashes, all
assignments, and all leakage groups before fitting. Test captures are never used
for fitting or post-calibration. The legacy sheet CSV path remains available for
older experiments but does not provide the new physical-capture provenance.

Do not promote a model based only on aggregate metrics. Review test metrics by
`capture_mode` and any collected device, print, marking-tool, and environment
subgroups. Report sample counts and uncertainty with every result.

## Gated lightweight bubble classifier benchmark

The repository now contains an offline HOG plus multinomial logistic benchmark,
not a deployed classifier. Install its optional dependency only in the
benchmark environment:

```powershell
python -m pip install -r requirements.classifier.txt
```

The bubble CSV contract is versioned `omr-bubble-crop-v1`. It requires unique
crop/capture/question/option keys, all inherited grouping identifiers,
canonical source bounds, final human label/status, crop path/hash, and hashes
of every labeling input. Each crop inherits its parent capture's split.
Crop-level random splitting is forbidden. Duplicate crop bytes, unknown
parents, parent metadata mismatch, invalid hashes, and group leakage fail.

```powershell
python scripts/check_classifier_readiness.py `
  prepared/captures.normalized.csv prepared/splits.json bubble_crops.csv `
  --output classifier_readiness.json
```

The default benchmark gate requires, for both `scanner` and `mobile`,
500 verified `empty` and 500 verified `filled` crops, 25 independent physical
sheets, and at least 50 test crops per required label/mode. These are
data-sufficiency gates, not accuracy targets. The report also checks availability
of OpenCV HOG and scikit-learn logistic regression.

If the gate is eventually met, benchmark a 32x32 OpenCV HOG plus logistic
regression candidate against the deterministic baseline:

```powershell
python scripts/train_bubble_classifier.py `
  prepared/captures.normalized.csv prepared/splits.json `
  exported-bubbles/bubble_crops.csv bubble-classifier.json `
  --model-version physical-shadow-v1

python scripts/evaluate_bubble_classifier.py `
  bubble-classifier.json prepared/captures.normalized.csv `
  prepared/splits.json exported-bubbles/bubble_crops.csv `
  --output bubble-benchmark.json
```

The trainer first requires the existing readiness gate and re-verifies the
group-safe split manifest. It fits scaling and logistic candidates on `train`,
uses only `calibration` to select logistic `C`, and evaluates the selected,
still-train-only estimator once on `test`. The untouched test report includes
multiclass confusion/per-class metrics, scanner/mobile metrics, and
filled-vs-not risk/coverage beside deterministic machine labels where those
labels are available.

The `omr-lightweight-classifier-export-v1` artifact is numeric JSON containing
the exact HOG configuration, scaling, class list, coefficients/intercepts,
seed, dataset/split/crop hashes, selection record, version, and test metrics.
Pickle/joblib is not accepted. The loader validates dimensions and provenance
and exposes only shadow predictions. Nothing enables this model in
`ReliableOMRRecognizer` or the service.

Run the risk calibrator workflow first, then the bubble benchmark. Freeze both
artifacts and run the answer-level shadow benchmark in `ACCURACY_VALIDATION.md`
before considering any promotion. Report scanner/mobile, sheet-design,
organization, device, print batch, and other legally collected subgroups with
sample counts and uncertainty.

Synthetic fixtures and templates exercise contracts only. They establish no
accuracy. CNN, MobileNet, EfficientNet, TensorFlow, and PyTorch remain
`BLOCKED_REAL_DATA` until representative real captures, locked group-safe
splits, and human-adjudicated benchmarks exist.

