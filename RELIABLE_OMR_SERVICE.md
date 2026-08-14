# Reliable OMR Python service

This package adds a review-first classical OpenCV pipeline alongside the legacy
OMRChecker CLI. It does not use YOLO, a vision LLM, or an unverified accuracy
claim. Production acceptance thresholds must be validated on representative
PaperCreator sheets.

## Install and run

```bash
python -m pip install -r requirements.service.txt
python -m uvicorn src.reliable_omr.service:app --host 0.0.0.0 --port 8000
```

`GET /health` reports OpenCV, ArUco, PDF, and calibrator capabilities.
`POST /process` is multipart form data:

- `file`: an image or PDF (PDF pages are returned as separate sheets)
- `sheet_definition`: the complete definition JSON string
- `capture_mode`: `scanner` or `mobile`

Set `OMR_API_KEY` to require the ASP.NET client's `X-Api-Key` header on
`POST /process`. When the environment variable is absent or blank, local
development remains unauthenticated. The health endpoint never exposes the key.

The default canonical contract is
`src/reliable_omr/definitions/reliable_a4_v1.json`; its JSON Schema is
`src/reliable_omr/sheet_definition.schema.json`. The definition fixes A4
millimetres, canonical pixels, four unique marker IDs, QR payload fields,
four 25-row answer blocks (100 A-D questions), eight roll-number positions, and
all quality/decision thresholds. Definition schema major version `1` is
currently supported.

QR contracts remain definition-driven and strict. `O1:` keeps the existing
compact/JSON identity fields. The optional
`definitions/reliable_a4_o2_v1.json` contract accepts exactly
`O2:<22-base64url-profile-token>.<6-base64url-signature>`: 32 ASCII bytes with
no padding. Python validates only syntax, length, and unpadded base64url and
returns `qr_payload.profile_token`, `qr_payload.signature`, and `qr_raw`.
ASP.NET must verify the HMAC and resolve the organization/profile identity from
its database; a syntactically valid O2 value is not authenticated by Python.

Library usage:

```python
from src.reliable_omr import ReliableOMRRecognizer

recognizer = ReliableOMRRecognizer(
    sheet_definition="my_sheet_definition.json",
    calibrator="risk_calibrator.json",  # optional
)
result = recognizer.recognize("scan.pdf", capture_mode="scanner")
payload = result.to_dict()
```

Each sheet contains:

- a bounded, metadata-free `rectified_image` JPEG preview (nullable when the
  library option is disabled);
- input quality measurements and gate failures;
- marker IDs, homography, residual error, and fallback diagnostics;
- per-bubble inner/ring/background features;
- explicit `filled`, `empty`, `multiple`, `ambiguous`, or `invalid` field
  states;
- question and roll-number answers with top/second scores, margins, and
  canonical-pixel `{x,y,width,height}` bounding boxes;
- structured review reasons;
- heuristic or calibrated risk, feature values, and contributions.

The top-level response includes `processor_version`. All HTTP response and
multipart names are snake_case and match the ASP.NET contract exactly.

If `cv2.aruco` is missing or all markers cannot be resolved, recognition returns
a structured invalid/review result through a named resize fallback. It never
silently treats that fallback as successful marker rectification.

Set `OMR_CALIBRATOR_PATH` to load one JSON calibrator for the service. If no
calibrator is configured, the response identifies the transparent heuristic
fallback and includes every contribution.

## Diagnostic test sheet

The Angular OMR page is the production print source. For repeatable processor
and API diagnostics, generate a 2480 x 3508 (300-DPI A4) marked fixture:

```bash
python scripts/generate_reliable_omr_test_sheet.py \
  outputs/reliable-a4-v1-marked-test.png --verify
```

Use `--blank` for an unmarked answer area and pass `--exam-id`,
`--template-version`, `--template-checksum`, and `--roll-number` as needed.
The PNG includes the four standard ArUco markers, compact QR identity, all 100
answer rows, and eight roll-number columns. It is synthetic integration data,
not evidence for the production accuracy claim.

## Calibrator training and evaluation

The training CSV needs `sheet_id`, `is_error` (`0`/`1`), and the feature columns
listed by `RISK_FEATURES` in `src/reliable_omr/calibration.py`. Capture device,
pen type, paper/print batch, or similar fields can be retained for subgroup
reporting. Repeated rows are aggregated by `sheet_id` before splitting.

```bash
python scripts/train_risk_calibrator.py data.csv risk.json \
  --subgroup capture_mode --subgroup pen_type --calibration auto

python scripts/evaluate_risk_calibrator.py risk.json held_out.csv \
  --subgroup capture_mode --subgroup pen_type
```

Training uses disjoint sheet-level train/calibration/test splits and records
SHA-256 split manifests in model metadata. Logistic fitting has a NumPy
implementation; scikit-learn is used when available for its logistic estimator
and isotonic calibration. A disjoint Platt-style sigmoid calibration remains
available without scikit-learn. Evaluation reports risk-coverage points,
95% Wilson lower bounds, calibration metrics, AUC, and subgroup summaries.

The calibrator JSON contains numeric coefficients only. Do not load pickled
estimators from untrusted sources.

For physical scanner/mobile capture collection, immutable image hashes,
connected leakage-group splits, and classifier readiness gates, follow
[`DATA_MODELING.md`](DATA_MODELING.md). For the 99.9% auto-accepted-answer
release gate, exact confidence bounds, locked benchmark manifests, and
production drift monitoring, follow
[`ACCURACY_VALIDATION.md`](ACCURACY_VALIDATION.md). Neither workflow reports a
production accuracy result until real human-labelled evidence is supplied.

The labeling workflow uses two independent reviewers per bubble/capture. An
agreement is `verified`; disagreement requires an independent third adjudicator
and is `adjudicated`. Stable capture, physical-sheet, session, device, print
batch, mode, question, and option identifiers are retained. Generate crops with
`scripts/export_bubble_crops.py`, capture risk features with
`scripts/export_capture_features.py`, connected splits with
`scripts/prepare_modeling_dataset.py`, then train/evaluate the risk calibrator
before the optional bubble classifier. Human truth is never inferred from the
service answer, status, confidence, or risk.

No learned bubble classifier is deployed in this service. The optional
`requirements.classifier.txt` environment can train and evaluate a gated 32x32
OpenCV HOG plus multinomial logistic candidate with
`scripts/train_bubble_classifier.py` and
`scripts/evaluate_bubble_classifier.py`. Its JSON artifact is
`shadow_only`; neither `ReliableOMRRecognizer` nor FastAPI loads it. Run the
locked answer-level shadow benchmark before any separate promotion decision.
Synthetic tests/templates are contract examples only, not accuracy evidence.
CNN/MobileNet/EfficientNet remain `BLOCKED_REAL_DATA`.
