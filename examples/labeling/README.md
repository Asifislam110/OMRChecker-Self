# Labeling templates (format only)

Every file in this directory is synthetic placeholder data provided only to
demonstrate file formats. None of it is a physical capture, measured benchmark,
or production accuracy evidence. Do not cite these rows, labels, hashes, paths,
or generated outputs as evidence of OMR performance.

Suggested real-data sequence:

1. Copy the templates outside this directory and replace every `PLACEHOLDER`
   value with immutable real capture data.
2. Have two independent reviewers label each bubble without seeing the machine
   answer. Agreement becomes `verified`; disagreement requires a third,
   independent adjudicator and becomes `adjudicated`.
3. Export crops with `scripts/export_bubble_crops.py`.
4. Export capture features with `scripts/export_capture_features.py`. The
   `is_error` value must come from `capture_verification.template.json`-shaped
   human verification, never from processor status, risk, or answer fields.
5. Prepare connected, group-safe splits before any fitting.

`captures.template.csv` intentionally uses nonexistent `PLACEHOLDER/...` paths.
The hash-looking values in other templates are placeholders, not hashes of real
evidence. The production validators will reject nonexistent files or mismatched
hashes.
