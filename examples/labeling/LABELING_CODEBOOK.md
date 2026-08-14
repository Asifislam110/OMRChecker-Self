# Synthetic labeling codebook

This is a format-only synthetic codebook. It is not a measured dataset and must
not be used as production accuracy evidence.

Bubble labels:

- `empty`: no intentional mark inside the bubble.
- `filled`: one clearly filled bubble.
- `partial`: an incomplete fill that is not simply an erased mark.
- `crossed`: a cross, tick, slash, or similar non-fill mark.
- `erased`: visible erasure residue or a changed/removed mark.
- `multiple`: the bubble belongs to a question where more than one option was
  intentionally selected; use this only when the labeling policy treats each
  crop as part of that multiple-mark outcome.

Each annotation has `reviewer_1_label` and `reviewer_2_label`. Reviewers work
independently and cannot use `machine_label` as truth. If labels agree, copy the
shared value to `label`, set `label_status=verified`, and leave
`adjudicator_id` empty. If they disagree, an independent third person chooses
the final `label`, records their ID, and sets `label_status=adjudicated`.

Identifiers are stable opaque values. `physical_sheet_id` follows the piece of
paper across rescans; `capture_session_id` follows one collection session on
one device/mode; `device_id` and `print_batch_id` identify the acquisition
device and print batch. Every crop inherits all four IDs and `capture_mode` from
its parent capture.

Bounding boxes use canonical rectified-image pixels with nonnegative `x/y` and
positive `width/height`, wholly inside the image. SHA-256 columns record exact
source bytes. Values made only of zeros in these templates are explicit
placeholders and will not validate against real files.
