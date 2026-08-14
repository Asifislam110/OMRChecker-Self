"""Generate a canonical diagnostic OMR PNG from the production definition."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reliable_omr.opencv import require_cv2  # noqa: E402
from src.reliable_omr.recognizer import ReliableOMRRecognizer  # noqa: E402
from src.reliable_omr.test_sheet import (  # noqa: E402
    cycling_answers,
    generate_canonical_test_sheet,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a pixel-exact 300-DPI diagnostic sheet. The Angular page "
            "remains the production print source."
        )
    )
    parser.add_argument("output_png", type=Path)
    parser.add_argument(
        "--blank",
        action="store_true",
        help="Leave answer bubbles blank instead of cycling A/B/C/D.",
    )
    parser.add_argument("--roll-number", default="12345678")
    parser.add_argument("--exam-id", default="test")
    parser.add_argument("--template-version", default="1")
    parser.add_argument("--template-checksum", default="deadbeef")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the generated pixels through ReliableOMRRecognizer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    answers = {} if args.blank else cycling_answers()
    image = generate_canonical_test_sheet(
        answers=answers,
        roll_number=args.roll_number,
        exam_id=args.exam_id,
        template_version=args.template_version,
        template_checksum=args.template_checksum,
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    cv2 = require_cv2()
    if not cv2.imwrite(str(args.output_png), image):
        raise SystemExit("OpenCV could not write {}".format(args.output_png))

    report = {
        "output": str(args.output_png),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "marked_answers": len(answers),
        "synthetic_accuracy_evidence": False,
    }
    if args.verify:
        result = ReliableOMRRecognizer(
            include_rectified_image=False
        ).recognize_image(image)
        report["verification"] = {
            "status": result.status.value,
            "question_count": len(result.questions),
            "roll_number": result.roll_number,
            "qr_payload": result.qr_payload,
            "qr_raw": result.qr_raw,
            "detected_marker_ids": result.rectification.detected_marker_ids,
            "matching_answers": sum(
                1
                for question in result.questions
                if answers.get(question.question) == question.answer
            ),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
