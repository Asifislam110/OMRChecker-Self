"""Generate deterministic canonical sheets for integration diagnostics.

The Angular SVG/HTML renderer remains the production print source. This module
creates a pixel-exact 300-DPI fixture from the same definition so teams can
exercise recognition and cross-stack contracts without treating synthetic data
as accuracy evidence.
"""

from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.reliable_omr.definition import (
    iter_answer_geometry,
    iter_roll_geometry,
    load_sheet_definition,
    mm_to_px,
    point_mm_to_px,
)
from src.reliable_omr.opencv import require_cv2


def compact_qr_payload(
    exam_id: str,
    template_version: str,
    template_checksum: str,
) -> str:
    payload = "O1:{}.{}.{}".format(
        str(exam_id).strip(),
        str(template_version).strip(),
        str(template_checksum).strip().lower()[:8],
    )
    if len(payload.encode("utf-8")) > 32:
        raise ValueError("Compact QR payload exceeds QR Version 2-L capacity")
    return payload


def generate_canonical_test_sheet(
    definition: Optional[Mapping[str, Any]] = None,
    answers: Optional[Mapping[int, str]] = None,
    roll_number: str = "12345678",
    exam_id: str = "test",
    template_version: str = "1",
    template_checksum: str = "deadbeef",
) -> np.ndarray:
    """Return a labelled canonical BGR sheet suitable for PNG output."""

    cv2 = require_cv2()
    loaded = load_sheet_definition(definition)
    page = loaded["page"]
    image = np.full(
        (page["canonical_height_px"], page["canonical_width_px"], 3),
        255,
        dtype=np.uint8,
    )
    black = (0, 0, 0)

    dictionary_id = getattr(cv2.aruco, loaded["aruco"]["dictionary"])
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    for marker in loaded["aruco"]["markers"]:
        x, y = _point(marker["top_left_mm"], loaded)
        size = int(round(mm_to_px(marker["size_mm"], loaded)))
        marker_image = cv2.aruco.generateImageMarker(
            dictionary, int(marker["id"]), size
        )
        image[y : y + size, x : x + size] = cv2.cvtColor(
            marker_image, cv2.COLOR_GRAY2BGR
        )

    _draw_qr(
        image,
        compact_qr_payload(
            exam_id, template_version, template_checksum
        ),
        loaded,
    )
    _draw_heading(image, loaded)
    _draw_answers(image, loaded, answers or {})
    _draw_roll_number(image, loaded, roll_number)

    footer_y = int(round(mm_to_px(273, loaded)))
    cv2.putText(
        image,
        "DIAGNOSTIC TEST SHEET - PRINT AT 100 PERCENT",
        (int(round(mm_to_px(55, loaded))), footer_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        black,
        1,
        cv2.LINE_AA,
    )
    return image


def cycling_answers(definition: Optional[Mapping[str, Any]] = None) -> Dict[int, str]:
    """Return A/B/C/D cycling answers for every configured question."""

    loaded = load_sheet_definition(definition)
    output: Dict[int, str] = {}
    for block in loaded["answer_blocks"]:
        options = list(block["options"])
        for offset in range(int(block["question_count"])):
            question = int(block["first_question"]) + offset
            output[question] = str(options[offset % len(options)])
    return output


def _point(point_mm, definition):
    return tuple(
        int(round(value)) for value in point_mm_to_px(point_mm, definition)
    )


def _draw_heading(image: np.ndarray, definition: Mapping[str, Any]) -> None:
    cv2 = require_cv2()
    cv2.putText(
        image,
        "PAPERCREATOR OMR - CANONICAL TEST",
        _point((67, 17), definition),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def _draw_answers(
    image: np.ndarray,
    definition: Mapping[str, Any],
    answers: Mapping[int, str],
) -> None:
    cv2 = require_cv2()
    outline = max(2, int(round(mm_to_px(0.3, definition))))
    by_question: Dict[int, list] = {}
    for bubble in iter_answer_geometry(definition):
        by_question.setdefault(int(bubble["question"]), []).append(bubble)

    for question, bubbles in sorted(by_question.items()):
        first_center = bubbles[0]["center_mm"]
        label_x = first_center[0] - 11
        label_y = first_center[1] + 0.8
        cv2.putText(
            image,
            str(question),
            _point((label_x, label_y), definition),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        selected = str(answers.get(question, "")).upper()
        for bubble in bubbles:
            center = _point(bubble["center_mm"], definition)
            radius = int(
                round(mm_to_px(bubble["diameter_mm"], definition) / 2.0)
            )
            cv2.circle(image, center, radius, (0, 0, 0), outline)
            if str(bubble["option"]).upper() == selected:
                cv2.circle(
                    image,
                    center,
                    max(2, int(round(radius * 0.58))),
                    (20, 20, 20),
                    -1,
                )

    for block in definition["answer_blocks"]:
        first_x, first_y = block["first_center_mm"]
        for option_index, option in enumerate(block["options"]):
            x = first_x + option_index * block["option_pitch_mm"] - 1
            cv2.putText(
                image,
                str(option),
                _point((x, first_y - 4), definition),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )


def _draw_roll_number(
    image: np.ndarray,
    definition: Mapping[str, Any],
    roll_number: str,
) -> None:
    cv2 = require_cv2()
    roll = definition["roll_number"]
    if len(roll_number) != int(roll["columns"]) or not roll_number.isdigit():
        raise ValueError(
            "roll_number must contain exactly {} digits".format(
                roll["columns"]
            )
        )
    outline = max(2, int(round(mm_to_px(0.25, definition))))
    for bubble in iter_roll_geometry(definition):
        center = _point(bubble["center_mm"], definition)
        radius = int(
            round(mm_to_px(bubble["diameter_mm"], definition) / 2.0)
        )
        cv2.circle(image, center, radius, (0, 0, 0), outline)
        if str(bubble["digit"]) == roll_number[bubble["column_index"]]:
            cv2.circle(
                image,
                center,
                max(2, int(round(radius * 0.58))),
                (20, 20, 20),
                -1,
            )


def _draw_qr(
    image: np.ndarray,
    payload: str,
    definition: Mapping[str, Any],
) -> None:
    cv2 = require_cv2()
    qr = cv2.QRCodeEncoder_create().encode(payload)
    qr_with_quiet_zone = cv2.copyMakeBorder(
        qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255
    )
    size = int(round(mm_to_px(definition["qr"]["size_mm"], definition)))
    rendered = cv2.resize(
        qr_with_quiet_zone, (size, size), interpolation=cv2.INTER_NEAREST
    )
    x, y = _point(definition["qr"]["top_left_mm"], definition)
    image[y : y + size, x : x + size] = cv2.cvtColor(
        rendered, cv2.COLOR_GRAY2BGR
    )
