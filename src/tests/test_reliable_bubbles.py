import numpy as np

from src.reliable_omr.bubbles import (
    classify_bubble,
    extract_bubble_features,
    resolve_mark_group,
)
from src.reliable_omr.definition import load_sheet_definition
from src.reliable_omr.types import MarkStatus


def _synthetic_bubble(filled):
    image = np.full((140, 140), 255, dtype=np.uint8)
    yy, xx = np.ogrid[:140, :140]
    distance = np.sqrt((xx - 70) ** 2 + (yy - 70) ** 2)
    image[(distance >= 17) & (distance <= 21)] = 0
    if filled:
        image[distance <= 13] = 35
    return image


def test_inner_disk_ignores_printed_ring_and_detects_fill():
    classification = load_sheet_definition()["classification"]
    empty = extract_bubble_features(
        _synthetic_bubble(False), [70, 70], 20, classification
    )
    filled = extract_bubble_features(
        _synthetic_bubble(True), [70, 70], 20, classification
    )

    assert empty.ring_darkness > 0.5
    assert empty.fill_score <= classification["empty_score_max"]
    assert filled.fill_score >= classification["filled_score_min"]
    assert filled.fill_score > empty.fill_score + 0.5


def _result(option, filled):
    definition = load_sheet_definition()
    features = extract_bubble_features(
        _synthetic_bubble(filled),
        [70, 70],
        20,
        definition["classification"],
    )
    return classify_bubble(
        option,
        [70, 70],
        20,
        features,
        definition["classification"],
    )


def test_question_resolution_exposes_filled_empty_and_multiple_states():
    classification = load_sheet_definition()["classification"]

    filled = resolve_mark_group(
        "q1",
        [_result("A", True), _result("B", False), _result("C", False)],
        classification,
        question=1,
    )
    empty = resolve_mark_group(
        "q2",
        [_result("A", False), _result("B", False), _result("C", False)],
        classification,
        question=2,
    )
    multiple = resolve_mark_group(
        "q3",
        [_result("A", True), _result("B", True), _result("C", False)],
        classification,
        question=3,
    )

    assert filled.status == MarkStatus.FILLED
    assert filled.answer == "A"
    assert empty.status == MarkStatus.EMPTY
    assert empty.answer is None
    assert multiple.status == MarkStatus.MULTIPLE
    assert multiple.answer is None
    assert multiple.selected_options == ["A", "B"]
    assert multiple.review_reasons[0].code == "multiple_marks"
