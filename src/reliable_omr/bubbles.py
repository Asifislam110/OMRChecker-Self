"""Robust local bubble features and explicit mark-state decisions."""

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from src.reliable_omr.definition import (
    iter_answer_geometry,
    iter_roll_geometry,
    mm_to_px,
    point_mm_to_px,
)
from src.reliable_omr.opencv import require_cv2
from src.reliable_omr.types import (
    BoundingBox,
    BubbleFeatures,
    BubbleResult,
    BubbleStatus,
    MarkStatus,
    QuestionResult,
    ReviewReason,
)


def _bounding_box(bubbles: Sequence[BubbleResult]) -> BoundingBox:
    """Return a canonical-pixel box covering every bubble in the field."""

    if not bubbles:
        return BoundingBox(x=0, y=0, width=0, height=0)
    left = math.floor(
        min(bubble.center_px[0] - bubble.radius_px for bubble in bubbles)
    )
    top = math.floor(
        min(bubble.center_px[1] - bubble.radius_px for bubble in bubbles)
    )
    right = math.ceil(
        max(bubble.center_px[0] + bubble.radius_px for bubble in bubbles)
    )
    bottom = math.ceil(
        max(bubble.center_px[1] + bubble.radius_px for bubble in bubbles)
    )
    return BoundingBox(
        x=max(0, left),
        y=max(0, top),
        width=max(0, right - max(0, left)),
        height=max(0, bottom - max(0, top)),
    )


def _to_gray(image: np.ndarray) -> np.ndarray:
    cv2 = require_cv2()
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def extract_bubble_features(
    gray: np.ndarray,
    center_px: Sequence[float],
    radius_px: float,
    classification: Mapping[str, float],
) -> BubbleFeatures:
    """Measure inner ink against the printed ring and nearby paper.

    The score is locally normalized, so a grey pencil mark on grey paper can
    still separate from its immediate background. The printed outline is
    measured independently and does not count as inner fill.
    """

    center_x, center_y = center_px
    outer_ratio = float(classification["background_outer_ratio"])
    extent = int(np.ceil(radius_px * outer_ratio)) + 1
    x0 = max(0, int(np.floor(center_x)) - extent)
    x1 = min(gray.shape[1], int(np.floor(center_x)) + extent + 1)
    y0 = max(0, int(np.floor(center_y)) - extent)
    y1 = min(gray.shape[0], int(np.floor(center_y)) + extent + 1)
    if x1 <= x0 or y1 <= y0:
        return BubbleFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    crop = gray[y0:y1, x0:x1].astype(np.float32) / 255.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    inner = distance <= radius_px * classification["inner_radius_ratio"]
    ring = (
        (distance >= radius_px * classification["ring_inner_ratio"])
        & (distance <= radius_px * classification["ring_outer_ratio"])
    )
    background = (
        (distance >= radius_px * classification["background_inner_ratio"])
        & (distance <= radius_px * classification["background_outer_ratio"])
    )
    expected_side = 2 * extent + 1
    valid_pixel_fraction = float(
        crop.shape[0] * crop.shape[1] / float(expected_side * expected_side)
    )
    if not np.any(inner) or not np.any(ring) or not np.any(background):
        return BubbleFeatures(
            0.0, 0.0, 0.0, 0.0, 0.0, round(valid_pixel_fraction, 4)
        )

    inner_light = float(np.mean(crop[inner]))
    ring_light = float(np.mean(crop[ring]))
    background_light = float(np.median(crop[background]))
    local_pixels = crop[distance <= radius_px * outer_ratio]
    low, high = np.percentile(local_pixels, [10, 90])
    local_range = max(0.15, float(high - low))

    inner_darkness = 1.0 - inner_light
    ring_darkness = 1.0 - ring_light
    background_darkness = 1.0 - background_light
    local_contrast = float(
        np.clip((background_light - inner_light) / local_range, 0.0, 1.0)
    )
    ink_above_background = max(0.0, inner_darkness - background_darkness)
    # The ring term only corrects substantial bleed into the inner disk; it
    # cannot by itself turn an empty outlined bubble into a filled bubble.
    bleed_penalty = max(0.0, ring_darkness - inner_darkness) * 0.08
    fill_score = float(
        np.clip(
            0.75 * local_contrast
            + 0.25 * ink_above_background
            - bleed_penalty,
            0.0,
            1.0,
        )
    )
    return BubbleFeatures(
        inner_darkness=round(inner_darkness, 6),
        ring_darkness=round(ring_darkness, 6),
        background_darkness=round(background_darkness, 6),
        local_contrast=round(local_contrast, 6),
        fill_score=round(fill_score, 6),
        valid_pixel_fraction=round(valid_pixel_fraction, 6),
    )


def classify_bubble(
    option: str,
    center_px: Sequence[float],
    radius_px: float,
    features: BubbleFeatures,
    classification: Mapping[str, float],
) -> BubbleResult:
    reasons: List[ReviewReason] = []
    score = features.fill_score
    if features.valid_pixel_fraction < 0.9:
        status = BubbleStatus.INVALID
        confidence = 0.0
        reasons.append(
            ReviewReason(
                code="bubble_roi_clipped",
                message="Bubble neighbourhood extends outside the canonical page",
                scope="bubble:{}".format(option),
                severity="error",
            )
        )
    elif score <= classification["empty_score_max"]:
        status = BubbleStatus.EMPTY
        confidence = 1.0 - 0.45 * (
            score / max(classification["empty_score_max"], 1e-6)
        )
    elif score >= classification["filled_score_min"]:
        status = BubbleStatus.FILLED
        confidence = 0.7 + 0.3 * (
            (score - classification["filled_score_min"])
            / max(1.0 - classification["filled_score_min"], 1e-6)
        )
    else:
        status = BubbleStatus.AMBIGUOUS
        confidence = 0.45
        reasons.append(
            ReviewReason(
                code="bubble_uncertain_density",
                message="Bubble fill score lies between empty and filled thresholds",
                scope="bubble:{}".format(option),
            )
        )
    return BubbleResult(
        option=str(option),
        center_px=[round(float(value), 3) for value in center_px],
        radius_px=round(float(radius_px), 3),
        status=status,
        confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
        features=features,
        review_reasons=reasons,
    )


def resolve_mark_group(
    field_id: str,
    bubbles: List[BubbleResult],
    classification: Mapping[str, float],
    question: int = None,
) -> QuestionResult:
    """Resolve one question/roll column using top score, runner-up, and margin."""

    valid = [
        bubble for bubble in bubbles if bubble.status != BubbleStatus.INVALID
    ]
    reasons: List[ReviewReason] = []
    if len(valid) < 2:
        reasons.append(
            ReviewReason(
                code="insufficient_valid_bubbles",
                message="Too few valid bubble regions to resolve this field",
                scope=field_id,
                severity="error",
            )
        )
        return QuestionResult(
            field_id=field_id,
            question=question,
            status=MarkStatus.INVALID,
            answer=None,
            selected_options=[],
            confidence=0.0,
            top_score=0.0,
            second_score=0.0,
            margin=0.0,
            bounding_box=_bounding_box(bubbles),
            bubbles=bubbles,
            review_reasons=reasons,
        )

    ranked = sorted(
        valid, key=lambda bubble: bubble.features.fill_score, reverse=True
    )
    top, second = ranked[0], ranked[1]
    top_score = float(top.features.fill_score)
    second_score = float(second.features.fill_score)
    margin = top_score - second_score
    multiple = [
        bubble
        for bubble in ranked
        if bubble.features.fill_score >= classification["multiple_score_min"]
    ]

    if len(multiple) >= 2:
        status = MarkStatus.MULTIPLE
        answer = None
        selected = [bubble.option for bubble in multiple]
        confidence = max(0.05, 0.4 - 0.5 * margin)
        reasons.append(
            ReviewReason(
                code="multiple_marks",
                message="More than one bubble exceeds the multiple-mark threshold",
                scope=field_id,
            )
        )
    elif top_score <= classification["empty_score_max"]:
        status = MarkStatus.EMPTY
        answer = None
        selected = []
        confidence = 1.0 - 0.5 * (
            top_score / max(classification["empty_score_max"], 1e-6)
        )
    elif (
        top_score >= classification["filled_score_min"]
        and margin >= classification["min_top_margin"]
    ):
        status = MarkStatus.FILLED
        answer = top.option
        selected = [top.option]
        confidence = min(
            0.99,
            0.65
            + 0.25
            * margin
            / max(classification["min_top_margin"], 1e-6)
            + 0.1 * top.confidence,
        )
    else:
        status = MarkStatus.AMBIGUOUS
        answer = None
        selected = [top.option]
        confidence = 0.45
        code = (
            "top_margin_too_small"
            if top_score >= classification["filled_score_min"]
            else "top_mark_uncertain"
        )
        reasons.append(
            ReviewReason(
                code=code,
                message=(
                    "Top mark does not have enough separation from the runner-up"
                    if code == "top_margin_too_small"
                    else "Top mark lies between empty and filled thresholds"
                ),
                scope=field_id,
            )
        )

    return QuestionResult(
        field_id=field_id,
        question=question,
        status=status,
        answer=answer,
        selected_options=selected,
        confidence=round(float(np.clip(confidence, 0.0, 1.0)), 4),
        top_score=round(top_score, 6),
        second_score=round(second_score, 6),
        margin=round(margin, 6),
        bounding_box=_bounding_box(bubbles),
        bubbles=bubbles,
        review_reasons=reasons,
    )


def _recognize_geometry_groups(
    image: np.ndarray,
    geometry: Iterable[Dict[str, Any]],
    definition: Mapping[str, Any],
    group_key: str,
) -> List[QuestionResult]:
    gray = _to_gray(image)
    classification = definition["classification"]
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for item in geometry:
        grouped[item[group_key]].append(item)

    results = []
    for group in sorted(grouped):
        items = sorted(grouped[group], key=lambda item: item["option_index"])
        bubbles = []
        for item in items:
            center = point_mm_to_px(item["center_mm"], definition)
            radius = mm_to_px(item["diameter_mm"], definition) / 2.0
            features = extract_bubble_features(
                gray, center, radius, classification
            )
            bubbles.append(
                classify_bubble(
                    item["option"], center, radius, features, classification
                )
            )
        field_id = items[0]["field_id"]
        question = int(group) if group_key == "question" else None
        results.append(
            resolve_mark_group(
                field_id, bubbles, classification, question=question
            )
        )
    return results


def recognize_answers(
    canonical_image: np.ndarray, definition: Mapping[str, Any]
) -> List[QuestionResult]:
    return _recognize_geometry_groups(
        canonical_image,
        iter_answer_geometry(definition),
        definition,
        "question",
    )


def recognize_roll_number(
    canonical_image: np.ndarray, definition: Mapping[str, Any]
) -> Tuple[str, List[QuestionResult]]:
    columns = _recognize_geometry_groups(
        canonical_image,
        iter_roll_geometry(definition),
        definition,
        "column_index",
    )
    value = "".join(
        column.answer if column.status == MarkStatus.FILLED else "?"
        for column in columns
    )
    return value, columns
