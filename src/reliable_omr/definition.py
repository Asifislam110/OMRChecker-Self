"""Loading, validation, and deterministic geometry expansion for sheet definitions."""

import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from jsonschema import Draft202012Validator


PACKAGE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = PACKAGE_DIR / "sheet_definition.schema.json"
DEFAULT_DEFINITION_PATH = PACKAGE_DIR / "definitions" / "reliable_a4_v1.json"
SUPPORTED_SCHEMA_MAJOR = 1


class SheetDefinitionError(ValueError):
    """Raised when a sheet definition is malformed or internally inconsistent."""


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    with SCHEMA_PATH.open("r", encoding="utf-8") as schema_file:
        return Draft202012Validator(json.load(schema_file))


def _json_path(error: Any) -> str:
    path = "$"
    for item in error.absolute_path:
        path += "[{}]".format(item) if isinstance(item, int) else ".{}".format(item)
    return path


def validate_sheet_definition(definition: Mapping[str, Any]) -> None:
    """Validate JSON shape plus geometric invariants not expressible in JSON Schema."""

    errors = sorted(
        _validator().iter_errors(definition),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            "{}: {}".format(_json_path(error), error.message) for error in errors[:8]
        )
        raise SheetDefinitionError("Invalid sheet definition: {}".format(details))

    major = int(str(definition["schema_version"]).split(".", 1)[0])
    if major != SUPPORTED_SCHEMA_MAJOR:
        raise SheetDefinitionError(
            "Unsupported schema major {}; expected {}".format(
                major, SUPPORTED_SCHEMA_MAJOR
            )
        )

    page = definition["page"]
    expected_width = round(
        page["width_mm"] * page["canonical_dpi"] / 25.4
    )
    expected_height = round(
        page["height_mm"] * page["canonical_dpi"] / 25.4
    )
    if (
        abs(page["canonical_width_px"] - expected_width) > 2
        or abs(page["canonical_height_px"] - expected_height) > 2
    ):
        raise SheetDefinitionError(
            "Canonical pixel dimensions do not match page millimetres and DPI"
        )

    markers = definition["aruco"]["markers"]
    marker_ids = [marker["id"] for marker in markers]
    marker_roles = [marker["role"] for marker in markers]
    expected_roles = {"top_left", "top_right", "bottom_right", "bottom_left"}
    if len(set(marker_ids)) != 4:
        raise SheetDefinitionError("All four ArUco marker IDs must be unique")
    if set(marker_roles) != expected_roles:
        raise SheetDefinitionError(
            "ArUco marker roles must contain each page corner exactly once"
        )
    for marker in markers:
        _validate_square_on_page(
            marker["top_left_mm"],
            marker["size_mm"],
            page,
            "ArUco marker {}".format(marker["id"]),
        )

    qr = definition["qr"]
    _validate_square_on_page(qr["top_left_mm"], qr["size_mm"], page, "QR code")
    field_names = [field["name"] for field in qr["payload_fields"]]
    if len(field_names) != len(set(field_names)):
        raise SheetDefinitionError("QR payload field names must be unique")
    for field in qr["payload_fields"]:
        if field.get("min_length", 0) > field.get("max_length", 512):
            raise SheetDefinitionError(
                "QR payload field '{}' has min_length above max_length".format(
                    field["name"]
                )
            )
    if qr["prefix"] == "O1:" and qr["encoding"] != "json":
        raise SheetDefinitionError("O1 QR payloads must use json encoding")
    if qr["prefix"] == "O2:":
        expected_fields = [
            {
                "name": "profile_token",
                "type": "string",
                "required": True,
                "min_length": 22,
                "max_length": 22,
            },
            {
                "name": "signature",
                "type": "string",
                "required": True,
                "min_length": 6,
                "max_length": 6,
            },
        ]
        if (
            qr["encoding"] != "base64url_profile"
            or qr["payload_fields"] != expected_fields
        ):
            raise SheetDefinitionError(
                "O2 QR definitions require the exact profile_token/signature "
                "base64url contract"
            )

    question_ids = set()
    for bubble in iter_answer_geometry(definition):
        question_id = bubble["question"]
        if question_id in question_ids and bubble["option_index"] == 0:
            raise SheetDefinitionError(
                "Answer blocks overlap at question {}".format(question_id)
            )
        question_ids.add(question_id)
        _validate_circle_on_page(
            bubble["center_mm"],
            bubble["diameter_mm"],
            page,
            "question {} option {}".format(question_id, bubble["option"]),
        )

    for bubble in iter_roll_geometry(definition):
        _validate_circle_on_page(
            bubble["center_mm"],
            bubble["diameter_mm"],
            page,
            "roll column {} digit {}".format(
                bubble["column_index"], bubble["digit"]
            ),
        )

    classification = definition["classification"]
    ratios = [
        classification["inner_radius_ratio"],
        classification["ring_inner_ratio"],
        classification["ring_outer_ratio"],
        classification["background_inner_ratio"],
        classification["background_outer_ratio"],
    ]
    if ratios != sorted(ratios) or len(set(ratios)) != len(ratios):
        raise SheetDefinitionError(
            "Classification mask radius ratios must be strictly increasing"
        )
    if not (
        classification["empty_score_max"]
        < classification["multiple_score_min"]
        <= classification["filled_score_min"]
    ):
        raise SheetDefinitionError(
            "Classification score thresholds must satisfy "
            "empty < multiple <= filled"
        )


def _validate_square_on_page(
    top_left: List[float],
    size: float,
    page: Mapping[str, Any],
    label: str,
) -> None:
    x, y = top_left
    if x < 0 or y < 0 or x + size > page["width_mm"] or y + size > page["height_mm"]:
        raise SheetDefinitionError("{} extends outside the A4 page".format(label))


def _validate_circle_on_page(
    center: List[float],
    diameter: float,
    page: Mapping[str, Any],
    label: str,
) -> None:
    radius = diameter / 2.0
    x, y = center
    if (
        x - radius < 0
        or y - radius < 0
        or x + radius > page["width_mm"]
        or y + radius > page["height_mm"]
    ):
        raise SheetDefinitionError("{} extends outside the A4 page".format(label))


DefinitionInput = Optional[
    Union[str, Path, Mapping[str, Any]]
]


def load_sheet_definition(source: DefinitionInput = None) -> Dict[str, Any]:
    """Load a definition from a mapping, JSON string, path, or the bundled default."""

    if source is None:
        source = DEFAULT_DEFINITION_PATH

    if isinstance(source, Mapping):
        definition = copy.deepcopy(dict(source))
    else:
        if isinstance(source, Path):
            path = source
        elif isinstance(source, str) and re.match(r"^\s*\{", source):
            try:
                definition = json.loads(source)
            except json.JSONDecodeError as exc:
                raise SheetDefinitionError(
                    "Sheet definition is not valid JSON: {}".format(exc)
                ) from exc
            validate_sheet_definition(definition)
            return definition
        else:
            path = Path(str(source))

        try:
            with path.open("r", encoding="utf-8") as definition_file:
                definition = json.load(definition_file)
        except (OSError, json.JSONDecodeError) as exc:
            raise SheetDefinitionError(
                "Could not load sheet definition '{}': {}".format(path, exc)
            ) from exc

    validate_sheet_definition(definition)
    return definition


def mm_to_px(value_mm: float, definition: Mapping[str, Any]) -> float:
    return value_mm * definition["page"]["canonical_dpi"] / 25.4


def point_mm_to_px(
    point_mm: Iterable[float], definition: Mapping[str, Any]
) -> List[float]:
    return [mm_to_px(value, definition) for value in point_mm]


def iter_answer_geometry(definition: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield every fixed answer bubble in stable question/option order."""

    for block in definition["answer_blocks"]:
        first_x, first_y = block["first_center_mm"]
        for question_offset in range(block["question_count"]):
            question = block["first_question"] + question_offset
            y = first_y + question_offset * block["question_pitch_mm"]
            for option_index, option in enumerate(block["options"]):
                x = first_x + option_index * block["option_pitch_mm"]
                yield {
                    "field_id": "q{}".format(question),
                    "question": question,
                    "option": option,
                    "option_index": option_index,
                    "center_mm": [x, y],
                    "diameter_mm": block["bubble_diameter_mm"],
                }


def iter_roll_geometry(definition: Mapping[str, Any]) -> Iterable[Dict[str, Any]]:
    """Yield every fixed roll-number bubble in column/digit order."""

    roll = definition["roll_number"]
    first_x, first_y = roll["first_center_mm"]
    for column_index in range(roll["columns"]):
        x = first_x + column_index * roll["column_pitch_mm"]
        for digit in range(roll["digits"]):
            y = first_y + digit * roll["digit_pitch_mm"]
            yield {
                "field_id": "roll_{}".format(column_index + 1),
                "column_index": column_index,
                "digit": digit,
                "option": str(digit),
                "option_index": digit,
                "center_mm": [x, y],
                "diameter_mm": roll["bubble_diameter_mm"],
            }
