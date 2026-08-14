import copy

import pytest

from src.reliable_omr.definition import (
    SheetDefinitionError,
    iter_answer_geometry,
    iter_roll_geometry,
    load_sheet_definition,
)


def test_default_definition_has_fixed_versioned_geometry():
    definition = load_sheet_definition()

    assert definition["schema_version"] == "1.0.0"
    assert definition["page"]["size"] == "A4"
    assert (definition["page"]["width_mm"], definition["page"]["height_mm"]) == (
        210,
        297,
    )
    assert [marker["id"] for marker in definition["aruco"]["markers"]] == [
        0,
        1,
        2,
        3,
    ]
    assert definition["aruco"]["dictionary"] == "DICT_4X4_50"
    assert [marker["role"] for marker in definition["aruco"]["markers"]] == [
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    ]
    assert definition["definition_id"] == "papercreator-a4-100q-v1"
    assert [
        (block["first_question"], block["question_count"])
        for block in definition["answer_blocks"]
    ] == [(1, 25), (26, 25), (51, 25), (76, 25)]
    assert len(list(iter_answer_geometry(definition))) == 100 * 4
    assert len(list(iter_roll_geometry(definition))) == 8 * 10
    assert {
        field["name"] for field in definition["qr"]["payload_fields"]
    } == {"exam_id", "template_version", "template_checksum"}


def test_definition_rejects_duplicate_marker_ids():
    definition = copy.deepcopy(load_sheet_definition())
    definition["aruco"]["markers"][1]["id"] = 0

    with pytest.raises(SheetDefinitionError, match="must be unique"):
        load_sheet_definition(definition)


def test_definition_rejects_inconsistent_canonical_pixels():
    definition = copy.deepcopy(load_sheet_definition())
    definition["page"]["canonical_width_px"] = 2000

    with pytest.raises(SheetDefinitionError, match="do not match"):
        load_sheet_definition(definition)


def test_definition_accepts_json_string():
    definition = load_sheet_definition()
    import json

    loaded = load_sheet_definition(json.dumps(definition))
    assert loaded["definition_id"] == definition["definition_id"]
