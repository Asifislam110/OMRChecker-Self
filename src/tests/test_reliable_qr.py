import copy
from pathlib import Path

import pytest

from src.reliable_omr.definition import (
    SheetDefinitionError,
    load_sheet_definition,
)
from src.reliable_omr.recognizer import QRPayloadError, parse_qr_payload


O2_DEFINITION = (
    Path(__file__).resolve().parents[1]
    / "reliable_omr"
    / "definitions"
    / "reliable_a4_o2_v1.json"
)


def test_o1_compact_and_json_payloads_remain_valid():
    definition = load_sheet_definition()

    assert parse_qr_payload(
        "O1:exam-1.2.deadbeef", definition
    ) == {
        "exam_id": "exam-1",
        "template_version": "2",
        "template_checksum": "deadbeef",
    }
    assert parse_qr_payload(
        'O1:{"exam_id":"exam-1","template_version":"2",'
        '"template_checksum":"deadbeef"}',
        definition,
    )["exam_id"] == "exam-1"


@pytest.mark.parametrize(
    "raw",
    [
        "O0:exam-1.2.deadbeef",
        "O1:exam-1..deadbeef",
        "O1:exam-1.2.deadbeef.extra",
        (
            'O1:{"exam_id":"exam-1","template_version":"2",'
            '"template_checksum":"deadbeef","unexpected":1}'
        ),
    ],
)
def test_o1_malformed_payloads_are_rejected(raw):
    with pytest.raises(QRPayloadError):
        parse_qr_payload(raw, load_sheet_definition())


def test_o2_exact_payload_returns_opaque_token_and_signature_only():
    definition = load_sheet_definition(O2_DEFINITION)
    raw = "O2:{}.".format("A" * 22) + "b_-012"

    assert len(raw.encode("ascii")) == 32
    assert parse_qr_payload(raw, definition) == {
        "profile_token": "A" * 22,
        "signature": "b_-012",
    }


@pytest.mark.parametrize(
    "raw",
    [
        "O2:{}.".format("A" * 21) + "b_-012",
        "O2:{}.".format("A" * 23) + "b_-012",
        "O2:{}.".format("A" * 22) + "b_+012",
        "O2:{}.".format("A" * 22) + "b_-01=",
        "O2:{}.".format("A" * 22) + "b_-012.extra",
        "O2:{}.".format("é" * 22) + "b_-012",
    ],
)
def test_o2_malformed_length_or_base64url_is_rejected(raw):
    with pytest.raises(QRPayloadError):
        parse_qr_payload(raw, load_sheet_definition(O2_DEFINITION))


def test_o2_definition_contract_is_exact_and_additional_properties_closed():
    definition = load_sheet_definition(O2_DEFINITION)
    wrong_fields = copy.deepcopy(definition)
    wrong_fields["qr"]["payload_fields"][0]["max_length"] = 23
    with pytest.raises(SheetDefinitionError, match="exact"):
        load_sheet_definition(wrong_fields)

    extra_property = copy.deepcopy(definition)
    extra_property["qr"]["payload_fields"][0]["hmac"] = True
    with pytest.raises(SheetDefinitionError, match="Additional properties"):
        load_sheet_definition(extra_property)
