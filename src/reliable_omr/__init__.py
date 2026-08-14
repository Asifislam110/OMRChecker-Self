"""Schema-driven, review-first OMR recognition.

This package is intentionally independent from OMRChecker's legacy CLI pipeline.
Applications should use :class:`ReliableOMRRecognizer` or the FastAPI app in
``src.reliable_omr.service``.
"""

from src.reliable_omr.definition import (
    DEFAULT_DEFINITION_PATH,
    SheetDefinitionError,
    load_sheet_definition,
)
from src.reliable_omr.recognizer import ReliableOMRRecognizer

__all__ = [
    "DEFAULT_DEFINITION_PATH",
    "ReliableOMRRecognizer",
    "SheetDefinitionError",
    "load_sheet_definition",
]
