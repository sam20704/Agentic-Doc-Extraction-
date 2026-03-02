import re
import logging
from typing import Dict, List

from ..models import DocumentExtractionResult

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Strip currency symbols, commas, whitespace and lowercase."""
    text = str(text).lower().strip()
    text = re.sub(r"[$€£¥,\s]", "", text)
    # Remove trailing zeros after decimal: "1234.00" → "1234"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


class GroundingTool:

    def __init__(self, extraction_result: DocumentExtractionResult):
        # Pre-compute normalized tokens once
        self._normalized_tokens: List[str] = [
            _normalize(r.text) for r in extraction_result.ocr_regions
        ]
        self._raw_tokens: List[str] = [
            r.text for r in extraction_result.ocr_regions
        ]

    def verify(self, value) -> Dict:
        """Check if a value appears in the OCR output."""
        target = _normalize(str(value))
        if not target:
            return {"grounded": False, "reason": "empty value"}

        for i, token in enumerate(self._normalized_tokens):
            if target == token:
                return {
                    "grounded": True,
                    "matched_token": self._raw_tokens[i],
                    "match_type": "exact",
                }

        # Fallback: check if target appears as substring of any token
        for i, token in enumerate(self._normalized_tokens):
            if target in token or token in target:
                return {
                    "grounded": True,
                    "matched_token": self._raw_tokens[i],
                    "match_type": "partial",
                }

        return {"grounded": False, "reason": f"'{value}' not found in OCR output"}
