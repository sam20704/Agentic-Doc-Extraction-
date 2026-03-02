import logging
from typing import List

from ..models import OCRRegion

logger = logging.getLogger(__name__)


class LayoutReaderProcessor:
    """
    Sorts OCR regions into reading order.
    Uses top-to-bottom, left-to-right heuristic.
    Replace with LayoutReader model for production accuracy.
    """

    def get_reading_order(self, ocr_regions: List[OCRRegion]) -> List[int]:
        """Returns list of original indices sorted in reading order."""
        indexed = [(i, r.bbox.y1, r.bbox.x1) for i, r in enumerate(ocr_regions)]
        indexed.sort(key=lambda x: (x[1], x[2]))
        return [idx for idx, _, _ in indexed]
