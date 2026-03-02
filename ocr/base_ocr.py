from abc import ABC, abstractmethod
from typing import Tuple, List

from ..models import OCRRegion, LayoutRegion


class BaseOCREngine(ABC):
    @abstractmethod
    def process_document(
        self, document_path: str
    ) -> Tuple[List[OCRRegion], List[LayoutRegion]]:
        pass
