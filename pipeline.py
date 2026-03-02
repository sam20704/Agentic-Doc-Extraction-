import logging

from .ocr.azure_engine import AzureDocumentIntelligenceEngine
from .layout.layout_reader import LayoutReaderProcessor
from .models import DocumentExtractionResult

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """
    OCR + Layout stage only.
    Does NOT instantiate VLM — that belongs to the agent layer.
    """

    def __init__(self):
        self.ocr = AzureDocumentIntelligenceEngine()
        self.layout = LayoutReaderProcessor()

    def run(self, document_path: str) -> DocumentExtractionResult:

        logger.info("Pipeline starting: %s", document_path)

        ocr_regions, layout_regions = self.ocr.process_document(document_path)
        reading_order = self.layout.get_reading_order(ocr_regions)

        # Build ordered text from reading order indices
        ordered_text = [ocr_regions[i].text for i in reading_order]

        # Derive page count from actual data
        page_count = (
            max(r.page_number for r in ocr_regions) if ocr_regions else 0
        )

        result = DocumentExtractionResult(
            ocr_regions=ocr_regions,
            layout_regions=layout_regions,
            ordered_text=ordered_text,
            region_images={},
            page_count=page_count,
        )

        logger.info(
            "Pipeline complete: %d regions, %d pages, %d ordered tokens",
            len(ocr_regions), page_count, len(ordered_text),
        )
        return result
