import logging
from typing import Tuple, List

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

from .base_ocr import BaseOCREngine
from ..config import get_config
from ..models import OCRRegion, LayoutRegion, BoundingBox, RegionType
from ..exceptions import OCRFailure

logger = logging.getLogger(__name__)


class AzureDocumentIntelligenceEngine(BaseOCREngine):

    def __init__(self):
        cfg = get_config().di
        if not cfg.endpoint or not cfg.api_key:
            raise OCRFailure("Azure DI endpoint and api_key must be configured")

        self.client = DocumentIntelligenceClient(
            endpoint=cfg.endpoint,
            credential=AzureKeyCredential(cfg.api_key),
        )
        self.model_id = cfg.model_id

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=8),
        retry=retry_if_exception_type(OCRFailure),   # ← only retry transient failures
        reraise=True,
    )
    def _analyze(self, path: str):
        try:
            with open(path, "rb") as f:
                poller = self.client.begin_analyze_document(
                    model_id=self.model_id,
                    analyze_request=f,
                    content_type="application/octet-stream",
                )
            return poller.result()
        except HttpResponseError as exc:
            logger.error("Azure DI HTTP error: %s (status=%s)", exc.message, exc.status_code)
            raise OCRFailure(f"Azure DI: {exc.message}") from exc

    def process_document(
        self, document_path: str
    ) -> Tuple[List[OCRRegion], List[LayoutRegion]]:

        result = self._analyze(document_path)

        ocr_regions: List[OCRRegion] = []
        layout_regions: List[LayoutRegion] = []
        region_id = 0
        page_count = 0

        for page in result.pages or []:
            page_count = max(page_count, page.page_number)
            for word in page.words or []:
                if word.polygon and len(word.polygon) >= 8:
                    bbox = BoundingBox.from_polygon(word.polygon)
                    ocr_regions.append(
                        OCRRegion(word.content, bbox, word.confidence or 0.0, page.page_number)
                    )

        for table in result.tables or []:
            if table.bounding_regions:
                br = table.bounding_regions[0]
                if br.polygon and len(br.polygon) >= 8:
                    bbox = BoundingBox.from_polygon(br.polygon)
                    layout_regions.append(
                        LayoutRegion(region_id, RegionType.TABLE, bbox, 1.0, br.page_number)
                    )
                    region_id += 1

        logger.info(
            "OCR complete: %d words, %d tables, %d pages",
            len(ocr_regions), len(layout_regions), page_count,
        )
        return ocr_regions, layout_regions
