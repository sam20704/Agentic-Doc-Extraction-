import base64
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logger.warning("PyMuPDF not installed — page rendering disabled")


class PageRenderer:
    """
    Renders PDF pages to base64 PNG images.
    Required for VLM image analysis to actually work.
    """

    def __init__(self, dpi: int = 200):
        self.dpi = dpi

    def render_pages(self, document_path: str) -> Dict[int, str]:
        """
        Returns {page_number: base64_png_string}
        """
        if not HAS_FITZ:
            logger.warning("Skipping render — PyMuPDF not available")
            return {}

        path = Path(document_path)
        if path.suffix.lower() not in (".pdf",):
            logger.info("Not a PDF, skipping page rendering")
            return {}

        page_images: Dict[int, str] = {}

        try:
            doc = fitz.open(document_path)
            zoom = self.dpi / 72
            matrix = fitz.Matrix(zoom, zoom)

            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=matrix)
                png_bytes = pix.tobytes("png")
                b64 = base64.b64encode(png_bytes).decode("utf-8")
                page_images[page_num + 1] = b64  # 1-indexed

            doc.close()
            logger.info("Rendered %d pages from %s", len(page_images), path.name)

        except Exception as exc:
            logger.error("Page rendering failed: %s", exc)

        return page_images
