from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

from pydantic import BaseModel


# ─── OCR / Layout primitives (WERE MISSING) ───

class RegionType(Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_polygon(cls, polygon: list) -> "BoundingBox":
        """
        Azure DI returns polygon as flat list [x1,y1, x2,y2, x3,y3, x4,y4].
        We take the axis-aligned bounding box.
        """
        xs = [polygon[i] for i in range(0, len(polygon), 2)]
        ys = [polygon[i] for i in range(1, len(polygon), 2)]
        return cls(x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys))


@dataclass
class OCRRegion:
    text: str
    bbox: BoundingBox
    confidence: float
    page_number: int


@dataclass
class LayoutRegion:
    region_id: int
    region_type: RegionType
    bbox: BoundingBox
    confidence: float
    page_number: int


# ─── Structured extraction schemas ───

class InvoiceLineItem(BaseModel):
    description: str = ""
    quantity: float = 0.0
    unit_price: float = 0.0
    line_total: float = 0.0


class InvoiceData(BaseModel):
    vendor_name: Optional[str] = None
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    line_items: List[InvoiceLineItem] = []
    subtotal: float = 0.0
    tax: float = 0.0
    total_amount: float = 0.0
    reasoning: Optional[str] = None


# ─── Pipeline output ───

@dataclass
class DocumentExtractionResult:
    ocr_regions: List[OCRRegion]
    layout_regions: List[LayoutRegion]
    ordered_text: List[str]
    region_images: Dict[int, Any]
    page_count: int
