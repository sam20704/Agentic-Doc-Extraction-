import logging
from typing import List

from ..tools.calculator import CalculatorTool
from ..tools.grounding import GroundingTool
from ..models import DocumentExtractionResult
from .state import AgentState

logger = logging.getLogger(__name__)


class ValidationAgent:

    def __init__(self, extraction_result: DocumentExtractionResult):
        self.calculator = CalculatorTool()
        self.grounding = GroundingTool(extraction_result)

    def run(self, state: AgentState) -> AgentState:
        invoice = state.draft_invoice
        if invoice is None:
            state.add_issue("No invoice data to validate")
            state.decision = "rejected"
            return state

        # ── Math verification ──
        math_result = self.calculator.verify_invoice(invoice)

        for issue in math_result["line_issues"]:
            state.add_issue(issue)

        if not math_result["subtotal_match"]:
            state.add_issue("Subtotal does not match sum of line items")

        if not math_result["total_match"]:
            state.add_issue("Total does not match subtotal + tax")

        # ── Grounding verification ──
        fields_to_ground = {
            "total_amount": invoice.total_amount,
            "vendor_name": invoice.vendor_name,
            "invoice_number": invoice.invoice_number,
        }

        for field_name, value in fields_to_ground.items():
            if value is None or value == "" or value == 0.0:
                continue
            result = self.grounding.verify(value)
            if not result["grounded"]:
                state.add_issue(f"'{field_name}' value '{value}' not grounded in OCR")

        # ── Decision ──
        if not state.has_issues:
            state.decision = "auto_approved"
            state.confidence_score = 0.98
        elif len(state.issues) <= 2:
            state.decision = "requires_review"
            state.confidence_score = 0.6
        else:
            state.decision = "rejected"
            state.confidence_score = 0.3

        logger.info(
            "Validation complete: decision=%s, confidence=%.2f, issues=%d",
            state.decision, state.confidence_score, len(state.issues),
        )
        return state
