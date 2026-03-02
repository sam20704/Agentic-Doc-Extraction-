import logging
from typing import Dict, List

from ..models import InvoiceData

logger = logging.getLogger(__name__)

TOLERANCE = 0.02  # accounts for rounding in tax/currency


class CalculatorTool:

    def verify_invoice(self, invoice: InvoiceData) -> Dict:
        issues: List[str] = []

        # 1. Verify each line item total
        for i, item in enumerate(invoice.line_items):
            expected = round(item.quantity * item.unit_price, 2)
            if abs(expected - item.line_total) > TOLERANCE:
                issues.append(
                    f"Line {i+1}: {item.quantity} × {item.unit_price} = "
                    f"{expected}, but got {item.line_total}"
                )

        # 2. Verify subtotal
        calculated_subtotal = round(
            sum(item.line_total for item in invoice.line_items), 2
        )
        subtotal_match = abs(calculated_subtotal - invoice.subtotal) < TOLERANCE
        if not subtotal_match:
            issues.append(
                f"Subtotal: sum of lines = {calculated_subtotal}, "
                f"stated = {invoice.subtotal}"
            )

        # 3. Verify total = subtotal + tax
        expected_total = round(invoice.subtotal + invoice.tax, 2)
        total_match = abs(expected_total - invoice.total_amount) < TOLERANCE
        if not total_match:
            issues.append(
                f"Total: {invoice.subtotal} + {invoice.tax} = "
                f"{expected_total}, stated = {invoice.total_amount}"
            )

        return {
            "calculated_subtotal": calculated_subtotal,
            "subtotal_match": subtotal_match,
            "total_match": total_match,
            "line_issues": issues,
            "passed": len(issues) == 0,
        }
