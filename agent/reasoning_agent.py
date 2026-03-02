import re
import logging
from typing import Optional

from .state import AgentState
from ..exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class ReasoningAgent:

    def answer_question(self, state: AgentState, question: str) -> str:
        invoice = state.draft_invoice
        if invoice is None:
            return "No invoice data available to reason about."

        q = question.lower().strip()

        # ── "What if we remove line item X?" ──
        if "remove line item" in q:
            idx = self._extract_line_number(q)
            if idx is None or idx < 0 or idx >= len(invoice.line_items):
                return (
                    f"Invalid line item number. "
                    f"Invoice has {len(invoice.line_items)} line items (1-indexed)."
                )
            removed = invoice.line_items[idx]
            new_subtotal = round(invoice.subtotal - removed.line_total, 2)
            # Keep same tax rate
            tax_rate = (
                invoice.tax / invoice.subtotal if invoice.subtotal > 0 else 0
            )
            new_tax = round(new_subtotal * tax_rate, 2)
            new_total = round(new_subtotal + new_tax, 2)
            return (
                f"Removing line {idx+1} ('{removed.description}', "
                f"${removed.line_total:.2f}):\n"
                f"  New subtotal: ${new_subtotal:.2f}\n"
                f"  Estimated tax: ${new_tax:.2f}\n"
                f"  New total: ${new_total:.2f}"
            )

        # ── "What is the largest line item?" ──
        if "largest" in q and "line" in q:
            if not invoice.line_items:
                return "No line items found."
            largest = max(invoice.line_items, key=lambda x: x.line_total)
            return (
                f"Largest line item: '{largest.description}' "
                f"at ${largest.line_total:.2f}"
            )

        # ── "What is the tax rate?" ──
        if "tax rate" in q:
            if invoice.subtotal > 0:
                rate = (invoice.tax / invoice.subtotal) * 100
                return f"Effective tax rate: {rate:.2f}%"
            return "Cannot compute tax rate: subtotal is zero."

        return (
            "I can answer questions like:\n"
            "- 'What if we remove line item 2?'\n"
            "- 'What is the largest line item?'\n"
            "- 'What is the tax rate?'"
        )

    @staticmethod
    def _extract_line_number(text: str) -> Optional[int]:
        """Safely extract a 1-indexed line number and convert to 0-indexed."""
        match = re.search(r"line\s*(?:item\s*)?\s*(\d+)", text)
        if match:
            return int(match.group(1)) - 1
        return None
