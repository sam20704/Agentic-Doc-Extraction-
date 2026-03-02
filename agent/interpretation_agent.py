import json
import logging

from ..models import InvoiceData
from ..tools.vlm_factory import AzureOpenAIVLM
from ..exceptions import AgentExecutionError
from .state import AgentState

logger = logging.getLogger(__name__)

INVOICE_SCHEMA = """
{
  "vendor_name": "string or null",
  "invoice_number": "string or null",
  "invoice_date": "string or null",
  "line_items": [
    {
      "description": "string",
      "quantity": number,
      "unit_price": number,
      "line_total": number
    }
  ],
  "subtotal": number,
  "tax": number,
  "total_amount": number,
  "reasoning": "string — explain how you derived each field"
}
"""

EXTRACTION_PROMPT = (
    "You are a precise document extraction engine. "
    "Extract structured invoice data from the OCR text provided. "
    "Return ONLY valid JSON matching this exact schema:\n"
    f"{INVOICE_SCHEMA}\n"
    "If a field is not found, use null for strings and 0.0 for numbers. "
    "In the 'reasoning' field, explain your extraction logic."
)


class InterpretationAgent:

    def __init__(self, vlm: AzureOpenAIVLM):
        self.vlm = vlm

    def run(self, state: AgentState) -> AgentState:
        full_text = " ".join(r.text for r in state.extraction_result.ocr_regions)

        if not full_text.strip():
            raise AgentExecutionError("No OCR text available for interpretation")

        # Truncate safely at word boundary
        max_chars = 6000
        if len(full_text) > max_chars:
            truncated = full_text[:max_chars]
            truncated = truncated[: truncated.rfind(" ")]
            full_text = truncated

        try:
            raw_response = self.vlm.analyze_text(
                system_prompt=EXTRACTION_PROMPT,
                user_content=full_text,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            parsed = InvoiceData.model_validate_json(raw_response)
            state.draft_invoice = parsed
            logger.info("Interpretation complete: %s", parsed.invoice_number)

        except json.JSONDecodeError as exc:
            logger.error("VLM returned invalid JSON: %s", exc)
            raise AgentExecutionError(f"Invalid JSON from VLM: {exc}") from exc
        except Exception as exc:
            logger.error("Interpretation failed: %s", exc)
            raise AgentExecutionError(str(exc)) from exc

        return state
