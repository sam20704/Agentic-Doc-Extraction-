from dataclasses import dataclass, field
from typing import List, Optional

from ..models import DocumentExtractionResult, InvoiceData


@dataclass
class AgentState:
    extraction_result: DocumentExtractionResult
    user_goal: str

    iteration: int = 0
    max_iterations: int = 3

    draft_invoice: Optional[InvoiceData] = None
    issues: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    decision: str = "pending"  # pending | auto_approved | requires_review | rejected

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    def add_issue(self, issue: str):
        self.issues.append(issue)

    def reset_for_retry(self):
        self.issues.clear()
        self.draft_invoice = None
        self.confidence_score = 0.0
        self.decision = "pending"
        self.iteration += 1
