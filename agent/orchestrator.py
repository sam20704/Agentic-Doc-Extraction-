import logging

from .interpretation_agent import InterpretationAgent
from .validation_agent import ValidationAgent
from .reasoning_agent import ReasoningAgent
from .state import AgentState
from ..pipeline import DocumentPipeline
from ..tools.vlm_factory import AzureOpenAIVLM
from ..exceptions import AgentExecutionError

logger = logging.getLogger(__name__)


class EnterpriseWorkflowEngine:

    def __init__(self):
        self.pipeline = DocumentPipeline()
        self.vlm = AzureOpenAIVLM()  # single instance, shared across agents

    def process(self, document_path: str, goal: str = "extract_invoice") -> AgentState:

        logger.info("Workflow starting: %s (goal=%s)", document_path, goal)

        # ── Stage 1: OCR + Layout ──
        extraction_result = self.pipeline.run(document_path)

        state = AgentState(
            extraction_result=extraction_result,
            user_goal=goal,
        )

        interpreter = InterpretationAgent(self.vlm)
        validator = ValidationAgent(extraction_result)

        # ── Stage 2: Interpret → Validate loop ──
        while state.iteration < state.max_iterations:

            try:
                state = interpreter.run(state)
            except AgentExecutionError as exc:
                logger.error("Interpretation failed on attempt %d: %s", state.iteration + 1, exc)
                state.add_issue(f"Interpretation error: {exc}")
                state.decision = "rejected"
                break

            state = validator.run(state)

            if state.decision == "auto_approved":
                logger.info("Auto-approved on iteration %d", state.iteration + 1)
                break

            if state.decision == "rejected":
                logger.warning("Rejected on iteration %d", state.iteration + 1)
                break

            # requires_review — retry if we have iterations left
            if state.iteration + 1 < state.max_iterations:
                logger.info(
                    "Issues found, retrying (%d/%d): %s",
                    state.iteration + 1, state.max_iterations, state.issues,
                )
                state.reset_for_retry()
            else:
                logger.warning("Max iterations reached, sending to review")
                break

        logger.info(
            "Workflow complete: decision=%s, confidence=%.2f",
            state.decision, state.confidence_score,
        )
        return state

    def ask(self, state: AgentState, question: str) -> str:
        """Post-extraction Q&A."""
        reasoner = ReasoningAgent()
        return reasoner.answer_question(state, question)
