"""
Document Intelligence Agent using LangChain + Retrieval.
"""
from typing import List, Dict, Optional

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from .tools import AnalyzeChart, AnalyzeTable, setup_tools
from prompts.templates import create_system_prompt
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    USE_RETRIEVAL,
)


class DocumentAgent:
    """Agent that answers questions about documents."""

    def __init__(
        self,
        ocr_regions: List,
        layout_regions: List,
        region_images: Dict[int, dict],
        ordered_text: List[dict],
        retriever=None,                    # ← NEW
    ):
        self.ocr_regions = ocr_regions
        self.layout_regions = layout_regions
        self.region_images = region_images
        self.ordered_text = ordered_text
        self.retriever = retriever         # ← NEW

        # Inject region images into tool globals
        setup_tools(region_images)

        self._initialize_agent()

    # ────────────────────────────────────
    # Agent initialisation
    # ────────────────────────────────────

    def _initialize_agent(self):
        """Build the LangChain tool-calling agent."""
        print("Initializing document agent...")

        # Decide prompt mode based on retrieval availability
        retrieval_active = USE_RETRIEVAL and self.retriever is not None

        system_prompt = create_system_prompt(
            self.ordered_text,
            self.layout_regions,
            use_retrieval=retrieval_active,
        )

        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            temperature=0,
        )

        tools = [AnalyzeChart, AnalyzeTable]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, tools, prompt)

        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True,
        )

        mode = "retrieval-augmented" if (USE_RETRIEVAL and self.retriever) else "full-context"
        print(f"✓ Document agent initialized ({mode} mode)")

    # ────────────────────────────────────
    # Query
    # ────────────────────────────────────

    def query(self, question: str) -> str:
        """
        Answer a user question about the document.

        If retrieval is enabled the most relevant chunks
        are prepended to the user message so the agent
        reasons over focused context instead of everything.
        """
        try:
            # Build the user-facing input
            if USE_RETRIEVAL and self.retriever is not None:
                context = self.retriever.format_retrieved_context(question)

                enhanced_input = (
                    f"══ RETRIEVED DOCUMENT EXCERPTS ══\n"
                    f"(These are the most relevant parts of the document "
                    f"for your question.)\n\n"
                    f"{context}\n\n"
                    f"══ USER QUESTION ══\n"
                    f"{question}\n\n"
                    f"Follow the mandatory reasoning workflow in the system prompt."
                )
            else:
                enhanced_input = question

            response = self.agent_executor.invoke({"input": enhanced_input})
            return response["output"]

        except Exception as e:
            return f"Error processing query: {str(e)}"

    # ────────────────────────────────────
    # Stats
    # ────────────────────────────────────

    def get_statistics(self) -> Dict[str, any]:
        """Return document + retriever statistics."""
        stats = {
            "total_ocr_regions": len(self.ocr_regions),
            "total_layout_regions": len(self.layout_regions),
            "tables": len([r for r in self.layout_regions if r.region_type == "table"]),
            "figures": len([r for r in self.layout_regions if r.region_type == "figure"]),
            "text_blocks": len([r for r in self.layout_regions if r.region_type == "text"]),
            "region_images": len(self.region_images),
            "retrieval_enabled": USE_RETRIEVAL and self.retriever is not None,
        }

        if self.retriever:
            stats["retriever"] = self.retriever.get_stats()

        return stats
