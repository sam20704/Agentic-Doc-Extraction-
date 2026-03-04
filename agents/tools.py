"""
LangChain tools for document analysis.
"""
from typing import Dict
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from prompts.templates import CHART_ANALYSIS_PROMPT, TABLE_ANALYSIS_PROMPT
from config.settings import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)


# Global state — set once by setup_tools()
_region_images: Dict[int, dict] = {}
_vlm = None


def setup_tools(region_images: Dict[int, dict]):
    """Inject region images and initialise the Vision LLM."""
    global _region_images, _vlm

    _region_images = region_images

    _vlm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT,
        temperature=0,
    )
    print("✓ Tools setup complete")


def _call_vlm(image_base64: str, prompt: str) -> str:
    """Send an image + prompt to Azure OpenAI Vision."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
            },
        ]
    )
    response = _vlm.invoke([message])
    return response.content


@tool
def AnalyzeChart(region_id: int) -> str:
    """Analyze a chart or figure region using Azure OpenAI Vision.
    Use this tool when you need to extract data from charts, graphs, or figures.

    Args:
        region_id: The ID of the layout region to analyze (must be a chart/figure type)

    Returns:
        Structured chart data as JSON string
    """
    if region_id not in _region_images:
        available = list(_region_images.keys())
        return (
            f"Error: Region {region_id} not found. "
            f"Available region IDs: {available}"
        )

    region_data = _region_images[region_id]

    if region_data["type"] != "figure":
        return (
            f"Warning: Region {region_id} is type '{region_data['type']}', "
            f"not a figure. Proceeding with visual analysis anyway."
        )

    result = _call_vlm(region_data["base64"], CHART_ANALYSIS_PROMPT)

    return (
        f"CHART_DATA (Region {region_id}, "
        f"Page {region_data.get('page', '?')}):\n{result}"
    )


@tool
def AnalyzeTable(region_id: int) -> str:
    """Extract structured data from a table region using Azure OpenAI Vision.
    Use this tool when you need to verify or deeply analyze tabular data.

    Args:
        region_id: The ID of the layout region to analyze (must be a table type)

    Returns:
        Structured table data as JSON string
    """
    if region_id not in _region_images:
        available = list(_region_images.keys())
        return (
            f"Error: Region {region_id} not found. "
            f"Available region IDs: {available}"
        )

    region_data = _region_images[region_id]

    if region_data["type"] != "table":
        return (
            f"Warning: Region {region_id} is type '{region_data['type']}', "
            f"not a table. Proceeding with visual analysis anyway."
        )

    result = _call_vlm(region_data["base64"], TABLE_ANALYSIS_PROMPT)

    return (
        f"TABLE_DATA (Region {region_id}, "
        f"Page {region_data.get('page', '?')}):\n{result}"
    )
