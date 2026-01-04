"""
PDF Summarization Agent.
"""
from .chains import PaperCompressionChain, OrganizedSummaryChain, OrganizedSummary
from .utils import url_requests_safely, html_to_markdown, parse_markdown

from langgraph.graph import StateGraph, START, END

import requests
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel, Field
import logging
import re
from rich.progress import track

logger = logging.getLogger(__name__)


# --- State Definition ---

class PaperState(BaseModel):
    """
    Represents the state of the summarization workflow.
    """
    source: str = Field(..., description="Path to the input PDF file.")
    md_text: str = Field(default="", description="Raw markdown text extracted from HTML.")
    summary: str = Field(default="", description="Generated summary of the paper.")
    organized_summary: Optional[OrganizedSummary] = Field(default=None, description="Organized Summary")

# --- Node Definition ---

def resolve_source(state: PaperState) -> Dict[str, Any]:
    """
    Resolve ambiguous source string into a concrete arXiv HTML URL.

    Args:
        state (PaperState): Current state containing the source string.

    Returns:
        Dict[str, Any]: Dict[str, Any]: Updated state with 'source' set to the valid HTML URL.
    """
    src = state.source.strip()

    # Match arXiv ID (e.g., '2101.12345' or 'arXiv:2101.12345')
    arxiv_match = re.fullmatch(r'(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)', src, re.IGNORECASE)

    if not arxiv_match:
        raise ValueError(f"Source does not match arXiv ID pattern: {src}")

    arxiv_id = arxiv_match.group(1)
    logger.info(f"Detected arXiv ID: {arxiv_id}.")

    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        response = requests.head(html_url, timeout=10)  # Use HEAD request to check existence without downloading content
        if response.status_code == 200:
            return {"source": html_url}
        else:
            raise RuntimeError(f"arXiv HTML page not available for {src}")
    except Exception as e:
        raise RuntimeError(f"Failed to access arXiv HTML URL {html_url}: {e}") from e


def parse_html_page(state: PaperState) -> Dict[str, Any]:
    """
    Parse the HTML page of an arXiv paper.

    Args:
        state (PaperState): Current state containing the html url.

    Returns:
        Dict[str, Any]: Updated state with 'paper' field populated.
    """
    source = state.source
    logger.debug(f"Parsing HTML page: {source}")

    response = url_requests_safely("https://arxiv.org/html/2512.23758v1")
    try:
        md_text = html_to_markdown(response.text)
    except Exception as e:
        raise RuntimeError(f"Failed to convert HTML to Markdown: {e}") from e

    return {"md_text": md_text}


def summarize_paper(state: PaperState) -> Dict[str, Any]:
    """
    Generate a concise, structured summary of an astrophysics paper using an LLM.

    The function first compresses the abstract, then iteratively compresses each section while maintaining cumulative context to avoid redundancy and ensure coherence. Character allocation per section is proportional to its length.

    Args:
        state (PaperState): Current state containing md_text.

    Returns:
        Dict[str, Any]: Updated state with 'summary' field populated.
    """
    md_text = state.md_text

    try:
        docs = parse_markdown(md_text)


        

    chain = PaperCompressionChain()
    context = ""
    # Compress section
    total_char_limit = 10000
    total_chars = paper.total_chars
    for sec in track(paper.sections, description="Summarizing sections..."):
        section_chunks = paper.get_chunks_by_section(sec)

        # Skip empty sections
        if not section_chunks:
            continue

        section_text = '\n\n'.join(chunk.content.strip() for chunk in section_chunks)

        num_chunk_char_limit = len(section_text) * total_char_limit / total_chars
        try:
            response = chain.invoke({"chunk": section_text, "num_chunk_char_limit": num_chunk_char_limit, "context": context})
            context += response + "\n\n"
        except Exception as e:
            logger.error(f"Error processing '{sec}': {e}")
            continue

    logger.info("Summarization completed!")
    return {"summary": context.strip()}

def organize_summarization(state: PaperState) -> Dict[str, Any]:
    """
    Organizes an unstructured astrophysics paper summary into a standardized JSON structure with six key sections: background, motivation, methodology, results, interpretation, and implication.

    Args:
        state (PaperState): Current state containing raw_text.
        llm (Runnable): Configured LLM chain for summarization.

    Returns:
        Dict[str, Any]: Updated state with 'organized_summary' field populated.
    """
    chain = OrganizedSummaryChain()

    try:
        response = chain.invoke({"summary": state.summary})
        logger.debug("Organized Summary!")
        return {"organized_summary": response}
    except Exception as e:
        logger.error(f"Failed to organize summary: {e}")
        return {}


# --- Graph Construction ---3

def arXivSummarizationAgent():
    """
    Creates the LangGraph workflow for paper summarization.

    Returns:
        StateGraph: LangGraph workflow ready for execution.
    """
    logger.debug("Creating LangGraph workflow for paper summarization.")

    # Initialize the state graph 
    workflow = StateGraph(PaperState)

    # Add nodes
    workflow.add_node("resolve_source", resolve_source)
    workflow.add_node("parse_html_page", parse_html_page)
    workflow.add_node("summarize_paper", summarize_paper)
    workflow.add_node("organize_summarization", organize_summarization)

    # Define the edges/flow between nodes
    workflow.add_edge(START, "resolve_source")
    workflow.add_edge("resolve_source", "parse_html_page")
    workflow.add_edge("parse_html_page", "summarize_paper")
    workflow.add_edge("summarize_paper", "organize_summarization")
    workflow.add_edge("organize_summarization", END)

    return workflow.compile()