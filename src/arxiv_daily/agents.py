"""
PDF Summarization Agent.
"""
from .chains import PaperCompressionChain, OrganizedSummaryChain, OrganizedSummary
from .utils import url_requests_safely, html_to_markdown, markdown_splitter

from langgraph.graph import StateGraph, START, END

import requests
from typing import List, Dict, Any, Optional
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

    response = url_requests_safely(source)
    try:
        md_text = html_to_markdown(response.text)
    except Exception as e:
        raise RuntimeError(f"Failed to convert HTML to Markdown: {e}") from e

    return {"md_text": md_text.strip()}


def summarize_paper(state: PaperState) -> Dict[str, Any]:
    """
    Generate a concise, structured summary of an astrophysics paper using an LLM.

    Strategy:
    - Split by # and ## headers to preserve section semantics.
    - Compress each section with limited context to balance coherence and redundancy.

    Args:
        state (PaperState): Current state containing md_text.

    Returns:
        Dict[str, Any]: Updated state with 'summary' field populated.
    """
    md_text = state.md_text

    # === Step 1: Split by headers ===
    headers = [
        ("#", "Header 1"),
        ("##", "Header 2")
    ]
    try:
        docs = markdown_splitter(
            md_text, 
            headers_to_split_on=headers, 
            return_each_line=False, 
            strip_headers=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to parse markdown: {e}") from e

    # Estimate word count
    total_word = sum(doc.metadata["word_count"] for doc in docs)
    TARGET_TOTAL_WORDS = 1500

    chain = PaperCompressionChain()
    context = ""

    for sec in track(docs, description="Summarizing sections..."):
        section_text = sec.page_content.strip()
        if not section_text:
            continue

        # Compute word budget for this section
        section_words = sec.metadata["word_count"]
        num_word_limit = int(section_words * TARGET_TOTAL_WORDS / total_word)
        header_title = (
            sec.metadata["header"].get("Header 2") or 
            sec.metadata["header"].get("Header 1") or 
            "Unnamed Section"
        )
        logger.debug(f"Section '{header_title}': {section_words} words → allocated {num_word_limit} words")

        try:
            sec_summary = chain.invoke({"chunk": section_text, "num_word_limit": num_word_limit, "context": context})
            context += sec_summary + "\n\n"
        except Exception as e:
            logger.error(f"Error processing '{sec}': {e}")
            continue

    logger.info("Summarization completed!")
    return {"summary": context}

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