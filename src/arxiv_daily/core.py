from . import llm_client
from .agents import arXivSummarizationAgent, PaperState
from .chains import KnowledgeGraphExtractor
from .utils import get_daily_arxiv_updates, arXivItem, get_arxiv_metadata

from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
import re
import frontmatter

logger = logging.getLogger(__name__)
load_dotenv()


def _run_new(
    channel: str = "astro-ph",
    category: Optional[Union[str, List[str]]] = None,
) -> Dict[str, List[arXivItem]]:
    """
    Fetch and group daily arXiv articles.

    Args:
        channel: arXiv channel (e.g., "astro-ph")
        category: Category ID(s) to filter by. 
    """
    # Fetch the latest daily arXiv articles from the specified channel (e.g., "astro-ph")
    articles = get_daily_arxiv_updates(channel)

    # Normalize `category` to a set of strings (or None)
    target_categories: Optional[Set[str]] = None
    if category is not None:
        if isinstance(category, str):
            target_categories = {category}
        else:
            target_categories = set(category)

    grouped: Dict[str, List[arXivItem]] = defaultdict(list)
    for article in articles:
        for subject in article.subjects:
            # Extract category ID from the subject string
            subjectID = None
            match = re.search(r'\(([^)]+)\)$', subject.strip())
            if match:
                subjectID = match.group(1)

            # Apply filtering only if categories are specified
            if target_categories is not None:
                if subjectID is None or subjectID not in target_categories:
                    continue

            grouped[subject].append(article)

    return dict(grouped)


def _run_summarize(
    arxivid: str,
    model: str = "deepseek-chat",
    model_provider: str = "deepseek",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning: Optional[bool] = None,
):
    llm_config: dict[str, Any] = {
        "model": model,
        "model_provider": model_provider,
    }
    if temperature is not None:
        llm_config["temperature"] = temperature
    if max_tokens is not None:
        llm_config["max_tokens"] = max_tokens
    if reasoning is not None:
        llm_config["reasoning"] = reasoning
    llm_client.basicConfig(**llm_config)

    # Compile into a runnable app
    app = arXivSummarizationAgent()

    # Run the workflow
    initial_state = PaperState(source=arxivid)
    return app.invoke(initial_state)


def _get_metadata(arxivid: str) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for a given arXiv paper.

    Args:
        arxivid: arXiv identifier

    Returns:
        Dict with title, authors, subjects, abstract, comments, date; None if failed.
    """
    return get_arxiv_metadata(arxivid)


def _run_extractor(
    arxivid: str,
    model: str = "deepseek-chat",
    model_provider: str = "deepseek",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    reasoning: Optional[bool] = None,
):
    """
    Extract knowledge graph relationships from a saved Markdown file.

    Args:
        arxivid: arXiv ID.
                 Searches current directory first, then ARXIV_SUMMARIZE_OUTPUT.
        model: LLM model name.
        model_provider: LLM provider.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        reasoning: Reasoning/thinking mode.

    Returns:
        Dict containing arxiv_id and extracted KnowledgeGraphExtraction result.
    """
    # Find the markdown file by arxiv_id
    candidate_file = Path(f"{arxivid}.md")
    
    # Priority 1: Check current directory
    if candidate_file.exists():
        md_file = candidate_file
    else:
        # Priority 2: Check ARXIV_SUMMARIZE_OUTPUT directory
        output_dir = os.environ.get("ARXIV_SUMMARIZE_OUTPUT")
        if output_dir:
            md_file = Path(output_dir) / candidate_file
        else:
            raise FileNotFoundError(
                f"File '{candidate_file}' not found in current directory.\n"
                f"Tip: Set ARXIV_SUMMARIZE_OUTPUT environment variable to specify the output directory, "
                f"or run 'arxiv summarize {arxivid}' first to generate the file."
            )

    # Parse frontmatter and body content using python-frontmatter
    post = frontmatter.load(md_file)
    summary_text = post.content.strip()
    # Remove the "# AI Summary" header if present
    summary_text = summary_text.lstrip("# AI Summary\n").strip()

    # Configure LLM
    llm_config: dict[str, Any] = {
        "model": model,
        "model_provider": model_provider,
    }
    if temperature is not None:
        llm_config["temperature"] = temperature
    if max_tokens is not None:
        llm_config["max_tokens"] = max_tokens
    if reasoning is not None:
        llm_config["reasoning"] = reasoning
    llm_client.basicConfig(**llm_config)

    # Run knowledge graph extraction
    chain = KnowledgeGraphExtractor()
    return chain.invoke({"summary": summary_text})


