from . import llm_client
from .agents import arXivSummarizationAgent, PaperState
from .utils import get_daily_arxiv_updates, arXivItem

from collections import defaultdict
from typing import Dict, List, Any, Optional, Union, Set
import logging
from dotenv import load_dotenv
import re

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
