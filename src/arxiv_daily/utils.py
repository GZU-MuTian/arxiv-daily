"""
Module to fetch and parse daily arXiv updates for a specified channel.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set
from bs4 import BeautifulSoup
from datetime import datetime
import requests
import logging
import time
import re

logger = logging.getLogger(__name__)


def url_requests_safely(
    url: str,
    max_retries: int = 3,
    timeout: float = 30.0,
    retry_delay: float = 1.0
) -> Optional[requests.Response]:
    """
    Safely fetch a URL with retry logic.

    Args:
        url (str): The URL to request.
        max_retries (int): Maximum number of retry attempts (default: 3).
        timeout (float): Request timeout in seconds (default: 30.0).
        retry_delay (float): Delay between retries in seconds (default: 1.0).
        logger (Optional[logging.Logger]): Logger instance for recording events.

    Returns:
        Optional[requests.Response]: The response object if successful; None otherwise.
    """
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response
            else:
                logger.warning("Attempt %d: Received status code %d", attempt + 1, response.status_code)
        except requests.exceptions.RequestException as e:
            logger.warning("Attempt %d failed with exception: %s", attempt + 1, e)

        # 如果不是最后一次尝试，则等待后重试
        if attempt < max_retries - 1:
            time.sleep(retry_delay)

    logger.error("All %d attempts failed for URL: %s", max_retries, url)
    return None


class arXivItem(BaseModel):
    """Represents a single paper entry from arXiv daily updates."""
    num: Optional[int] = Field(None, description="Sequential number")
    arXivID: str = Field(..., description="arXiv identifier")
    title: str = Field("", description="Title of the paper")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    abstract: str = Field("", description="Abstract text of the paper")
    comments: str = Field("", description="Comments")
    subjects: List[str] = Field(default_factory=list, description="Subject categories")
    formats: Dict[str, str] = Field(
        default_factory=dict,
        description="Available formats"
    )

def get_daily_arxiv_updates(channel: str = "astro-ph") -> List[arXivItem]:
    """
    Fetch and parse the latest daily updates from arXiv for a given channel.

    Args:
        channel (str): The channel of arXiv to search. Defaults to 'astro-ph'.
    
    Returns:
        List[arXivItem]: A list of arXivItem objects containing the latest updates from arXiv.
    """
    articles: List[arXivItem] = []

    # HTTP request
    logger.info("Fetching daily updates for channel: %s", channel)
    base_url = f"https://arxiv.org/list/{channel}/new"
    response = url_requests_safely(base_url)
    if response is None:
        return articles
    
    # Parse HTML using BeautifulSoup
    bsObj = BeautifulSoup(response.text, "html.parser")

    h3_tags = bsObj.find_all('h3')
    # Remove the first <h3> if there are exactly 5
    if len(h3_tags) == 5:
        del h3_tags[0]

    # Validate that the update is for today
    date_pattern = re.compile(r'Showing new listings for (.+?)$')
    date_match = date_pattern.search(h3_tags[0].text)
    if date_match:
        scraping_date = date_match.group(1).split(',')[1].strip()
        try:
            # Parse date string into a date object
            date = datetime.strptime(scraping_date, "%d %B %Y").date()
            today = datetime.today().date()
            if date != today:
                logger.info("No new updates for today (scraping date: %s).", today)
                return articles
        except:
            logger.warning(f"{scraping_date} parsing failed.")
    else:
        logger.warning(f"Failed to extract date from header: {h3_tags[0].text}")

    # Parse the number of new submissions
    number_pattern = re.compile(r'showing \d+ of (\d+) entries')
    new_submissions_match = number_pattern.search(h3_tags[1].text)
    if new_submissions_match:
        new_submissions_number = int(new_submissions_match.group(1))
    else:
        new_submissions_number = 0
    cross_lists_match = number_pattern.search(h3_tags[2].text)
    if cross_lists_match:
        cross_lists_number = int(cross_lists_match.group(1))
    else:
        cross_lists_number = 0
    total_number = new_submissions_number + cross_lists_number
    logger.info("New submissions: %d, Cross-lists: %d → Total: %d", new_submissions_number, cross_lists_number, total_number)
    
    # Locate all paper entries
    dt_list = bsObj.find_all("dt")
    dd_list = bsObj.find_all("dd")
    pairs = zip(dt_list[:total_number], dd_list[:total_number])
    for dt, dd in pairs:
        try:
            # --- Extract sequential number ---
            num_a = dt.find("a", {"name": re.compile(r'^item\d+$')})
            num = int(num_a.get_text().strip()[1:-1]) if num_a else 0

            # --- Extract arXiv ID ---
            arxiv_a = dt.find("a", {"title": "Abstract"})
            arXivID = arxiv_a.get_text().strip() if arxiv_a else ""

            # --- Access metadata container ---
            meta = dd.find("div", class_="meta")
            if not meta:
                logger.debug("Skipping entry %s: missing .meta div", arXivID)
                continue
            
            # --- Title ---
            title_div = meta.find("div", class_="list-title")
            title = title_div.get_text().replace("Title:", "", 1).strip() if title_div else ""

            # --- Authors ---
            authors_div = meta.find("div", class_="list-authors")
            authors = []
            if authors_div:
                authors_text = authors_div.get_text()
                authors = [name.strip() for name in authors_text.split(",") if name.strip()]
            
            # --- Abstract ---
            abstract_p = meta.find("p", class_="mathjax")
            abstract = abstract_p.get_text().strip() if abstract_p else ""

            # --- Comments (optional) ---
            comments_div = meta.find("div", class_="list-comments mathjax")
            comments = ""
            if comments_div:
                comments = re.sub(r'^\s*Comments:\s*', '', comments_div.get_text(), flags=re.IGNORECASE).strip()

            # --- Subjects ---
            subjects_div = meta.find("div", class_="list-subjects")
            subjects = []
            if subjects_div:
                subjects_text = re.sub(r'^\s*Subjects:\s*', '', subjects_div.get_text(), flags=re.IGNORECASE)
                subjects = [s.strip() for s in subjects_text.split(";") if s.strip()]

            # --- Available Formats ---
            formats_dict = {}
            for link in dt.find_all("a", href=True):
                href = link.get("href", "")
                title_attr = link.get("title", "").strip()
                if title_attr != "View HTML":
                    href = "https://arxiv.org" + href
                formats_dict[title_attr] = href

        except Exception as e:
            logger.error("Error parsing article entry (arXiv ID: %s)", arXivID)
            continue

        # --- Construct and store the item ---
        item = arXivItem(
            num=num,
            arXivID=arXivID,
            title=title,
            authors=authors,
            abstract=abstract,
            comments=comments,
            subjects=subjects,
            formats=formats_dict,
        )
        articles.append(item)

    return articles


class Chunk(BaseModel):
    """
    Represents a logical segment (e.g., paragraph, subsection) of a paper.
    """
    section: Optional[str] = Field(None, description="Name of the top-level section this chunk belongs to (e.g., 'Introduction').")
    paragraph: Optional[str] = Field(None, description="paragraph")
    content: str = Field(default="", description="The actual textual content of the chunk.")
    metadata: dict = Field(default_factory=dict, description="Additional metadata information.")

class Paper(BaseModel):
    """
    Represents a structured academic paper with metadata and chunked content.
    """
    source: str = Field(..., description="Path or URI to the original document (e.g., PDF file).")
    title: str = Field(default="", description="Title of the paper.")
    author: str = Field(default="", description="List of author names.")
    abstract: str = Field(default="", description="Abstract text of the paper.")
    chunks: List[Chunk] = Field(default_factory=list, description="List of textual chunks representing the body of the paper.")

    def add_chunk(self, chunk: Chunk):
        """Add a new chunk to the paper's content."""
        self.chunks.append(chunk)

    def get_chunks_by_section(self, section: str) -> List[Chunk]:
        """
        Retrieve all chunks belonging to a specific section.

        Args:
            section: The section name to filter by.

        Returns:
            A list of Chunk objects matching the given section.
        """
        return [c for c in self.chunks if c.section == section]
    
    @property
    def sections(self) -> List[str]:
        """
        Return a sorted list of unique section names present in the paper.
        """
        unique_sections: Set[str] = set()
        for chunk in self.chunks:
            unique_sections.add(chunk.section)
        return sorted(unique_sections)

    @property
    def total_chars(self) -> int:
        """Return the total number of characters across all chunks."""
        return sum(len(chunk.content) for chunk in self.chunks)


def load_arxiv_html_page(url: str) -> Paper:
    """
    Parse the HTML page of an arXiv paper into a structured Paper object.

    Args:
        url (str): URL of the arXiv HTML page.

    Returns:
        Paper: A structured representation of the parsed paper.
    """
    paper = Paper(source=url)

    response = url_requests_safely(url)

    # Parse HTML content
    assert response is not None  # for type checker
    bsObj = BeautifulSoup(response.text, "html.parser")

    # title parsing
    title_tags = bsObj.find("h1", class_="ltx_title")
    if title_tags:
        paper.title = title_tags.get_text(separator=' ', strip=True)
    else:
        logger.warning(f"Title parsing failed.")

    # abstract parsing
    abstract_tags = bsObj.find("div", class_="ltx_abstract")
    if abstract_tags:
        paper.abstract = abstract_tags.get_text(separator=' ', strip=True)
    else:
        logger.warning(f"Abstract parsing failed.")

    # section parsing
    section_tags = bsObj.find_all("section", class_="ltx_section")
    if section_tags:
        for section in section_tags:
            section_id = section.get("id", default="")
            paragraphs = section.find_all("p", class_="ltx_p")
            for p in paragraphs:
                paragraph_id = p.get("id", default="")
                content = p.get_text(separator=' ', strip=True)
                chunk = Chunk(section=section_id, paragraph=paragraph_id, content=content)
                paper.add_chunk(chunk)
    else:
        logging.warning(f"Section parsing failed.")

    return paper
