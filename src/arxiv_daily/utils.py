"""
Module to fetch and parse daily arXiv updates for a specified channel.
"""
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, TypedDict, NamedTuple, Any
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from datetime import datetime
from uuid import uuid4
import requests
import logging
import time
import re

logger = logging.getLogger(__name__)


def parse_arxiv_id(source: str) -> str:
    """
    Parse an arXiv ID from a source string.

    Accepts formats like '2101.12345', 'arXiv:2101.12345', or 'arXiv:2101.12345v1'.

    Args:
        source: Raw source string potentially containing an arXiv ID.

    Returns:
        The extracted arXiv ID (e.g., '2101.12345').

    Raises:
        ValueError: If no valid arXiv ID pattern is found.
    """
    src = source.strip()
    arxiv_match = re.fullmatch(r'(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?)', src, re.IGNORECASE)
    if not arxiv_match:
        raise ValueError(f"Source does not match arXiv ID pattern: {source}")
    return arxiv_match.group(1)


def build_markdown_content(
    arxiv_id: str,
    paper_meta: dict,
    summary: Any,
) -> str:
    """
    Build Markdown content for an arXiv paper summary.

    Args:
        arxiv_id: The arXiv identifier.
        paper_meta: Dictionary containing paper metadata (title, authors, subjects, comments, date).
        summary: OrganizedSummary object with the paper summary sections.

    Returns:
        Formatted Markdown content string.
    """
    title = paper_meta.get("title", "")
    authors = paper_meta.get("authors", [])
    abstract = paper_meta.get("abstract", "")
    subjects = paper_meta.get("subjects", [])
    comments = paper_meta.get("comments", "")
    pub_date = paper_meta.get("date", "")

    # Build tags
    tags = ["arxiv", "ai-summary"]
    # Add subjects as tags for classification — extract abbreviations
    if subjects:
        for s in subjects:
            # Extract abbreviation inside parenthesis, e.g. "astro-ph.IM" from "Instrumentation and Methods for Astrophysics (astro-ph.IM)"
            match = re.search(r'\(([^)]+)\)', s)
            if match:
                abbr = match.group(1)
                # Remove characters not allowed in Obsidian tags (dots)
                tag = abbr.replace(".", "_")
                tags.append(tag)

    # Format authors
    authors_str = ", ".join(authors[:3]) + (", et al." if len(authors) > 3 else "")

    # Build frontmatter and title
    subjects_str = ", ".join(subjects) if subjects else ""
    md_content = f"""---
title: "{title}"
arxivId: "{arxiv_id}"
authors: "{authors_str}"
date: {pub_date.replace('/', '-') if pub_date else ''}
tags:
{chr(10).join(f'  - {t}' for t in tags)}
comments: "{comments}"
subjects: "{subjects_str}"
abstract: "{abstract}"
---
# AI Summary
"""

    # Summary sections
    if hasattr(summary, "model_dump"):
        summary_dict = summary.model_dump()
    elif isinstance(summary, dict):
        summary_dict = summary
    else:
        summary_dict = {}

    for k, v in summary_dict.items():
        name = k.replace("_", " ").title()
        md_content += f"## {name}\n{v}\n"

    return md_content


def url_requests_safely(
    url: str,
    max_retries: int = 3,
    timeout: float = 30.0,
    retry_delay: float = 1.0
) -> requests.Response:
    """
    Safely fetch a URL with retry logic.

    Args:
        url (str): The URL to request.
        max_retries (int): Maximum number of retry attempts (default: 3).
        timeout (float): Request timeout in seconds (default: 30.0).
        retry_delay (float): Delay between retries in seconds (default: 1.0).

    Returns:
        requests.Response: The successful response (status code 200).

    Raises:
        RuntimeError: If all retry attempts fail.
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

    raise RuntimeError("All %d attempts failed for URL: %s", max_retries, url)


def get_arxiv_metadata(arxivid: str) -> Optional[Dict[str, Any]]:
    """
    Fetch paper metadata from arXiv abstract page.

    Args:
        arxivid: arXiv identifier

    Returns:
        Dict with title, authors, subjects, abstract, comments, date; None if failed.
    """
    url = f"https://arxiv.org/abs/{arxivid}"
    try:
        response = url_requests_safely(url, max_retries=2)
        bsObj = BeautifulSoup(response.text, "html.parser")

        # Title - use meta tag for stability
        title_meta = bsObj.find("meta", attrs={"name": "citation_title"})
        title = title_meta.get("content", "").strip() if title_meta else ""

        # Authors - use meta tags for stability
        author_metas = bsObj.find_all("meta", attrs={"name": "citation_author"})
        authors = [a.get("content", "").strip() for a in author_metas if a.get("content")]

        # Abstract - use meta tag for stability
        abstract_meta = bsObj.find("meta", attrs={"name": "citation_abstract"})
        abstract = abstract_meta.get("content", "").strip() if abstract_meta else ""

        # Date - use meta tag
        date_meta = bsObj.find("meta", attrs={"name": "citation_date"})
        date_str = date_meta.get("content", "") if date_meta else ""

        # Comments - match both "tablecell comments" and "tablecell comments mathjax"
        comments_td = bsObj.find("td", class_=lambda c: c and "tablecell" in c and "comments" in c)
        comments = ""
        if comments_td:
            comments = comments_td.get_text().strip()

        # Subjects
        subjects_td = bsObj.find("td", class_="tablecell subjects")
        subjects = []
        if subjects_td:
            subjects_text = subjects_td.get_text()
            subjects = [s.strip() for s in subjects_text.split(";") if s.strip()]

        return {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "comments": comments,
            "subjects": subjects,
            "date": date_str,
        }
    except Exception as e:
        logger.warning(f"Failed to fetch metadata for {arxivid}: {e}")
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


LTX_TAG_SKIP_CLASSES = {
    "ltx_tag_section",
    "ltx_tag_subsection",
    "ltx_tag_subsubsection",
}

def parse_ltx_para(tag: Tag) -> str:
    """
    Recursively parse a LaTeX paragraph tag and convert its contents to plain text or Markdown.

    Args:
        tag: The BeautifulSoup Tag to parse

    Returns:
        Normalized string representation of the paragraph content
    """
    parts: List[str] = []

    for child in tag.children:
        if isinstance(child, NavigableString):
            # Handle plain text nodes
            parts.append(str(child))
        elif isinstance(child, Tag):
            if child.name == "math":
                # Extract math formula from alttext attribute
                formula = child.get("alttext", "")
                display = child.get("display", "inline")
                if display == "block":
                    parts.append(f"\n$$\n{formula}\n$$\n")
                else:
                    parts.append(f"${formula}$")
            elif child.name == "cite":
                # Extract citation text without tags
                cite_text = child.get_text()
                parts.append(cite_text)
            elif child.name == "span":
                classes = child.get("class", [])
                if set(classes) & LTX_TAG_SKIP_CLASSES:
                    continue
                else:
                    # Process other spans recursively (e.g., equation numbers, emphasis, etc.)
                    parts.append(parse_ltx_para(child))
            else:
                # Recursively process other nested tags
                parts.append(parse_ltx_para(child))
        else:
            # Log unexpected node types for debugging
            logger.debug(f"Unexpected node type in paragraph: {type(child)}")

    # Normalize whitespace for prose; math formulas from alttext are assumed clean
    text = "".join(parts)
    return re.sub(r'\s+', ' ', text).strip()


def html_to_markdown(html: str) -> str:
    """
    Convert LaTeX-generated HTML to Markdown format.

    Args:
        html: Input HTML string generated from LaTeX

    Returns:
        Converted Markdown string with proper heading levels and content
    """
    md_lines: List[str] = []
    soup = BeautifulSoup(html, "html.parser")

    # ====== 1. Document Title ======
    title_tag = soup.find('title')
    if title_tag:
        title_text = title_tag.get_text(strip=True)
        md_lines.append(f"# {title_text}\n")

    # ====== 2. Abstract Section ======
    abstract_div = soup.find("div", class_="ltx_abstract")
    if abstract_div:
        for p in abstract_div.find_all("p", class_="ltx_p"):
            md_text = parse_ltx_para(p)
            md_lines.append(md_text)

    # ====== 3. Main Content Sections ======
    sections = soup.select("section.ltx_section, section.ltx_subsection, section.ltx_subsubsection")
    logger.debug(f"Found {len(sections)} content sections to process")

    for section in sections:
        selector = "h2.ltx_title, h3.ltx_title, h4.ltx_title, h5.ltx_title, h6.ltx_title"
        heading_tag = section.select_one(selector)

        if heading_tag:
            heading_text = parse_ltx_para(heading_tag)
            level = {'h2': 2, 'h3': 3, 'h4': 4, 'h5': 5, 'h6': 6}.get(heading_tag.name, 2)
            md_lines.append(f"{'#' * level} {heading_text}\n")
            logger.debug(f"Processed heading (level {level}): {heading_text}")

        # Process paragraphs directly under this section
        # for para_div in section.find_all("div", class_="ltx_para"):
        for para_div in section.select(":scope > div.ltx_para"):
            # Ensure we're only processing paragraphs directly in this section
            para_md = parse_ltx_para(para_div)
            if para_md:
                md_lines.append(para_md + "\n")

    return "\n".join(md_lines).strip()


class ParagraphMetadata(TypedDict):
    """
    Metadata for a paragraph chunk.
    """
    paragraph_id: str
    header: Dict[str, str]
    prev_paragraph_id: str
    next_paragraph_id: str
    create_at: str
    create_by: str
    word_count: int

def markdown_splitter(
    markdown_text: str,
    headers_to_split_on: Optional[List[tuple]] = None,
    return_each_line: bool = True,
    strip_headers: bool = True,
    create_at: Optional[str] = None,
    create_by: str = "arxiv-daily"
) -> List[Document]:
    """
    Parse a Markdown file into a list of LangChain Document objects with enriched metadata.

    Args:
        markdown_text (str): markdown text.
        headers_to_split_on (Optional[List[Tuple[str, str]]]): List of (header_prefix, header_name) tuples. Defaults to H1–H4.
        return_each_line (bool): Whether to split each line as a separate document. Default: True.
        strip_headers (bool: Strip split headers from the content of the chunk
        create_at (Optional[str]): Creation date in 'YYYY-MM-DD'. Defaults to today.
        create_by: Creator identifier.

    Returns:
        List[Document]: List of documents with custom metadata from ParagraphMetadata.
    """
    documents: List[Document] = []

    if create_at is None:
        create_at = datetime.now().strftime("%Y-%m-%d")

    # Define header patterns for MarkdownHeaderTextSplitter
    _DEFAULT_HEADERS_TO_SPLIT_ON = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4")
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on or _DEFAULT_HEADERS_TO_SPLIT_ON,
        return_each_line=return_each_line,
        strip_headers=strip_headers
    )
    try:
        raw_docs = splitter.split_text(markdown_text)
    except Exception as e:
        logger.error(f"Failed to split markdown: {e}")
        return documents

    for doc in raw_docs:
        paragraph = doc.page_content

        # Estimate word count
        word_count = len(paragraph.split())

        para_id = str(uuid4())

        meta = ParagraphMetadata(
            paragraph_id=para_id,
            header=doc.metadata,
            create_at=create_at,
            create_by=create_by,
            word_count=word_count,
            prev_paragraph_id="",
            next_paragraph_id=""
        )

        documents.append(Document(page_content=doc.page_content, metadata=meta))

    # Second pass: link prev/next paragraph IDs
    for idx, doc in enumerate(documents):
        if idx > 0:
            doc.metadata["prev_paragraph_id"] = documents[idx - 1].metadata["paragraph_id"]
        if idx < len(documents) - 1:
            doc.metadata["next_paragraph_id"] = documents[idx + 1].metadata["paragraph_id"]

    logger.info(f"Parsed {len(documents)} chunks'")
    return documents


class TocEntry(NamedTuple):
    num: str
    title: str

def generate_toc(docs: List[Document]) -> List[TocEntry]:
    """
    Generate a hierarchical table of contents (TOC) from document headers (supports levels 1–6).

    Args:
        docs: List of Document objects containing header metadata.
              
    Returns:
        List of TocEntry objects representing the hierarchical TOC
    """
    toc: List[TocEntry] = []  # Final list of TOC entries
    # Counters for heading levels (index 0 = Header 1, index 5 = Header 6)
    counters = [0, 0, 0, 0, 0, 0]
    # Track seen titles per level to avoid duplicates
    seen: Dict[int, set] = {level: set() for level in range(1, 7)}
    
    # Process all documents in order
    for doc in docs:
        header = doc.metadata.get("header", {})
        # Iterate through all header fields in the metadata
        for key, title in header.items():
            # Extract heading level from key (e.g., "Header 3" → 3)
            level = int(key.split()[-1])

            # Skip if this exact title has already been processed at this level
            if title in seen[level]:
                continue

            # Increment counter for the current heading level
            counters[level - 1] += 1

            # Reset all counters for deeper levels (level+1 and beyond)
            for lower_level in range(level, 6):
                counters[lower_level] = 0

            # Record this title as seen at its level to prevent future duplicates
            seen[level].add(title)

            # Build the hierarchical number string (e.g., "1.2.3")
            num = ".".join(str(counters[i]) for i in range(level))

            entry = TocEntry(num=num, title=title)
            # Append the TOC entry
            toc.append(entry)
    return toc

