from .core import _run_new, _run_summarize, _get_metadata
from .chains import OrganizedSummary
from .utils import parse_arxiv_id, build_markdown_content

from typing import Optional, Literal, List
from pathlib import Path
from datetime import datetime
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
import logging
import logging.config

# Global console and logger
logger = logging.getLogger(__name__)
console = Console(highlight=False)


# --- CLI Command ---

app = typer.Typer(
    help="AI for arXiv.",
    rich_markup_mode="rich"
)

# Define allowed log levels using Literal
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

@app.callback()
def main(
    log_level: LogLevel = typer.Option("ERROR", "--log-level", "-v", help="Enable logging with specified level (e.g., DEBUG, INFO, WARNING). ERROR by default."),
) -> None:
    """
    Global options for all commands.
    
    The --log-level option applies to every subcommand automatically.
    """
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(message)s",
                "datefmt": "[%X]",
            }
        },
        "handlers": {
            "rich": {
                "()": RichHandler,
                "level": log_level,
                "rich_tracebacks": False,
                "show_time": True,
                "show_path": True,
                "formatter": "default",
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["rich"],
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)


@app.command(help="Fetch and display the latest arXiv preprints for a given channel.")
def new(
    channel: str = typer.Option('astro-ph', "--channel", "-c", help="arXiv channel to monitor."),
    category: Optional[List[str]] = typer.Option(
        None,
        "--category", "-t",
        envvar="ARXIV_CATEGORY",
        help="Filter by arXiv category ID(s)."
    )
) -> None:
    """
    Main CLI entry point to fetch daily arXiv articles for the specified channel
    """
    console.print(Rule(f"{channel}", characters="=", style="dim"))

    # Parse category input (from CLI or envvar)
    parsed_category: Optional[List[str]] = None
    if category:
        if len(category) == 1 and ',' in category[0]:
            parsed_category = [v.strip() for v in category[0].split(',') if v.strip()]
        else:
            parsed_category = [v.strip() for v in category if v.strip()]

    grouped_articles = _run_new(
        channel=channel,
        category=parsed_category
    )

    # Output results
    for subject, articles in sorted(grouped_articles.items()):
        console.print(Rule(f"{subject}", characters="-", style="dim"))
        # Print each article under this subject
        for article in articles:
            # arXiv ID
            console.print(f"🆔 [blue]{article.arXivID}[/blue]")

            # Title
            console.print(f"📄 [bold]{article.title}[/bold]")

            # Authors
            if article.authors:
                authors_display = ", ".join(article.authors[:3])
                if len(article.authors) > 3:
                    authors_display += ", et al."
                console.print(f"👥 {authors_display}")

            # Comments
            if article.comments:
                console.print(f"💬 [italic dim]{article.comments}[/italic dim]")

            # Formats — make them clickable!
            if article.formats:
                format_links = []
                for title, href in article.formats.items():
                    format_links.append(f"[link={href}][blue]{title}[/blue][/link]")
                console.print(f"🔗 {', '.join(format_links)}")

            # Abstract (truncated)
            abstract_display = (
                article.abstract[:800] + "..." if len(article.abstract) > 800 else article.abstract
            )
            console.print(f"📝 {abstract_display}\n")


@app.command(help="Fetch metadata for an arXiv paper by its identifier.")
def meta(
    arxivid: str = typer.Argument(..., help="arXiv identifier (e.g., 2401.12345, arXiv:2401.12345)."),
) -> None:
    """
    Fetch and display metadata for a given arXiv paper.
    """
    try:
        arxiv_id = parse_arxiv_id(arxivid)
    except ValueError as e:
        console.print(f"[bold red]Failed to fetch metadata for {arxivid}[/bold red]")
        raise typer.Exit(1)

    metadata = _get_metadata(arxiv_id)
    if metadata is None:
        console.print(f"[bold red]Failed to fetch metadata for {arxiv_id}[/bold red]")
        raise typer.Exit(1)

    console.print(Rule(arxiv_id, style="dim"))

    # Title
    title = metadata.get("title", "")
    if title:
        console.print(f"📄 [bold]{title}[/bold]")

    # Authors
    authors = metadata.get("authors", [])
    if authors:
        authors_display = ", ".join(authors[:3])
        if len(authors) > 3:
            authors_display += ", et al."
        console.print(f"👥 {authors_display}")

    # Date
    date = metadata.get("date", "")
    if date:
        console.print(f"📅 {date}")

    # Subjects
    subjects = metadata.get("subjects", [])
    if subjects:
        console.print(f"🏷️  {'; '.join(subjects)}")

    # Comments
    comments = metadata.get("comments", "")
    if comments:
        console.print(f"💬 [italic dim]{comments}[/italic dim]")

    # Abstract
    abstract = metadata.get("abstract", "")
    if abstract:
        console.print(f"📝 {abstract}")


@app.command(help="Extract key insights from arXiv paper.")
def summarize(
    arxivid: str = typer.Argument(..., help="arXiv identifier."),
    model: str = typer.Option("deepseek-chat", "--model", "-m", envvar="ARXIV_SUMMARIZE_MODEL", help="Model name."),
    model_provider: str = typer.Option("deepseek", "--provider", "-p", envvar="ARXIV_SUMMARIZE_MODEL_PROVIDER", help="Model provider."),
    temperature: Optional[float] = typer.Option(None, "--temp", "-t", help="Sampling temperature."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum number of output tokens."),
    reasoning: Optional[bool] = typer.Option(None, "--reasoning", help="Controls the reasoning/thinking mode for supported models."),
    output: Optional[str] = typer.Option(None, "--output", "-o", envvar="ARXIV_SUMMARIZE_OUTPUT", help="Output directory path (Obsidian-friendly)."),
) -> None:
    try:
        arxiv_id = parse_arxiv_id(arxivid)
    except ValueError as e:
        console.print(f"[bold red]Failed to fetch metadata for {arxivid}[/bold red]")
        raise typer.Exit(1)

    results = _run_summarize(
        arxivid=arxiv_id,
        model=model,
        model_provider=model_provider,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning
    )

    summary: OrganizedSummary = results.get("organized_summary") or OrganizedSummary()

    # Console output
    console.print(Rule(arxiv_id, style="dim"))
    for k, v in summary.model_dump().items():
        name = k.replace("_", " ").title()
        console.print(f"[bold cyan]{name}[/bold cyan]", style="bold")
        console.print(f"{v}\n")

    # Write to file
    if output:
        # Extract metadata
        paper_meta = _get_metadata(arxiv_id) or {}

        # Build markdown content
        md_content = build_markdown_content(
            arxiv_id=arxiv_id,
            paper_meta=paper_meta,
            summary=summary,
        )

        md_file = Path(output) / f"{arxiv_id}.md"
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(md_content, encoding="utf-8")
        console.print(f"📝 Saved to [green]{md_file}[/green]")


if __name__ == "__main__":
    app()