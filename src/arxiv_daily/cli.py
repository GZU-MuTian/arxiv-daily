from .core import _run_new, _run_summarize, _get_metadata, _run_extractor
from .chains import OrganizedSummary
from .utils import parse_arxiv_id, build_markdown_content, create_concept_file, format_concept_entry

from typing import Optional, Literal, List
from pathlib import Path
from datetime import datetime
import sys
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
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
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        envvar="ARXIV_NEW_OUTPUT",
        help="Output directory for Obsidian daily note."
    ),
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

    # Save to Obsidian daily note
    if output:
        today_str = datetime.today().strftime("%Y-%m-%d")
        daily_file = Path(output) / f"{today_str}.md"
        daily_file.parent.mkdir(parents=True, exist_ok=True)

        # Build daily note content
        seen: set[str] = set()
        lines: List[str] = []
        for articles in grouped_articles.values():
            for article in articles:
                if article.arXivID not in seen:
                    seen.add(article.arXivID)
                    lines.append(f"- [[{parse_arxiv_id(article.arXivID)}]] {article.title}")

        daily_file.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"📝 Daily note saved to [green]{daily_file}[/green]")


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


@app.command(help="Extract knowledge graph relationships from arXiv paper.")
def extractor(
    arxivid: str = typer.Argument(..., help="arXiv identifier."),
    model: str = typer.Option("deepseek-chat", "--model", "-m", envvar="ARXIV_SUMMARIZE_MODEL", help="Model name."),
    model_provider: str = typer.Option("deepseek", "--provider", "-p", envvar="ARXIV_SUMMARIZE_MODEL_PROVIDER", help="Model provider."),
    temperature: Optional[float] = typer.Option(None, "--temp", "-t", help="Sampling temperature."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum number of output tokens."),
    reasoning: Optional[bool] = typer.Option(None, "--reasoning", help="Controls the reasoning/thinking mode for supported models."),
    output: Optional[str] = typer.Option(None, "--output", "-o", envvar="ARXIV_EXTRACTOR_OUTPUT", help="Output directory path (Obsidian-friendly)."),
) -> None:
    """
    Extract knowledge graph relationships from an arXiv paper.
    """
    try:
        arxiv_id = parse_arxiv_id(arxivid)
    except ValueError as e:
        console.print(f"[bold red]Failed to parse arXiv ID: {arxivid}[/bold red]")
        raise typer.Exit(1)

    try:
        results = _run_extractor(
            arxivid=arxiv_id,
            model=model,
            model_provider=model_provider,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning
        )
    except Exception as e:
        console.print(f"[bold red]Failed to extract knowledge graph.[/bold red]")
        raise typer.Exit(1)
    
    relationships = results.relationships
    if not relationships:
        console.print("No relationships extracted.")
        return
    
    # Create a table for better visualization
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("#", style="dim", width=3, justify="center")
    table.add_column("Concept", style="cyan", min_width=20)
    table.add_column("Category", style="green", min_width=15)
    table.add_column("Relation", style="yellow", min_width=12)
    table.add_column("Description", style="white", min_width=30, ratio=1)
    
    for i, rel in enumerate(relationships, 1):
        table.add_row(
            str(i),
            rel.concept,
            rel.category,
            rel.relation,
            rel.description
        )
    
    console.print(table)

    # Save concept files to output directory
    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_concepts = []
        for rel in relationships:
            # Create safe filename from concept name (normalize spaces to hyphens)
            safe_concept = rel.concept.replace(" ", "-")
            filename = safe_concept + ".md"
            concept_file = output_dir / filename
            
            # Format entry line
            entry_line = format_concept_entry(
                arxiv_id=arxiv_id,
                relation=rel.relation,
                description=rel.description,
            )
            
            if concept_file.exists():
                # Check if arxiv_id already exists in file to avoid duplicates
                existing_content = concept_file.read_text(encoding="utf-8")
                arxiv_link = f"[[{arxiv_id}]]"
                if arxiv_link in existing_content:
                    logger.debug(f"Skipping {rel.concept}: {arxiv_id} already exists")
                    continue
                # Append to existing file
                with open(concept_file, "a", encoding="utf-8") as f:
                    f.write(entry_line + "\n")
            else:
                # Create new file with frontmatter
                md_content = create_concept_file(
                    concept=rel.concept,
                    category=rel.category,
                )
                concept_file.write_text(md_content + "\n" + entry_line + "\n", encoding="utf-8")
            
            saved_concepts.append(rel.concept)
        
        console.print(f"\n📝 Saved {len(saved_concepts)} concept(s) to [green]{output_dir}[/green]")


if __name__ == "__main__":
    app()