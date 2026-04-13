from .core import _run_new, _run_summarize
from .chains import OrganizedSummary

from typing import Optional, Literal, List
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


@app.command(help="Extract key insights from arXiv paper.")
def summarize(
    arxivid: str = typer.Argument(..., help="arXiv identifier."),
    model: str = typer.Option("deepseek-chat", "--model", "-m", envvar="ARXIV_SUMMARIZE_MODEL", help="Model name."),
    model_provider: str = typer.Option("deepseek", "--provider", "-p", envvar="ARXIV_SUMMARIZE_MODEL_PROVIDER", help="Model provider."),
    temperature: Optional[float] = typer.Option(None, "--temp", "-t", help="Sampling temperature."),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Maximum number of output tokens."),
    reasoning: Optional[bool] = typer.Option(None, "--reasoning", help="Controls the reasoning/thinking mode for supported models."),
) -> None:
    results = _run_summarize(
        arxivid=arxivid,
        model=model,
        model_provider=model_provider,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning=reasoning
    )

    # Output results
    console.print(Rule("Organized Summary", style="dim"))
    summary: OrganizedSummary = results.get("organized_summary", "") 
    for k, v in summary.model_dump().items():
        name = k.replace("_", " ").title()
        console.print(f"{name}: ", style="bold white")
        console.print(f"{v}\n", style="white")


if __name__ == "__main__":
    app()