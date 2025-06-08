"""Command-line interface for the conversation filter."""
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

from convo_filter.core.filter import ConversationFilter
from convo_filter.utils.config import FilterConfig
from convo_filter.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def read_topics_from_file(file_path: str) -> list[str]:
    """Read topics from a file, one per line.

    Args:
        file_path: Path to the topics file

    Returns:
        List of topics
    """
    topics = []
    with open(file_path) as f:
        for file_line in f:
            line = file_line.strip()
            if line and not line.startswith("#"):
                topics.append(line)
    return topics


class MutuallyExclusiveCommand(click.Command):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # First parse the args normally
        result = super().parse_args(ctx, args)

        # Then check for mutual exclusivity
        if ctx.params.get("topics_file") and ctx.params.get("topics"):
            raise click.UsageError("Cannot specify both --topics-file and --topics")
        return result


@click.group()
def cli() -> None:
    """Conversation Filter - Filter PDF conversations by topic using Claude AI."""
    pass


@cli.command(cls=MutuallyExclusiveCommand)
@click.option(
    "--topics-file",
    type=click.Path(exists=True),
    help="Path to a file containing topics to search for, one per line",
)
@click.option(
    "--topics",
    multiple=True,
    help="Topics to search for (can be specified multiple times)",
)
@click.option(
    "--pdf",
    type=click.Path(exists=True),
    required=True,
    help="Path to the PDF file to analyze",
)
@click.option(
    "--model",
    default="claude-3-haiku-20240307",
    help="Claude model to use",
)
@click.option(
    "--batch-size",
    type=int,
    default=2,
    help="Number of pages to process in each batch",
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    help="Maximum number of worker threads",
)
@click.option(
    "--no-group",
    is_flag=True,
    help="Disable grouping of consecutive pages",
)
@click.option(
    "--privacy",
    is_flag=True,
    help="Enable privacy mode (randomize output filenames)",
)
@click.option(
    "--no-pdf",
    is_flag=True,
    help="Disable PDF output (output text only)",
)
@click.option(
    "--strictness",
    type=click.Choice(["low", "medium", "high"]),
    default="high",
    help="How strict to be in matching topics",
)
@click.option(
    "--output",
    type=click.Path(),
    help="Output file path (default: filtered_convo_<input>.pdf)",
)
def filter(  # noqa: PLR0913
    topics_file: str | None,
    topics: tuple[str, ...],
    pdf: str,
    model: str,
    batch_size: int,
    max_workers: int,
    no_group: bool,
    privacy: bool,
    no_pdf: bool,
    strictness: str,
    output: str | None,
) -> None:
    """Filter a PDF file for specific topics."""
    if not topics_file and not topics:
        raise click.UsageError("Either --topics-file or --topics must be specified")

    # Read topics from file if specified
    if topics_file:
        with open(topics_file) as f:
            file_topics = [line.strip() for line in f if line.strip()]
        topics = tuple(file_topics)

    # Create configuration
    config = FilterConfig(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        topics=list(topics),
        pdf_path=pdf,
        model=model,
        batch_size=batch_size,
        max_workers=max_workers,
        group_consecutive_pages=not no_group,
        privacy_mode=privacy,
        output_pdf=not no_pdf,
        strictness=strictness,
        output_path=output,
        pages_per_batch=batch_size,
    )

    # Run the filter
    filter = ConversationFilter(config)
    filter.process()


def do_setup(
    config_path: Path,
    confirm_func: Callable[[str], bool],
    console: Console,
) -> bool:
    """Set up the configuration file.

    Args:
        config_path: Path to the configuration file
        confirm_func: Function to call for user confirmation
        console: Rich console for output

    Returns:
        True if setup was successful, False otherwise
    """
    config_content = '''"""Configuration for Conversation Filter."""

# Your Anthropic API key
ANTHROPIC_API_KEY = "your-api-key-here"

# Default topics to filter by
DEFAULT_TOPICS = [
    "meeting notes",
    "project updates",
    "action items"
]

# Claude model to use
MODEL_NAME = "claude-3-haiku-20240307"

# Performance and cost optimization settings
BATCH_SIZE = 10      # Number of topics to check in one API call
MAX_WORKERS = 2      # Number of parallel workers
PAGES_PER_BATCH = 4  # Number of pages to process in one API call

# Feature flags
GROUP_CONSECUTIVE_PAGES = True  # Group consecutive pages discussing the same topic
PRIVACY_MODE = True            # Hide sensitive information in output
OUTPUT_PDF = True              # Output as PDF instead of text
STRICTNESS = "high"            # How strict to be when matching topics: 
                                 "low", "medium", or "high"
'''
    if config_path.exists():
        if not confirm_func("Config file already exists. Overwrite?"):
            return False
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        console.print(
            Panel.fit(
                "[green]Configuration file created successfully![/green]\n\n"
                "Please edit config.py and add your Anthropic API key.",
                title="Success",
                border_style="green",
            )
        )
        return True
    except Exception as err:
        logger.error(f"Error creating config file: {err}")
        raise click.ClickException(str(err)) from err


@cli.command()
def setup() -> None:
    """Set up the configuration file."""
    do_setup(Path("config.py"), click.confirm, console)


if __name__ == "__main__":
    cli()
