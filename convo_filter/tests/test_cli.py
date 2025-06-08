"""Smoke tests for the CLI interface."""
import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from rich.console import Console

from convo_filter.cli.main import cli

# Add constants for magic values
EXPECTED_EXIT_CODE = 0
EXPECTED_EXIT_CODE_ERROR = 1
EXPECTED_EXIT_CODE_USAGE_ERROR = 2


@pytest.fixture(autouse=True)
def cleanup_output_files() -> Generator[None, None, None]:
    """Clean up output files after each test."""
    yield
    # Clean up all possible output file patterns
    patterns = [
        "filtered_convo_*.pdf.zip",
        "filtered_convo_*.txt.zip",
        "filtered_convo_*.pdf",
        "filtered_convo_*.txt",
        "custom_output.pdf",
        "custom_output.txt",
        "custom_output.pdf.zip",
        "custom_output.txt.zip",
    ]
    for pattern in patterns:
        for file in Path(".").glob(pattern):
            try:
                os.remove(file)
            except OSError:
                pass


def test_cli_filter_smoke(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
    topics_path = tmp_path / "topics.txt"
    topics_path.write_text("topic1\ntopic2\n")

    runner = CliRunner()
    with patch("convo_filter.cli.main.ConversationFilter") as mock_filter_class:
        mock_filter_instance = mock_filter_class.return_value
        mock_filter_instance.process.return_value = None
        result = runner.invoke(
            cli,
            [
                "filter",
                "--pdf",
                str(pdf_path),
                "--topics-file",
                str(topics_path),
            ],
            env={"ANTHROPIC_API_KEY": "fake-key"},
        )
        assert result.exit_code == EXPECTED_EXIT_CODE


def test_cli_filter_no_api_key(tmp_path: Path) -> None:
    """Test CLI behavior when no API key is provided."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
    topics_path = tmp_path / "topics.txt"
    topics_path.write_text("topic1\ntopic2\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["filter", "--pdf", str(pdf_path), "--topics-file", str(topics_path)],
        env={},
    )
    assert result.exit_code == EXPECTED_EXIT_CODE_ERROR


def test_cli_filter_both_topics_options(tmp_path: Path) -> None:
    """Test CLI behavior when both --topics-file and --topics are provided."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
    topics_path = tmp_path / "topics.txt"
    topics_path.write_text("topic1\ntopic2\n")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "filter",
            "--pdf",
            str(pdf_path),
            "--topics-file",
            str(topics_path),
            "--topics",
            "topic3",
            "--topics",
            "topic4",
        ],
        env={"ANTHROPIC_API_KEY": "fake-key"},
    )
    assert result.exit_code == EXPECTED_EXIT_CODE_USAGE_ERROR
    assert "Cannot specify both --topics-file and --topics" in result.output


def test_cli_filter_default_topics(tmp_path: Path) -> None:
    """Test CLI behavior when no topics are provided."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")

    runner = CliRunner()
    with patch("convo_filter.cli.main.ConversationFilter") as mock_filter_class:
        mock_filter_instance = mock_filter_class.return_value
        mock_filter_instance.process.return_value = None
        result = runner.invoke(
            cli,
            ["filter", "--pdf", str(pdf_path)],
            env={"ANTHROPIC_API_KEY": "fake-key"},
        )
        assert result.exit_code == EXPECTED_EXIT_CODE_USAGE_ERROR
        assert "Either --topics-file or --topics must be specified" in result.output


def test_cli_filter_invalid_pdf(tmp_path: Path) -> None:
    """Test CLI behavior with an invalid PDF file."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"Not a PDF file")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["filter", "--pdf", str(pdf_path), "--topics", "topic1"],
        env={"ANTHROPIC_API_KEY": "fake-key"},
    )
    assert result.exit_code == 1


def test_cli_setup(tmp_path: Path) -> None:
    """Test the setup command."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["setup"], input="y\n")
        assert result.exit_code == 0
        assert "Configuration file created successfully" in result.output


def test_cli_setup_existing_config(tmp_path: Path) -> None:
    """Test setup command when config file already exists."""
    from convo_filter.cli.main import do_setup

    config_path = tmp_path / "config.py"
    config_path.write_text("# Existing config")

    # Test with 'n' to not overwrite
    result = do_setup(config_path, lambda prompt: False, Console())
    assert result is False
    assert config_path.read_text() == "# Existing config"

    # Test with 'y' to overwrite
    result = do_setup(config_path, lambda prompt: True, Console())
    assert result is True
    assert "your-api-key-here" in config_path.read_text()
