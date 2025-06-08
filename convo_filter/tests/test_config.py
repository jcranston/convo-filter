"""Tests for the FilterConfig class."""
import os
from collections.abc import Generator
from pathlib import Path

import pytest

from convo_filter.utils.config import FilterConfig


@pytest.fixture(autouse=True)
def cleanup_test_files() -> Generator[None, None, None]:
    """Clean up test files after each test."""
    yield
    # Clean up test files
    patterns = [
        "test.pdf",
        "file.txt",
        "filtered_convo_*.pdf.zip",
        "filtered_convo_*.txt.zip",
        "filtered_convo_*.pdf",
        "filtered_convo_*.txt",
    ]
    for pattern in patterns:
        for file in Path(".").glob(pattern):
            try:
                os.remove(file)
            except OSError:
                pass


def test_valid_config(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
    config = FilterConfig(
        api_key="test-key",
        topics=["topic1", "topic2"],
        pdf_path=str(pdf_path),
    )
    assert config.api_key == "test-key"
    assert config.topics == ["topic1", "topic2"]
    assert config.pdf_path == str(pdf_path)


def test_invalid_pdf_path() -> None:
    with pytest.raises(ValueError):
        FilterConfig(api_key="k", topics=["t"], pdf_path="/not/a/real/file.pdf")


def test_invalid_pdf_extension(tmp_path: Path) -> None:
    txt_path = tmp_path / "file.txt"
    txt_path.write_text("not a pdf")
    with pytest.raises(ValueError):
        FilterConfig(api_key="k", topics=["t"], pdf_path=str(txt_path))


def test_no_topics(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
    with pytest.raises(ValueError):
        FilterConfig(api_key="k", topics=[], pdf_path=str(pdf_path))


def test_invalid_strictness(tmp_path: Path) -> None:
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")
    with pytest.raises(ValueError):
        FilterConfig(
            api_key="k", topics=["t"], pdf_path=str(pdf_path), strictness="super-strict"
        )
