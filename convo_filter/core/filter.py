"""Core conversation filtering functionality."""
from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

import anthropic
from pydantic import BaseModel, Field
from pypdf import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer

from convo_filter.utils.config import FilterConfig
from convo_filter.utils.logging import get_logger

logger = get_logger(__name__)

# Add constants for magic values
MAX_REQUESTS_PER_MINUTE = 50
PAGE_GROUP_THRESHOLD = 2
MAX_TOKENS_PER_CHUNK = 4000


class PageMatch(BaseModel):
    """Represents a match found in a page."""

    page_num: int
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class TopicMatch(BaseModel):
    """Represents a match for a specific topic."""

    topic: str
    pages: list[PageMatch]
    is_consecutive: bool = False


class ConversationFilter:
    """Main class for filtering conversations by topic using Claude AI."""

    def __init__(self, config: FilterConfig) -> None:
        """Initialize the conversation filter.

        Args:
            config: Configuration object containing all necessary settings
        """
        self.config = config
        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.api_call_count = 0
        self.api_call_lock = threading.Lock()
        self.results_lock = threading.Lock()

        # Rate limiting setup
        self.requests_timestamps: list[float] = []
        self.max_requests_per_minute = MAX_REQUESTS_PER_MINUTE

        # Set up output paths
        self._setup_output_paths()

    def _setup_output_paths(self) -> None:
        """Set up the output file paths based on configuration."""
        if not self.config.output_path:
            if self.config.privacy_mode:
                random_id = str(uuid.uuid4())[:8]
                base_filename = f"filtered_convo_{random_id}"
            else:
                base_name = Path(self.config.pdf_path).stem
                base_filename = f"filtered_convo_{base_name}"

            self.output_path = (
                f"{base_filename}.{'pdf' if self.config.output_pdf else 'txt'}"
            )
        else:
            self.output_path = self.config.output_path

        # Add .zip extension if not present
        if not self.output_path.endswith(".zip"):
            self.compressed_output_path = f"{self.output_path}.zip"
        else:
            self.compressed_output_path = self.output_path
            self.output_path = self.output_path[:-4]

    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting to stay under API limits."""
        with self.api_call_lock:
            current_time = time.time()
            one_minute_ago = current_time - 60

            # Remove old timestamps
            self.requests_timestamps = [
                t for t in self.requests_timestamps if t > one_minute_ago
            ]

            if (
                len(self.requests_timestamps) >= self.max_requests_per_minute
                and self.requests_timestamps
            ):
                # Force mypy to accept our types with assertions
                assert isinstance(min(self.requests_timestamps), int | float)
                oldest_timestamp_f = float(min(self.requests_timestamps))
                assert isinstance(current_time, int | float)
                current_time_f = float(current_time)
                # Split calculation into steps
                base_time = oldest_timestamp_f + 60
                wait_time = base_time - current_time_f  # type: ignore
                if wait_time > 0:
                    logger.info(f"Rate limit: waiting {wait_time:.2f}s...")
                    time.sleep(wait_time + 0.1)

                    current_time = time.time()
                    one_minute_ago = current_time - 60
                    self.requests_timestamps = [
                        t for t in self.requests_timestamps if t > one_minute_ago
                    ]

            self.requests_timestamps.append(current_time)
            self.api_call_count += 1

    def extract_text_from_pdf(self) -> dict[int, str]:
        """Extract text from PDF, handling compressed PDFs if needed.

        Returns:
            Dictionary mapping page numbers to their text content
        """
        logger.info("Processing input file...")

        temp_dir = tempfile.mkdtemp()
        extracted_pdf_path: str | None = None

        try:
            if self.config.pdf_path.endswith(".zip"):
                logger.info("Detected compressed PDF. Extracting...")
                with zipfile.ZipFile(self.config.pdf_path, "r") as zip_ref:
                    pdf_files = [f for f in zip_ref.namelist() if f.endswith(".pdf")]

                    if not pdf_files:
                        raise ValueError("No PDF file found in the zip archive")

                    pdf_file = pdf_files[0]
                    extracted_pdf_path = os.path.join(temp_dir, pdf_file)
                    zip_ref.extract(pdf_file, temp_dir)

                    if not self.config.privacy_mode:
                        logger.info(f"Extracted PDF: {pdf_file}")
            else:
                extracted_pdf_path = os.path.join(
                    temp_dir, os.path.basename(self.config.pdf_path)
                )
                shutil.copy2(self.config.pdf_path, extracted_pdf_path)

            if not extracted_pdf_path:
                raise ValueError("Failed to extract or copy PDF file")

            logger.info("Extracting text from PDF...")
            reader = PdfReader(extracted_pdf_path)
            pages_text: dict[int, str] = {}
            total_pages = len(reader.pages)
            logger.info(f"Total pages in PDF: {total_pages}")
            for i, page in enumerate(reader.pages):
                if total_pages > PAGE_GROUP_THRESHOLD and i % 10 == 0:
                    percent_done = (i / total_pages) * 100
                    logger.info(
                        f"Extracting text: {i}/{total_pages} pages "
                        f"({percent_done:.1f}%)..."
                    )
                page_num = i + 1
                text = page.extract_text()
                if text and text.strip():
                    pages_text[page_num] = text.strip()
            logger.info(f"Extracted text from {len(pages_text)} pages with content.")
            return pages_text
        except Exception as err:
            logger.error(f"Error extracting text from PDF: {err}")
            raise
        finally:
            shutil.rmtree(temp_dir)

    def filter_conversation(self, pages_text: dict[int, str]) -> dict[str, TopicMatch]:
        """Filter pages by specified topics using the Claude API.

        Args:
            pages_text: Dictionary of page numbers to page text

        Returns:
            Dictionary mapping topics to lists of matching pages
        """
        if not pages_text:
            raise ValueError("No pages to process")

        # Always start with a fresh results dictionary
        results: dict[str, list[dict[str, Any]]] = {
            topic: [] for topic in self.config.topics
        }
        page_items = [(page_num, text) for page_num, text in pages_text.items()]

        total_pages = len(page_items)
        logger.info(
            "Processing %d pages for %d topics...",
            total_pages,
            len(self.config.topics),
        )

        num_workers = min(self.config.max_workers, max(1, (os.cpu_count() or 1) - 1))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            # Process pages in batches
            for i in range(0, len(page_items), self.config.pages_per_batch):
                batch = page_items[i : i + self.config.pages_per_batch]
                future = executor.submit(self._process_page_batch, batch)
                futures.append(future)
            # Wait for all batches to complete and merge results
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_results = future.result()
                    for topic, topic_matches in batch_results.items():
                        results[topic].extend(topic_matches)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    raise

        # Group consecutive pages if enabled
        if self.config.group_consecutive_pages:
            results = self._group_consecutive_pages(results)

        # Convert results to TopicMatch objects
        topic_matches_dict: dict[str, TopicMatch] = {}
        for topic, matches in results.items():
            page_matches: list[PageMatch] = []
            for match in matches:
                page_matches.append(
                    PageMatch(
                        page_num=match["page_num"],
                        text=match["text"],
                        confidence=match["confidence"],
                    )
                )
            topic_matches_dict[topic] = TopicMatch(
                topic=topic,
                pages=page_matches,
                is_consecutive=any(m.get("is_consecutive", False) for m in matches),
            )

        return topic_matches_dict

    def _process_page_batch(
        self, page_batch: list[tuple[int, str]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Process a batch of pages for topic matching.

        Args:
            page_batch: List of (page_num, text) tuples to process

        Returns:
            Dictionary mapping topics to lists of matching pages
        """
        prompt = self._create_prompt(page_batch)
        self._enforce_rate_limit()

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=4000,
                temperature=0,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )
            if not response.content or not isinstance(
                response.content[0], anthropic.types.TextBlock
            ):
                raise ValueError("Unexpected response format from Claude API")
            content = response.content[0].text
            matches: dict[str, list[dict[str, Any]]] = json.loads(content)
            # Ensure all topics are present as keys
            for topic in self.config.topics:
                if topic not in matches:
                    matches[topic] = []
            return matches
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    def _create_prompt(self, page_batch: list[tuple[int, str]]) -> str:
        """Create the prompt for the Claude API.

        Args:
            page_batch: List of (page_num, text) tuples to process

        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        for page_num, text in page_batch:
            prompt_parts.append(f"Page {page_num}:\n{text}\n")
        return "\n".join(prompt_parts)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the Claude API.

        Returns:
            System prompt string
        """
        topics_str = ", ".join(self.config.topics)
        return f"""You are a helpful assistant that analyzes text for specific
topics. For each page of text provided, identify if it contains content related
to any of these topics: {topics_str}. Return a JSON object where each key is a
topic and the value is a list of matches.
Each match should have:
- page_num: The page number
- text: The relevant text excerpt
- confidence: A float between 0 and 1 indicating confidence in the match
- is_consecutive: A boolean indicating if this page is consecutive with the
previous match

Example response format:
{{
    "topic1": [
        {{
            "page_num": 1,
            "text": "relevant text excerpt",
            "confidence": 0.95,
            "is_consecutive": false
        }}
    ],
    "topic2": []
}}"""

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks that fit within token limits.

        Args:
            text: Text to split into chunks

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        words = text.split()
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0
        for word in words:
            if current_length + 1 > MAX_TOKENS_PER_CHUNK:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(word)
            current_length += 1
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _group_consecutive_pages(
        self, results: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Group consecutive pages in the results.

        Args:
            results: Dictionary mapping topics to lists of matching pages

        Returns:
            Updated results with consecutive pages grouped
        """
        grouped_results: dict[str, list[dict[str, Any]]] = {}
        for topic, matches in results.items():
            if not matches:
                grouped_results[topic] = []
                continue

            # Sort matches by page number
            sorted_matches = sorted(matches, key=lambda x: x["page_num"])
            grouped_matches: list[dict[str, Any]] = []
            current_group: dict[str, Any] | None = None
            end_page: int | None = None

            for match in sorted_matches:
                if current_group is None:
                    current_group = match.copy()
                    current_group["is_consecutive"] = False
                    end_page = match["page_num"]
                elif end_page is not None and match["page_num"] == end_page + 1:
                    # Extend current group
                    current_group["text"] += f"\n\n{match['text']}"
                    current_group["confidence"] = max(
                        current_group["confidence"], match["confidence"]
                    )
                    current_group["is_consecutive"] = True
                    end_page = match["page_num"]
                else:
                    if current_group is not None and end_page is not None:
                        current_group["end_page"] = end_page
                        grouped_matches.append(current_group)
                    current_group = match.copy()
                    current_group["is_consecutive"] = False
                    end_page = match["page_num"]

            if current_group is not None and end_page is not None:
                current_group["end_page"] = end_page
                grouped_matches.append(current_group)

            grouped_results[topic] = grouped_matches

        return grouped_results

    def create_pdf(self, results: dict[str, TopicMatch]) -> None:
        """Create a PDF file with the filtered results.

        Args:
            results: Dictionary mapping topics to TopicMatch objects
        """
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
        )
        styles = getSampleStyleSheet()
        story: list[Flowable] = []

        for topic, topic_match in results.items():
            if not topic_match.pages:
                continue

            # Add topic header
            story.append(Paragraph(f"Topic: {topic}", styles["Heading1"]))
            story.append(Spacer(1, 12))

            # Add each page match
            for page_match in topic_match.pages:
                story.append(
                    Paragraph(f"Page {page_match.page_num}", styles["Heading2"])
                )
                story.append(Paragraph(page_match.text, styles["Normal"]))
                story.append(Spacer(1, 12))

        doc.build(story)

    def write_results_to_file(self, results: dict[str, TopicMatch]) -> None:
        """Write the filtered results to a text file.

        Args:
            results: Dictionary mapping topics to TopicMatch objects
        """
        with open(self.output_path, "w", encoding="utf-8") as f:
            for topic, topic_match in results.items():
                if not topic_match.pages:
                    continue

                f.write(f"Topic: {topic}\n")
                f.write("=" * 80 + "\n\n")

                for page_match in topic_match.pages:
                    f.write(f"Page {page_match.page_num}\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"{page_match.text}\n\n")

    def process(self) -> None:
        """Process the PDF file and generate filtered output."""
        try:
            # Extract text from PDF
            pages_text = self.extract_text_from_pdf()

            # Filter conversation
            results = self.filter_conversation(pages_text)

            # Generate output
            if self.config.output_pdf:
                self.create_pdf(results)
            else:
                self.write_results_to_file(results)

            logger.info(f"Processing complete. Output saved to {self.output_path}")

            # Cleanup output file if privacy_mode is enabled
            if self.config.privacy_mode and os.path.exists(self.output_path):
                os.remove(self.output_path)

        except Exception as e:
            logger.error(f"Error processing file: {e}")
            raise


def extract_text_from_pdf(pdf_path: str) -> dict[int, str]:
    """Extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary mapping page numbers to their text content
    """
    return _extract_text_from_pdf(pdf_path)


def _extract_text_from_pdf(pdf_path: str) -> dict[int, str]:
    """Internal function to extract text from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dictionary mapping page numbers to their text content
    """
    reader = PdfReader(pdf_path)
    pages_text: dict[int, str] = {}
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages_text[i + 1] = text.strip()
    return pages_text


def filter_conversation(  # noqa: PLR0913
    pages_text: dict[int, str],
    topics: list[str],
    api_key: str,
    model: str = "claude-3-haiku-20240307",
    batch_size: int = 2,
    max_workers: int = 1,
    group_consecutive_pages: bool = True,
    strictness: str = "high",
) -> dict[str, TopicMatch]:
    """Filter a conversation by topics using the Claude API.

    Args:
        pages_text: Dictionary mapping page numbers to their text content
        topics: List of topics to filter for
        api_key: Anthropic API key
        model: Claude model to use
        batch_size: Number of pages to process in each batch
        max_workers: Maximum number of worker threads
        group_consecutive_pages: Whether to group consecutive pages
        strictness: How strict to be in matching topics

    Returns:
        Dictionary mapping topics to TopicMatch objects
    """
    config = FilterConfig(
        api_key=api_key,
        topics=topics,
        pdf_path="",  # Not needed for this function
        model=model,
        pages_per_batch=batch_size,
        max_workers=max_workers,
        group_consecutive_pages=group_consecutive_pages,
        strictness=strictness,
    )
    filter_instance = ConversationFilter(config)
    return filter_instance.filter_conversation(pages_text)
