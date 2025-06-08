"""Logging configuration for the conversation filter."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str, level: int | None = None) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Name of the logger
        level: Optional logging level (defaults to INFO)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    if level is None:
        level = logging.INFO

    # Only configure if not already configured
    if not logger.handlers:
        logger.setLevel(level)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Add formatter to handler
        handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(handler)

    return logger
