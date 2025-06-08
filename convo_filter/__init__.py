"""Conversation Filter - A tool to filter PDF conversations by topic using Claude AI."""

from convo_filter.core.filter import ConversationFilter
from convo_filter.utils.config import FilterConfig

__version__ = "0.1.0"
__all__ = ["ConversationFilter", "FilterConfig"]
