"""Conversation Filter - A tool to filter PDF conversations by topic using Claude AI."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "ConversationFilter":
        from convo_filter.core.filter import ConversationFilter

        return ConversationFilter
    if name == "FilterConfig":
        from convo_filter.utils.config import FilterConfig

        return FilterConfig
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["ConversationFilter", "FilterConfig"]
