from __future__ import annotations


class LLMUnavailableError(RuntimeError):
    """Raised when LLM service is required but unavailable."""

