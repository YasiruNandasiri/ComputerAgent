"""
Unified LLM exception hierarchy.

Providers translate native exceptions into these; the Coordinator only
imports from here, staying provider-agnostic.
"""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-related errors."""


class LLMRateLimitError(LLMError):
    """Raised only after retries are exhausted (HTTP 429 / quota exceeded)."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMContextWindowError(LLMError):
    """The prompt exceeds the model's context window — retrying is pointless."""


class LLMTransientError(LLMError):
    """Transient connection / 5xx / timeout — may succeed if retried."""
