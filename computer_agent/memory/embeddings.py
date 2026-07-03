"""
Embedding generation for memory indexing.
Uses the Anthropic API (or a local sentence-transformers model as fallback)
to embed task descriptions for pgvector similarity search.
"""

from __future__ import annotations

import hashlib

from computer_agent.logging_setup import get_logger

logger = get_logger(__name__)

# Simple in-process cache for repeated embeddings
_cache: dict[str, list[float]] = {}


async def embed_text(text: str) -> list[float] | None:
    """
    Generate a 1536-dimensional embedding for the given text.
    Tries Anthropic voyage embeddings first; falls back to sentence-transformers.
    Returns None if embedding is unavailable.
    """
    # Cache hit
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in _cache:
        return _cache[cache_key]

    embedding = await _try_sentence_transformers(text)

    if embedding:
        _cache[cache_key] = embedding

    return embedding


async def _try_sentence_transformers(text: str) -> list[float] | None:
    """Use sentence-transformers (local, no API cost)."""
    try:
        import asyncio


        # Use a 768-dim model, padded to 1536 for pgvector schema compat
        model = _get_st_model()
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, lambda: model.encode(text).tolist())

        # Pad or truncate to 1536 dimensions (pgvector schema requirement)
        if len(vec) < 1536:
            vec = vec + [0.0] * (1536 - len(vec))
        elif len(vec) > 1536:
            vec = vec[:1536]

        return vec
    except Exception as e:
        logger.debug("sentence_transformers_failed", error=str(e))
        return None


_st_model = None


def _get_st_model():
    """Lazy-load the sentence transformer model."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("embedding_model_loaded", model="all-MiniLM-L6-v2")
    return _st_model
