"""Embedding provider abstraction: Voyage AI (primary) + Jina (standby).

Design rules from the architecture doc (§5.1):
- Never mix Voyage and Jina vectors in one index — each provider has its own table.
- "Failover" means a Jina *standby index* built alongside the primary, NOT hot-swapping
  providers mid-query.  Retries on transient errors stay within the same provider.
- Both providers are called with output_dimension=512 (Matryoshka truncation) so the
  vector columns in both tables have the same fixed width.
- input_type is "document" at ingest time, "query" at query time.

Public API:
  embed_texts(texts, input_type, provider, api_key)  → list[list[float]]
  embed_chunks(chunks, provider, api_key)            → list[EmbeddedChunk]
  get_provider(name)                                 → EmbedProvider
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal, cast

import httpx

from crawler.chunk import Chunk
from crawler.config import (
    JINA_EMBED_BATCH_SIZE,
    JINA_EMBED_DIMS,
    JINA_EMBED_MODEL,
    JINA_EMBED_URL,
    VOYAGE_EMBED_BATCH_SIZE,
    VOYAGE_EMBED_DIMS,
    VOYAGE_EMBED_DTYPE,
    VOYAGE_EMBED_MODEL,
    VOYAGE_INPUT_TYPE_DOC,
    VOYAGE_INPUT_TYPE_QUERY,
)

logger = logging.getLogger(__name__)

ProviderName = Literal["voyage", "jina"]
InputType = Literal["document", "query"]

# ── Typed output ──────────────────────────────────────────────────────────────


@dataclass
class EmbeddedChunk:
    """A Chunk enriched with its embedding vector."""

    chunk_id: str
    url: str
    title: str
    page_type: str
    content_hash: str
    text: str
    chunk_index: int
    source: str
    is_faq: bool
    vector: list[float]
    provider: str
    model: str
    dims: int

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        vector: list[float],
        provider: str,
        model: str,
        dims: int,
    ) -> EmbeddedChunk:
        return cls(
            chunk_id=chunk.chunk_id,
            url=chunk.url,
            title=chunk.title,
            page_type=chunk.page_type,
            content_hash=chunk.content_hash,
            text=chunk.text,
            chunk_index=chunk.chunk_index,
            source=chunk.source,
            is_faq=chunk.is_faq,
            vector=vector,
            provider=provider,
            model=model,
            dims=dims,
        )

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "url": self.url,
            "title": self.title,
            "page_type": self.page_type,
            "content_hash": self.content_hash,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "source": self.source,
            "is_faq": self.is_faq,
            "vector": self.vector,
        }


# ── Provider base ─────────────────────────────────────────────────────────────


@dataclass
class EmbedProvider:
    """Provider configuration + embed method."""

    name: str
    model: str
    dims: int
    batch_size: int
    _api_key: str = field(default="", repr=False)

    def embed_batch(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        raise NotImplementedError


# ── Voyage provider ───────────────────────────────────────────────────────────


@dataclass
class VoyageProvider(EmbedProvider):
    def embed_batch(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        import voyageai

        client = voyageai.Client(api_key=self._api_key or None)
        voyage_input = (
            VOYAGE_INPUT_TYPE_DOC if input_type == "document" else VOYAGE_INPUT_TYPE_QUERY
        )
        result = client.embed(
            texts,
            model=self.model,
            input_type=voyage_input,
            output_dimension=self.dims,
            output_dtype=VOYAGE_EMBED_DTYPE,
        )
        return cast("list[list[float]]", result.embeddings)


# ── Jina provider ─────────────────────────────────────────────────────────────


@dataclass
class JinaProvider(EmbedProvider):
    def embed_batch(self, texts: list[str], input_type: InputType) -> list[list[float]]:
        task = "retrieval.passage" if input_type == "document" else "retrieval.query"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
            "task": task,
            "dimensions": self.dims,
            "normalized": True,
        }
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(JINA_EMBED_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return [item["embedding"] for item in data["data"]]


# ── Factory ───────────────────────────────────────────────────────────────────


def get_provider(name: ProviderName, api_key: str | None = None) -> EmbedProvider:
    """Return a configured provider instance, resolving the API key from env if not given."""
    if name == "voyage":
        key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        return VoyageProvider(
            name="voyage",
            model=VOYAGE_EMBED_MODEL,
            dims=VOYAGE_EMBED_DIMS,
            batch_size=VOYAGE_EMBED_BATCH_SIZE,
            _api_key=key,
        )
    if name == "jina":
        key = api_key or os.environ.get("JINA_API_KEY", "")
        return JinaProvider(
            name="jina",
            model=JINA_EMBED_MODEL,
            dims=JINA_EMBED_DIMS,
            batch_size=JINA_EMBED_BATCH_SIZE,
            _api_key=key,
        )
    raise ValueError(f"Unknown provider: {name!r}. Choose 'voyage' or 'jina'.")


# ── Core batching logic ───────────────────────────────────────────────────────

_RETRY_DELAYS = [2.0, 8.0, 32.0]  # exponential backoff for transient errors


def embed_texts(
    texts: list[str],
    input_type: InputType = "document",
    provider: EmbedProvider | None = None,
    api_key: str | None = None,
) -> list[list[float]]:
    """Embed a list of texts, batching and retrying transient errors.

    Returns one float list per text in the same order as *texts*.
    """
    if not texts:
        return []

    p = provider or get_provider("voyage", api_key=api_key)
    all_vectors: list[list[float]] = []

    for batch_start in range(0, len(texts), p.batch_size):
        batch = texts[batch_start : batch_start + p.batch_size]
        batch_vectors = _embed_with_retry(p, batch, input_type)
        all_vectors.extend(batch_vectors)
        logger.debug(
            "Embedded batch %d–%d via %s (%d dims)",
            batch_start,
            batch_start + len(batch),
            p.name,
            p.dims,
        )

    if len(all_vectors) != len(texts):
        raise RuntimeError(f"Vector count mismatch: expected {len(texts)}, got {len(all_vectors)}")
    return all_vectors


def _embed_with_retry(
    provider: EmbedProvider,
    texts: list[str],
    input_type: InputType,
) -> list[list[float]]:
    last_exc: Exception | None = None
    for attempt, delay in enumerate([0.0, *_RETRY_DELAYS]):
        if delay:
            logger.warning(
                "Retrying embed (attempt %d) after %.0fs for %s",
                attempt,
                delay,
                provider.name,
            )
            time.sleep(delay)
        try:
            return provider.embed_batch(texts, input_type)
        except httpx.HTTPStatusError as exc:
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                raise
            last_exc = exc
            logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("Embed attempt %d failed: %s", attempt + 1, exc)
    raise RuntimeError(f"Embedding failed after retries: {last_exc}") from last_exc


def embed_chunks(
    chunks: list[Chunk],
    provider: EmbedProvider | None = None,
    api_key: str | None = None,
) -> list[EmbeddedChunk]:
    """Embed a list of Chunks, returning EmbeddedChunk objects in the same order."""
    p = provider or get_provider("voyage", api_key=api_key)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts, input_type="document", provider=p)
    return [
        EmbeddedChunk.from_chunk(chunk, vector, provider=p.name, model=p.model, dims=p.dims)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
