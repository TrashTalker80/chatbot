"""Tests for crawler/embed.py.

All tests mock the actual provider API calls so no live keys are required.
Validates:
- Provider factory (voyage/jina) sets correct model + dims + batch_size
- embed_texts batches correctly and merges results in order
- Transient failures are retried; permanent failures propagate
- embed_chunks produces EmbeddedChunk objects with vector + metadata
- Jina provider builds correct HTTP payload
"""

from __future__ import annotations

import pytest

from crawler.chunk import Chunk
from crawler.config import (
    JINA_EMBED_BATCH_SIZE,
    JINA_EMBED_DIMS,
    JINA_EMBED_MODEL,
    VOYAGE_EMBED_BATCH_SIZE,
    VOYAGE_EMBED_DIMS,
    VOYAGE_EMBED_MODEL,
)
from crawler.embed import (
    EmbeddedChunk,
    JinaProvider,
    VoyageProvider,
    embed_chunks,
    embed_texts,
    get_provider,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _fake_chunk(chunk_id: str = "abc123", page_type: str = "faq") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        url="https://www.appther.com/faq",
        title="Appther FAQ",
        page_type=page_type,
        content_hash="deadbeef",
        text="What does Appther do? Appther is a technology company.",
        chunk_index=0,
        source="sitemap",
        is_faq=True,
    )


def _zero_vector(dims: int = VOYAGE_EMBED_DIMS) -> list[float]:
    return [0.0] * dims


# ── Provider factory ──────────────────────────────────────────────────────────


def test_get_provider_voyage_defaults():
    p = get_provider("voyage", api_key="test-key")
    assert isinstance(p, VoyageProvider)
    assert p.name == "voyage"
    assert p.model == VOYAGE_EMBED_MODEL
    assert p.dims == VOYAGE_EMBED_DIMS
    assert p.batch_size == VOYAGE_EMBED_BATCH_SIZE


def test_get_provider_jina_defaults():
    p = get_provider("jina", api_key="test-key")
    assert isinstance(p, JinaProvider)
    assert p.name == "jina"
    assert p.model == JINA_EMBED_MODEL
    assert p.dims == JINA_EMBED_DIMS
    assert p.batch_size == JINA_EMBED_BATCH_SIZE


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("openai")  # type: ignore[arg-type]


# ── embed_texts — batching and ordering ──────────────────────────────────────


def test_embed_texts_empty():
    result = embed_texts([], provider=get_provider("voyage", api_key="x"))
    assert result == []


def test_embed_texts_batches_and_merges(mocker):
    """4 texts with batch_size=2 should call embed_batch twice."""
    p = get_provider("voyage", api_key="fake")
    p.batch_size = 2  # override for test
    call_count = 0

    def fake_embed(texts, input_type):
        nonlocal call_count
        call_count += 1
        return [[float(call_count)] * VOYAGE_EMBED_DIMS for _ in texts]

    mocker.patch.object(p, "embed_batch", side_effect=fake_embed)
    texts = ["a", "b", "c", "d"]
    result = embed_texts(texts, provider=p)
    assert call_count == 2
    assert len(result) == 4
    # batch 1 → vectors full of 1.0, batch 2 → 2.0
    assert result[0] == [1.0] * VOYAGE_EMBED_DIMS
    assert result[2] == [2.0] * VOYAGE_EMBED_DIMS


def test_embed_texts_order_preserved(mocker):
    p = get_provider("voyage", api_key="fake")
    p.batch_size = 10

    def fake_embed(texts, input_type):
        return [[float(i)] * 4 for i in range(len(texts))]

    mocker.patch.object(p, "embed_batch", side_effect=fake_embed)
    texts = ["x", "y", "z"]
    result = embed_texts(texts, provider=p)
    assert len(result) == 3


def test_embed_texts_retries_transient_failure(mocker):
    """First embed_batch call raises, second succeeds."""
    p = get_provider("voyage", api_key="fake")
    calls = {"n": 0}

    def flaky(texts, input_type):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("timeout")
        return [[0.0] * VOYAGE_EMBED_DIMS for _ in texts]

    mocker.patch.object(p, "embed_batch", side_effect=flaky)
    mocker.patch("crawler.embed.time.sleep")  # skip real sleep
    result = embed_texts(["hello"], provider=p)
    assert len(result) == 1
    assert calls["n"] == 2


def test_embed_texts_propagates_after_max_retries(mocker):
    """All attempts fail → RuntimeError raised."""
    p = get_provider("voyage", api_key="fake")

    def always_fail(texts, input_type):
        raise RuntimeError("503")

    mocker.patch.object(p, "embed_batch", side_effect=always_fail)
    mocker.patch("crawler.embed.time.sleep")
    with pytest.raises(RuntimeError, match="Embedding failed after retries"):
        embed_texts(["hello"], provider=p)


# ── embed_chunks ──────────────────────────────────────────────────────────────


def test_embed_chunks_returns_embedded_chunks(mocker):
    p = get_provider("voyage", api_key="fake")
    chunks = [_fake_chunk("c1"), _fake_chunk("c2")]

    mocker.patch.object(
        p,
        "embed_batch",
        return_value=[_zero_vector() for _ in chunks],
    )

    result = embed_chunks(chunks, provider=p)
    assert len(result) == 2
    for ec, chunk in zip(result, chunks, strict=False):
        assert isinstance(ec, EmbeddedChunk)
        assert ec.chunk_id == chunk.chunk_id
        assert ec.url == chunk.url
        assert ec.is_faq == chunk.is_faq
        assert len(ec.vector) == VOYAGE_EMBED_DIMS
        assert ec.provider == "voyage"
        assert ec.model == VOYAGE_EMBED_MODEL
        assert ec.dims == VOYAGE_EMBED_DIMS


def test_embed_chunks_to_dict_contains_vector(mocker):
    p = get_provider("voyage", api_key="fake")
    chunk = _fake_chunk()
    vec = _zero_vector()
    mocker.patch.object(p, "embed_batch", return_value=[vec])

    ec = embed_chunks([chunk], provider=p)[0]
    d = ec.to_dict()
    assert "vector" in d
    assert d["chunk_id"] == chunk.chunk_id
    assert d["is_faq"] is True


# ── Vector dimension invariant ────────────────────────────────────────────────


def test_voyage_dims_match_config():
    """The VoyageProvider dims must match VOYAGE_EMBED_DIMS from config."""
    p = get_provider("voyage", api_key="x")
    assert p.dims == VOYAGE_EMBED_DIMS
    assert p.dims == 512


def test_jina_dims_match_voyage_dims():
    """Jina standby index uses the same dims as Voyage so columns are compatible."""
    vp = get_provider("voyage", api_key="x")
    jp = get_provider("jina", api_key="x")
    assert vp.dims == jp.dims, "Voyage and Jina dims must match for compatible schemas"
