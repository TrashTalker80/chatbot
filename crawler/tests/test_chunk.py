"""Tests for crawler/chunk.py — Markdown → embeddable Chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

from crawler.chunk import (
    Chunk,
    _approx_tokens,
    _make_chunk_id,
    _split_by_headings,
    _split_section,
    chunk_document,
    chunk_faq_pairs,
)
from crawler.config import CHARS_PER_TOKEN, CHUNK_MAX_TOKENS
from crawler.extract import FaqPair, extract
from crawler.normalize import NormalizedDoc, compute_content_hash

FIXTURES = Path(__file__).parent / "fixtures"


def _html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _make_doc(
    url: str = "https://www.appther.com/test",
    markdown: str = "# Title\n\nSome paragraph text.",
    page_type: str = "other",
    source: str = "sitemap",
    faq_pairs: list[FaqPair] | None = None,
) -> NormalizedDoc:
    return NormalizedDoc(
        url=url,
        original_url=url,
        title="Test Page",
        markdown=markdown,
        page_type=page_type,
        content_hash=compute_content_hash(markdown),
        source=source,
        faq_pairs=faq_pairs or [],
    )


def _long_paragraph(words: int = 200) -> str:
    """Generate a paragraph of approximately *words* words."""
    sentence = "Appther delivers Odoo ERP implementations for growing businesses. "
    reps = max(1, words // len(sentence.split()))
    return sentence * reps


# ── TestApproxTokens ──────────────────────────────────────────────────────────


class TestApproxTokens:
    def test_empty_returns_one(self):
        assert _approx_tokens("") == 1

    def test_proportional_to_length(self):
        text = "a" * (CHARS_PER_TOKEN * 100)
        assert _approx_tokens(text) == 100

    def test_returns_int(self):
        assert isinstance(_approx_tokens("hello world"), int)


# ── TestSplitByHeadings ───────────────────────────────────────────────────────


class TestSplitByHeadings:
    def test_single_section_no_heading(self):
        md = "Just a paragraph with no heading."
        sections = _split_by_headings(md)
        assert len(sections) == 1
        assert sections[0] == md

    def test_h1_creates_new_section(self):
        md = "# Title\n\nParagraph one.\n\n# Second\n\nParagraph two."
        sections = _split_by_headings(md)
        assert len(sections) == 2
        assert sections[0].startswith("# Title")
        assert sections[1].startswith("# Second")

    def test_h2_creates_new_section(self):
        md = "# Top\n\nIntro text.\n\n## Sub One\n\nContent.\n\n## Sub Two\n\nMore."
        sections = _split_by_headings(md)
        assert len(sections) == 3

    def test_empty_string_returns_empty(self):
        assert _split_by_headings("") == []

    def test_no_empty_sections(self):
        md = "# A\n\n# B\n\nContent"
        sections = _split_by_headings(md)
        for s in sections:
            assert s.strip()

    def test_section_includes_heading_text(self):
        md = "## Implementation\n\nWe follow a proven methodology."
        sections = _split_by_headings(md)
        assert "## Implementation" in sections[0]
        assert "proven methodology" in sections[0]

    def test_article_fixture_splits_by_h2(self):
        art_html = _html("article_page.html")
        ext = extract(art_html)
        sections = _split_by_headings(ext.markdown)
        # article_page.html has h1 + 3× h2 = 4 sections
        assert len(sections) >= 2


# ── TestSplitSection ─────────────────────────────────────────────────────────


class TestSplitSection:
    def test_short_section_returns_one_chunk(self):
        text = "Short paragraph."
        chunks = _split_section(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_section_split_into_multiple(self):
        long_text = _long_paragraph(500)
        chunks = _split_section(long_text)
        assert len(chunks) > 1

    def test_chunks_within_max_size(self):
        long_text = _long_paragraph(800)
        max_chars = CHUNK_MAX_TOKENS * CHARS_PER_TOKEN * 2  # allow 2x for merged tail
        for chunk in _split_section(long_text):
            assert len(chunk) <= max_chars, f"chunk too long: {len(chunk)} chars"

    def test_no_content_lost(self):
        long_text = _long_paragraph(500)
        chunks = _split_section(long_text)
        # All non-overlap text is present across chunks (overlap is a subset)
        joined = " ".join(chunks)
        # Every sentence from the original should appear in joined
        for sentence in long_text.split(". ")[:5]:
            assert sentence.strip()[:20] in joined

    def test_each_chunk_non_empty(self):
        text = "paragraph one.\n\nparagraph two.\n\nparagraph three."
        for chunk in _split_section(text):
            assert chunk.strip()

    def test_returns_list(self):
        result = _split_section("some text")
        assert isinstance(result, list)


# ── TestChunkFaqPairs ─────────────────────────────────────────────────────────


class TestChunkFaqPairs:
    def _pairs(self) -> list[FaqPair]:
        return [
            FaqPair(
                question="How long does implementation take?",
                answer="Eight to sixteen weeks depending on modules.",
            ),
            FaqPair(
                question="What does it cost?",
                answer="Fifteen to eighty thousand dollars.",
            ),
        ]

    def test_one_chunk_per_pair(self):
        chunks = chunk_faq_pairs(
            self._pairs(),
            url="https://www.appther.com/faq",
            title="FAQ",
            page_type="faq",
            content_hash="abc",
            source="sitemap",
        )
        assert len(chunks) == 2

    def test_is_faq_true(self):
        chunks = chunk_faq_pairs(
            self._pairs(),
            url="https://www.appther.com/faq",
            title="FAQ",
            page_type="faq",
            content_hash="abc",
            source="sitemap",
        )
        for c in chunks:
            assert c.is_faq is True

    def test_text_format(self):
        chunks = chunk_faq_pairs(
            self._pairs(),
            url="https://www.appther.com/faq",
            title="FAQ",
            page_type="faq",
            content_hash="abc",
            source="sitemap",
        )
        for c in chunks:
            assert c.text.startswith("Q: ")
            assert "\nA: " in c.text

    def test_chunk_index_sequential(self):
        chunks = chunk_faq_pairs(
            self._pairs(),
            url="https://www.appther.com/faq",
            title="FAQ",
            page_type="faq",
            content_hash="abc",
            source="sitemap",
            start_index=5,
        )
        assert chunks[0].chunk_index == 5
        assert chunks[1].chunk_index == 6

    def test_metadata_on_every_chunk(self):
        chunks = chunk_faq_pairs(
            self._pairs(),
            url="https://www.appther.com/faq",
            title="FAQ",
            page_type="faq",
            content_hash="testhash",
            source="sitemap",
        )
        for c in chunks:
            assert c.url == "https://www.appther.com/faq"
            assert c.title == "FAQ"
            assert c.page_type == "faq"
            assert c.content_hash == "testhash"
            assert c.chunk_id

    def test_empty_pairs_returns_empty(self):
        chunks = chunk_faq_pairs(
            [],
            url="https://www.appther.com/faq",
            title="FAQ",
            page_type="faq",
            content_hash="abc",
            source="sitemap",
        )
        assert chunks == []


# ── TestChunkDocument ─────────────────────────────────────────────────────────


class TestChunkDocument:
    def test_returns_list_of_chunks(self):
        doc = _make_doc(markdown="# Title\n\nSome text.")
        result = chunk_document(doc)
        assert isinstance(result, list)
        for c in result:
            assert isinstance(c, Chunk)

    def test_empty_doc_returns_empty(self):
        doc = _make_doc(markdown="")
        result = chunk_document(doc)
        assert result == []

    def test_metadata_present_on_all_chunks(self):
        art_html = _html("article_page.html")
        ext = extract(art_html, "https://www.appther.com/services/odoo")
        doc = NormalizedDoc(
            url="https://www.appther.com/services/odoo",
            original_url="https://www.appther.com/services/odoo",
            title=ext.title,
            markdown=ext.markdown,
            page_type="service",
            content_hash=compute_content_hash(ext.markdown),
            source="sitemap",
        )
        for chunk in chunk_document(doc):
            assert chunk.url, "url missing"
            assert chunk.title, "title missing"
            assert chunk.page_type, "page_type missing"
            assert chunk.content_hash, "content_hash missing"
            assert chunk.chunk_id, "chunk_id missing"

    def test_faq_pairs_become_first_chunks(self):
        faq_html = _html("faq_page.html")
        ext = extract(faq_html, "https://www.appther.com/faq")
        doc = NormalizedDoc(
            url="https://www.appther.com/faq",
            original_url="https://www.appther.com/faq",
            title=ext.title,
            markdown=ext.markdown,
            page_type="faq",
            content_hash=compute_content_hash(ext.markdown),
            source="sitemap",
            faq_pairs=ext.faq_pairs,
        )
        chunks = chunk_document(doc)
        faq_chunks = [c for c in chunks if c.is_faq]
        assert len(faq_chunks) == 4  # fixture has 4 FAQ pairs

        # FAQ chunks come before markdown chunks (lower chunk_index)
        non_faq = [c for c in chunks if not c.is_faq]
        if faq_chunks and non_faq:
            assert max(c.chunk_index for c in faq_chunks) < min(c.chunk_index for c in non_faq)

    def test_faq_answer_intact_in_chunk(self):
        faq_html = _html("faq_page.html")
        ext = extract(faq_html, "https://www.appther.com/faq")
        doc = NormalizedDoc(
            url="https://www.appther.com/faq",
            original_url="https://www.appther.com/faq",
            title=ext.title,
            markdown=ext.markdown,
            page_type="faq",
            content_hash=compute_content_hash(ext.markdown),
            source="sitemap",
            faq_pairs=ext.faq_pairs,
        )
        chunks = chunk_document(doc)
        impl_chunk = next(c for c in chunks if c.is_faq and "implementation take" in c.text)
        # Full answer must be present — not truncated
        assert "eight and sixteen weeks" in impl_chunk.text
        assert "migrated" in impl_chunk.text

    def test_chunk_ids_unique(self):
        art_html = _html("article_page.html")
        ext = extract(art_html)
        doc = NormalizedDoc(
            url="https://www.appther.com/services/odoo",
            original_url="https://www.appther.com/services/odoo",
            title=ext.title,
            markdown=ext.markdown,
            page_type="service",
            content_hash=compute_content_hash(ext.markdown),
            source="sitemap",
        )
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_index_sequential_from_zero(self):
        doc = _make_doc(markdown="# A\n\nContent A.\n\n# B\n\nContent B.")
        chunks = chunk_document(doc)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_to_dict_contains_required_keys(self):
        doc = _make_doc(markdown="# Title\n\nText.")
        chunks = chunk_document(doc)
        required_keys = {
            "chunk_id",
            "url",
            "title",
            "page_type",
            "content_hash",
            "text",
            "chunk_index",
            "source",
            "is_faq",
            "token_count",
        }
        for c in chunks:
            d = c.to_dict()
            assert required_keys.issubset(d.keys()), f"missing keys: {required_keys - d.keys()}"

    def test_large_doc_chunked_within_size(self):
        # Build a document large enough to require splitting (>600 tokens)
        long_md = "# Overview\n\n"
        long_md += _long_paragraph(800)
        doc = _make_doc(markdown=long_md)
        chunks = chunk_document(doc)
        max_allowed = CHUNK_MAX_TOKENS * CHARS_PER_TOKEN * 2  # allow 2x for merged tails
        for c in chunks:
            assert len(c.text) <= max_allowed, (
                f"Chunk too long: {len(c.text)} chars (max {max_allowed})"
            )

    def test_doc_only_faq_no_markdown(self):
        pairs = [FaqPair("Q?", "Answer.")]
        doc = NormalizedDoc(
            url="https://www.appther.com/faq",
            original_url="https://www.appther.com/faq",
            title="FAQ",
            markdown="",
            page_type="faq",
            content_hash="x",
            source="sitemap",
            faq_pairs=pairs,
        )
        chunks = chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].is_faq


# ── TestChunkOverlap ──────────────────────────────────────────────────────────


class TestChunkOverlap:
    def test_overlap_content_in_adjacent_chunks(self):
        """Content near a chunk boundary should appear in both adjacent chunks."""
        long_text = _long_paragraph(700)
        all_chunks = _split_section(long_text)

        if len(all_chunks) < 2:
            pytest.skip("Text too short to produce multiple chunks; increase length")

        # The end of chunk 0 and the start of chunk 1 should share some words
        end_words = set(all_chunks[0].split()[-20:])
        start_words = set(all_chunks[1].split()[:20])
        overlap_count = len(end_words & start_words)
        assert overlap_count > 0, "Expected overlap between adjacent chunks"

    def test_chunk_id_stable(self):
        text = "Fixed content for stability test."
        id1 = _make_chunk_id("https://www.appther.com/test", 0, text)
        id2 = _make_chunk_id("https://www.appther.com/test", 0, text)
        assert id1 == id2

    def test_chunk_id_differs_by_index(self):
        text = "Same text content."
        id0 = _make_chunk_id("https://www.appther.com/test", 0, text)
        id1 = _make_chunk_id("https://www.appther.com/test", 1, text)
        assert id0 != id1

    def test_chunk_id_differs_by_url(self):
        text = "Same text content."
        id_a = _make_chunk_id("https://www.appther.com/a", 0, text)
        id_b = _make_chunk_id("https://www.appther.com/b", 0, text)
        assert id_a != id_b
