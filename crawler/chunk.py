"""Markdown → text chunks with metadata, ready for embedding.

Design:
- chunk_document()           — split one NormalizedDoc into Chunk list:
                                 1. chunk FAQ pairs first (each pair = 1 chunk)
                                 2. split remaining markdown by headings
                                 3. size-constrained sliding-window over each section
- chunk_faq_pairs()          — convert FaqPair list → Chunk list (1 pair / chunk)
- _split_by_headings()       — break markdown at ## / # boundaries
- _split_section()           — sliding window: target 400–600 tokens, 65-token overlap
- fetch_and_chunk_llms_txt() — fetch llms.txt / llms-full.txt and emit overview chunks
- _make_chunk_id()           — stable sha256-based ID so chunks can be upserted idempotently

Token counting uses a character approximation (CHARS_PER_TOKEN = 4) so no tokenizer
dependency is needed. This is "good enough" for 400–600-token target windows — a ±20%
error in the count only shifts a chunk boundary, it never loses content.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import httpx

from crawler.config import (
    CHARS_PER_TOKEN,
    CHUNK_MAX_TOKENS,
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    LLMS_FULL_URL,
    LLMS_TXT_URL,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
)
from crawler.extract import FaqPair
from crawler.normalize import NormalizedDoc

# Markdown heading line — captures prefix (##) and rest of line
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    """A single embeddable text unit with full provenance metadata."""

    chunk_id: str
    url: str
    title: str
    page_type: str
    content_hash: str
    text: str
    chunk_index: int
    source: str = "sitemap"
    is_faq: bool = False

    @property
    def token_count(self) -> int:
        return _approx_tokens(self.text)

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
            "token_count": self.token_count,
        }


# ── Token / character helpers ─────────────────────────────────────────────────


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _max_chars() -> int:
    return CHUNK_MAX_TOKENS * CHARS_PER_TOKEN


def _min_chars() -> int:
    return CHUNK_MIN_TOKENS * CHARS_PER_TOKEN


def _overlap_chars() -> int:
    return CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN


# ── Chunk ID ─────────────────────────────────────────────────────────────────


def _make_chunk_id(url: str, index: int, text: str) -> str:
    """Stable 16-char hex ID: sha256(url + index + first 64 chars of text)."""
    key = f"{url}::{index}::{text[:64]}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ── Heading-based sectioning ──────────────────────────────────────────────────


def _split_by_headings(markdown: str) -> list[str]:
    """Break markdown into sections at heading boundaries (# and ##).

    Each section starts with the heading line. Returns non-empty sections only.
    """
    sections: list[str] = []
    current_lines: list[str] = []

    for line in markdown.splitlines():
        if _HEADING_RE.match(line) and current_lines:
            section = "\n".join(current_lines).strip()
            if section:
                sections.append(section)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        section = "\n".join(current_lines).strip()
        if section:
            sections.append(section)

    return sections or [markdown.strip()] if markdown.strip() else []


# ── Sliding-window split within a section ────────────────────────────────────


def _extract_overlap(text: str, overlap_chars: int) -> str:
    """Take the last *overlap_chars* characters but trim to a sentence boundary.

    Falls back to the raw tail if no sentence boundary is found within the last
    30% of the overlap window.
    """
    if len(text) <= overlap_chars:
        return text

    tail = text[-overlap_chars:]
    # Try to start overlap at a sentence end + space boundary
    boundary = re.search(r"(?<=[.!?])\s+(?=\S)", tail)
    if boundary:
        return tail[boundary.end() :]
    return tail


def _split_section(section: str) -> list[str]:
    """Split one section into target-size chunks with overlap.

    Splits preferring paragraph boundaries (blank-line separated), falling back
    to raw character splits if a single paragraph exceeds max size.
    """
    max_c = _max_chars()
    min_c = _min_chars()
    overlap_c = _overlap_chars()

    # Fast path: section fits in one chunk
    if len(section) <= max_c:
        return [section]

    # Split into paragraphs at blank lines
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", section) if p.strip()]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    def flush(include_overlap: bool) -> str:
        text = "\n\n".join(current_parts)
        if include_overlap:
            overlap = _extract_overlap(text, overlap_c)
            current_parts.clear()
            if overlap:
                current_parts.append(overlap)
        else:
            current_parts.clear()
        return text

    for para in paragraphs:
        # A single paragraph is too big → hard-split it
        if len(para) > max_c:
            if current_parts:
                chunks.append(flush(include_overlap=True))
                current_len = sum(len(p) for p in current_parts)

            pos = 0
            while pos < len(para):
                piece = para[pos : pos + max_c]
                if pos + max_c < len(para):
                    chunks.append(piece)
                    overlap = _extract_overlap(piece, overlap_c)
                    pos += max_c - len(overlap)
                else:
                    current_parts.append(piece)
                    current_len = len(piece)
                    pos = len(para)
            continue

        candidate_len = current_len + (2 if current_parts else 0) + len(para)
        if candidate_len > max_c and current_len >= min_c:
            chunks.append(flush(include_overlap=True))
            current_len = sum(len(p) for p in current_parts)

        current_parts.append(para)
        current_len = sum(len(p) for p in current_parts)

    if current_parts:
        remaining = "\n\n".join(current_parts).strip()
        if remaining:
            # Merge tiny tail into last chunk rather than emitting a stub
            if chunks and len(remaining) < min_c:
                last = chunks[-1]
                if len(last) + len(remaining) + 2 <= max_c * 2:
                    chunks[-1] = last + "\n\n" + remaining
                else:
                    chunks.append(remaining)
            else:
                chunks.append(remaining)

    return chunks or [section]


# ── FAQ pair chunking ─────────────────────────────────────────────────────────


def chunk_faq_pairs(
    faq_pairs: list[FaqPair],
    url: str,
    title: str,
    page_type: str,
    content_hash: str,
    source: str,
    start_index: int = 0,
) -> list[Chunk]:
    """Each FAQ Q+A pair becomes exactly one chunk (preserving answer integrity)."""
    chunks: list[Chunk] = []
    for i, pair in enumerate(faq_pairs):
        text = pair.to_text()
        idx = start_index + i
        chunks.append(
            Chunk(
                chunk_id=_make_chunk_id(url, idx, text),
                url=url,
                title=title,
                page_type=page_type,
                content_hash=content_hash,
                text=text,
                chunk_index=idx,
                source=source,
                is_faq=True,
            )
        )
    return chunks


# ── Document-level chunking ───────────────────────────────────────────────────


def chunk_document(doc: NormalizedDoc) -> list[Chunk]:
    """Split a NormalizedDoc into embeddable Chunks.

    Order:
      1. FAQ pairs (each → 1 chunk, preserving Q+A integrity)
      2. Remaining markdown split by headings then size-windowed

    FAQ pair chunks come first so they get low chunk_index values and are
    easy to identify (is_faq=True) for retrieval boosting in Step 5.
    """
    if doc.is_empty and not doc.faq_pairs:
        return []

    chunks: list[Chunk] = []
    meta = {
        "url": doc.url,
        "title": doc.title,
        "page_type": doc.page_type,
        "content_hash": doc.content_hash,
        "source": doc.source,
    }

    # 1. FAQ pairs
    faq_chunks = chunk_faq_pairs(
        doc.faq_pairs,
        start_index=0,
        **meta,
    )
    chunks.extend(faq_chunks)

    # 2. Main markdown
    if doc.markdown.strip():
        sections = _split_by_headings(doc.markdown)
        idx = len(chunks)
        for section in sections:
            for piece in _split_section(section):
                if not piece.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=_make_chunk_id(doc.url, idx, piece),
                        url=doc.url,
                        title=doc.title,
                        page_type=doc.page_type,
                        content_hash=doc.content_hash,
                        text=piece,
                        chunk_index=idx,
                        source=doc.source,
                        is_faq=False,
                    )
                )
                idx += 1

    return chunks


# ── llms.txt / llms-full.txt ingestion ───────────────────────────────────────


def fetch_and_chunk_llms_txt(
    client: httpx.Client | None = None,
    prefer_full: bool = True,
) -> list[Chunk]:
    """Fetch llms.txt (or llms-full.txt) and emit overview chunks.

    These chunks are tagged `page_type="overview"` and `source="overview"`.
    The dedup/down-weight step in normalize_documents() handles any overlap with
    specific crawled pages — but we still chunk here so the overview perspective
    is always represented in the index.
    """
    urls_to_try: list[tuple[str, str]] = []
    if prefer_full:
        urls_to_try = [(LLMS_FULL_URL, "llms-full"), (LLMS_TXT_URL, "llms")]
    else:
        urls_to_try = [(LLMS_TXT_URL, "llms"), (LLMS_FULL_URL, "llms-full")]

    close_client = False
    if client is None:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        close_client = True

    text: str = ""
    source_url: str = ""
    try:
        for url, _label in urls_to_try:
            try:
                resp = client.get(url)
                if resp.status_code == 200 and resp.text.strip():
                    text = resp.text
                    source_url = url
                    break
            except httpx.HTTPError:
                continue
    finally:
        if close_client:
            client.close()

    if not text.strip():
        return []

    # Chunk the overview text just like regular markdown
    hash_val = hashlib.sha256(text.encode("utf-8")).hexdigest()
    base_doc = NormalizedDoc(
        url=source_url,
        original_url=source_url,
        title="Appther — Overview",
        markdown=text,
        page_type="overview",
        content_hash=hash_val,
        source="overview",
    )

    # Use the same heading-based chunker but tag each chunk
    chunks = chunk_document(base_doc)
    for chunk in chunks:
        chunk.source = "overview"
        chunk.page_type = "overview"

    return chunks
