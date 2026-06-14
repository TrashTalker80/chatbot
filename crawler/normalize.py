"""URL canonicalization, content hashing, and near-duplicate collapse.

Design:
- canonicalize_url()        — lowercase scheme/host/path + strip tracking params
                               (and optionally follow redirects via an httpx.Client).
                               This produces the stable document ID used for both
                               citations and dedup keys.
- compute_content_hash()     — SHA-256 of whitespace-normalized markdown; used for
                               incremental re-embed (Step 3) and dedup.
- build_normalized_doc()     — combine a FetchResult + ExtractResult + sitemap
                               metadata into a NormalizedDoc.
- dedupe_by_url()             — collapse exact-URL duplicates (e.g. `/industries/`
                               redirecting to `/industry/`, which is itself also in
                               the sitemap), keeping the richer-metadata entry.
- find_near_duplicate_groups() / collapse_near_duplicates()
                              — MinHash/LSH near-duplicate collapse for templated
                               pages (~25 `hire-*`, location pages, near-identical
                               `services/*`).
- normalize_documents()      — end-to-end orchestrator over the above.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse, urlunparse

import httpx
from datasketch import MinHash, MinHashLSH

from crawler.config import MINHASH_NUM_PERM, NEAR_DUP_THRESHOLD
from crawler.extract import ExtractResult, FaqPair
from crawler.models import FetchResult
from crawler.urls import infer_page_type, normalize_url

logger = logging.getLogger(__name__)

# Source priority for dedup/dup-collapse: lower wins. Real crawled pages outrank
# the homepage-BFS fallback, which outranks llms.txt overview chunks (the
# blueprint's "dedupe/down-weight overview chunks" rule).
_SOURCE_PRIORITY: dict[str, int] = {
    "sitemap": 0,
    "blog-sitemap": 0,
    "bfs": 1,
    "overview": 2,
}

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class NormalizedDoc:
    """A single page after URL canonicalization, extraction, and content hashing."""

    url: str
    original_url: str
    title: str
    markdown: str
    page_type: str
    content_hash: str
    source: str = "sitemap"
    lastmod: str | None = None
    priority: float | None = None
    faq_pairs: list[FaqPair] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.markdown.strip()

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "original_url": self.original_url,
            "title": self.title,
            "page_type": self.page_type,
            "content_hash": self.content_hash,
            "source": self.source,
            "lastmod": self.lastmod,
            "priority": self.priority,
            "faq_count": len(self.faq_pairs),
            "word_count": len(self.markdown.split()),
        }


# ── URL canonicalization ─────────────────────────────────────────────────────


def canonicalize_url(url: str, client: httpx.Client | None = None) -> str:
    """Return the canonical form of *url*: lowercase scheme/host/path, tracking

    params stripped, fragment dropped. When *client* is given, follows redirects
    first and canonicalizes the final URL — this turns `/industries/` into
    `/industry/` (per the sitemap audit) and collapses mixed-case path duplicates
    such as `…company-DelhiNCR` vs `…company-delhincr`.
    """
    if client is not None:
        try:
            response = client.get(normalize_url(url), follow_redirects=True)
            url = str(response.url)
        except httpx.HTTPError as exc:
            logger.debug("canonicalize_url: request failed for %s: %s", url, exc)

    normalized = normalize_url(url)
    return _lowercase_path(normalized)


def _lowercase_path(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path.lower(), parsed.params, parsed.query, "")
    )


# ── Content hashing ───────────────────────────────────────────────────────────


def compute_content_hash(text: str) -> str:
    """SHA-256 of whitespace-normalized, lowercased text.

    Used for Step 3's incremental re-embed: only re-embed chunks whose
    content_hash changed since the last run.
    """
    normalized = _WHITESPACE_RE.sub(" ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ── Building NormalizedDoc ────────────────────────────────────────────────────


def build_normalized_doc(
    fetch: FetchResult,
    extract: ExtractResult,
    source: str = "sitemap",
    lastmod: str | None = None,
    priority: float | None = None,
    client: httpx.Client | None = None,
) -> NormalizedDoc:
    """Build a NormalizedDoc from a fetched page's FetchResult + ExtractResult.

    The canonical URL is derived from `fetch.final_url` (redirects already
    followed by fetch.py); `client` is only needed if you want canonicalize_url
    to re-verify redirects independently of the fetch step.
    """
    canonical = canonicalize_url(fetch.final_url)
    return NormalizedDoc(
        url=canonical,
        original_url=fetch.url,
        title=extract.title,
        markdown=extract.markdown,
        page_type=infer_page_type(canonical),
        content_hash=compute_content_hash(extract.markdown),
        source=source,
        lastmod=lastmod,
        priority=priority,
        faq_pairs=extract.faq_pairs,
    )


# ── Exact-URL dedup ───────────────────────────────────────────────────────────


def dedupe_by_url(docs: list[NormalizedDoc]) -> list[NormalizedDoc]:
    """Collapse exact-URL duplicates, keeping the richer/higher-priority entry.

    Two different sitemap entries can canonicalize to the same final URL
    (e.g. `/industries/` 301s to `/industry/`, which is itself also listed).
    """
    seen: dict[str, NormalizedDoc] = {}
    for doc in docs:
        existing = seen.get(doc.url)
        if existing is None:
            seen[doc.url] = doc
            continue
        if _source_rank(doc) < _source_rank(existing):
            seen[doc.url] = doc

    return list(seen.values())


def _source_rank(doc: NormalizedDoc) -> tuple[int, float]:
    return (_SOURCE_PRIORITY.get(doc.source, 99), -(doc.priority or 0.5))


# ── Near-duplicate detection (MinHash/LSH) ────────────────────────────────────


def minhash_of(text: str, num_perm: int = MINHASH_NUM_PERM) -> MinHash:
    """Build a MinHash signature from 3-word shingles of *text* (lowercased)."""
    m = MinHash(num_perm=num_perm)
    words = re.findall(r"\w+", text.lower())

    if len(words) < 3:
        for word in words:
            m.update(word.encode("utf-8"))
        return m

    for i in range(len(words) - 2):
        shingle = " ".join(words[i : i + 3])
        m.update(shingle.encode("utf-8"))
    return m


def find_near_duplicate_groups(
    docs: list[NormalizedDoc],
    threshold: float = NEAR_DUP_THRESHOLD,
) -> list[list[str]]:
    """Group URLs whose content has >= threshold Jaccard similarity (MinHash/LSH).

    Returns a list of groups (each a sorted list of >= 2 URLs); singletons are
    omitted. Groups are sorted largest-first.
    """
    non_empty = [d for d in docs if not d.is_empty]
    if len(non_empty) < 2:
        return []

    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
    hashes: dict[str, MinHash] = {}

    for doc in non_empty:
        m = minhash_of(doc.markdown)
        hashes[doc.url] = m
        lsh.insert(doc.url, m)

    adjacency: dict[str, set[str]] = {url: set() for url in hashes}
    for url, m in hashes.items():
        for neighbor in lsh.query(m):
            neighbor = str(neighbor)
            if neighbor != url:
                adjacency[url].add(neighbor)
                adjacency[neighbor].add(url)

    visited: set[str] = set()
    groups: list[list[str]] = []
    for url in adjacency:
        if url in visited:
            continue
        component = _connected_component(url, adjacency, visited)
        if len(component) > 1:
            groups.append(sorted(component))

    return sorted(groups, key=len, reverse=True)


def _connected_component(
    start: str, adjacency: dict[str, set[str]], visited: set[str]
) -> list[str]:
    component: list[str] = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        component.append(node)
        stack.extend(adjacency[node] - visited)
    return component


def collapse_near_duplicates(
    docs: list[NormalizedDoc],
    threshold: float = NEAR_DUP_THRESHOLD,
) -> list[NormalizedDoc]:
    """Collapse near-duplicate documents, keeping one representative per group.

    The representative is chosen by source priority (sitemap > bfs > overview),
    then by sitemap `priority` (higher wins), then by URL for stability.
    """
    groups = find_near_duplicate_groups(docs, threshold=threshold)
    if not groups:
        return docs

    by_url = {doc.url: doc for doc in docs}
    grouped_urls: set[str] = set()
    winners: list[NormalizedDoc] = []

    for group in groups:
        candidates = [by_url[url] for url in group]
        winner = min(candidates, key=lambda d: (*_source_rank(d), d.url))
        winners.append(winner)
        grouped_urls.update(group)
        logger.info(
            "Near-dup group (%d pages): keeping %s, dropping %s",
            len(group),
            winner.url,
            [u for u in group if u != winner.url],
        )

    result = [doc for doc in docs if doc.url not in grouped_urls]
    result.extend(winners)
    return result


# ── End-to-end orchestration ──────────────────────────────────────────────────


def normalize_documents(
    triples: list[tuple[FetchResult, ExtractResult, dict]],
    client: httpx.Client | None = None,
    collapse_dupes: bool = True,
    dup_threshold: float = NEAR_DUP_THRESHOLD,
) -> list[NormalizedDoc]:
    """Build, dedupe, and near-dup-collapse a batch of fetched/extracted pages.

    Each triple is (fetch_result, extract_result, meta) where *meta* may contain
    "source", "lastmod", "priority" keys (all optional).
    """
    docs: list[NormalizedDoc] = []
    for fetch, extract_result, meta in triples:
        if not extract_result.has_content:
            logger.debug("Skipping %s — no extractable content", fetch.url)
            continue
        docs.append(
            build_normalized_doc(
                fetch,
                extract_result,
                source=meta.get("source", "sitemap"),
                lastmod=meta.get("lastmod"),
                priority=meta.get("priority"),
                client=client,
            )
        )

    docs = dedupe_by_url(docs)

    if collapse_dupes:
        docs = collapse_near_duplicates(docs, threshold=dup_threshold)

    return docs
