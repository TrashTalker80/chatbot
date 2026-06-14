"""Sitemap-aware URL discovery for appther.com.

Discovery pipeline:
  1. Fetch sitemap.xml and detect whether it is a sitemap index or a urlset.
  2. If it is a sitemap index, fetch every child sitemap (including
     blog-sitemap.xml) and merge the URL sets.
  3. Apply robots.txt filtering and strip tracking query params.
  4. Deduplicate by normalized URL.
  5. Fall back to a bounded BFS from the homepage when the sitemap is
     completely unavailable or returns no URLs.

CLI usage:
  python -m crawler.discovery               # full run, writes staging/discovery/
  python -m crawler.discovery --dry-run     # count only, no I/O
  python -m crawler.discovery --output urls.jsonl
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import logging
import time
import xml.etree.ElementTree as ET
from collections import deque
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from crawler.config import (
    BASE_URL,
    BFS_MAX_DEPTH,
    BFS_MAX_URLS,
    DEFAULT_CRAWL_DELAY_SECONDS,
    SITEMAP_URL,
)
from crawler.http_client import create_client
from crawler.models import DiscoveredURL
from crawler.robots import RobotsChecker
from crawler.urls import infer_page_type, is_internal, normalize_url

logger = logging.getLogger(__name__)

# Sitemap XML namespace
_SM_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


# ── Public API ────────────────────────────────────────────────────────────────


def discover_urls(
    client: httpx.Client,
    robots: RobotsChecker,
    sitemap_url: str = SITEMAP_URL,
    include_bfs_fallback: bool = True,
) -> list[DiscoveredURL]:
    """Discover all crawlable URLs for appther.com.

    Returns a deduplicated, robots-filtered, tracking-clean list of
    DiscoveredURL objects, each tagged with page_type and source.

    Falls back to a bounded BFS from BASE_URL when the sitemap is
    unavailable or yields nothing.
    """
    urls = _fetch_sitemap_tree(client, sitemap_url)

    if not urls and include_bfs_fallback:
        logger.warning("Sitemap returned no URLs — falling back to BFS from %s", BASE_URL)
        urls = _bfs_discover(client, robots)

    return _filter_and_dedupe(urls, robots)


# ── Sitemap parsing ───────────────────────────────────────────────────────────


def _fetch_sitemap_tree(
    client: httpx.Client,
    sitemap_url: str = SITEMAP_URL,
) -> list[DiscoveredURL]:
    """Fetch the root sitemap and recursively retrieve all child sitemaps.

    Handles both sitemap indexes (<sitemapindex>) and plain urlsets (<urlset>).
    A single level of indirection is followed (index → children); deeper nesting
    is not expected for appther.com and is skipped with a warning.
    """
    logger.info("Fetching root sitemap: %s", sitemap_url)
    try:
        response = client.get(sitemap_url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Failed to fetch root sitemap (%s): %s", sitemap_url, exc)
        return []

    content = response.text

    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        logger.error("Malformed XML in root sitemap: %s", exc)
        return []

    local_tag = _local_tag(root.tag)

    if local_tag == "sitemapindex":
        return _expand_sitemap_index(client, root)
    elif local_tag == "urlset":
        source = _source_from_url(sitemap_url)
        return _parse_urlset(root, source=source)
    else:
        logger.warning("Unknown root element <%s> in %s — skipping", local_tag, sitemap_url)
        return []


def _expand_sitemap_index(
    client: httpx.Client,
    index_root: ET.Element,
) -> list[DiscoveredURL]:
    """Fetch every child sitemap referenced in a <sitemapindex>."""
    child_urls = _parse_sitemap_index(index_root)
    logger.info("Sitemap index lists %d child sitemaps", len(child_urls))

    all_urls: list[DiscoveredURL] = []
    for child_url in child_urls:
        source = _source_from_url(child_url)
        logger.info("Fetching child sitemap [%s]: %s", source, child_url)
        try:
            response = client.get(child_url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Failed to fetch child sitemap %s: %s", child_url, exc)
            continue

        try:
            child_root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            logger.warning("Malformed XML in %s: %s", child_url, exc)
            continue

        child_local = _local_tag(child_root.tag)
        if child_local != "urlset":
            logger.warning(
                "Expected <urlset> in %s, got <%s> — skipping nested index",
                child_url,
                child_local,
            )
            continue

        entries = _parse_urlset(child_root, source=source)
        logger.info("  → %d URLs from %s", len(entries), child_url)
        all_urls.extend(entries)

    return all_urls


def _parse_sitemap_index(root: ET.Element) -> list[str]:
    """Extract child sitemap <loc> values from a <sitemapindex> element."""
    locs = []
    for sitemap_elem in root:
        if _local_tag(sitemap_elem.tag) != "sitemap":
            continue
        loc_elem = _find_child(sitemap_elem, "loc")
        if loc_elem is not None and loc_elem.text:
            locs.append(loc_elem.text.strip())
    return locs


def _parse_urlset(root: ET.Element, source: str = "sitemap") -> list[DiscoveredURL]:
    """Extract URL entries with metadata from a <urlset> element."""
    entries: list[DiscoveredURL] = []
    for url_elem in root:
        if _local_tag(url_elem.tag) != "url":
            continue
        loc_elem = _find_child(url_elem, "loc")
        if loc_elem is None or not loc_elem.text:
            continue

        raw_url = loc_elem.text.strip()
        url = normalize_url(raw_url)

        lastmod_elem = _find_child(url_elem, "lastmod")
        priority_elem = _find_child(url_elem, "priority")
        changefreq_elem = _find_child(url_elem, "changefreq")

        priority: float | None = None
        if priority_elem is not None and priority_elem.text:
            with contextlib.suppress(ValueError):
                priority = float(priority_elem.text.strip())

        entries.append(
            DiscoveredURL(
                url=url,
                lastmod=_elem_text(lastmod_elem),
                priority=priority,
                changefreq=_elem_text(changefreq_elem),
                source=source,
                page_type=infer_page_type(url),
            )
        )
    return entries


# ── BFS fallback ──────────────────────────────────────────────────────────────


def _bfs_discover(
    client: httpx.Client,
    robots: RobotsChecker,
    base_url: str = BASE_URL,
    max_depth: int = BFS_MAX_DEPTH,
    max_urls: int = BFS_MAX_URLS,
    crawl_delay: float = DEFAULT_CRAWL_DELAY_SECONDS,
) -> list[DiscoveredURL]:
    """Bounded breadth-first crawl from *base_url*.

    This is only used when the sitemap is completely unavailable.
    Fetches pages to extract links but discards the HTML — full content
    storage is handled by fetch.py.
    """
    logger.info(
        "BFS fallback: starting from %s (max_depth=%d, max_urls=%d)",
        base_url,
        max_depth,
        max_urls,
    )

    discovered: list[DiscoveredURL] = []
    visited: set[str] = set()
    # queue entries: (normalized_url, depth)
    queue: deque[tuple[str, int]] = deque([(normalize_url(base_url), 0)])

    while queue and len(discovered) < max_urls:
        url, depth = queue.popleft()

        if url in visited:
            continue
        visited.add(url)

        if not robots.is_allowed(url):
            logger.debug("BFS: robots disallows %s", url)
            continue

        try:
            response = client.get(url)
            if response.status_code != 200:
                logger.debug("BFS: HTTP %d for %s", response.status_code, url)
                continue
            html = response.text
            # Record the final URL after redirects as the canonical one
            final_url = normalize_url(str(response.url))
        except Exception as exc:
            logger.warning("BFS: fetch failed for %s: %s", url, exc)
            continue

        discovered.append(
            DiscoveredURL(url=final_url, source="bfs", page_type=infer_page_type(final_url))
        )

        if depth < max_depth:
            for link_url in _extract_internal_links(html, base_url):
                if link_url not in visited:
                    queue.append((link_url, depth + 1))

        time.sleep(crawl_delay)

    logger.info("BFS discovered %d URLs", len(discovered))
    return discovered


def _extract_internal_links(html: str, base_url: str) -> list[str]:
    """Parse <a href> links from HTML and return normalized internal URLs."""
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        href = str(a_tag["href"]).strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        abs_url = urljoin(base_url, href)
        if is_internal(abs_url, base_url):
            links.append(normalize_url(abs_url))
    return links


# ── Filtering and deduplication ───────────────────────────────────────────────


def _filter_and_dedupe(
    urls: list[DiscoveredURL],
    robots: RobotsChecker,
) -> list[DiscoveredURL]:
    """Apply robots.txt filter and deduplicate by normalized URL.

    Deduplication keeps the entry with the richest metadata (sitemap entries
    carry lastmod/priority; BFS entries do not). When the same URL appears
    in multiple sources, the first occurrence in input order wins (caller
    should put higher-priority sources first).

    Tracking params are already stripped by normalize_url() during parsing;
    this function just applies the robots check and dedup.
    """
    seen: dict[str, DiscoveredURL] = {}
    filtered_count = 0
    dup_count = 0

    for entry in urls:
        if not robots.is_allowed(entry.url):
            logger.debug("robots.txt disallows %s", entry.url)
            filtered_count += 1
            continue

        if entry.url in seen:
            dup_count += 1
            # Keep the richer entry (prefer sitemap over bfs for metadata)
            existing = seen[entry.url]
            if existing.source == "bfs" and entry.source != "bfs":
                seen[entry.url] = entry
            continue

        seen[entry.url] = entry

    logger.info(
        "filter_and_dedupe: %d in → %d out (%d robots-filtered, %d duplicates)",
        len(urls),
        len(seen),
        filtered_count,
        dup_count,
    )
    return list(seen.values())


def summarize(urls: list[DiscoveredURL]) -> dict[str, int]:
    """Return counts by source and by page_type for reporting."""
    by_source: dict[str, int] = collections.Counter(u.source for u in urls)
    by_type: dict[str, int] = collections.Counter(u.page_type or "unknown" for u in urls)
    return {"total": len(urls), "by_source": dict(by_source), "by_page_type": dict(by_type)}


# ── XML helpers ───────────────────────────────────────────────────────────────


def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix from a tag name.

    ET represents namespaced tags as "{namespace}localname". This returns
    just the local part so callers can match without knowing the namespace.
    """
    return tag.split("}")[-1] if "}" in tag else tag


def _find_child(parent: ET.Element, local: str) -> ET.Element | None:
    """Find first direct child matching *local* tag name (namespace-agnostic)."""
    for child in parent:
        if _local_tag(child.tag) == local:
            return child
    return None


def _elem_text(elem: ET.Element | None) -> str | None:
    """Return stripped text of *elem*, or None if elem is None or empty."""
    return elem.text.strip() if elem is not None and elem.text else None


def _source_from_url(url: str) -> str:
    """Infer a short source label from a child sitemap URL."""
    lower = url.lower()
    if "blog" in lower:
        return "blog-sitemap"
    return "sitemap"


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Discover all crawlable URLs on appther.com via sitemap + BFS fallback."
    )
    parser.add_argument(
        "--sitemap",
        default=SITEMAP_URL,
        help="Root sitemap URL (default: appther.com/sitemap.xml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URL count and breakdown without saving to disk.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Write discovered URLs as JSONL to FILE (default: staging/discovery/).",
    )
    parser.add_argument(
        "--no-bfs",
        action="store_true",
        help="Disable BFS fallback if sitemap is unavailable.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    with create_client() as client:
        robots = RobotsChecker()
        robots.load(client=client)

        urls = discover_urls(
            client=client,
            robots=robots,
            sitemap_url=args.sitemap,
            include_bfs_fallback=not args.no_bfs,
        )

    info = summarize(urls)
    print(f"\nDiscovered {info['total']} URLs")
    print("\nBy source:")
    for source, count in sorted(info["by_source"].items()):
        print(f"  {source}: {count}")
    print("\nBy page_type:")
    for ptype, count in sorted(info["by_page_type"].items(), key=lambda x: -x[1]):
        print(f"  {ptype}: {count}")

    if args.dry_run:
        return

    # Determine output path
    from crawler.staging import save_discovery

    output_path = args.output
    saved = save_discovery(urls, output=output_path)
    print(f"\nSaved to {saved}")


if __name__ == "__main__":
    main()
