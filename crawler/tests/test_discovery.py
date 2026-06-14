"""Tests for crawler.discovery — sitemap parsing, filtering, deduplication, BFS fallback.

All HTTP calls are intercepted by pytest-httpx so no network access is needed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

from pytest_httpx import HTTPXMock

from crawler.config import BASE_URL, SITEMAP_URL
from crawler.discovery import (
    _extract_internal_links,
    _filter_and_dedupe,
    _local_tag,
    _parse_sitemap_index,
    _parse_urlset,
    _source_from_url,
    discover_urls,
    summarize,
)
from crawler.http_client import create_client
from crawler.models import DiscoveredURL
from crawler.robots import RobotsChecker

# ── Helpers ───────────────────────────────────────────────────────────────────


def _robots_from_fixture(robots_txt: str) -> RobotsChecker:
    """Build a RobotsChecker pre-loaded with fixture content (no network)."""
    checker = RobotsChecker()
    checker.load(content=robots_txt)
    return checker


# ── XML helpers ───────────────────────────────────────────────────────────────


class TestLocalTag:
    def test_strips_namespace(self):
        assert _local_tag("{http://www.sitemaps.org/schemas/sitemap/0.9}urlset") == "urlset"

    def test_no_namespace_unchanged(self):
        assert _local_tag("urlset") == "urlset"

    def test_sitemapindex(self):
        ns_tag = "{http://www.sitemaps.org/schemas/sitemap/0.9}sitemapindex"
        assert _local_tag(ns_tag) == "sitemapindex"


class TestSourceFromUrl:
    def test_blog_sitemap(self):
        assert _source_from_url("https://www.appther.com/blog-sitemap.xml") == "blog-sitemap"

    def test_page_sitemap(self):
        assert _source_from_url("https://www.appther.com/page-sitemap.xml") == "sitemap"

    def test_root_sitemap(self):
        assert _source_from_url("https://www.appther.com/sitemap.xml") == "sitemap"


# ── Sitemap index parsing ─────────────────────────────────────────────────────


class TestParseSitemapIndex:
    def test_extracts_child_urls(self, sitemap_index_xml: str):
        root = ET.fromstring(sitemap_index_xml)
        child_urls = _parse_sitemap_index(root)
        assert len(child_urls) == 2
        assert any("page-sitemap" in u for u in child_urls)
        assert any("blog-sitemap" in u for u in child_urls)

    def test_returns_list_of_strings(self, sitemap_index_xml: str):
        root = ET.fromstring(sitemap_index_xml)
        child_urls = _parse_sitemap_index(root)
        assert all(isinstance(u, str) for u in child_urls)


# ── Urlset parsing ────────────────────────────────────────────────────────────


class TestParseUrlset:
    def test_parses_all_urls(self, page_sitemap_xml: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        # page_sitemap has 10 <url> entries
        assert len(entries) == 10

    def test_all_entries_are_discovered_url(self, page_sitemap_xml: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        assert all(isinstance(e, DiscoveredURL) for e in entries)

    def test_lastmod_captured(self, page_sitemap_xml: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        homepage = next(e for e in entries if e.url.endswith("appther.com/"))
        assert homepage.lastmod == "2026-05-15"

    def test_priority_captured(self, page_sitemap_xml: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        homepage = next(e for e in entries if e.url.endswith("appther.com/"))
        assert homepage.priority == 1.0

    def test_changefreq_captured(self, page_sitemap_xml: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        homepage = next(e for e in entries if e.url.endswith("appther.com/"))
        assert homepage.changefreq == "weekly"

    def test_source_propagated(self, blog_sitemap_xml: str):
        root = ET.fromstring(blog_sitemap_xml)
        entries = _parse_urlset(root, source="blog-sitemap")
        assert all(e.source == "blog-sitemap" for e in entries)

    def test_page_type_inferred(self, page_sitemap_xml: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        faq = next((e for e in entries if "/faq" in e.url), None)
        assert faq is not None
        assert faq.page_type == "faq"

    def test_tracking_params_stripped_from_url(self, page_sitemap_xml: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        # The page sitemap contains a URL with utm_source — it must be cleaned
        for entry in entries:
            assert "utm_source" not in entry.url
            assert "utm_medium" not in entry.url


# ── Filter and deduplication ──────────────────────────────────────────────────


class TestFilterAndDedupe:
    def test_removes_disallowed_paths(self, page_sitemap_xml: str, robots_txt: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        robots = _robots_from_fixture(robots_txt)
        filtered = _filter_and_dedupe(entries, robots)
        urls = {e.url for e in filtered}
        assert not any("/thank-you" in u for u in urls)
        assert not any("/wp-admin" in u for u in urls)

    def test_deduplicates_after_tracking_strip(self, page_sitemap_xml: str, robots_txt: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        robots = _robots_from_fixture(robots_txt)
        filtered = _filter_and_dedupe(entries, robots)
        # /services/odoo-erp/ appears twice (once clean, once with utm_source)
        # After strip+dedup there should be exactly one
        odoo_urls = [e for e in filtered if "/services/odoo-erp/" in e.url]
        assert len(odoo_urls) == 1

    def test_keeps_allowed_urls(self, page_sitemap_xml: str, robots_txt: str):
        root = ET.fromstring(page_sitemap_xml)
        entries = _parse_urlset(root, source="sitemap")
        robots = _robots_from_fixture(robots_txt)
        filtered = _filter_and_dedupe(entries, robots)
        urls = {e.url for e in filtered}
        assert any("/faq" in u for u in urls)
        assert any("/services/odoo-erp/" in u for u in urls)
        assert any("/privacy-policy" in u for u in urls)

    def test_prefers_sitemap_over_bfs_on_dup(self, robots_txt: str):
        robots = _robots_from_fixture(robots_txt)
        sitemap_entry = DiscoveredURL(
            url="https://www.appther.com/faq",
            lastmod="2026-03-01",
            priority=0.7,
            source="sitemap",
        )
        bfs_entry = DiscoveredURL(
            url="https://www.appther.com/faq",
            source="bfs",
        )
        # bfs first, then sitemap — sitemap should win (richer metadata)
        result = _filter_and_dedupe([bfs_entry, sitemap_entry], robots)
        assert len(result) == 1
        assert result[0].source == "sitemap"
        assert result[0].lastmod == "2026-03-01"


# ── Full discovery pipeline (with HTTP mocking) ───────────────────────────────


class TestDiscoverUrlsHappyPath:
    def test_merges_page_and_blog_sitemaps(
        self,
        httpx_mock: HTTPXMock,
        sitemap_index_xml: str,
        page_sitemap_xml: str,
        blog_sitemap_xml: str,
        robots_txt: str,
    ):
        httpx_mock.add_response(url=SITEMAP_URL, text=sitemap_index_xml)
        httpx_mock.add_response(
            url="https://www.appther.com/page-sitemap.xml", text=page_sitemap_xml
        )
        httpx_mock.add_response(
            url="https://www.appther.com/blog-sitemap.xml", text=blog_sitemap_xml
        )

        with create_client() as client:
            robots = _robots_from_fixture(robots_txt)
            urls = discover_urls(
                client, robots, sitemap_url=SITEMAP_URL, include_bfs_fallback=False
            )

        assert len(urls) > 0

    def test_blog_urls_present(
        self,
        httpx_mock: HTTPXMock,
        sitemap_index_xml: str,
        page_sitemap_xml: str,
        blog_sitemap_xml: str,
        robots_txt: str,
    ):
        httpx_mock.add_response(url=SITEMAP_URL, text=sitemap_index_xml)
        httpx_mock.add_response(
            url="https://www.appther.com/page-sitemap.xml", text=page_sitemap_xml
        )
        httpx_mock.add_response(
            url="https://www.appther.com/blog-sitemap.xml", text=blog_sitemap_xml
        )

        with create_client() as client:
            robots = _robots_from_fixture(robots_txt)
            urls = discover_urls(
                client, robots, sitemap_url=SITEMAP_URL, include_bfs_fallback=False
            )

        all_urls = {e.url for e in urls}
        assert any("/blog/" in u for u in all_urls), "No blog URLs found in discovery output"
        assert any("odoo-implementation-tips" in u for u in all_urls)
        assert any("erp-roi-case-study" in u for u in all_urls)

    def test_tracking_params_stripped(
        self,
        httpx_mock: HTTPXMock,
        sitemap_index_xml: str,
        page_sitemap_xml: str,
        blog_sitemap_xml: str,
        robots_txt: str,
    ):
        httpx_mock.add_response(url=SITEMAP_URL, text=sitemap_index_xml)
        httpx_mock.add_response(
            url="https://www.appther.com/page-sitemap.xml", text=page_sitemap_xml
        )
        httpx_mock.add_response(
            url="https://www.appther.com/blog-sitemap.xml", text=blog_sitemap_xml
        )

        with create_client() as client:
            robots = _robots_from_fixture(robots_txt)
            urls = discover_urls(
                client, robots, sitemap_url=SITEMAP_URL, include_bfs_fallback=False
            )

        for entry in urls:
            assert "utm_" not in entry.url, f"Tracking param in URL: {entry.url}"
            assert "gclid" not in entry.url
            assert "fbclid" not in entry.url

    def test_disallowed_paths_excluded(
        self,
        httpx_mock: HTTPXMock,
        sitemap_index_xml: str,
        page_sitemap_xml: str,
        blog_sitemap_xml: str,
        robots_txt: str,
    ):
        httpx_mock.add_response(url=SITEMAP_URL, text=sitemap_index_xml)
        httpx_mock.add_response(
            url="https://www.appther.com/page-sitemap.xml", text=page_sitemap_xml
        )
        httpx_mock.add_response(
            url="https://www.appther.com/blog-sitemap.xml", text=blog_sitemap_xml
        )

        with create_client() as client:
            robots = _robots_from_fixture(robots_txt)
            urls = discover_urls(
                client, robots, sitemap_url=SITEMAP_URL, include_bfs_fallback=False
            )

        all_urls = {e.url for e in urls}
        assert not any("/thank-you" in u for u in all_urls)
        assert not any("/wp-admin" in u for u in all_urls)
        assert not any("/vendor/" in u for u in all_urls)

    def test_deduplication_across_sources(
        self,
        httpx_mock: HTTPXMock,
        sitemap_index_xml: str,
        page_sitemap_xml: str,
        blog_sitemap_xml: str,
        robots_txt: str,
    ):
        httpx_mock.add_response(url=SITEMAP_URL, text=sitemap_index_xml)
        httpx_mock.add_response(
            url="https://www.appther.com/page-sitemap.xml", text=page_sitemap_xml
        )
        httpx_mock.add_response(
            url="https://www.appther.com/blog-sitemap.xml", text=blog_sitemap_xml
        )

        with create_client() as client:
            robots = _robots_from_fixture(robots_txt)
            urls = discover_urls(
                client, robots, sitemap_url=SITEMAP_URL, include_bfs_fallback=False
            )

        all_url_strs = [e.url for e in urls]
        # No duplicates allowed
        assert len(all_url_strs) == len(set(all_url_strs)), "Duplicate URLs found"

    def test_page_types_assigned(
        self,
        httpx_mock: HTTPXMock,
        sitemap_index_xml: str,
        page_sitemap_xml: str,
        blog_sitemap_xml: str,
        robots_txt: str,
    ):
        httpx_mock.add_response(url=SITEMAP_URL, text=sitemap_index_xml)
        httpx_mock.add_response(
            url="https://www.appther.com/page-sitemap.xml", text=page_sitemap_xml
        )
        httpx_mock.add_response(
            url="https://www.appther.com/blog-sitemap.xml", text=blog_sitemap_xml
        )

        with create_client() as client:
            robots = _robots_from_fixture(robots_txt)
            urls = discover_urls(
                client, robots, sitemap_url=SITEMAP_URL, include_bfs_fallback=False
            )

        type_map = {e.url: e.page_type for e in urls}
        assert any(t == "faq" for t in type_map.values())
        assert any(t == "blog" for t in type_map.values())
        assert any(t == "service" for t in type_map.values())
        assert any(t == "case-study" for t in type_map.values())

    def test_sitemap_source_labeled_correctly(
        self,
        httpx_mock: HTTPXMock,
        sitemap_index_xml: str,
        page_sitemap_xml: str,
        blog_sitemap_xml: str,
        robots_txt: str,
    ):
        httpx_mock.add_response(url=SITEMAP_URL, text=sitemap_index_xml)
        httpx_mock.add_response(
            url="https://www.appther.com/page-sitemap.xml", text=page_sitemap_xml
        )
        httpx_mock.add_response(
            url="https://www.appther.com/blog-sitemap.xml", text=blog_sitemap_xml
        )

        with create_client() as client:
            robots = _robots_from_fixture(robots_txt)
            urls = discover_urls(
                client, robots, sitemap_url=SITEMAP_URL, include_bfs_fallback=False
            )

        blog_entries = [e for e in urls if "/blog/" in e.url]
        assert all(e.source == "blog-sitemap" for e in blog_entries)

        page_entries = [e for e in urls if e.source == "sitemap"]
        assert len(page_entries) > 0


# ── Sitemap error → BFS fallback ──────────────────────────────────────────────


class TestBfsFallback:
    def test_bfs_activates_when_sitemap_fails(
        self,
        httpx_mock: HTTPXMock,
        robots_txt: str,
    ):
        # Sitemap returns 500 → BFS should activate
        httpx_mock.add_response(url=SITEMAP_URL, status_code=500)
        # Catch-all: BFS homepage fetch gets minimal HTML (no links = one page, then stops)
        httpx_mock.add_response(text="<html><body><h1>Appther</h1></body></html>", status_code=200)

        with create_client() as client:
            robots = _robots_from_fixture(robots_txt)
            urls = discover_urls(client, robots, sitemap_url=SITEMAP_URL, include_bfs_fallback=True)

        assert len(urls) > 0

    def test_bfs_respects_robots(self, robots_txt: str):
        robots = _robots_from_fixture(robots_txt)
        entries = [
            DiscoveredURL(url="https://www.appther.com/services/", source="bfs"),
            DiscoveredURL(url="https://www.appther.com/thank-you", source="bfs"),
            DiscoveredURL(url="https://www.appther.com/wp-admin/", source="bfs"),
        ]
        filtered = _filter_and_dedupe(entries, robots)
        urls = {e.url for e in filtered}
        assert "https://www.appther.com/services/" in urls
        assert not any("/thank-you" in u for u in urls)
        assert not any("/wp-admin" in u for u in urls)


# ── Internal link extraction ──────────────────────────────────────────────────


class TestExtractInternalLinks:
    def test_extracts_internal_links(self, homepage_html: str):
        links = _extract_internal_links(homepage_html, BASE_URL)
        assert any("/services/odoo-erp/" in u for u in links)
        assert any("/faq" in u for u in links)
        assert any("/case-study/" in u for u in links)

    def test_excludes_external_links(self, homepage_html: str):
        links = _extract_internal_links(homepage_html, BASE_URL)
        assert not any("example.com" in u for u in links)

    def test_strips_tracking_params_from_links(self, homepage_html: str):
        links = _extract_internal_links(homepage_html, BASE_URL)
        for url in links:
            assert "utm_source" not in url

    def test_excludes_mailto_and_tel(self):
        html = '<a href="mailto:info@appther.com">Email</a><a href="tel:+1234">Call</a>'
        links = _extract_internal_links(html, BASE_URL)
        assert not any("mailto:" in u for u in links)
        assert not any("tel:" in u for u in links)

    def test_excludes_anchor_only_links(self):
        html = '<a href="#section">Jump</a><a href="/services/">Services</a>'
        links = _extract_internal_links(html, BASE_URL)
        # "#section" should be filtered; "/services/" should be present
        assert not any(u.endswith("#section") for u in links)
        assert any("/services/" in u for u in links)


# ── Summarize ─────────────────────────────────────────────────────────────────


class TestSummarize:
    def test_total_count(self):
        urls = [
            DiscoveredURL(url="https://www.appther.com/faq", source="sitemap", page_type="faq"),
            DiscoveredURL(
                url="https://www.appther.com/blog/post/",
                source="blog-sitemap",
                page_type="blog",
            ),
        ]
        info = summarize(urls)
        assert info["total"] == 2

    def test_by_source(self):
        urls = [
            DiscoveredURL(url="https://www.appther.com/faq", source="sitemap"),
            DiscoveredURL(url="https://www.appther.com/blog/post/", source="blog-sitemap"),
            DiscoveredURL(url="https://www.appther.com/services/", source="sitemap"),
        ]
        info = summarize(urls)
        assert info["by_source"]["sitemap"] == 2
        assert info["by_source"]["blog-sitemap"] == 1

    def test_by_page_type(self):
        urls = [
            DiscoveredURL(url="https://www.appther.com/faq", page_type="faq"),
            DiscoveredURL(url="https://www.appther.com/faq2", page_type="faq"),
            DiscoveredURL(url="https://www.appther.com/blog/post/", page_type="blog"),
        ]
        info = summarize(urls)
        assert info["by_page_type"]["faq"] == 2
        assert info["by_page_type"]["blog"] == 1
