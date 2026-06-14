"""Tests for crawler/normalize.py — URL canonicalization, hashing, near-dup collapse."""

from __future__ import annotations

from pathlib import Path

from crawler.extract import extract
from crawler.normalize import (
    NormalizedDoc,
    canonicalize_url,
    collapse_near_duplicates,
    compute_content_hash,
    dedupe_by_url,
    find_near_duplicate_groups,
    minhash_of,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _make_doc(
    url: str,
    markdown: str = "Hello world content here for testing.",
    source: str = "sitemap",
    priority: float = 1.0,
    page_type: str = "other",
) -> NormalizedDoc:
    return NormalizedDoc(
        url=url,
        original_url=url,
        title="Test Page",
        markdown=markdown,
        page_type=page_type,
        content_hash=compute_content_hash(markdown),
        source=source,
        priority=priority,
    )


# ── TestComputeContentHash ────────────────────────────────────────────────────


class TestComputeContentHash:
    def test_returns_hex_string(self):
        h = compute_content_hash("hello")
        assert isinstance(h, str)
        assert all(c in "0123456789abcdef" for c in h)

    def test_sha256_length(self):
        h = compute_content_hash("hello")
        assert len(h) == 64

    def test_deterministic(self):
        assert compute_content_hash("test text") == compute_content_hash("test text")

    def test_whitespace_normalized(self):
        h1 = compute_content_hash("  hello   world  ")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_newlines_treated_as_whitespace(self):
        h1 = compute_content_hash("hello\nworld")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_case_insensitive(self):
        h1 = compute_content_hash("Hello World")
        h2 = compute_content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        assert compute_content_hash("foo") != compute_content_hash("bar")

    def test_empty_string(self):
        h = compute_content_hash("")
        assert len(h) == 64


# ── TestCanonicalizeUrl ───────────────────────────────────────────────────────


class TestCanonicalizeUrl:
    def test_lowercases_path(self):
        result = canonicalize_url("https://www.appther.com/Company-DelhiNCR")
        assert result == "https://www.appther.com/company-delhincr"

    def test_strips_utm_params(self):
        result = canonicalize_url("https://www.appther.com/faq?utm_source=google")
        assert "utm_source" not in result

    def test_strips_fragment(self):
        result = canonicalize_url("https://www.appther.com/services#anchor")
        assert "#" not in result

    def test_preserves_path_structure(self):
        result = canonicalize_url("https://www.appther.com/services/odoo-erp")
        assert "/services/odoo-erp" in result

    def test_lowercase_scheme(self):
        result = canonicalize_url("HTTPS://www.appther.com/faq")
        assert result.startswith("https://")

    def test_lowercase_host(self):
        result = canonicalize_url("https://WWW.APPTHER.COM/faq")
        assert "WWW" not in result

    def test_no_client_no_redirect_following(self):
        # Without client, canonicalize_url only lowercases — no HTTP call
        result = canonicalize_url("https://www.appther.com/Industries")
        assert "Industries" not in result

    def test_with_mock_redirect(self, httpx_mock):
        httpx_mock.add_response(
            url="https://www.appther.com/industries/",
            status_code=301,
            headers={"Location": "https://www.appther.com/industry/"},
        )
        httpx_mock.add_response(
            url="https://www.appther.com/industry/",
            status_code=200,
            text="<html><body>Industry page</body></html>",
        )

        import httpx

        with httpx.Client() as client:
            result = canonicalize_url(
                "https://www.appther.com/industries/", client=client
            )
        assert "industry" in result
        assert "industries" not in result

    def test_with_mock_request_error_falls_back(self, httpx_mock):
        """When the HTTP request fails, canonicalize_url falls back to local normalization."""
        import httpx

        httpx_mock.add_exception(httpx.ConnectError("timeout"))

        with httpx.Client() as client:
            result = canonicalize_url("https://www.appther.com/FAQ", client=client)
        # Should still return the lowercase version (local normalization fallback)
        assert result == "https://www.appther.com/faq"


# ── TestMinHashOf ─────────────────────────────────────────────────────────────


class TestMinHashOf:
    def test_identical_texts_jaccard_one(self):
        text = "The quick brown fox jumps over the lazy dog"
        m1 = minhash_of(text)
        m2 = minhash_of(text)
        assert m1.jaccard(m2) > 0.99

    def test_different_texts_lower_jaccard(self):
        m1 = minhash_of("Python is a programming language")
        m2 = minhash_of("Odoo ERP implementation services in India")
        assert m1.jaccard(m2) < 0.5

    def test_short_text_does_not_raise(self):
        m = minhash_of("hi")
        assert m is not None

    def test_empty_text_does_not_raise(self):
        m = minhash_of("")
        assert m is not None

    def test_near_duplicate_high_jaccard(self):
        base = "Appther delivers Odoo ERP implementations for manufacturing retail and services. "
        base = base * 10  # repeat for more shingles
        dup = base + " Also serving London."
        m1 = minhash_of(base)
        m2 = minhash_of(dup)
        assert m1.jaccard(m2) > 0.7

    def test_num_perm_respected(self):
        m = minhash_of("hello world", num_perm=64)
        assert len(m.hashvalues) == 64


# ── TestFindNearDuplicateGroups ───────────────────────────────────────────────


class TestFindNearDuplicateGroups:
    def test_empty_list_returns_empty(self):
        assert find_near_duplicate_groups([]) == []

    def test_single_doc_returns_empty(self):
        doc = _make_doc("https://example.com/a", "some content text here")
        assert find_near_duplicate_groups([doc]) == []

    def test_identical_docs_grouped(self):
        text = ("Appther delivers end-to-end Odoo ERP implementations. " * 30)
        doc_a = _make_doc("https://www.appther.com/a", text)
        doc_b = _make_doc("https://www.appther.com/b", text)
        groups = find_near_duplicate_groups([doc_a, doc_b], threshold=0.85)
        assert len(groups) == 1
        assert sorted(groups[0]) == [
            "https://www.appther.com/a",
            "https://www.appther.com/b",
        ]

    def test_distinct_docs_not_grouped(self):
        art_html = _html("article_page.html")
        faq_html = _html("faq_page.html")
        doc_a = _make_doc(
            "https://www.appther.com/services/odoo",
            extract(art_html).markdown,
        )
        doc_b = _make_doc(
            "https://www.appther.com/faq",
            extract(faq_html).markdown,
        )
        groups = find_near_duplicate_groups([doc_a, doc_b], threshold=0.85)
        assert groups == []

    def test_groups_sorted_largest_first(self):
        text = ("duplicate content " * 40)
        docs = [_make_doc(f"https://www.appther.com/{i}", text) for i in range(5)]
        docs.append(
            _make_doc("https://www.appther.com/x", "completely different content about zebras")
        )
        groups = find_near_duplicate_groups(docs, threshold=0.85)
        if len(groups) > 1:
            assert len(groups[0]) >= len(groups[1])

    def test_empty_docs_excluded(self):
        doc_a = _make_doc("https://www.appther.com/a", "")
        doc_b = _make_doc("https://www.appther.com/b", "")
        groups = find_near_duplicate_groups([doc_a, doc_b], threshold=0.5)
        assert groups == []


# ── TestCollapseNearDuplicates ────────────────────────────────────────────────


class TestCollapseNearDuplicates:
    def _dup_pair(self):
        text = ("Appther delivers Odoo ERP implementations for growing businesses. " * 20)
        doc_a = _make_doc("https://www.appther.com/a", text, source="sitemap", priority=1.0)
        doc_b = _make_doc("https://www.appther.com/b", text + " Extra.", source="bfs", priority=0.5)
        return doc_a, doc_b

    def test_no_dupes_returns_same_count(self):
        doc_a = _make_doc("https://www.appther.com/a", "unique content about Odoo " * 30)
        doc_b = _make_doc(
            "https://www.appther.com/b",
            "completely different content about Python " * 30,
        )
        result = collapse_near_duplicates([doc_a, doc_b], threshold=0.85)
        assert len(result) == 2

    def test_near_dup_collapsed_to_one(self):
        doc_a, doc_b = self._dup_pair()
        result = collapse_near_duplicates([doc_a, doc_b], threshold=0.5)
        assert len(result) == 1

    def test_sitemap_wins_over_bfs(self):
        doc_a, doc_b = self._dup_pair()
        result = collapse_near_duplicates([doc_a, doc_b], threshold=0.5)
        assert result[0].source == "sitemap"
        assert result[0].url == "https://www.appther.com/a"

    def test_higher_priority_wins_when_same_source(self):
        text = ("Appther delivers Odoo ERP implementations for growing businesses. " * 20)
        low = _make_doc("https://www.appther.com/a", text, source="sitemap", priority=0.3)
        high = _make_doc("https://www.appther.com/b", text, source="sitemap", priority=0.9)
        result = collapse_near_duplicates([low, high], threshold=0.5)
        assert len(result) == 1
        assert result[0].url == "https://www.appther.com/b"

    def test_non_dup_doc_retained(self):
        doc_a, doc_b = self._dup_pair()
        doc_c = _make_doc(
            "https://www.appther.com/faq",
            extract(_html("faq_page.html")).markdown,
        )
        result = collapse_near_duplicates([doc_a, doc_b, doc_c], threshold=0.5)
        urls = {d.url for d in result}
        assert "https://www.appther.com/faq" in urls

    def test_overview_source_loses_to_sitemap(self):
        text = ("Appther delivers Odoo ERP implementations for growing businesses. " * 20)
        overview = _make_doc("https://www.appther.com/a", text, source="overview", priority=1.0)
        sitemap = _make_doc("https://www.appther.com/b", text, source="sitemap", priority=0.5)
        result = collapse_near_duplicates([overview, sitemap], threshold=0.5)
        assert result[0].source == "sitemap"


# ── TestDedupeByUrl ───────────────────────────────────────────────────────────


class TestDedupeByUrl:
    def test_unique_urls_unchanged(self):
        docs = [
            _make_doc("https://www.appther.com/a"),
            _make_doc("https://www.appther.com/b"),
        ]
        result = dedupe_by_url(docs)
        assert len(result) == 2

    def test_duplicate_url_collapsed(self):
        docs = [
            _make_doc("https://www.appther.com/a", source="sitemap"),
            _make_doc("https://www.appther.com/a", source="bfs"),
        ]
        result = dedupe_by_url(docs)
        assert len(result) == 1

    def test_sitemap_preferred_over_bfs(self):
        docs = [
            _make_doc("https://www.appther.com/a", source="bfs"),
            _make_doc("https://www.appther.com/a", source="sitemap"),
        ]
        result = dedupe_by_url(docs)
        assert result[0].source == "sitemap"

    def test_empty_list(self):
        assert dedupe_by_url([]) == []

    def test_order_preserved_for_distinct_urls(self):
        docs = [
            _make_doc("https://www.appther.com/a"),
            _make_doc("https://www.appther.com/b"),
            _make_doc("https://www.appther.com/c"),
        ]
        result = dedupe_by_url(docs)
        assert [d.url for d in result] == [
            "https://www.appther.com/a",
            "https://www.appther.com/b",
            "https://www.appther.com/c",
        ]
