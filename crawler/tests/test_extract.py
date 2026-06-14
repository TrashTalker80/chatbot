"""Tests for crawler/extract.py — HTML → Markdown + FAQ pair extraction."""

from __future__ import annotations

from pathlib import Path

import pytest  # noqa: F401  (used via @pytest.fixture)

from crawler.extract import (
    ExtractResult,
    FaqPair,
    _extract_faq_pairs,
    _extract_title,
    _extract_with_bs4_fallback,
    extract,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _html(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def article_html() -> str:
    return _html("article_page.html")


@pytest.fixture()
def faq_html() -> str:
    return _html("faq_page.html")


# ── TestExtractMainContent ────────────────────────────────────────────────────


class TestExtractMainContent:
    def test_returns_extract_result(self, article_html):
        result = extract(article_html, "https://www.appther.com/services/odoo")
        assert isinstance(result, ExtractResult)

    def test_has_content(self, article_html):
        result = extract(article_html, "https://www.appther.com/services/odoo")
        assert result.has_content

    def test_title_stripped(self, article_html):
        result = extract(article_html, "https://www.appther.com/services/odoo")
        # " | Appther" suffix must be removed
        assert result.title == "Odoo ERP Implementation Services"
        assert "Appther" not in result.title

    def test_url_stored(self, article_html):
        url = "https://www.appther.com/services/odoo"
        result = extract(article_html, url)
        assert result.url == url

    def test_nav_links_not_as_standalone_lines(self, article_html):
        result = extract(article_html, "")
        lines = [ln.strip().lower() for ln in result.markdown.splitlines() if ln.strip()]
        nav_only_lines = {"home", "contact", "industries"}
        assert not any(ln in nav_only_lines for ln in lines), (
            f"Nav link text as standalone line: {[ln for ln in lines if ln in nav_only_lines]}"
        )

    def test_footer_not_in_markdown(self, article_html):
        result = extract(article_html, "")
        assert "privacy policy" not in result.markdown.lower()
        assert "terms" not in result.markdown.lower() or "terms" in result.title.lower()

    def test_cta_banner_not_in_markdown(self, article_html):
        result = extract(article_html, "")
        assert "get free consultation" not in result.markdown.lower()

    def test_main_content_preserved(self, article_html):
        result = extract(article_html, "")
        md = result.markdown.lower()
        assert "odoo erp" in md
        assert "five-phase" in md or "methodology" in md

    def test_word_count_reasonable(self, article_html):
        result = extract(article_html, "")
        # Fixture has ~170 words of real content
        assert result.word_count >= 50

    def test_faq_page_has_content(self, faq_html):
        result = extract(faq_html, "https://www.appther.com/faq")
        assert result.has_content

    def test_faq_title_extracted(self, faq_html):
        result = extract(faq_html, "")
        assert result.title == "Frequently Asked Questions"

    def test_empty_html_returns_empty(self):
        result = extract("<html><body></body></html>", "")
        # Should not raise; may or may not have content
        assert isinstance(result, ExtractResult)

    def test_minimal_html_does_not_raise(self):
        result = extract("<p>Hello world</p>", "")
        assert isinstance(result, ExtractResult)

    def test_extraction_method_set(self, article_html):
        result = extract(article_html, "")
        assert result.extraction_method in ("trafilatura", "bs4-fallback")

    def test_bs4_fallback_strips_boilerplate(self):
        html = """<html><body>
        <nav><a href='/'>Home</a></nav>
        <main><h1>Title</h1><p>Body text here.</p></main>
        <footer>Copyright</footer>
        </body></html>"""
        md = _extract_with_bs4_fallback(html)
        assert "Body text here" in md
        assert "Copyright" not in md

    def test_bs4_fallback_headings_converted(self):
        html = """<html><body><main>
        <h1>Top</h1><h2>Sub</h2><p>Text</p></main></body></html>"""
        md = _extract_with_bs4_fallback(html)
        assert "# Top" in md
        assert "## Sub" in md


# ── TestExtractFaqPairs ───────────────────────────────────────────────────────


class TestExtractFaqPairs:
    def test_returns_four_pairs(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        assert len(pairs) == 4

    def test_all_are_faq_pair_instances(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        for p in pairs:
            assert isinstance(p, FaqPair)

    def test_details_pattern_question(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        questions = [p.question for p in pairs]
        assert "How long does an Odoo ERP implementation take?" in questions

    def test_details_pattern_full_answer_preserved(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        impl_pair = next(p for p in pairs if "implementation take" in p.question)
        # Full answer must contain key phrases — not truncated
        assert "eight and sixteen weeks" in impl_pair.answer
        assert "migrated" in impl_pair.answer

    def test_class_based_accordion_question(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        questions = [p.question for p in pairs]
        assert "Do you provide post-go-live support?" in questions

    def test_class_based_accordion_full_answer(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        support_pair = next(p for p in pairs if "post-go-live" in p.question)
        assert "monthly support retainers" in support_pair.answer
        assert "expert on call" in support_pair.answer

    def test_definition_list_question(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        questions = [p.question for p in pairs]
        assert "Which industries do you specialize in?" in questions

    def test_definition_list_full_answer(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        industry_pair = next(p for p in pairs if "industries" in p.question)
        assert "manufacturing" in industry_pair.answer
        assert "professional services" in industry_pair.answer

    def test_no_duplicate_questions(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        questions = [p.question for p in pairs]
        assert len(questions) == len(set(questions))

    def test_answer_whitespace_normalized(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        for p in pairs:
            # No raw newlines or tab indentation in answer
            assert "\n" not in p.answer
            assert "\t" not in p.answer
            # No double spaces
            assert "  " not in p.answer

    def test_question_whitespace_normalized(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        for p in pairs:
            assert "  " not in p.question
            assert "\n" not in p.question

    def test_to_text_format(self, faq_html):
        pairs = _extract_faq_pairs(faq_html)
        for p in pairs:
            text = p.to_text()
            assert text.startswith("Q: ")
            assert "\nA: " in text

    def test_no_faq_pairs_on_article_page(self, article_html):
        pairs = _extract_faq_pairs(article_html)
        assert pairs == []

    def test_empty_html_returns_empty_list(self):
        pairs = _extract_faq_pairs("<html><body></body></html>")
        assert pairs == []


# ── TestGetTitle ──────────────────────────────────────────────────────────────


class TestGetTitle:
    def test_strips_site_name_pipe_separator(self):
        html = "<html><head><title>FAQ | Appther</title></head></html>"
        assert _extract_title(html) == "FAQ"

    def test_strips_site_name_dash_separator(self):
        html = "<html><head><title>Services - Appther</title></head></html>"
        assert _extract_title(html) == "Services"

    def test_strips_site_name_em_dash(self):
        html = "<html><head><title>About Us — Appther</title></head></html>"
        assert _extract_title(html) == "About Us"

    def test_falls_back_to_h1(self):
        html = "<html><body><h1>My Page</h1></body></html>"
        assert _extract_title(html) == "My Page"

    def test_empty_title_falls_back_to_h1(self):
        html = "<html><head><title></title></head><body><h1>H1 Title</h1></body></html>"
        assert _extract_title(html) == "H1 Title"

    def test_no_title_no_h1_returns_empty(self):
        html = "<html><body><p>text</p></body></html>"
        assert _extract_title(html) == ""

    def test_article_fixture_title(self, article_html):
        assert _extract_title(article_html) == "Odoo ERP Implementation Services"

    def test_faq_fixture_title(self, faq_html):
        assert _extract_title(faq_html) == "Frequently Asked Questions"
