"""Tests for crawler.urls — normalization, tracking-param stripping, page-type inference."""

import pytest

from crawler.urls import (
    infer_page_type,
    is_internal,
    normalize_url,
    strip_tracking_params,
    url_to_hash,
    url_to_slug,
)


class TestStripTrackingParams:
    def test_strips_utm_source(self):
        url = "https://www.appther.com/services/?utm_source=google"
        assert "utm_source" not in strip_tracking_params(url)

    def test_strips_utm_medium(self):
        url = "https://www.appther.com/?utm_medium=cpc"
        assert "utm_medium" not in strip_tracking_params(url)

    def test_strips_gclid(self):
        url = "https://www.appther.com/faq?gclid=abc123"
        assert "gclid" not in strip_tracking_params(url)

    def test_strips_fbclid(self):
        url = "https://www.appther.com/?fbclid=xyz"
        assert "fbclid" not in strip_tracking_params(url)

    def test_strips_multiple_tracking_params(self):
        url = "https://www.appther.com/?utm_source=google&utm_medium=cpc&gclid=abc"
        result = strip_tracking_params(url)
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "gclid" not in result

    def test_strips_nonstandard_utm_prefix(self):
        url = "https://www.appther.com/?utm_custom_field=value"
        assert "utm_custom_field" not in strip_tracking_params(url)

    def test_preserves_legitimate_params(self):
        url = "https://www.appther.com/search?q=odoo+erp&page=2"
        result = strip_tracking_params(url)
        assert "q=odoo" in result
        assert "page=2" in result

    def test_no_change_when_no_params(self):
        url = "https://www.appther.com/faq"
        assert strip_tracking_params(url) == url

    def test_mixed_tracking_and_real_params(self):
        url = "https://www.appther.com/?q=erp&utm_source=google&page=1"
        result = strip_tracking_params(url)
        assert "utm_source" not in result
        assert "q=erp" in result
        assert "page=1" in result


class TestNormalizeUrl:
    def test_lowercases_scheme(self):
        url = "HTTPS://www.appther.com/"
        assert normalize_url(url).startswith("https://")

    def test_lowercases_host(self):
        url = "https://WWW.APPTHER.COM/page"
        assert "www.appther.com" in normalize_url(url)

    def test_strips_tracking_params(self):
        url = "https://www.appther.com/services/?utm_source=google"
        result = normalize_url(url)
        assert "utm_source" not in result

    def test_drops_fragment(self):
        url = "https://www.appther.com/faq#section-3"
        assert "#" not in normalize_url(url)

    def test_stable_output_called_twice(self):
        url = "https://www.appther.com/services/odoo-erp/"
        assert normalize_url(url) == normalize_url(normalize_url(url))

    def test_preserves_path(self):
        url = "https://www.appther.com/case-study/odoo-retail/"
        assert "/case-study/odoo-retail/" in normalize_url(url)


class TestInferPageType:
    @pytest.mark.parametrize(
        "path, expected",
        [
            ("/", "home"),
            ("/faq", "faq"),
            ("/faq/", "faq"),
            ("/case-study/odoo-retail/", "case-study"),
            ("/blog/some-post/", "blog"),
            ("/services/odoo-erp/", "service"),
            ("/industry/retail/", "industry"),
            ("/industries/retail/", "industry"),  # legacy path
            ("/hire-react-developer/", "hire"),
            ("/hire-python-developer", "hire"),
            ("/privacy-policy", "legal"),
            ("/terms", "legal"),
            ("/cookie-policy", "legal"),
            ("/about", "company"),
            ("/contact", "company"),
            ("/team", "company"),
            ("/some-random-page", "other"),
        ],
    )
    def test_page_type_patterns(self, path: str, expected: str):
        url = f"https://www.appther.com{path}"
        assert infer_page_type(url) == expected

    def test_full_url_input(self):
        assert infer_page_type("https://www.appther.com/faq") == "faq"

    def test_unknown_path_returns_other(self):
        assert infer_page_type("https://www.appther.com/some-unknown-slug") == "other"


class TestIsInternal:
    BASE = "https://www.appther.com"

    def test_same_host_is_internal(self):
        assert is_internal("https://www.appther.com/services/", self.BASE) is True

    def test_different_host_is_external(self):
        assert is_internal("https://example.com/", self.BASE) is False

    def test_relative_url_is_internal(self):
        assert is_internal("/services/odoo-erp/", self.BASE) is True

    def test_subdomain_is_external(self):
        assert is_internal("https://blog.appther.com/post", self.BASE) is False

    def test_case_insensitive_host(self):
        assert is_internal("https://WWW.APPTHER.COM/faq", self.BASE) is True


class TestUrlHelpers:
    def test_url_to_slug_simple(self):
        slug = url_to_slug("https://www.appther.com/services/odoo-erp/")
        assert "services" in slug
        assert "odoo" in slug

    def test_url_to_slug_homepage(self):
        assert url_to_slug("https://www.appther.com/") == "index"

    def test_url_to_hash_is_16_chars(self):
        h = url_to_hash("https://www.appther.com/faq")
        assert len(h) == 16

    def test_url_to_hash_is_stable(self):
        url = "https://www.appther.com/services/"
        assert url_to_hash(url) == url_to_hash(url)

    def test_url_to_hash_different_urls_differ(self):
        assert url_to_hash("https://www.appther.com/faq") != url_to_hash(
            "https://www.appther.com/services/"
        )

    def test_url_to_hash_normalizes_before_hashing(self):
        # Tracking params should not affect the hash
        url_clean = "https://www.appther.com/services/"
        url_tracked = "https://www.appther.com/services/?utm_source=google"
        assert url_to_hash(url_clean) == url_to_hash(url_tracked)
