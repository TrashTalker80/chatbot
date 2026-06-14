"""Tests for crawler.fetch — retry logic, redirect handling, error cases.

Uses pytest-httpx to intercept all httpx calls without real network I/O.
time.sleep is patched throughout to keep the test suite fast.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
from pytest_httpx import HTTPXMock

from crawler.fetch import _retry_wait, fetch_all, fetch_page
from crawler.http_client import create_client
from crawler.models import DiscoveredURL, FetchResult

# ── fetch_page (single URL) ───────────────────────────────────────────────────


class TestFetchPage:
    def test_success_returns_html(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url="https://www.appther.com/faq", text="<html>FAQ</html>")
        with create_client() as client:
            result = fetch_page("https://www.appther.com/faq", client)
        assert result.ok
        assert result.status_code == 200
        assert result.html == "<html>FAQ</html>"

    def test_final_url_after_redirect(self, httpx_mock: HTTPXMock):
        # /industries/ redirects to /industry/
        httpx_mock.add_response(
            url="https://www.appther.com/industries/",
            status_code=301,
            headers={"Location": "https://www.appther.com/industry/"},
        )
        httpx_mock.add_response(
            url="https://www.appther.com/industry/",
            text="<html>Industry</html>",
        )
        with create_client() as client:
            result = fetch_page("https://www.appther.com/industries/", client)
        assert result.ok
        assert "/industry/" in result.final_url
        assert result.url == "https://www.appther.com/industries/"

    def test_404_is_permanent_error_no_retry(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url="https://www.appther.com/gone", status_code=404)
        with create_client() as client, patch("crawler.fetch.time.sleep") as mock_sleep:
            result = fetch_page("https://www.appther.com/gone", client, max_retries=3)
        assert result.status_code == 404
        assert not result.ok
        assert result.is_permanent_error
        mock_sleep.assert_not_called()

    def test_429_retries_then_succeeds(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://www.appther.com/services/",
            status_code=429,
            headers={"Retry-After": "0"},
        )
        httpx_mock.add_response(
            url="https://www.appther.com/services/",
            text="<html>Services</html>",
        )
        with create_client() as client, patch("crawler.fetch.time.sleep"):
            result = fetch_page("https://www.appther.com/services/", client, max_retries=3)
        assert result.ok
        assert result.status_code == 200

    def test_500_retries_then_succeeds(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url="https://www.appther.com/page", status_code=500)
        httpx_mock.add_response(url="https://www.appther.com/page", text="<html>OK</html>")
        with create_client() as client, patch("crawler.fetch.time.sleep"):
            result = fetch_page("https://www.appther.com/page", client, max_retries=3)
        assert result.ok

    def test_500_exhausts_retries(self, httpx_mock: HTTPXMock):
        # 4 consecutive 500s (1 initial + 3 retries)
        for _ in range(4):
            httpx_mock.add_response(url="https://www.appther.com/down", status_code=500)
        with create_client() as client, patch("crawler.fetch.time.sleep"):
            result = fetch_page("https://www.appther.com/down", client, max_retries=3)
        assert result.status_code == 500
        assert not result.ok

    def test_tracking_params_stripped_before_request(self, httpx_mock: HTTPXMock):
        # The mock expects the clean URL; if tracking params were NOT stripped,
        # the request would go to a different URL and pytest-httpx would raise.
        httpx_mock.add_response(
            url="https://www.appther.com/services/",
            text="<html>Services</html>",
        )
        with create_client() as client:
            result = fetch_page("https://www.appther.com/services/?utm_source=google", client)
        assert result.ok

    def test_fetched_at_is_populated(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url="https://www.appther.com/faq", text="<html>FAQ</html>")
        with create_client() as client:
            result = fetch_page("https://www.appther.com/faq", client)
        assert result.fetched_at != ""
        # Should be a valid ISO timestamp
        assert "T" in result.fetched_at or "-" in result.fetched_at

    def test_render_method_is_httpx(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url="https://www.appther.com/faq", text="<html>FAQ</html>")
        with create_client() as client:
            result = fetch_page("https://www.appther.com/faq", client)
        assert result.render_method == "httpx"

    def test_ok_false_for_404(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(url="https://www.appther.com/gone", status_code=404)
        with create_client() as client:
            result = fetch_page("https://www.appther.com/gone", client)
        assert not result.ok

    def test_headers_captured(self, httpx_mock: HTTPXMock):
        httpx_mock.add_response(
            url="https://www.appther.com/faq",
            text="<html>FAQ</html>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )
        with create_client() as client:
            result = fetch_page("https://www.appther.com/faq", client)
        assert "content-type" in {k.lower() for k in result.headers}


# ── Network errors ────────────────────────────────────────────────────────────


class TestNetworkErrors:
    def test_timeout_retries(self, httpx_mock: HTTPXMock):
        httpx_mock.add_exception(
            httpx.TimeoutException("connect timeout"),
            url="https://www.appther.com/slow",
        )
        httpx_mock.add_response(url="https://www.appther.com/slow", text="<html>OK</html>")
        with create_client() as client, patch("crawler.fetch.time.sleep"):
            result = fetch_page("https://www.appther.com/slow", client, max_retries=2)
        assert result.ok

    def test_timeout_exhausts_retries(self, httpx_mock: HTTPXMock):
        for _ in range(3):  # initial + 2 retries
            httpx_mock.add_exception(
                httpx.TimeoutException("timeout"),
                url="https://www.appther.com/dead",
            )
        with create_client() as client, patch("crawler.fetch.time.sleep"):
            result = fetch_page("https://www.appther.com/dead", client, max_retries=2)
        assert not result.ok
        assert result.error is not None
        assert "timeout" in result.error.lower()

    def test_network_error_returns_error_result(self, httpx_mock: HTTPXMock):
        httpx_mock.add_exception(
            httpx.NetworkError("connection refused"),
            url="https://www.appther.com/unreachable",
        )
        with create_client() as client, patch("crawler.fetch.time.sleep"):
            result = fetch_page("https://www.appther.com/unreachable", client, max_retries=0)
        assert not result.ok
        assert result.error is not None


# ── Retry-After header ────────────────────────────────────────────────────────


class TestRetryWait:
    def test_respects_retry_after_header(self):
        wait = _retry_wait(retry_after="5", attempt=0, backoff_base=2.0)
        assert wait == 5.0

    def test_falls_back_to_backoff_when_no_header(self):
        wait = _retry_wait(retry_after=None, attempt=2, backoff_base=2.0)
        assert wait == 4.0  # 2 ** 2

    def test_invalid_retry_after_falls_back(self):
        wait = _retry_wait(retry_after="invalid", attempt=1, backoff_base=2.0)
        assert wait == 2.0  # 2 ** 1

    def test_zero_retry_after(self):
        wait = _retry_wait(retry_after="0", attempt=0, backoff_base=2.0)
        assert wait == 0.0


# ── fetch_all (batch) ─────────────────────────────────────────────────────────


class TestFetchAll:
    def test_fetches_all_urls(self, httpx_mock: HTTPXMock):
        urls = [
            DiscoveredURL(url="https://www.appther.com/faq", source="sitemap"),
            DiscoveredURL(url="https://www.appther.com/services/odoo-erp/", source="sitemap"),
        ]
        httpx_mock.add_response(url="https://www.appther.com/faq", text="<html>FAQ</html>")
        httpx_mock.add_response(
            url="https://www.appther.com/services/odoo-erp/",
            text="<html>Services</html>",
        )
        with create_client() as client, patch("crawler.fetch.time.sleep"):
            results = fetch_all(urls, client, crawl_delay=0)
        assert len(results) == 2
        assert all(r.ok for r in results)

    def test_order_preserved(self, httpx_mock: HTTPXMock):
        urls = [
            DiscoveredURL(url="https://www.appther.com/page-a", source="sitemap"),
            DiscoveredURL(url="https://www.appther.com/page-b", source="sitemap"),
        ]
        httpx_mock.add_response(url="https://www.appther.com/page-a", text="<html>A</html>")
        httpx_mock.add_response(url="https://www.appther.com/page-b", text="<html>B</html>")
        with create_client() as client, patch("crawler.fetch.time.sleep"):
            results = fetch_all(urls, client, crawl_delay=0)
        assert "page-a" in results[0].url
        assert "page-b" in results[1].url

    def test_crawl_delay_applied_between_requests(self, httpx_mock: HTTPXMock):
        urls = [
            DiscoveredURL(url="https://www.appther.com/page-a", source="sitemap"),
            DiscoveredURL(url="https://www.appther.com/page-b", source="sitemap"),
        ]
        httpx_mock.add_response(url="https://www.appther.com/page-a", text="<html>A</html>")
        httpx_mock.add_response(url="https://www.appther.com/page-b", text="<html>B</html>")
        with create_client() as client, patch("crawler.fetch.time.sleep") as mock_sleep:
            fetch_all(urls, client, crawl_delay=1.5)
        # sleep should be called once (between first and second request)
        mock_sleep.assert_called_once_with(1.5)

    def test_playwright_urls_use_playwright_method(self, httpx_mock: HTTPXMock):
        urls = [DiscoveredURL(url="https://www.appther.com/js-only/", source="sitemap")]
        with (
            create_client() as client,
            patch("crawler.fetch.time.sleep"),
            patch("crawler.fetch.fetch_page") as mock_fetch,
        ):
            mock_fetch.return_value = FetchResult(
                url="https://www.appther.com/js-only/",
                final_url="https://www.appther.com/js-only/",
                status_code=200,
                html="<html>JS</html>",
                render_method="playwright",
            )
            fetch_all(
                urls,
                client,
                playwright_urls={"https://www.appther.com/js-only/"},
                crawl_delay=0,
            )
        # Verify fetch_page was called with use_playwright=True
        call_kwargs = mock_fetch.call_args
        assert call_kwargs.kwargs.get("use_playwright") is True or call_kwargs.args[2] is True


# ── FetchResult properties ────────────────────────────────────────────────────


class TestFetchResultProperties:
    def test_ok_true_for_200_with_html(self):
        r = FetchResult(
            url="https://x.com/", final_url="https://x.com/", status_code=200, html="<html>"
        )
        assert r.ok is True

    def test_ok_false_when_no_html(self):
        r = FetchResult(
            url="https://x.com/", final_url="https://x.com/", status_code=200, html=None
        )
        assert r.ok is False

    def test_is_permanent_error_for_404(self):
        r = FetchResult(url="https://x.com/", final_url="https://x.com/", status_code=404)
        assert r.is_permanent_error is True

    def test_is_permanent_error_false_for_429(self):
        r = FetchResult(url="https://x.com/", final_url="https://x.com/", status_code=429)
        assert r.is_permanent_error is False

    def test_is_transient_error_for_500(self):
        r = FetchResult(url="https://x.com/", final_url="https://x.com/", status_code=500)
        assert r.is_transient_error is True

    def test_is_transient_error_for_429(self):
        r = FetchResult(url="https://x.com/", final_url="https://x.com/", status_code=429)
        assert r.is_transient_error is True

    def test_is_transient_error_for_error_string(self):
        r = FetchResult(
            url="https://x.com/", final_url="https://x.com/", status_code=0, error="timeout"
        )
        assert r.is_transient_error is True
