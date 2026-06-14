"""HTTP page fetcher with retry, polite delay, and optional Playwright fallback.

Design:
- fetch_page()    — fetch a single URL; returns FetchResult.
- fetch_all()     — fetch a list of DiscoveredURLs with inter-request delay.
- Retry policy    — exponential backoff on 429 / 5xx / network errors; respects
                    Retry-After header. 4xx (except 429) are permanent failures,
                    not retried.
- Playwright flag — per-URL opt-in for JS-rendered pages. The flag must be
                    explicitly set in the caller (discovery decides which pages
                    need it based on the render spike findings in crawler/README.md).
- Tracking params are stripped from the URL before every request so we never
  fetch duplicate content via decorated URLs.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import httpx

from crawler.config import BACKOFF_BASE, DEFAULT_CRAWL_DELAY_SECONDS, MAX_RETRIES
from crawler.models import DiscoveredURL, FetchResult
from crawler.urls import normalize_url, strip_tracking_params

logger = logging.getLogger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_page(
    url: str,
    client: httpx.Client,
    use_playwright: bool = False,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE,
) -> FetchResult:
    """Fetch a single URL and return a FetchResult.

    The URL is normalized and tracking params are stripped before the request.
    The final URL (after redirects) is recorded in FetchResult.final_url.
    """
    clean_url = strip_tracking_params(url)

    if use_playwright:
        logger.debug("fetch_page[playwright]: %s", clean_url)
        return _fetch_with_playwright(clean_url)

    logger.debug("fetch_page[httpx]: %s", clean_url)
    return _fetch_with_httpx(clean_url, client, max_retries=max_retries, backoff_base=backoff_base)


def fetch_all(
    urls: list[DiscoveredURL],
    client: httpx.Client,
    crawl_delay: float = DEFAULT_CRAWL_DELAY_SECONDS,
    playwright_urls: set[str] | None = None,
    max_retries: int = MAX_RETRIES,
) -> list[FetchResult]:
    """Fetch a list of DiscoveredURLs sequentially with a polite inter-request delay.

    Args:
        urls:            URLs to fetch (from discovery).
        client:          Shared httpx.Client (one TCP connection pool for all requests).
        crawl_delay:     Seconds to sleep between requests; use robots.crawl_delay() here.
        playwright_urls: Set of normalized URLs that require Playwright rendering.
        max_retries:     Per-request retry budget.

    Returns:
        List of FetchResult in the same order as *urls*.
    """
    playwright_set = playwright_urls or set()
    results: list[FetchResult] = []

    for i, discovered in enumerate(urls):
        use_playwright = normalize_url(discovered.url) in playwright_set
        result = fetch_page(
            discovered.url,
            client,
            use_playwright=use_playwright,
            max_retries=max_retries,
        )
        results.append(result)

        level = logging.DEBUG if result.ok else logging.WARNING
        logger.log(
            level,
            "[%d/%d] %s → HTTP %d%s",
            i + 1,
            len(urls),
            discovered.url,
            result.status_code,
            f" (error: {result.error})" if result.error else "",
        )

        if i < len(urls) - 1:
            time.sleep(crawl_delay)

    ok_count = sum(1 for r in results if r.ok)
    logger.info("fetch_all: %d/%d pages fetched successfully", ok_count, len(results))
    return results


# ── httpx implementation ──────────────────────────────────────────────────────


def _fetch_with_httpx(
    url: str,
    client: httpx.Client,
    max_retries: int = MAX_RETRIES,
    backoff_base: float = BACKOFF_BASE,
) -> FetchResult:
    """Fetch with httpx, retrying on transient errors."""
    last_result: FetchResult | None = None

    for attempt in range(max_retries + 1):
        fetched_at = datetime.now(UTC).isoformat()
        try:
            response = client.get(url)
            result = FetchResult(
                url=url,
                final_url=normalize_url(str(response.url)),
                status_code=response.status_code,
                html=response.text if response.status_code == 200 else None,
                headers=dict(response.headers),
                render_method="httpx",
                fetched_at=fetched_at,
            )
        except httpx.TimeoutException as exc:
            result = FetchResult(
                url=url,
                final_url=url,
                status_code=0,
                error=f"timeout: {exc}",
                render_method="httpx",
                fetched_at=fetched_at,
            )
        except httpx.NetworkError as exc:
            result = FetchResult(
                url=url,
                final_url=url,
                status_code=0,
                error=f"network: {exc}",
                render_method="httpx",
                fetched_at=fetched_at,
            )
        except httpx.HTTPError as exc:
            result = FetchResult(
                url=url,
                final_url=url,
                status_code=0,
                error=f"http: {exc}",
                render_method="httpx",
                fetched_at=fetched_at,
            )

        last_result = result

        if result.is_permanent_error:
            logger.debug("Permanent error HTTP %d for %s — not retrying", result.status_code, url)
            return result

        if not result.is_transient_error:
            # Success or unexpected state — return immediately
            return result

        if attempt >= max_retries:
            break

        wait = _retry_wait(
            result.headers.get("retry-after") if result.headers else None,
            attempt,
            backoff_base,
        )
        logger.warning(
            "Transient error (HTTP %d / %s) for %s — retry %d/%d in %.1fs",
            result.status_code,
            result.error or "n/a",
            url,
            attempt + 1,
            max_retries,
            wait,
        )
        time.sleep(wait)

    return last_result  # type: ignore[return-value]


def _retry_wait(retry_after: str | None, attempt: int, backoff_base: float) -> float:
    """Compute how long to wait before the next retry.

    Respects the Retry-After response header when present; otherwise uses
    exponential backoff: backoff_base^attempt seconds.
    """
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            from email.utils import parsedate_to_datetime

            try:
                delta = parsedate_to_datetime(retry_after) - datetime.now(UTC)
                return max(0.0, delta.total_seconds())
            except (TypeError, ValueError):
                pass
    return backoff_base**attempt


# ── Playwright fallback ───────────────────────────────────────────────────────


def _fetch_with_playwright(url: str) -> FetchResult:
    """Fetch a URL using headless Chromium via Playwright.

    This is the fallback for pages that are genuinely JS-only. Based on the
    render spike (see crawler/README.md), appther.com pages render cleanly
    without JS execution, so this path is effectively dead code for the
    current corpus. It is kept for pages that might change in future crawls.

    Raises ImportError if playwright is not installed (install with
    `pip install playwright && playwright install chromium`).
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return FetchResult(
            url=url,
            final_url=url,
            status_code=0,
            error="playwright not installed; run: pip install playwright && playwright install chromium",  # noqa: E501
            render_method="playwright",
            fetched_at=datetime.now(UTC).isoformat(),
        )

    fetched_at = datetime.now(UTC).isoformat()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            status = response.status if response else 0
            html = page.content()
            final_url = page.url
            browser.close()

        return FetchResult(
            url=url,
            final_url=normalize_url(final_url),
            status_code=status,
            html=html if status == 200 else None,
            render_method="playwright",
            fetched_at=fetched_at,
        )
    except Exception as exc:
        logger.error("Playwright fetch failed for %s: %s", url, exc)
        return FetchResult(
            url=url,
            final_url=url,
            status_code=0,
            error=str(exc),
            render_method="playwright",
            fetched_at=fetched_at,
        )
