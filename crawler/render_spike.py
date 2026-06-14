"""Render spike: compare httpx vs Playwright text extraction on appther.com.

Run this BEFORE building the full ingestion pipeline to decide whether
headless JS rendering is actually needed. If the text captured by plain
httpx is substantially complete, Playwright is the exception (not the rule).

Usage:
  python -m crawler.render_spike                    # uses default PROBE_URLS
  python -m crawler.render_spike --output report.md
  python -m crawler.render_spike --url /faq --url /services/odoo-erp/

Findings should be pasted into the "## Render spike" section of crawler/README.md.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from crawler.config import BASE_URL, USER_AGENT
from crawler.http_client import create_client

logger = logging.getLogger(__name__)

# Representative pages: one JS-heavy candidate (homepage), one content page,
# one FAQ accordion (most likely to need rendering if answers are JS-loaded).
PROBE_URLS: list[str] = [
    "/",
    "/services/odoo-erp/",
    "/faq",
]


@dataclass
class ProbeResult:
    url: str
    httpx_chars: int
    httpx_key_phrases: list[str]
    playwright_chars: int | None
    playwright_key_phrases: list[str] | None
    playwright_available: bool

    @property
    def needs_playwright(self) -> bool:
        """Heuristic: Playwright finds >20% more text than httpx."""
        if not self.playwright_available or self.playwright_chars is None:
            return False
        ratio = self.playwright_chars / max(self.httpx_chars, 1)
        return ratio > 1.20

    def summary_line(self) -> str:
        pw_str = (
            f"{self.playwright_chars:,} chars"
            if self.playwright_chars is not None
            else "N/A (playwright not installed)"
        )
        verdict = "⚠ playwright gains >20%" if self.needs_playwright else "✓ httpx sufficient"
        return (
            f"  URL: {self.url}\n"
            f"    httpx:      {self.httpx_chars:,} chars\n"
            f"    playwright: {pw_str}\n"
            f"    verdict:    {verdict}"
        )


def _visible_text(html: str) -> str:
    """Extract visible text from HTML, stripping nav/footer/script/style."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def _key_phrases(text: str) -> list[str]:
    """Return a short list of content-indicator phrases found in *text*."""
    indicators = [
        "odoo",
        "erp",
        "implementation",
        "consultation",
        "faq",
        "how much",
        "contact",
        "case study",
        "appther",
    ]
    lower = text.lower()
    return [p for p in indicators if p in lower]


def probe_url(full_url: str) -> ProbeResult:
    """Fetch *full_url* with httpx (and Playwright if available) and compare."""
    # httpx fetch
    with create_client() as client:
        try:
            response = client.get(full_url)
            httpx_html = response.text if response.status_code == 200 else ""
        except Exception as exc:
            logger.warning("httpx failed for %s: %s", full_url, exc)
            httpx_html = ""

    httpx_text = _visible_text(httpx_html)
    httpx_chars = len(httpx_text)
    httpx_phrases = _key_phrases(httpx_text)

    # Playwright fetch (optional)
    playwright_chars: int | None = None
    playwright_phrases: list[str] | None = None
    playwright_available = False

    try:
        from playwright.sync_api import sync_playwright

        playwright_available = True
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers={"User-Agent": USER_AGENT})
            page.goto(full_url, wait_until="networkidle", timeout=30_000)
            pw_html = page.content()
            browser.close()

        pw_text = _visible_text(pw_html)
        playwright_chars = len(pw_text)
        playwright_phrases = _key_phrases(pw_text)

    except ImportError:
        logger.info("playwright not installed — skipping browser comparison for %s", full_url)
    except Exception as exc:
        logger.warning("playwright failed for %s: %s", full_url, exc)
        playwright_available = True  # installed, but failed this specific page

    return ProbeResult(
        url=full_url,
        httpx_chars=httpx_chars,
        httpx_key_phrases=httpx_phrases,
        playwright_chars=playwright_chars,
        playwright_key_phrases=playwright_phrases,
        playwright_available=playwright_available,
    )


def run_spike(paths: list[str], base_url: str = BASE_URL) -> list[ProbeResult]:
    results = []
    for path in paths:
        full_url = urljoin(base_url, path) if not path.startswith("http") else path
        logger.info("Probing %s ...", full_url)
        results.append(probe_url(full_url))
    return results


def format_report(results: list[ProbeResult]) -> str:
    lines = [
        "## Render spike findings",
        "",
        "Comparing httpx+BeautifulSoup vs Playwright text extraction on representative pages.",
        "",
    ]
    any_needs_pw = any(r.needs_playwright for r in results)
    for r in results:
        lines.append(r.summary_line())
        lines.append("")

    if any_needs_pw:
        lines.append(
            "**Decision:** At least one page requires Playwright. "
            "Flag those URLs in `fetch.py:fetch_all(playwright_urls=...)` and run "
            "`playwright install chromium`."
        )
    else:
        lines.append(
            "**Decision:** All pages render sufficiently with plain httpx. "
            "Playwright is the exception-only fallback; not needed as default."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare httpx vs Playwright text extraction.")
    parser.add_argument(
        "--url",
        action="append",
        dest="urls",
        metavar="PATH",
        help="URL path to probe (repeatable). Defaults to PROBE_URLS.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        help="Write Markdown report to FILE instead of stdout.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    paths = args.urls or PROBE_URLS
    results = run_spike(paths)
    report = format_report(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
