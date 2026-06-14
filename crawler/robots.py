"""robots.txt compliance layer.

Design notes:
- RobotsChecker is lazy-loaded: the first call to is_allowed() or crawl_delay()
  triggers a fetch unless load() was called explicitly.
- load(content=...) accepts a pre-fetched string to enable offline testing
  without patching urllib.
- We always check using USER_AGENT first, then fall back to "*".
  appther.com explicitly allows GPTBot and ClaudeBot, so the check for our
  custom agent will hit the fallback "*" rules, which is fine.
"""

from __future__ import annotations

import logging
import urllib.robotparser

import httpx

from crawler.config import DEFAULT_CRAWL_DELAY_SECONDS, ROBOTS_URL, USER_AGENT

logger = logging.getLogger(__name__)


class RobotsChecker:
    def __init__(
        self,
        robots_url: str = ROBOTS_URL,
        user_agent: str = USER_AGENT,
    ) -> None:
        self._robots_url = robots_url
        self._user_agent = user_agent
        self._parser: urllib.robotparser.RobotFileParser | None = None

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self, content: str | None = None, client: httpx.Client | None = None) -> None:
        """Parse robots.txt.

        Args:
            content: Raw robots.txt text. When given, no network call is made.
            client:  httpx.Client to use for fetching. Falls back to stdlib
                     urllib when None and content is also None.
        """
        parser = urllib.robotparser.RobotFileParser()
        if content is not None:
            parser.parse(content.splitlines())
        elif client is not None:
            try:
                response = client.get(self._robots_url)
                parser.parse(response.text.splitlines())
            except Exception as exc:
                logger.warning("Failed to fetch robots.txt via httpx (%s) — allowing all", exc)
                parser.parse([])
        else:
            try:
                parser.set_url(self._robots_url)
                parser.read()
            except Exception as exc:
                logger.warning("Failed to fetch robots.txt via urllib (%s) — allowing all", exc)
                parser.parse([])
        self._parser = parser

    @property
    def _get_parser(self) -> urllib.robotparser.RobotFileParser:
        if self._parser is None:
            self.load()
        return self._parser

    # ── Public API ────────────────────────────────────────────────────────────

    def is_allowed(self, url: str) -> bool:
        """Return True if the URL may be crawled according to robots.txt."""
        try:
            return self._get_parser.can_fetch(self._user_agent, url)
        except Exception as exc:
            logger.warning("robots check raised for %s: %s — allowing", url, exc)
            return True

    def crawl_delay(self) -> float:
        """Return the Crawl-delay from robots.txt, or the configured default."""
        try:
            delay = self._get_parser.crawl_delay(self._user_agent)
            if delay is None:
                delay = self._get_parser.crawl_delay("*")
            return float(delay) if delay is not None else DEFAULT_CRAWL_DELAY_SECONDS
        except Exception:
            return DEFAULT_CRAWL_DELAY_SECONDS
