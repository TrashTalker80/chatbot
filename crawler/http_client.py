"""httpx client factory.

Creates an httpx.Client with the project's standard headers and timeouts.
Keeping construction in one place makes it easy to swap settings globally
and to mock in tests (pytest-httpx intercepts all clients transparently).
"""

from __future__ import annotations

import httpx

from crawler.config import REQUEST_TIMEOUT_SECONDS, USER_AGENT

# Accept header that resembles a real browser enough to avoid bot-detection
# on marketing sites, while still using our honest User-Agent string.
_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
_ACCEPT_LANGUAGE = "en-US,en;q=0.9"


def create_client(
    user_agent: str = USER_AGENT,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    follow_redirects: bool = True,
    max_redirects: int = 10,
) -> httpx.Client:
    """Return a configured httpx.Client.

    The client is intentionally synchronous: the crawler is a batch job
    (not a web server), and sync keeps the code simple while still
    supporting proper connection pooling and keep-alive.
    """
    return httpx.Client(
        headers={
            "User-Agent": user_agent,
            "Accept": _ACCEPT,
            "Accept-Language": _ACCEPT_LANGUAGE,
        },
        timeout=httpx.Timeout(timeout),
        follow_redirects=follow_redirects,
        max_redirects=max_redirects,
        limits=httpx.Limits(
            max_connections=10,
            max_keepalive_connections=5,
            keepalive_expiry=30,
        ),
    )
