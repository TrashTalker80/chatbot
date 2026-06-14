"""Data models for the crawler pipeline."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass
class DiscoveredURL:
    """A URL discovered during the crawl, with optional sitemap metadata."""

    url: str
    lastmod: str | None = None
    priority: float | None = None
    changefreq: str | None = None
    # sitemap | blog-sitemap | bfs | overview — used for refresh cadence decisions
    source: str = "sitemap"
    page_type: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DiscoveredURL:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, line: str) -> DiscoveredURL:
        return cls.from_dict(json.loads(line))


@dataclass
class FetchResult:
    """The outcome of fetching a single URL."""

    url: str  # original requested URL (as queued)
    final_url: str  # after following all redirects
    status_code: int
    html: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    # "httpx" or "playwright" — recorded so Step 2 knows how the page was rendered
    render_method: str = "httpx"
    fetched_at: str = ""  # ISO 8601
    error: str | None = None

    @property
    def ok(self) -> bool:
        """True when the page was fetched successfully with content."""
        return self.status_code == 200 and self.html is not None

    @property
    def is_permanent_error(self) -> bool:
        """4xx errors (except 429) should not be retried."""
        return 400 <= self.status_code < 500 and self.status_code != 429

    @property
    def is_transient_error(self) -> bool:
        """5xx + 429 + network errors should be retried with backoff."""
        return self.status_code in {429, 500, 502, 503, 504} or bool(self.error)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> FetchResult:
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})
