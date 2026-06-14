"""URL utilities: normalization, tracking-param stripping, page-type inference."""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from crawler.config import BASE_URL, PAGE_TYPE_PATTERNS, TRACKING_PARAMS


def strip_tracking_params(url: str) -> str:
    """Remove known tracking query parameters from a URL.

    Strips any param in TRACKING_PARAMS and any param whose name starts with
    'utm_' (catches non-standard UTM variants not in the static set).
    """
    parsed = urlparse(url)
    if not parsed.query:
        return url
    params = parse_qs(parsed.query, keep_blank_values=True)
    clean = {k: v for k, v in params.items() if not _is_tracking_param(k)}
    clean_query = urlencode(clean, doseq=True)
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, clean_query, parsed.fragment)
    )


def _is_tracking_param(name: str) -> bool:
    lower = name.lower()
    return lower in TRACKING_PARAMS or lower.startswith("utm_")


def normalize_url(url: str) -> str:
    """Canonical form: lowercase scheme+host, stripped tracking params, no fragment.

    Does NOT follow redirects — that happens in fetch.py. Call this to
    produce a stable key for deduplication before any network call.
    """
    parsed = urlparse(url)
    # Lowercase scheme and host for stable comparison
    normalized = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            "",  # drop fragment — fragments are client-side only
        )
    )
    return strip_tracking_params(normalized)


def is_internal(url: str, base: str = BASE_URL) -> bool:
    """Return True if *url* belongs to the same host as *base*."""
    base_host = urlparse(base).netloc.lower()
    url_parsed = urlparse(url)
    if not url_parsed.netloc:
        # Relative URL — always internal
        return True
    return url_parsed.netloc.lower() == base_host


def infer_page_type(url: str) -> str:
    """Map a URL path to a page_type tag using the configured patterns.

    Returns "other" when no pattern matches. The tag is stored in chunk
    metadata for retrieval filtering and citation display.
    """
    path = urlparse(url).path
    for pattern, page_type in PAGE_TYPE_PATTERNS:
        if re.match(pattern, path):
            return page_type
    return "other"


def url_to_slug(url: str) -> str:
    """Produce a short, human-readable slug from a URL path.

    Used as a display name in logs and as the directory name under staging/raw/.
    """
    path = urlparse(url).path.strip("/")
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", path)[:120] if path else "index"
    return slug or "index"


def url_to_hash(url: str) -> str:
    """Stable 16-char hex ID for a URL — used as a unique staging filename."""
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()[:16]
