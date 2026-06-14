"""Persist crawl outputs to local filesystem staging area.

Directory layout:
  <staging_dir>/
    discovery/
      discovered_urls.jsonl   ← one DiscoveredURL JSON per line
    raw/
      <url-hash>.html         ← raw HTML from FetchResult
      <url-hash>.meta.json    ← FetchResult metadata (no html field)

The staging dir is a durable checkpoint between pipeline stages:
  discovery.py → staging → fetch.py → staging → clean/chunk (Step 2)

S3 sync is handled in Step 3 when the LanceDB index is built.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from crawler.config import STAGING_DIR
from crawler.models import DiscoveredURL, FetchResult
from crawler.urls import url_to_hash

logger = logging.getLogger(__name__)


# ── Discovery persistence ─────────────────────────────────────────────────────


def save_discovery(
    urls: list[DiscoveredURL],
    output: str | None = None,
    staging_dir: str = STAGING_DIR,
) -> Path:
    """Write discovered URLs to a JSONL file.

    Returns the path that was written.
    """
    path = Path(output) if output else Path(staging_dir) / "discovery" / "discovered_urls.jsonl"

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        for url in urls:
            fh.write(url.to_json() + "\n")

    logger.info("Saved %d discovered URLs to %s", len(urls), path)
    return path


def load_discovery(
    output: str | None = None,
    staging_dir: str = STAGING_DIR,
) -> list[DiscoveredURL]:
    """Load previously saved discovered URLs from JSONL."""
    path = Path(output) if output else Path(staging_dir) / "discovery" / "discovered_urls.jsonl"
    if not path.exists():
        logger.warning("Discovery file not found: %s", path)
        return []
    with path.open(encoding="utf-8") as fh:
        return [DiscoveredURL.from_json(line) for line in fh if line.strip()]


# ── Raw fetch persistence ─────────────────────────────────────────────────────


def save_fetch_result(
    result: FetchResult,
    staging_dir: str = STAGING_DIR,
) -> Path:
    """Persist a FetchResult to disk.

    Writes two files:
      raw/<hash>.html        — full HTML content (only when result.ok)
      raw/<hash>.meta.json   — all fields except html (always written)

    Returns the path to the .meta.json file.
    """
    raw_dir = Path(staging_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    file_id = url_to_hash(result.url)

    # Metadata (without html to keep it small)
    meta = result.to_dict()
    meta.pop("html", None)
    meta_path = raw_dir / f"{file_id}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # HTML content
    if result.ok and result.html:
        html_path = raw_dir / f"{file_id}.html"
        html_path.write_text(result.html, encoding="utf-8")

    logger.debug("Staged %s → %s", result.url, meta_path)
    return meta_path


def load_fetch_result(url: str, staging_dir: str = STAGING_DIR) -> FetchResult | None:
    """Load a previously staged FetchResult by URL."""
    raw_dir = Path(staging_dir) / "raw"
    file_id = url_to_hash(url)
    meta_path = raw_dir / f"{file_id}.meta.json"

    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    html_path = raw_dir / f"{file_id}.html"
    html = html_path.read_text(encoding="utf-8") if html_path.exists() else None
    return FetchResult(**meta, html=html)


def list_staged_pages(staging_dir: str = STAGING_DIR) -> list[Path]:
    """Return paths to all .meta.json files in the raw staging directory."""
    raw_dir = Path(staging_dir) / "raw"
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.glob("*.meta.json"))
