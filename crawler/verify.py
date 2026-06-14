"""Crawl reconciliation, per-run reporting, and 404/410 de-index wiring.

After each crawl run this module:
  1. Reconciles every discovered URL against the actual fetch results and
     classifies each URL as fetched / redirected / failed_permanent /
     failed_transient / not_attempted.
  2. Emits a structured JSON report with per-run counts and a delta vs the
     previous run; raises a drop_alert flag when too many pages vanish.
  3. De-indexes URLs that returned permanent HTTP errors (404/410) from the
     LanceDB index so stale chunks are never served.
  4. Persists a failures.jsonl for targeted re-runs of only the failed URLs.

Public API
----------
  reconcile(discovered, fetch_results, robots_filtered_count, previous_report)
      → CrawlReport

  fetch_llms_txt_urls(client)
      → list[str]   (https:// links extracted from llms.txt / llms-full.txt)

  classify_crawl_cadence(url: DiscoveredURL)
      → Literal["weekly", "monthly", "yearly"]

  deindex_permanent_failures(report, index_uri, table_name)
      → list[str]   (de-indexed URLs)

  save_report(report, staging_dir)    → Path
  load_latest_report(staging_dir)     → CrawlReport | None
  save_failures(report, staging_dir)  → Path
  load_failures(staging_dir)          → list[dict]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx

from crawler.config import (
    DROP_ALERT_THRESHOLD,
    FAILURES_FILENAME,
    LANCE_TABLE_NAME,
    LLMS_FULL_URL,
    LLMS_TXT_URL,
    REPORTS_SUBDIR,
    REQUEST_TIMEOUT_SECONDS,
    STAGING_DIR,
    USER_AGENT,
)
from crawler.models import DiscoveredURL, FetchResult

logger = logging.getLogger(__name__)

UrlStatus = Literal[
    "fetched",
    "redirected",
    "failed_permanent",
    "failed_transient",
    "not_attempted",
]


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class UrlRecord:
    """Per-URL outcome of a single crawl run."""

    url: str
    status: UrlStatus
    http_code: int = 0
    final_url: str | None = None
    error: str | None = None
    suggested_cadence: str | None = None
    # Set to "deindexed" when the URL's chunks are removed from LanceDB.
    action: str | None = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "status": self.status,
            "http_code": self.http_code,
            "final_url": self.final_url,
            "error": self.error,
            "suggested_cadence": self.suggested_cadence,
            "action": self.action,
        }

    @classmethod
    def from_dict(cls, d: dict) -> UrlRecord:
        return cls(
            url=d["url"],
            status=d["status"],
            http_code=d.get("http_code", 0),
            final_url=d.get("final_url"),
            error=d.get("error"),
            suggested_cadence=d.get("suggested_cadence"),
            action=d.get("action"),
        )


@dataclass
class CrawlReport:
    """Aggregated metrics for one complete crawl run."""

    run_id: str
    # URL counts
    discovered: int
    robots_excluded: int
    fetched: int
    redirected: int
    failed_permanent: int
    failed_transient: int
    not_attempted: int
    deindexed: int
    # Delta vs previous run (None on first run)
    page_count_delta: int | None
    drop_alert: bool
    url_records: list[UrlRecord] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        """Pages that returned content this run (fetched + redirected)."""
        return self.fetched + self.redirected

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "discovered": self.discovered,
            "robots_excluded": self.robots_excluded,
            "fetched": self.fetched,
            "redirected": self.redirected,
            "failed_permanent": self.failed_permanent,
            "failed_transient": self.failed_transient,
            "not_attempted": self.not_attempted,
            "deindexed": self.deindexed,
            "page_count_delta": self.page_count_delta,
            "drop_alert": self.drop_alert,
            "url_records": [r.to_dict() for r in self.url_records],
        }

    @classmethod
    def from_dict(cls, d: dict) -> CrawlReport:
        records = [UrlRecord.from_dict(r) for r in d.get("url_records", [])]
        return cls(
            run_id=d["run_id"],
            discovered=d["discovered"],
            robots_excluded=d.get("robots_excluded", 0),
            fetched=d["fetched"],
            redirected=d["redirected"],
            failed_permanent=d["failed_permanent"],
            failed_transient=d["failed_transient"],
            not_attempted=d["not_attempted"],
            deindexed=d.get("deindexed", 0),
            page_count_delta=d.get("page_count_delta"),
            drop_alert=d["drop_alert"],
            url_records=records,
        )


# ── Core reconciliation ───────────────────────────────────────────────────────


def reconcile(
    discovered: list[DiscoveredURL],
    fetch_results: list[FetchResult],
    robots_filtered_count: int = 0,
    previous_report: CrawlReport | None = None,
) -> CrawlReport:
    """Build a CrawlReport by reconciling discovered URLs against fetch results.

    Each discovered URL is classified using FetchResult.status_code:
      - fetched          : HTTP 200 with html present, final_url == url
      - redirected       : HTTP 200 with html present, final_url != url
      - failed_permanent : 4xx (excluding 429) — should be de-indexed
      - failed_transient : 5xx / 429 / network error after all retries
      - not_attempted    : URL in discovered but absent from fetch_results
                           (e.g. the pipeline was interrupted)

    The drop_alert flag fires when the current ok_count (fetched + redirected)
    has fallen more than DROP_ALERT_THRESHOLD (5%) below the previous run's
    ok_count.  First-run reports never fire the alert.
    """
    run_id = datetime.now(UTC).isoformat()

    # Build a lookup from original requested URL → FetchResult.
    # fetch_results are keyed by FetchResult.url which equals the URL as queued
    # (tracking params already stripped by fetch_page before the request).
    fetch_by_url: dict[str, FetchResult] = {r.url: r for r in fetch_results}

    url_records: list[UrlRecord] = []
    counts: dict[str, int] = {
        "fetched": 0,
        "redirected": 0,
        "failed_permanent": 0,
        "failed_transient": 0,
        "not_attempted": 0,
    }

    for disc in discovered:
        cadence = classify_crawl_cadence(disc)
        result = fetch_by_url.get(disc.url)

        if result is None:
            counts["not_attempted"] += 1
            url_records.append(
                UrlRecord(
                    url=disc.url,
                    status="not_attempted",
                    suggested_cadence=cadence,
                )
            )
            continue

        status = _classify_fetch_result(result)
        counts[status] += 1
        url_records.append(
            UrlRecord(
                url=disc.url,
                status=status,
                http_code=result.status_code,
                final_url=result.final_url if result.final_url != disc.url else None,
                error=result.error,
                suggested_cadence=cadence,
            )
        )

    curr_ok = counts["fetched"] + counts["redirected"]
    delta, alert = _compute_delta_and_alert(curr_ok, previous_report)

    if alert:
        logger.warning(
            "DROP ALERT: ok pages dropped from %d to %d (>%.0f%% drop)",
            previous_report.ok_count if previous_report else 0,
            curr_ok,
            DROP_ALERT_THRESHOLD * 100,
        )

    report = CrawlReport(
        run_id=run_id,
        discovered=len(discovered),
        robots_excluded=robots_filtered_count,
        fetched=counts["fetched"],
        redirected=counts["redirected"],
        failed_permanent=counts["failed_permanent"],
        failed_transient=counts["failed_transient"],
        not_attempted=counts["not_attempted"],
        deindexed=0,
        page_count_delta=delta,
        drop_alert=alert,
        url_records=url_records,
    )

    _log_summary(report)
    return report


def _classify_fetch_result(result: FetchResult) -> UrlStatus:
    if result.is_transient_error:
        return "failed_transient"
    if result.is_permanent_error:
        return "failed_permanent"
    if result.ok:
        # HTTP 200 with content — check for redirect
        if result.final_url and result.final_url != result.url:
            return "redirected"
        return "fetched"
    # Any other non-200 non-error state (301/302 that was followed to a non-200
    # final page, or an unexpected code) → treat as transient if 5xx, else permanent.
    if 400 <= result.status_code < 500:
        return "failed_permanent"
    return "failed_transient"


def _compute_delta_and_alert(
    curr_ok: int,
    previous_report: CrawlReport | None,
) -> tuple[int | None, bool]:
    if previous_report is None:
        return None, False
    prev_ok = previous_report.ok_count
    delta = curr_ok - prev_ok
    if prev_ok == 0:
        return delta, False
    drop_ratio = (prev_ok - curr_ok) / prev_ok
    alert = drop_ratio > DROP_ALERT_THRESHOLD
    return delta, alert


def _log_summary(report: CrawlReport) -> None:
    logger.info(
        "Reconciliation: discovered=%d robots_excluded=%d "
        "fetched=%d redirected=%d failed_permanent=%d failed_transient=%d "
        "not_attempted=%d | drop_alert=%s",
        report.discovered,
        report.robots_excluded,
        report.fetched,
        report.redirected,
        report.failed_permanent,
        report.failed_transient,
        report.not_attempted,
        report.drop_alert,
    )


# ── llms.txt URL extraction ───────────────────────────────────────────────────

_HTTPS_URL_RE = re.compile(r"https://[^\s\)\]\>\"\']+")


def fetch_llms_txt_urls(client: httpx.Client | None = None) -> list[str]:
    """Fetch llms.txt (or llms-full.txt) and return the https:// URLs it references.

    These URLs are used as an additional cross-check in reconciliation: the
    caller can merge them with the sitemap-discovered URL list so the report
    flags any llms.txt-referenced page that was not crawled.

    Returns an empty list if the file cannot be fetched.
    """
    close_client = False
    if client is None:
        client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        close_client = True

    try:
        for url in (LLMS_FULL_URL, LLMS_TXT_URL):
            try:
                resp = client.get(url)
                if resp.status_code == 200 and resp.text.strip():
                    found = _HTTPS_URL_RE.findall(resp.text)
                    # Strip trailing punctuation that may have been captured
                    cleaned = [u.rstrip(".,;:)]}") for u in found]
                    unique = list(dict.fromkeys(cleaned))
                    logger.info(
                        "fetch_llms_txt_urls: found %d unique URLs from %s",
                        len(unique),
                        url,
                    )
                    return unique
            except httpx.HTTPError as exc:
                logger.debug("fetch_llms_txt_urls: could not fetch %s: %s", url, exc)
    finally:
        if close_client:
            client.close()

    return []


# ── Cadence classification ────────────────────────────────────────────────────


def classify_crawl_cadence(url: DiscoveredURL) -> Literal["weekly", "monthly", "yearly"]:
    """Map changefreq / priority to a suggested re-crawl cadence.

    This is purely informational — it appears in the per-URL report entry
    and does not affect which URLs are crawled on any given run (all URLs
    are always attempted; the index's content_hash handles cost efficiency).
    """
    freq = (url.changefreq or "").lower()
    if freq in {"always", "hourly", "daily", "weekly"}:
        return "weekly"
    if freq == "monthly":
        return "monthly"
    if freq in {"yearly", "never"}:
        return "yearly"

    # Fall back to priority
    p = url.priority
    if p is not None:
        if p >= 0.7:
            return "weekly"
        if p >= 0.4:
            return "monthly"
        return "yearly"

    return "weekly"  # safe default


# ── De-index wiring ───────────────────────────────────────────────────────────


def deindex_permanent_failures(
    report: CrawlReport,
    index_uri: str,
    table_name: str = LANCE_TABLE_NAME,
    _delete_fn=None,
) -> list[str]:
    """Remove chunks for permanently-failed URLs (404/410) from the LanceDB index.

    Only ``failed_permanent`` URLs are de-indexed — transient failures are
    retried on the next run and must not lose their index entries.

    Annotates each affected UrlRecord with action="deindexed" and updates
    report.deindexed in-place.

    Args:
        _delete_fn: Callable with the same signature as
            ``crawler.index.delete_chunks_for_urls``.  Injected in tests to
            avoid a real LanceDB/boto3 dependency; leave as None in production.

    Returns the list of de-indexed URLs.
    """
    if _delete_fn is None:
        from crawler.index import delete_chunks_for_urls as _delete_fn

    urls_to_deindex = [rec.url for rec in report.url_records if rec.status == "failed_permanent"]

    if not urls_to_deindex:
        return []

    deleted = _delete_fn(urls_to_deindex, uri=index_uri, table_name=table_name)
    logger.info(
        "deindex_permanent_failures: removed %d chunks for %d URLs",
        deleted,
        len(urls_to_deindex),
    )

    # Annotate records and update report counter
    for rec in report.url_records:
        if rec.url in set(urls_to_deindex):
            rec.action = "deindexed"
    report.deindexed = len(urls_to_deindex)

    return urls_to_deindex


# ── Report persistence ────────────────────────────────────────────────────────


def save_report(report: CrawlReport, staging_dir: str = STAGING_DIR) -> Path:
    """Write the report as a timestamped JSON file under <staging_dir>/reports/.

    The run_id (ISO 8601) is used in the filename so reports sort chronologically.
    Returns the path written.
    """
    reports_dir = Path(staging_dir) / REPORTS_SUBDIR
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize run_id for use in a filename (replace colons and dots)
    safe_id = report.run_id.replace(":", "-").replace("+", "").replace(".", "-")
    path = reports_dir / f"report_{safe_id}.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved crawl report to %s", path)
    return path


def load_latest_report(staging_dir: str = STAGING_DIR) -> CrawlReport | None:
    """Return the most recent CrawlReport from <staging_dir>/reports/, or None.

    Reports are sorted lexicographically by filename; since filenames embed the
    ISO-8601 run_id they sort chronologically.
    """
    reports_dir = Path(staging_dir) / REPORTS_SUBDIR
    if not reports_dir.exists():
        return None

    candidates = sorted(reports_dir.glob("report_*.json"))
    if not candidates:
        return None

    latest = candidates[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        return CrawlReport.from_dict(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load latest report from %s: %s", latest, exc)
        return None


# ── Failures persistence ──────────────────────────────────────────────────────


def save_failures(report: CrawlReport, staging_dir: str = STAGING_DIR) -> Path:
    """Persist URLs that failed (permanently or transiently) to <staging_dir>/failures.jsonl.

    The file is overwritten on every run so it always reflects the most recent
    set of failures.  The pipeline's --targeted flag reads this file to re-fetch
    only the failing URLs.
    """
    failures_path = Path(staging_dir) / FAILURES_FILENAME
    failures_path.parent.mkdir(parents=True, exist_ok=True)

    failure_statuses = {"failed_permanent", "failed_transient", "not_attempted"}
    entries = [
        {
            "url": rec.url,
            "reason": rec.status,
            "status_code": rec.http_code,
            "error": rec.error,
        }
        for rec in report.url_records
        if rec.status in failure_statuses
    ]

    with failures_path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")

    logger.info(
        "Saved %d failure entries to %s",
        len(entries),
        failures_path,
    )
    return failures_path


def load_failures(staging_dir: str = STAGING_DIR) -> list[dict]:
    """Load the failures list from the previous run.

    Returns a list of dicts with keys: url, reason, status_code, error.
    Returns an empty list if the file does not exist.
    """
    failures_path = Path(staging_dir) / FAILURES_FILENAME
    if not failures_path.exists():
        return []

    entries: list[dict] = []
    with failures_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed failures entry: %s", exc)
    return entries
