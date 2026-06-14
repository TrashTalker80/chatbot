"""Full crawl pipeline orchestrator — Steps 1–4.

Orchestrates the complete ingestion and verification pipeline:
  1. Discover URLs from sitemap (or load failures.jsonl for a targeted re-run)
  2. Fetch all discovered URLs with retry/backoff
  3. Extract + normalize + deduplicate + chunk each successful page
  4. Embed chunks and upsert to LanceDB index (skipped with --skip-embed)
  5. Reconcile fetch results against expected URL set → CrawlReport
  6. De-index permanently-failed URLs (404/410) from the index
  7. Persist report to staging/reports/ and failures list to staging/failures.jsonl

Exit codes:
  0 — pipeline completed; no drop alert
  1 — pipeline completed but drop_alert is True (unexpected page-count drop)
  2 — unrecoverable error during the pipeline

CLI:
  python -m crawler.pipeline
  python -m crawler.pipeline --targeted              # re-fetch only previous failures
  python -m crawler.pipeline --skip-embed            # discover + fetch + verify only
  python -m crawler.pipeline --dry-run               # discover only, no I/O writes
  python -m crawler.pipeline --index-uri s3://...    # custom LanceDB URI
  python -m crawler.pipeline --staging-dir /tmp/s    # custom staging directory
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from crawler.config import (
    DEFAULT_CRAWL_DELAY_SECONDS,
    JINA_EMBED_DIMS,
    LANCE_JINA_TABLE_NAME,
    LANCE_TABLE_NAME,
    STAGING_DIR,
    VOYAGE_EMBED_DIMS,
)

logger = logging.getLogger(__name__)


# ── Pipeline ──────────────────────────────────────────────────────────────────


def run_pipeline(
    index_uri: str,
    staging_dir: str = STAGING_DIR,
    targeted: bool = False,
    skip_embed: bool = False,
    dry_run: bool = False,
) -> int:
    """Execute the full crawl pipeline.  Returns an exit code (0/1/2)."""

    from crawler.chunk import chunk_document, fetch_and_chunk_llms_txt
    from crawler.discovery import discover_urls
    from crawler.embed import embed_chunks, get_provider
    from crawler.extract import extract
    from crawler.fetch import fetch_all
    from crawler.http_client import create_client
    from crawler.index import upsert_chunks
    from crawler.normalize import normalize_documents
    from crawler.robots import RobotsChecker
    from crawler.staging import save_discovery, save_fetch_result
    from crawler.verify import (
        deindex_permanent_failures,
        load_failures,
        load_latest_report,
        reconcile,
        save_failures,
        save_report,
    )

    logger.info("=== Crawl pipeline starting (targeted=%s skip_embed=%s) ===", targeted, skip_embed)

    # ── Step 1 — Discover or load targeted failures ───────────────────────────
    with create_client() as client:
        robots = RobotsChecker()
        robots.load(client=client)
        crawl_delay = robots.crawl_delay() or DEFAULT_CRAWL_DELAY_SECONDS

        if targeted:
            failures = load_failures(staging_dir=staging_dir)
            if not failures:
                logger.warning("--targeted requested but failures.jsonl is empty or missing.")
            discovered_list = [_make_discovered_url_from_failure(f) for f in failures]
            logger.info("Targeted re-run: %d failure URLs loaded", len(discovered_list))
        else:
            discovered_list = discover_urls(client=client, robots=robots)
            logger.info("Discovered %d URLs from sitemap", len(discovered_list))

        if dry_run:
            logger.info("--dry-run: stopping after discovery (%d URLs)", len(discovered_list))
            print(f"[dry-run] Discovered {len(discovered_list)} URLs.")
            return 0

        if not dry_run:
            save_discovery(discovered_list, staging_dir=staging_dir)

        # ── Step 2 — Fetch ────────────────────────────────────────────────────
        logger.info("Fetching %d URLs...", len(discovered_list))
        fetch_results = fetch_all(
            discovered_list,
            client=client,
            crawl_delay=crawl_delay,
        )

        for result in fetch_results:
            save_fetch_result(result, staging_dir=staging_dir)

    ok_results = [r for r in fetch_results if r.ok]
    logger.info("%d / %d pages fetched successfully", len(ok_results), len(fetch_results))

    # ── Step 3a — Extract ─────────────────────────────────────────────────────

    disc_by_url = {d.url: d for d in discovered_list}
    triples = []
    for result in ok_results:
        extract_result = extract(result.html or "", url=result.final_url)
        if not extract_result.has_content:
            logger.debug("No content extracted from %s — skipping", result.url)
            continue
        disc = disc_by_url.get(result.url)
        meta = {
            "source": disc.source if disc else "sitemap",
            "lastmod": disc.lastmod if disc else None,
            "priority": disc.priority if disc else None,
        }
        triples.append((result, extract_result, meta))

    # ── Step 3b — Normalize + chunk ───────────────────────────────────────────
    docs = normalize_documents(triples)
    logger.info("Normalized %d documents", len(docs))

    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))

    # Append llms.txt overview chunks
    with create_client() as client:
        llms_chunks = fetch_and_chunk_llms_txt(client=client)
    all_chunks.extend(llms_chunks)
    logger.info("Total chunks: %d (including %d overview)", len(all_chunks), len(llms_chunks))

    # ── Step 3c — Embed + index ───────────────────────────────────────────────
    if not skip_embed and all_chunks:
        voyage_key = os.environ.get("VOYAGE_API_KEY", "")
        jina_key = os.environ.get("JINA_API_KEY", "")

        # Primary: Voyage
        logger.info("Embedding %d chunks via Voyage...", len(all_chunks))
        voyage_provider = get_provider("voyage", api_key=voyage_key)
        voyage_embedded = embed_chunks(all_chunks, provider=voyage_provider)
        upsert_result = upsert_chunks(
            voyage_embedded,
            uri=index_uri,
            table_name=LANCE_TABLE_NAME,
            dims=VOYAGE_EMBED_DIMS,
        )
        logger.info("Voyage index upsert: %s", upsert_result)

        # Standby: Jina (build alongside primary; separate table)
        if jina_key:
            logger.info("Embedding %d chunks via Jina standby...", len(all_chunks))
            jina_provider = get_provider("jina", api_key=jina_key)
            jina_embedded = embed_chunks(all_chunks, provider=jina_provider)
            jina_result = upsert_chunks(
                jina_embedded,
                uri=index_uri,
                table_name=LANCE_JINA_TABLE_NAME,
                dims=JINA_EMBED_DIMS,
            )
            logger.info("Jina standby upsert: %s", jina_result)
        else:
            logger.info("JINA_API_KEY not set — skipping Jina standby index.")
    elif skip_embed:
        logger.info("--skip-embed: skipping embed and index steps.")
    else:
        logger.warning("No chunks to embed — index not updated.")

    # ── Step 4 — Verify ───────────────────────────────────────────────────────
    previous_report = load_latest_report(staging_dir=staging_dir)
    report = reconcile(
        discovered=discovered_list,
        fetch_results=fetch_results,
        previous_report=previous_report,
    )

    # De-index permanent failures
    if not skip_embed:
        deindex_permanent_failures(report, index_uri=index_uri, table_name=LANCE_TABLE_NAME)
        deindex_permanent_failures(report, index_uri=index_uri, table_name=LANCE_JINA_TABLE_NAME)

    # Persist report + failures
    report_path = save_report(report, staging_dir=staging_dir)
    save_failures(report, staging_dir=staging_dir)

    logger.info("=== Pipeline complete. Report: %s ===", report_path)
    _print_summary(report)

    return 1 if report.drop_alert else 0


def _make_discovered_url_from_failure(failure: dict):
    from crawler.models import DiscoveredURL

    return DiscoveredURL(url=failure["url"])


def _print_summary(report) -> None:
    print(
        f"\nCrawl report [{report.run_id}]\n"
        f"  Discovered : {report.discovered}\n"
        f"  Fetched    : {report.fetched}\n"
        f"  Redirected : {report.redirected}\n"
        f"  Perm-failed: {report.failed_permanent}\n"
        f"  Transient  : {report.failed_transient}\n"
        f"  Not tried  : {report.not_attempted}\n"
        f"  De-indexed : {report.deindexed}\n"
        f"  Delta      : {report.page_count_delta}\n"
        f"  Drop alert : {'YES ⚠' if report.drop_alert else 'no'}\n"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the full Appther RAG ingestion pipeline (Steps 1–4)."
    )
    parser.add_argument(
        "--index-uri",
        default=os.environ.get("LANCE_INDEX_URI", "./lance_index"),
        help="LanceDB URI (local path or s3://...). Overridden by LANCE_INDEX_URI env var.",
    )
    parser.add_argument(
        "--staging-dir",
        default=os.environ.get("STAGING_DIR", STAGING_DIR),
        help="Staging directory for reports, raw HTML, and failures list.",
    )
    parser.add_argument(
        "--targeted",
        action="store_true",
        help="Re-fetch only URLs listed in staging/failures.jsonl (skip discovery).",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Discover and fetch pages but skip embedding and indexing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery only and print the URL count; no fetching or I/O.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    exit_code = run_pipeline(
        index_uri=args.index_uri,
        staging_dir=args.staging_dir,
        targeted=args.targeted,
        skip_embed=args.skip_embed,
        dry_run=args.dry_run,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
