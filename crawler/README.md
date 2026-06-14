# Crawler

Ingestion pipeline for appther.com.

## Render spike

Run `python -m crawler.render_spike` (with Playwright optionally installed) to compare
plain-httpx text extraction against a headless-browser render on three representative pages:
`/` (homepage), `/services/odoo-erp/` (service page), `/faq` (accordion — most likely to need JS).

**Decision rule:** if Playwright captures >20% more text than httpx on any page, flag that URL
in `fetch.py:fetch_all(playwright_urls=...)` and install `playwright install chromium`.

Expected result for appther.com: server-side rendered — httpx + BeautifulSoup is sufficient;
Playwright is the exception-only fallback, not the default.

Paste the rendered report here once the spike is run against production:

```
(run `python -m crawler.render_spike --output crawler/render_spike_report.md` and paste output)
```

## Modules (Steps 1–4)

Checkmark = implemented and tested.

| File | Status | Purpose |
|---|---|---|
| `config.py` | ✓ | Central constants: URLs, timeouts, retry params, page-type patterns |
| `models.py` | ✓ | `DiscoveredURL` and `FetchResult` dataclasses with JSON serialization |
| `urls.py` | ✓ | `strip_tracking_params`, `normalize_url`, `infer_page_type`, `url_to_slug/hash` |
| `robots.py` | ✓ | `RobotsChecker` wrapping `urllib.robotparser`; offline-loadable for tests |
| `http_client.py` | ✓ | `create_client()` factory — shared httpx.Client with headers + timeouts |
| `discovery.py` | ✓ | Sitemap index fetch + merge (`sitemap.xml` + `blog-sitemap.xml`), robots filter, BFS fallback |
| `fetch.py` | ✓ | Sequential fetch with retry/backoff; Playwright per-URL opt-in |
| `staging.py` | ✓ | JSONL + HTML disk cache for discovery and fetch results |
| `render_spike.py` | ✓ | CLI tool: compare httpx vs Playwright text coverage; run before Step 2 |
| `extract.py` | ✓ | trafilatura main-content → Markdown; FAQ-pair extractor (3 patterns) |
| `normalize.py` | ✓ | URL canonicalization, SHA-256 content hash, MinHash near-dup collapse |
| `chunk.py` | ✓ | Heading-first, 400–600 tok, 50–80 overlap; FAQ pairs → dedicated chunks; llms.txt ingestion |
| `embed.py` | ✓ | Voyage primary + Jina standby; batching; int8/512-dim |
| `index.py` | ✓ | LanceDB on S3: build, upsert, delete; model metadata pinning |
| `verify.py` | ✓ | Post-run reconciliation report, drop alert, 404/410 de-index, failures persistence |
| `pipeline.py` | ✓ | End-to-end orchestrator (Steps 1–4): discovery → fetch → embed → verify; CLI entry-point |
