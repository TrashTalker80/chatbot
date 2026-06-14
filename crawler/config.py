"""Central configuration for the crawler. Override via environment variables where noted."""

import os

# ── Site ──────────────────────────────────────────────────────────────────────
BASE_URL = "https://www.appther.com"
SITEMAP_URL = "https://www.appther.com/sitemap.xml"
ROBOTS_URL = "https://www.appther.com/robots.txt"
LLMS_TXT_URL = "https://www.appther.com/llms.txt"
LLMS_FULL_URL = "https://www.appther.com/llms-full.txt"

# ── HTTP ──────────────────────────────────────────────────────────────────────
USER_AGENT = "AppTherChatbotCrawler/1.0 (+https://github.com/YOUR_ORG/appther-chatbot)"
REQUEST_TIMEOUT_SECONDS: float = 30.0

# Polite inter-request delay; overridden by robots.txt Crawl-delay when present.
DEFAULT_CRAWL_DELAY_SECONDS: float = 1.0

MAX_RETRIES: int = 3
BACKOFF_BASE: float = 2.0  # wait = BACKOFF_BASE ** attempt (seconds)

# ── URL filtering ─────────────────────────────────────────────────────────────
# Tracking query params stripped from every URL before queuing or storing.
# Also strips anything matching the "utm_*" prefix even if not in this set.
TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "gclid",
        "fbclid",
        "msclkid",
    }
)

# ── Page-type patterns (regex on URL path, first match wins) ──────────────────
# Tag is stored per-chunk for metadata filtering and display in citations.
PAGE_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"^/$", "home"),
    (r"^/faq(/|$)", "faq"),
    (r"^/case-study/", "case-study"),
    (r"^/blog/", "blog"),
    (r"^/services/", "service"),
    (r"^/industry/", "industry"),
    (r"^/industries/", "industry"),  # legacy path that 301s to /industry/
    (r"^/hire-", "hire"),
    (r"^/(privacy-policy|terms|cookie|legal)", "legal"),
    (r"^/(contact|about|team|company|careers)", "company"),
]

# ── BFS fallback limits ───────────────────────────────────────────────────────
BFS_MAX_DEPTH: int = 4
BFS_MAX_URLS: int = 200

# ── Staging ───────────────────────────────────────────────────────────────────
STAGING_DIR: str = os.getenv("STAGING_DIR", "staging")

# ── Normalization / dedup (Step 2) ──────────────────────────────────────────────
MINHASH_NUM_PERM: int = 128
NEAR_DUP_THRESHOLD: float = 0.85

# ── Chunking (Step 2) ────────────────────────────────────────────────────────────
CHUNK_MIN_TOKENS: int = 400
CHUNK_MAX_TOKENS: int = 600
CHUNK_OVERLAP_TOKENS: int = 65
# Approximate characters per token for size budgeting (no tokenizer dependency).
CHARS_PER_TOKEN: int = 4

# ── Embeddings (Step 3) ──────────────────────────────────────────────────────────
# Primary provider: Voyage AI
VOYAGE_EMBED_MODEL: str = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3.5")
VOYAGE_RERANK_MODEL: str = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5")
VOYAGE_EMBED_DIMS: int = 512  # Matryoshka truncation
VOYAGE_EMBED_DTYPE: str = "float"  # "float" for local; "int8" when supported
VOYAGE_EMBED_BATCH_SIZE: int = 128  # max texts per API call
VOYAGE_INPUT_TYPE_DOC: str = "document"
VOYAGE_INPUT_TYPE_QUERY: str = "query"

# Standby provider: Jina AI
JINA_EMBED_MODEL: str = os.getenv("JINA_EMBED_MODEL", "jina-embeddings-v3")
JINA_EMBED_DIMS: int = 512  # Matryoshka truncation to match Voyage
JINA_EMBED_BATCH_SIZE: int = 64
JINA_EMBED_URL: str = "https://api.jina.ai/v1/embeddings"

# ── Vector store (Step 3) ────────────────────────────────────────────────────────
LANCE_TABLE_NAME: str = "chunks"
LANCE_JINA_TABLE_NAME: str = "chunks_jina"
# Index metadata key written into a sidecar JSON in the LanceDB directory
LANCE_META_FILENAME: str = ".index_meta.json"

# ── Verification (Step 4) ────────────────────────────────────────────────────────
# Alert when ok-page count drops more than this fraction vs the previous run.
DROP_ALERT_THRESHOLD: float = 0.05  # 5%
# Subdirectory under STAGING_DIR where per-run JSON reports are saved.
REPORTS_SUBDIR: str = "reports"
# Filename under STAGING_DIR that lists URLs failing the last fetch (for --targeted re-runs).
FAILURES_FILENAME: str = "failures.jsonl"
# Maximum concurrent HTTP requests.  Currently 1 (sequential) — polite for ~80-100 URLs.
# Raise this and add a ThreadPoolExecutor in fetch.py only when throughput becomes an issue.
MAX_CONCURRENT_REQUESTS: int = 1
