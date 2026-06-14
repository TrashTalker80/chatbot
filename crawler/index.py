"""LanceDB index builder and query helpers.

Architecture:
- One primary LanceDB table ('chunks') built with Voyage embeddings (512-dim float32).
- One standby table ('chunks_jina') built with Jina embeddings (same 512-dim schema).
- Both tables live under *uri* (a local path or s3:// URI).
- Index metadata (.index_meta.json) pins model + dims + build timestamp so the query
  layer can assert it reads with the same model used at ingest.

Key functions:
  build_index(embedded_chunks, uri, table_name, …)
      Create or overwrite a LanceDB table from EmbeddedChunk objects.

  upsert_chunks(embedded_chunks, uri, table_name, …)
      Incremental upsert: add/replace chunks whose content_hash changed; delete chunks
      whose doc URLs have vanished.  Idempotent — safe to call on every crawl run.

  delete_chunks_for_urls(urls, uri, table_name)
      Remove all chunks for the given canonical URLs (called when a page 404s).

  smoke_query(query_vector, fts_query, uri, table_name, top_k)
      Run a vector search + FTS search and return the results (for CLI smoke test).

  read_index_meta(uri, table_name) → dict
  write_index_meta(uri, table_name, meta)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

from crawler.config import (
    LANCE_META_FILENAME,
    LANCE_TABLE_NAME,
)
from crawler.embed import EmbeddedChunk

logger = logging.getLogger(__name__)

# ── PyArrow schema ────────────────────────────────────────────────────────────

_VECTOR_DIM = 512


def _build_schema(dims: int = _VECTOR_DIM) -> pa.Schema:
    return pa.schema(
        [
            pa.field("chunk_id", pa.string()),
            pa.field("url", pa.string()),
            pa.field("title", pa.string()),
            pa.field("page_type", pa.string()),
            pa.field("content_hash", pa.string()),
            pa.field("text", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("source", pa.string()),
            pa.field("is_faq", pa.bool_()),
            pa.field("vector", pa.list_(pa.float32(), dims)),
        ]
    )


def _rows_from_embedded(embedded: list[EmbeddedChunk]) -> list[dict]:
    rows = []
    for e in embedded:
        rows.append(
            {
                "chunk_id": e.chunk_id,
                "url": e.url,
                "title": e.title,
                "page_type": e.page_type,
                "content_hash": e.content_hash,
                "text": e.text,
                "chunk_index": e.chunk_index,
                "source": e.source,
                "is_faq": e.is_faq,
                "vector": e.vector,
            }
        )
    return rows


# ── Index metadata ────────────────────────────────────────────────────────────


def _meta_path(uri: str, table_name: str) -> Path:
    """Local path to the sidecar metadata JSON for this table."""
    base = Path(uri) if not uri.startswith("s3://") else Path("/tmp/lance_meta")
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{table_name}{LANCE_META_FILENAME}"


def write_index_meta(uri: str, table_name: str, meta: dict) -> None:
    path = _meta_path(uri, table_name)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote index metadata to %s", path)


def read_index_meta(uri: str, table_name: str) -> dict:
    path = _meta_path(uri, table_name)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _make_meta(embedded: list[EmbeddedChunk], dims: int) -> dict:
    if not embedded:
        return {"chunk_count": 0, "built_at": _utcnow()}
    first = embedded[0]
    return {
        "provider": first.provider,
        "model": first.model,
        "dims": dims,
        "chunk_count": len(embedded),
        "built_at": _utcnow(),
    }


def _utcnow() -> str:
    return datetime.now(tz=UTC).isoformat()


# ── LanceDB connection helper ─────────────────────────────────────────────────


def _connect(uri: str, storage_options: dict | None = None):
    import lancedb

    if uri.startswith("s3://"):
        aws_opts: dict = {
            "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        }
        aws_opts.update(storage_options or {})
        return lancedb.connect(uri, storage_options=aws_opts)
    return lancedb.connect(uri)


def _table_exists(db, table_name: str) -> bool:
    """Return True if *table_name* exists in *db*, handling different LanceDB return types."""
    result = db.list_tables()
    # LanceDB 0.33+ returns ListTablesResponse with a .tables attribute
    names = result.tables if hasattr(result, "tables") else list(result)
    return table_name in names


# ── Build (full overwrite) ────────────────────────────────────────────────────


def build_index(
    embedded: list[EmbeddedChunk],
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    dims: int = _VECTOR_DIM,
    storage_options: dict | None = None,
) -> None:
    """Create (or overwrite) *table_name* with all embedded chunks.

    Full rebuild — used on the first crawl or when the embedding model changes.
    For incremental updates use upsert_chunks().
    """
    if not embedded:
        logger.warning("build_index called with 0 chunks — skipping.")
        return

    db = _connect(uri, storage_options)
    schema = _build_schema(dims)
    rows = _rows_from_embedded(embedded)

    if _table_exists(db, table_name):
        db.drop_table(table_name)
        logger.info("Dropped existing table %r for full rebuild.", table_name)

    tbl = db.create_table(table_name, data=rows, schema=schema)
    _create_fts_index(tbl)

    meta = _make_meta(embedded, dims)
    write_index_meta(uri, table_name, meta)
    logger.info(
        "Built index %r with %d chunks (model=%s, dims=%d)",
        table_name,
        len(embedded),
        meta.get("model"),
        dims,
    )


def _create_fts_index(tbl) -> None:
    try:
        tbl.create_fts_index("text", replace=True)
        logger.debug("FTS index created on 'text' column.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("FTS index creation failed (non-fatal): %s", exc)


# ── Upsert (incremental) ──────────────────────────────────────────────────────


def upsert_chunks(
    embedded: list[EmbeddedChunk],
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    dims: int = _VECTOR_DIM,
    storage_options: dict | None = None,
) -> dict[str, int]:
    """Incremental upsert: add/replace changed chunks; leave unchanged ones alone.

    Returns a summary dict: {"added": N, "updated": N, "skipped": N}.
    Creates the table if it doesn't exist yet (first run).
    """
    if not embedded:
        return {"added": 0, "updated": 0, "skipped": 0}

    db = _connect(uri, storage_options)
    schema = _build_schema(dims)

    if not _table_exists(db, table_name):
        tbl = db.create_table(table_name, data=_rows_from_embedded(embedded), schema=schema)
        _create_fts_index(tbl)
        meta = _make_meta(embedded, dims)
        write_index_meta(uri, table_name, meta)
        return {"added": len(embedded), "updated": 0, "skipped": 0}

    tbl = db.open_table(table_name)
    existing_hashes = _fetch_existing_hashes(tbl)

    to_add: list[EmbeddedChunk] = []
    to_update: list[EmbeddedChunk] = []
    skipped = 0

    for e in embedded:
        prev_hash = existing_hashes.get(e.chunk_id)
        if prev_hash is None:
            to_add.append(e)
        elif prev_hash != e.content_hash:
            to_update.append(e)
        else:
            skipped += 1

    if to_update:
        ids_to_delete = [e.chunk_id for e in to_update]
        _delete_by_chunk_ids(tbl, ids_to_delete)

    new_rows = _rows_from_embedded(to_add + to_update)
    if new_rows:
        tbl.add(new_rows)
        _create_fts_index(tbl)

    meta = read_index_meta(uri, table_name)
    meta["chunk_count"] = tbl.count_rows()
    meta["last_upsert"] = _utcnow()
    meta["last_added"] = len(to_add)
    meta["last_updated"] = len(to_update)
    write_index_meta(uri, table_name, meta)

    logger.info(
        "Upsert complete — added=%d updated=%d skipped=%d (table=%r)",
        len(to_add),
        len(to_update),
        skipped,
        table_name,
    )
    return {"added": len(to_add), "updated": len(to_update), "skipped": skipped}


def _fetch_existing_hashes(tbl) -> dict[str, str]:
    """Return {chunk_id: content_hash} for every row currently in the table."""
    try:
        rows = tbl.search().select(["chunk_id", "content_hash"]).to_list()
        return {r["chunk_id"]: r["content_hash"] for r in rows}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch existing hashes: %s", exc)
        return {}


def _delete_by_chunk_ids(tbl, chunk_ids: list[str]) -> None:
    if not chunk_ids:
        return
    ids_sql = ", ".join(f"'{cid}'" for cid in chunk_ids)
    tbl.delete(f"chunk_id IN ({ids_sql})")


# ── Delete by URL (called on 404/410) ────────────────────────────────────────


def delete_chunks_for_urls(
    urls: list[str],
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    storage_options: dict | None = None,
) -> int:
    """Delete all chunks whose *url* is in *urls*.  Returns the number of deleted rows."""
    if not urls:
        return 0

    db = _connect(uri, storage_options)
    if not _table_exists(db, table_name):
        return 0

    tbl = db.open_table(table_name)
    urls_sql = ", ".join(f"'{u}'" for u in urls)
    before = tbl.count_rows()
    tbl.delete(f"url IN ({urls_sql})")
    deleted = before - tbl.count_rows()
    logger.info("Deleted %d chunks for %d URLs from table %r", deleted, len(urls), table_name)
    return deleted


# ── Smoke query (CLI validation) ──────────────────────────────────────────────


def smoke_query(
    query_vector: list[float],
    fts_query: str,
    uri: str,
    table_name: str = LANCE_TABLE_NAME,
    top_k: int = 5,
    storage_options: dict | None = None,
) -> dict[str, list[dict]]:
    """Run vector + FTS searches and return their results (for --smoke CLI flag)."""
    db = _connect(uri, storage_options)
    tbl = db.open_table(table_name)

    vector_results = tbl.search(query_vector, query_type="vector").limit(top_k).to_list()
    try:
        fts_results = tbl.search(fts_query, query_type="fts").limit(top_k).to_list()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FTS search failed: %s", exc)
        fts_results = []

    def _clean(rows: list[dict]) -> list[dict]:
        return [
            {
                "chunk_id": r.get("chunk_id"),
                "url": r.get("url"),
                "title": r.get("title"),
                "page_type": r.get("page_type"),
                "text": r.get("text", "")[:120],
                "score": r.get("_distance") or r.get("_score"),
            }
            for r in rows
        ]

    return {"vector": _clean(vector_results), "fts": _clean(fts_results)}


# ── CLI entry-point ───────────────────────────────────────────────────────────


def _cli_smoke(uri: str, table_name: str = LANCE_TABLE_NAME) -> None:
    import sys

    meta = read_index_meta(uri, table_name)
    if not meta:
        print(f"No index metadata found at {uri!r} for table {table_name!r}.")
        sys.exit(1)

    print(f"Index metadata: {json.dumps(meta, indent=2)}")

    db = _connect(uri)
    if not _table_exists(db, table_name):
        print(f"Table {table_name!r} not found.")
        sys.exit(1)

    tbl = db.open_table(table_name)
    count = tbl.count_rows()
    print(f"Table row count: {count}")

    # Use a zero vector as a stand-in (real smoke uses a real embedding)
    dims = meta.get("dims", _VECTOR_DIM)
    zero_vec = [0.0] * dims
    results = smoke_query(
        query_vector=zero_vec,
        fts_query="Appther ERP implementation",
        uri=uri,
        table_name=table_name,
    )
    print(f"Sample vector results ({len(results['vector'])} rows):")
    for r in results["vector"]:
        print(f"  [{r['page_type']}] {r['url']} — {r['text'][:80]!r}")
    print(f"Sample FTS results ({len(results['fts'])} rows):")
    for r in results["fts"]:
        print(f"  [{r['page_type']}] {r['url']} — {r['text'][:80]!r}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LanceDB index utilities")
    parser.add_argument("--smoke", action="store_true", help="Run smoke query against the index")
    parser.add_argument("--uri", default="./lance_index", help="LanceDB URI (local path or s3://)")
    parser.add_argument("--table", default=LANCE_TABLE_NAME, help="Table name")
    args = parser.parse_args()

    if args.smoke:
        _cli_smoke(uri=args.uri, table_name=args.table)
