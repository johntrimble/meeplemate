#!/usr/bin/env python
"""Dump everything `mm-ingest import-documents` wrote for one game, as stable JSON.

Intended use is a before/after check around a change that should not alter what
lands in the database — the ingest layout refactor, for example:

    python script/snapshot_game_state.py forbiddenisland > /tmp/before.json
    # ... migrate the package, re-import ...
    python script/snapshot_game_state.py forbiddenisland > /tmp/after.json
    diff /tmp/before.json /tmp/after.json

Output is sorted and hashed rather than raw, so the diff stays readable: a
changed chunk shows up as one changed hash line rather than a wall of markdown.
Pass --show-content to include the actual text for keys you want to inspect.
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from meeplemate.config import Config

KV_NAMESPACES = {
    "game_data": "game_info",
    "parent_chunks": "document_store",
    "full_pages": "full_page_store",
    "example_questions": "game_questions",
}


def digest(value: bytes | str) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.md5(value).hexdigest()[:16]


def canonical(value: object) -> object:
    """Recursively sort dict keys, so ordering does not affect the hash."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def fingerprint(raw: bytes | str) -> str:
    """`<meaning>:<bytes>` — the first half ignores key order, the second does not.

    Re-serialising the same data with keys in a different order is common and
    harmless (it happened when `game_version` moved out of `get_page_metadata`).
    Splitting the two lets a diff say which kind of change occurred: differing
    right-hand sides alone mean a reordering, differing left-hand sides mean the
    content actually changed.
    """
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        meaning = digest(json.dumps(canonical(json.loads(raw)), sort_keys=True))
    except (ValueError, TypeError):
        meaning = digest(raw)
    return f"{meaning}:{digest(raw)}"


async def snapshot(game_id: str, show_content: bool) -> dict:
    # Building the config emits log lines; keep stdout clean so the JSON pipes.
    with contextlib.redirect_stdout(sys.stderr):
        url = Config().pg.build_url()
    engine = create_async_engine(url)
    # Record the target. MM_PG__URL silently overrides the configured database
    # (import-dev.sh and import-prod.sh both set it), so a before/after pair
    # taken against different databases must fail loudly rather than compare.
    out: dict = {
        "game_id": game_id,
        "database": {
            "host": url.host, "port": url.port,
            "name": url.database, "user": url.username,
        },
    }
    try:
        async with engine.connect() as conn:
            row = await conn.execute(
                text("select value from langchain_key_value_stores "
                     "where namespace='current_game_version' and key=:k"),
                {"k": game_id},
            )
            raw = row.scalar()
            if raw is None:
                raise SystemExit(f"{game_id} has no current version in the database")
            game_key = json.loads(raw)
            out["game_key"] = game_key
            game_version = game_key.split("#", 1)[1]

            # --- key/value stores -------------------------------------------
            for label, namespace in KV_NAMESPACES.items():
                # game_questions is keyed by game_id; the rest by game_key.
                prefix = game_id if namespace == "game_questions" else game_key
                rows = await conn.execute(
                    text("select key, value from langchain_key_value_stores "
                         "where namespace=:ns and (key = :p or key like :like) order by key"),
                    {"ns": namespace, "p": prefix, "like": f"{prefix}#%"},
                )
                entries = {}
                for key, value in rows.fetchall():
                    entries[key] = json.loads(value) if show_content else fingerprint(value)
                out[label] = {"count": len(entries), "entries": entries}

            # --- vector store ------------------------------------------------
            rows = await conn.execute(
                text("select langchain_id, content, langchain_metadata::text "
                     "from rules_vectors where game_version = cast(:v as uuid) "
                     "order by langchain_id"),
                {"v": game_version},
            )
            vectors = {}
            for langchain_id, content, metadata in rows.fetchall():
                vectors[langchain_id] = (
                    {"content": content, "metadata": json.loads(metadata)}
                    if show_content
                    else f"{digest(content)}/{fingerprint(metadata)}"
                )
            out["child_chunks"] = {"count": len(vectors), "entries": vectors}

            # Embeddings are floats; a separate rollup keeps any drift visible
            # without burying the rest of the diff.
            row = await conn.execute(
                text("select md5(string_agg(embedding::text, ',' order by langchain_id)) "
                     "from rules_vectors where game_version = cast(:v as uuid)"),
                {"v": game_version},
            )
            out["embeddings_digest"] = (row.scalar() or "")[:16]

            # --- bm25 index ---------------------------------------------------
            row = await conn.execute(
                text("select tsv_config, k1, b, avgdl, doc_count, term_count, posting_count "
                     "from bm25_index_meta where game_version = cast(:v as uuid)"),
                {"v": game_version},
            )
            meta = row.fetchone()
            out["bm25"] = (
                {
                    "tsv_config": meta[0], "k1": float(meta[1]), "b": float(meta[2]),
                    "avgdl": round(float(meta[3]), 6), "doc_count": meta[4],
                    "term_count": meta[5], "posting_count": meta[6],
                }
                if meta
                else None
            )
    finally:
        await engine.dispose()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game_id")
    parser.add_argument("--show-content", action="store_true",
                        help="Include full text instead of hashes. Verbose; use to inspect a specific difference.")
    args = parser.parse_args()
    result = asyncio.run(snapshot(args.game_id, args.show_content))
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
