"""BM25 postings index over parent chunks; retire the child-level FTS column

The hybrid search this replaces never worked. ``langchain-postgres`` builds its
sparse arm with ``plainto_tsquery``, which ANDs every term, so a query like
"Warpstorm Scroll text and effect" only matches a chunk containing *all four*
lexemes. Measured against this corpus, 82% of real eval queries matched zero
documents — the lexical arm was silently contributing nothing, and "hybrid"
search was pure vector search wearing a hat.

Two consequences follow, and both are fixed here.

First, recall. Postgres full-text ranking (``ts_rank_cd``) has no IDF, so even
when the AND query did match, a rare proper noun carried no more weight than
"rules" or "item". BM25 exists precisely to weight by term rarity, and Postgres
cannot express it without the corpus statistics this migration stores.

Second, the cutoff. ``find_cutoff_adaptive_k`` looks for the largest gap in a
descending score list. RRF scores are ``1/(rank + 60)``, whose deltas decrease
monotonically, so over a single populated list the largest gap is *always* at
index 0 and the function returned exactly ``post_k_buffer + 1 = 6`` for any
input. Retrieval was pinned at six candidates per query.

The new index is built per ``game_version``, which is immutable once imported.
That is what makes it worth precomputing: ``N``, ``df``, ``avgdl`` and every
per-document length factor are fixed at build time, so query time reduces to
one index-only range scan per query term plus a sum. Only IDF is computed per
query, from ``bm25_df`` — which is stored per *rulebook* so that a future
"select the expansions you own" feature gets exact statistics for any subset,
since document frequency is additive over disjoint document sets.

Surrogate integer keys throughout. ``langchain_id`` averages 101 characters,
and a covering btree cannot deduplicate ``INCLUDE`` payloads, so using it as
the posting key costs 73MB against 22MB for an ``int``.

No backfill. Existing game versions keep working (vector-only) until they are
re-imported or ``mm-ingest rebuild-bm25`` is run; ``bm25_index_meta`` exists so
that state is detectable rather than silent.

Revision ID: 0003_bm25_index
Revises: 0002_account_deletion
Create Date: 2026-08-08

"""
from typing import Sequence, Union

from alembic import op

revision: str = "0003_bm25_index"
down_revision: Union[str, None] = "0002_account_deletion"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- retire the child-level FTS machinery -------------------------------
    # Dropping the column on the partitioned parent cascades to every partition
    # and takes the index with it; the explicit DROP INDEX is for idempotency
    # and to keep the intent legible.
    op.execute("DROP INDEX IF EXISTS public.rules_vectors_content_tsv_gin;")
    op.execute("ALTER TABLE public.rules_vectors DROP COLUMN IF EXISTS content_tsv;")

    # --- BM25 index ---------------------------------------------------------
    # Raw SQL rather than op.create_table, following the rules_vectors
    # precedent in 0001_initial: these tables are not in db/models.py (see the
    # include_object filter in alembic/env.py), so autogenerate must never see
    # them and there is no ORM metadata to mirror.

    # Parent chunks. `did` is a per-version surrogate assigned at build time.
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.bm25_doc (
            game_version uuid     NOT NULL,
            did          integer  NOT NULL,
            langchain_id text     NOT NULL,
            rulebook_id  smallint NOT NULL,
            dl           integer  NOT NULL,
            CONSTRAINT bm25_doc_pkey PRIMARY KEY (game_version, did)
        );
    """)
    # Enforces one did per parent (a build invariant) and gives the
    # verification harness a reverse lookup.
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS bm25_doc_langchain_id
            ON public.bm25_doc (game_version, langchain_id);
    """)

    # Term dictionary.
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.bm25_term (
            game_version uuid    NOT NULL,
            tid          integer NOT NULL,
            lexeme       text    NOT NULL,
            CONSTRAINT bm25_term_pkey PRIMARY KEY (game_version, tid)
        );
    """)
    # Query time goes lexeme -> tid; INCLUDE keeps that lookup index-only.
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS bm25_term_lexeme
            ON public.bm25_term (game_version, lexeme) INCLUDE (tid);
    """)

    # Postings. `w` is the BM25 length factor
    #   (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
    # precomputed against the version's global avgdl. IDF is *not* baked in:
    # it depends on which rulebooks a query covers, so it is applied at query
    # time from bm25_df.
    #
    # This PK is the only hot index in the schema. Every query term is one
    # contiguous range scan on the (game_version, tid) prefix, and INCLUDE (w)
    # makes it index-only so the heap is never touched. That is also why the
    # surrogate keys matter: an INCLUDE payload is stored per tuple and cannot
    # be deduplicated, so a text key here would dominate the index size.
    # There is deliberately no index on `did` — nothing looks up by document.
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.bm25_posting (
            game_version uuid    NOT NULL,
            tid          integer NOT NULL,
            did          integer NOT NULL,
            w            real    NOT NULL,
            CONSTRAINT bm25_posting_pkey PRIMARY KEY (game_version, tid, did) INCLUDE (w)
        );
    """)

    # Document frequency per (term, rulebook). Summing over the selected
    # rulebooks gives exact df for any subset, because df is additive over
    # disjoint document sets. `tid` leads the key because it drives the query;
    # the rulebook set is a filter applied within each term's range.
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.bm25_df (
            game_version uuid     NOT NULL,
            tid          integer  NOT NULL,
            rulebook_id  smallint NOT NULL,
            df           integer  NOT NULL,
            CONSTRAINT bm25_df_pkey PRIMARY KEY (game_version, tid, rulebook_id) INCLUDE (df)
        );
    """)

    # Per-rulebook corpus statistics. `doc_count` sums to N; `len_sum` lets a
    # subset recompute its own avgdl if the global one ever stops being a good
    # enough approximation (measured spread across rulebooks: 27-34 tokens).
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.bm25_rulebook (
            game_version  uuid     NOT NULL,
            rulebook_id   smallint NOT NULL,
            rulebook_name text     NOT NULL,
            doc_count     integer  NOT NULL,
            len_sum       bigint   NOT NULL,
            CONSTRAINT bm25_rulebook_pkey PRIMARY KEY (game_version, rulebook_id)
        );
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS bm25_rulebook_name
            ON public.bm25_rulebook (game_version, rulebook_name);
    """)

    # Not a performance structure. This row is what makes two silent failure
    # modes loud: a version indexed with a different text search config than
    # the searcher queries with (lexing drift matches nothing and looks like a
    # ranking problem), and a version that was never indexed at all (which,
    # given there is no backfill, is the expected state for every game until it
    # is re-imported).
    op.execute("""
        CREATE TABLE IF NOT EXISTS public.bm25_index_meta (
            game_version  uuid             NOT NULL,
            game_id       text             NOT NULL,
            tsv_config    text             NOT NULL,
            k1            real             NOT NULL,
            b             real             NOT NULL,
            avgdl         double precision NOT NULL,
            doc_count     integer          NOT NULL,
            term_count    integer          NOT NULL,
            posting_count bigint           NOT NULL,
            built_at      timestamptz      NOT NULL DEFAULT now(),
            CONSTRAINT bm25_index_meta_pkey PRIMARY KEY (game_version)
        );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS public.bm25_index_meta;")
    op.execute("DROP TABLE IF EXISTS public.bm25_rulebook;")
    op.execute("DROP TABLE IF EXISTS public.bm25_df;")
    op.execute("DROP TABLE IF EXISTS public.bm25_posting;")
    op.execute("DROP TABLE IF EXISTS public.bm25_term;")
    op.execute("DROP TABLE IF EXISTS public.bm25_doc;")

    # Restores the column and index but NOT their contents — tsvectors are only
    # ever written by the vector store's INSERT path, so every existing row
    # comes back with content_tsv IS NULL. Downgrading requires re-importing
    # every game version to get child-level FTS working again.
    op.execute("ALTER TABLE public.rules_vectors ADD COLUMN IF NOT EXISTS content_tsv tsvector;")
    op.execute("""
        CREATE INDEX IF NOT EXISTS rules_vectors_content_tsv_gin
            ON public.rules_vectors
            USING gin (content_tsv);
    """)
