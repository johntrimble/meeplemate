"""BM25 lexical retrieval over parent chunks.

Two halves that must agree on one thing: how text is turned into lexemes.
``Bm25IndexBuilder`` writes postings using ``to_tsvector(TSV_CONFIG, ...)`` and
``Bm25Searcher`` lexes queries with the same call. If those ever diverge the
index matches nothing and the symptom is indistinguishable from bad ranking, so
both read the constant from this module and the builder records it in
``bm25_index_meta`` for the searcher to check.

Why parent chunks rather than the child chunks the vector search uses: child
chunks exist to stop dense embeddings diluting over long text, a bottleneck
BM25 does not have. Splitting instead destroys term co-occurrence, which *is*
BM25's signal — a query naming three things scores a document containing all
three far above one containing one, and that only works if the three can land
in the same document.

Why the index is precomputed: a game version is immutable once imported, so
``N``, ``df``, ``avgdl`` and every per-document length factor are fixed at build
time. Only IDF varies per query, because it depends on which rulebooks the
query covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from structlog import get_logger

logger = get_logger(__name__)


#: Text search configuration. Shared by the build and the query so lexing is
#: identical on both sides; changing it invalidates every existing index.
TSV_CONFIG = "pg_catalog.english"

#: Standard BM25 term-frequency saturation and length-normalisation constants.
BM25_K1 = 1.2
BM25_B = 0.75

#: The docstore namespace holding parent chunks. Must track the ``docstore``
#: component in ``meeplemate/config.py``.
DOCSTORE_NAMESPACE = "document_store"

_BM25_TABLES = (
    "bm25_posting",
    "bm25_df",
    "bm25_rulebook",
    "bm25_term",
    "bm25_doc",
    "bm25_index_meta",
)


def docstore_key_prefix(game_id: str, game_version: str) -> str:
    """Prefix of every parent-chunk docstore key for one game version.

    Keys are ``{game_id}#{game_version}#{document_key}#{page_ordinal}#{chunk_idx}``
    (see ``meeplemate/ingest/gamepackage.py``), so this prefix selects exactly
    the parents of one version and nothing else.
    """
    return f"{game_id}#{game_version}#"


@dataclass
class Bm25IndexBuilder:
    """Builds the BM25 postings index for a game version, in SQL.

    The build reads parent text straight out of the docstore and never
    round-trips it through Python: lexing has to happen in Postgres to match
    the query side, and pulling megabytes of markdown out only to send it back
    would be pure loss.
    """

    engine: AsyncEngine
    namespace: str = DOCSTORE_NAMESPACE
    tsv_config: str = TSV_CONFIG
    k1: float = BM25_K1
    b: float = BM25_B

    async def abuild(self, game_id: str, game_version: str) -> dict[str, int]:
        """(Re)build the index for one game version. Returns row counts."""
        if not game_version:
            raise ValueError("game_version is required to build a BM25 index")

        prefix = docstore_key_prefix(game_id, game_version)
        params = {
            "gv": game_version,
            "game_id": game_id,
            "ns": self.namespace,
            "prefix": prefix,
            "k1": self.k1,
            "b": self.b,
            "tsv_config": self.tsv_config,
        }

        # One connection for the whole build: the temp tables below are
        # session-scoped, and the six target tables must flip together so a
        # reader never sees a half-replaced index.
        async with self.engine.begin() as conn:
            for table in _BM25_TABLES:
                await conn.execute(
                    text(f"DELETE FROM public.{table} WHERE game_version = CAST(:gv AS uuid)"),
                    {"gv": game_version},
                )

            # Parent chunks, tokenised once. `value` is bytea holding UTF-8
            # JSON (PostgresJSONStore._encode does json.dumps(...).encode()),
            # hence convert_from before the jsonb cast.
            #
            # starts_with, not LIKE: every game_id contains '_', which LIKE
            # treats as a single-character wildcard, so 'munchkin#...' would
            # also match a hypothetical 'munchkinX#...'.
            await conn.execute(
                text(f"""
                CREATE TEMP TABLE _bm25_parent ON COMMIT DROP AS
                SELECT
                    (row_number() OVER (ORDER BY kv.key))::int AS did,
                    kv.key AS langchain_id,
                    coalesce(
                        (convert_from(kv.value, 'UTF8')::jsonb)
                            -> 'kwargs' -> 'metadata' ->> 'rulebook_name',
                        ''
                    ) AS rulebook_name,
                    to_tsvector(
                        '{self.tsv_config}',
                        coalesce(
                            (convert_from(kv.value, 'UTF8')::jsonb) -> 'kwargs' ->> 'page_content',
                            ''
                        )
                    ) AS tsv
                FROM public.langchain_key_value_stores kv
                WHERE kv.namespace = :ns
                  AND starts_with(kv.key, :prefix)
                """),
                {"ns": self.namespace, "prefix": prefix},
            )

            # unnest(tsvector) yields one row per (document, lexeme), so df
            # downstream is count(*) rather than count(DISTINCT did).
            await conn.execute(text("""
                CREATE TEMP TABLE _bm25_tok ON COMMIT DROP AS
                SELECT p.did,
                       p.rulebook_name,
                       t.lexeme,
                       coalesce(array_length(t.positions, 1), 1)::int AS tf
                FROM _bm25_parent p,
                     LATERAL unnest(p.tsv) AS t(lexeme, positions, weights);
            """))
            await conn.execute(text("CREATE INDEX ON _bm25_tok (lexeme);"))
            await conn.execute(text("CREATE INDEX ON _bm25_tok (did);"))

            # Rulebook surrogate ids.
            await conn.execute(text("""
                CREATE TEMP TABLE _bm25_book ON COMMIT DROP AS
                SELECT (row_number() OVER (ORDER BY rulebook_name))::smallint AS rulebook_id,
                       rulebook_name
                FROM (SELECT DISTINCT rulebook_name FROM _bm25_parent) s;
            """))

            # LEFT JOIN, not INNER: a parent whose text is all stopwords (a
            # heading-only chunk) produces no tokens but is still part of the
            # corpus, so it must count toward N and avgdl.
            await conn.execute(
                text("""
                INSERT INTO public.bm25_doc (game_version, did, langchain_id, rulebook_id, dl)
                SELECT CAST(:gv AS uuid), p.did, p.langchain_id, bk.rulebook_id, coalesce(l.dl, 0)
                FROM _bm25_parent p
                JOIN _bm25_book bk ON bk.rulebook_name = p.rulebook_name
                LEFT JOIN (
                    SELECT did, sum(tf)::int AS dl FROM _bm25_tok GROUP BY did
                ) l ON l.did = p.did;
                """),
                {"gv": game_version},
            )

            await conn.execute(
                text("""
                INSERT INTO public.bm25_term (game_version, tid, lexeme)
                SELECT CAST(:gv AS uuid), (row_number() OVER (ORDER BY lexeme))::int, lexeme
                FROM (SELECT DISTINCT lexeme FROM _bm25_tok) s;
                """),
                {"gv": game_version},
            )

            await conn.execute(
                text("""
                INSERT INTO public.bm25_rulebook
                    (game_version, rulebook_id, rulebook_name, doc_count, len_sum)
                SELECT CAST(:gv AS uuid), bk.rulebook_id, bk.rulebook_name,
                       count(d.did)::int, coalesce(sum(d.dl), 0)::bigint
                FROM _bm25_book bk
                LEFT JOIN public.bm25_doc d
                       ON d.game_version = CAST(:gv AS uuid) AND d.rulebook_id = bk.rulebook_id
                GROUP BY bk.rulebook_id, bk.rulebook_name;
                """),
                {"gv": game_version},
            )

            # The length factor only. IDF is applied at query time because it
            # depends on the rulebook subset being searched.
            await conn.execute(
                text("""
                WITH corpus AS (
                    SELECT sum(dl)::float8 / nullif(count(*), 0) AS avgdl
                    FROM public.bm25_doc WHERE game_version = CAST(:gv AS uuid)
                )
                INSERT INTO public.bm25_posting (game_version, tid, did, w)
                SELECT CAST(:gv AS uuid), t.tid, k.did,
                       ( (k.tf * (:k1 + 1.0))
                         / (k.tf + :k1 * (1.0 - :b
                                          + :b * d.dl
                                            / nullif((SELECT avgdl FROM corpus), 0))) )::real
                FROM _bm25_tok k
                JOIN public.bm25_term t
                  ON t.game_version = CAST(:gv AS uuid) AND t.lexeme = k.lexeme
                JOIN public.bm25_doc d
                  ON d.game_version = CAST(:gv AS uuid) AND d.did = k.did;
                """),
                {"gv": game_version, "k1": self.k1, "b": self.b},
            )

            await conn.execute(
                text("""
                INSERT INTO public.bm25_df (game_version, tid, rulebook_id, df)
                SELECT CAST(:gv AS uuid), t.tid, bk.rulebook_id, count(*)::int
                FROM _bm25_tok k
                JOIN public.bm25_term t
                  ON t.game_version = CAST(:gv AS uuid) AND t.lexeme = k.lexeme
                JOIN _bm25_book bk ON bk.rulebook_name = k.rulebook_name
                GROUP BY t.tid, bk.rulebook_id;
                """),
                {"gv": game_version},
            )

            row = (await conn.execute(
                text("""
                INSERT INTO public.bm25_index_meta
                    (game_version, game_id, tsv_config, k1, b, avgdl,
                     doc_count, term_count, posting_count)
                SELECT CAST(:gv AS uuid), :game_id, :tsv_config, :k1, :b,
                       coalesce((SELECT sum(dl)::float8 / nullif(count(*), 0)
                                 FROM public.bm25_doc WHERE game_version = CAST(:gv AS uuid)), 0),
                       (SELECT count(*) FROM public.bm25_doc     WHERE game_version = CAST(:gv AS uuid)),
                       (SELECT count(*) FROM public.bm25_term    WHERE game_version = CAST(:gv AS uuid)),
                       (SELECT count(*) FROM public.bm25_posting WHERE game_version = CAST(:gv AS uuid))
                RETURNING doc_count, term_count, posting_count, avgdl;
                """),
                params,
            )).one()

        # VACUUM cannot run inside a transaction block, and it is not optional:
        # without the visibility map set, the INCLUDE(w) index-only scan falls
        # back to heap fetches and the query is several times slower.
        await self._avacuum()

        counts = {
            "doc_count": row.doc_count,
            "term_count": row.term_count,
            "posting_count": row.posting_count,
        }
        logger.info(
            "Built BM25 index",
            game_id=game_id,
            game_version=game_version,
            avgdl=round(row.avgdl, 2),
            **counts,
        )
        if row.doc_count == 0:
            logger.warning(
                "BM25 index built with zero documents — is the docstore populated?",
                game_id=game_id,
                game_version=game_version,
                namespace=self.namespace,
                key_prefix=prefix,
            )
        return counts

    async def apurge(self, game_version: str) -> None:
        """Remove every trace of one game version's index."""
        if not game_version:
            return
        async with self.engine.begin() as conn:
            for table in _BM25_TABLES:
                await conn.execute(
                    text(f"DELETE FROM public.{table} WHERE game_version = CAST(:gv AS uuid)"),
                    {"gv": game_version},
                )
        logger.info("Purged BM25 index", game_version=game_version)

    async def _avacuum(self) -> None:
        async with self.engine.connect() as conn:
            autocommit = await conn.execution_options(isolation_level="AUTOCOMMIT")
            await autocommit.execute(
                text("VACUUM (ANALYZE) " + ", ".join(f"public.{t}" for t in _BM25_TABLES))
            )


@dataclass
class Bm25Searcher:
    """Scores parent chunks against a query using the precomputed index."""

    engine: AsyncEngine
    tsv_config: str = TSV_CONFIG
    # game_versions whose index has already been checked (or reported missing),
    # so the warning fires once per process rather than once per query.
    _checked: set[str] = field(default_factory=set, repr=False)

    async def asearch(
        self,
        query: str,
        game_version: str,
        *,
        k: int = 50,
        rulebooks: Optional[Sequence[str]] = None,
    ) -> list[tuple[str, float]]:
        """Return ``(parent langchain_id, score)`` ranked best-first.

        ``rulebooks`` restricts the corpus to those rulebook names, and the
        corpus statistics are recomputed for that subset rather than reused
        from the whole game — document frequency is additive over disjoint
        document sets, so the result is exact rather than approximate.
        """
        if not query.strip() or not game_version:
            return []

        await self._acheck_index(game_version)

        params: dict[str, object] = {"gv": game_version, "q": query, "k": k}
        if rulebooks:
            params["books"] = list(rulebooks)
            sql = self._SUBSET_SQL
        else:
            sql = self._FULL_SQL

        async with self.engine.connect() as conn:
            rows = (await conn.execute(text(sql.format(cfg=self.tsv_config)), params)).all()
        return [(r.langchain_id, float(r.score)) for r in rows]

    async def _acheck_index(self, game_version: str) -> None:
        """Warn once per version if the index is missing or lexed differently.

        Both failures are otherwise silent: a version with no index simply
        contributes nothing to fusion, and one built with a different text
        search config matches nothing, which looks exactly like poor ranking.
        """
        if game_version in self._checked:
            return
        self._checked.add(game_version)
        async with self.engine.connect() as conn:
            row = (await conn.execute(
                text("""SELECT tsv_config, doc_count FROM public.bm25_index_meta
                        WHERE game_version = CAST(:gv AS uuid)"""),
                {"gv": game_version},
            )).one_or_none()
        if row is None:
            logger.warning(
                "No BM25 index for this game version; lexical retrieval is disabled. "
                "Run `mm-ingest rebuild-bm25` or re-import the game.",
                game_version=game_version,
            )
        elif row.tsv_config != self.tsv_config:
            logger.error(
                "BM25 index was built with a different text search config; "
                "queries will not match. Rebuild the index.",
                game_version=game_version,
                index_tsv_config=row.tsv_config,
                searcher_tsv_config=self.tsv_config,
            )

    # to_tsvector on the QUERY, not plainto_tsquery. plainto_tsquery ANDs every
    # term, which is what made the previous lexical arm match nothing for 82%
    # of real queries. Here the query is just a bag of lexemes and BM25 decides
    # how much each is worth.
    #
    # sum(idf * w) is float8 because ln() returns double precision and promotes
    # the product — do not "simplify" by factoring idf out of the sum, as
    # sum(real) returns real.
    #
    # The (score DESC, did ASC) ordering makes ties deterministic, which the
    # eval harness relies on for reproducible runs.
    _FULL_SQL = """
        WITH q AS (
            SELECT DISTINCT t.tid
            FROM unnest(to_tsvector('{cfg}', :q)) AS v(lexeme, positions, weights)
            JOIN public.bm25_term t
              ON t.game_version = CAST(:gv AS uuid) AND t.lexeme = v.lexeme
        ),
        corpus AS (
            SELECT sum(doc_count)::float8 AS n
            FROM public.bm25_rulebook WHERE game_version = CAST(:gv AS uuid)
        ),
        idf AS (
            SELECT d.tid,
                   ln(1.0 + ((SELECT n FROM corpus) - sum(d.df)::float8 + 0.5)
                            / (sum(d.df)::float8 + 0.5)) AS idf
            FROM public.bm25_df d
            JOIN q ON q.tid = d.tid
            WHERE d.game_version = CAST(:gv AS uuid)
            GROUP BY d.tid
        ),
        scored AS (
            SELECT p.did, sum(i.idf * p.w) AS score
            FROM idf i
            JOIN public.bm25_posting p
              ON p.game_version = CAST(:gv AS uuid) AND p.tid = i.tid
            GROUP BY p.did
            ORDER BY score DESC, p.did ASC
            LIMIT :k
        )
        SELECT doc.langchain_id, s.score
        FROM scored s
        JOIN public.bm25_doc doc
          ON doc.game_version = CAST(:gv AS uuid) AND doc.did = s.did
        ORDER BY s.score DESC, s.did ASC
    """

    # Same shape, restricted to a rulebook subset. Kept as a separate statement
    # rather than a `(:books IS NULL OR ...)` predicate so the common full-corpus
    # path stays free of the bm25_doc join.
    _SUBSET_SQL = """
        WITH books AS (
            SELECT rulebook_id, doc_count
            FROM public.bm25_rulebook
            WHERE game_version = CAST(:gv AS uuid) AND rulebook_name = ANY(:books)
        ),
        q AS (
            SELECT DISTINCT t.tid
            FROM unnest(to_tsvector('{cfg}', :q)) AS v(lexeme, positions, weights)
            JOIN public.bm25_term t
              ON t.game_version = CAST(:gv AS uuid) AND t.lexeme = v.lexeme
        ),
        corpus AS (
            SELECT sum(doc_count)::float8 AS n FROM books
        ),
        idf AS (
            SELECT d.tid,
                   ln(1.0 + ((SELECT n FROM corpus) - sum(d.df)::float8 + 0.5)
                            / (sum(d.df)::float8 + 0.5)) AS idf
            FROM public.bm25_df d
            JOIN q ON q.tid = d.tid
            JOIN books bk ON bk.rulebook_id = d.rulebook_id
            WHERE d.game_version = CAST(:gv AS uuid)
            GROUP BY d.tid
        ),
        scored AS (
            SELECT p.did, sum(i.idf * p.w) AS score
            FROM idf i
            JOIN public.bm25_posting p
              ON p.game_version = CAST(:gv AS uuid) AND p.tid = i.tid
            JOIN public.bm25_doc doc
              ON doc.game_version = CAST(:gv AS uuid) AND doc.did = p.did
            JOIN books bk ON bk.rulebook_id = doc.rulebook_id
            GROUP BY p.did
            ORDER BY score DESC, p.did ASC
            LIMIT :k
        )
        SELECT doc.langchain_id, s.score
        FROM scored s
        JOIN public.bm25_doc doc
          ON doc.game_version = CAST(:gv AS uuid) AND doc.did = s.did
        ORDER BY s.score DESC, s.did ASC
    """
