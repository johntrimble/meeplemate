from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional, Sequence, TypedDict, cast
import os

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_postgres import PGEngine
from meeplemate.postgres.bm25 import Bm25IndexBuilder, Bm25Searcher
from meeplemate.postgres.vectorstore import PartitionedPGVectorStore
from langchain_postgres.v2.indexes import DistanceStrategy
import yaml

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.engine import URL, make_url

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph

from meeplemate.chatloop import ChatLoopService, build_chatloop_service
from meeplemate.component_system import System, afactory, factory
from meeplemate.game_service import GameService
from meeplemate.postgres.store import PostgresJSONStore, PostgresSerializableStore
from meeplemate.qa_graph import QAService, build_qa_service
from meeplemate.llm_models import load_tgi_chat_model, load_tokenizer, load_lightweight_tokenizer, load_approximate_tokenizer, wrap_embeddings_with_instructions
from meeplemate.failover_chat_model import FailoverChatModel
from meeplemate.db.datalayer import BaseDataLayer
from meeplemate.db.repository import PostgresDataLayer

from meeplemate.search import (
    ChunkSearchService, build_chunk_search_service_2
)
from meeplemate.server.deps import ApiDeps, CorsConfig
from meeplemate.server.rate_limit import RateLimitConfig, RateLimiter
from meeplemate.tracing import TraceSink, build_trace_sink


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Loads settings from a YAML file pointed to by MM_CONFIG_FILE.

    Slotted below env vars / .env in the priority chain so that environment
    variables always win over YAML values.
    """

    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)
        config_file = os.environ.get('MM_CONFIG_FILE')
        self._data: dict[str, Any] = {}
        if config_file:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            with open(config_path, 'r') as f:
                self._data = yaml.safe_load(f) or {}

    def get_field_value(self, field, field_name):
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._data


class IngestConfig(BaseModel):
    """Configuration for document ingestion."""
    chunk_size: int = Field(default=500, ge=1, description="Chunk size for document splitting")
    chunk_overlap: int = Field(default=50, ge=0, description="Chunk overlap for document splitting")
    child_chunk_size: int = Field(default=125, ge=0, description="Child chunk size for finer splitting (0 to disable)")
    child_chunk_overlap: int = Field(default=12, ge=0, description="Child chunk overlap for finer splitting")


class PGConfig(BaseModel):
    """Configuration for PostgreSQL connection.

    Provide either `url` (a full connection string), individual components, or both.
    Individual components override the corresponding parts of `url` when both are given.
    """
    url: Optional[str] = Field(default=None, description="Full PostgreSQL connection URL (without password)")
    username: Optional[str] = Field(default=None)
    password: Optional[SecretStr] = Field(default=None)
    host: Optional[str] = Field(default=None)
    port: Optional[int] = Field(default=None)
    database: Optional[str] = Field(default=None)
    query: dict[str, str] = Field(default_factory=dict, description="Query parameters, e.g. {'ssl': 'require'}")
    pool_size: int = Field(default=10, ge=1, description="SQLAlchemy connection pool size")
    max_overflow: int = Field(default=20, ge=0, description="SQLAlchemy max overflow connections beyond pool_size")
    pool_pre_ping: Optional[bool] = Field(default=None, description="Whether to enable SQLAlchemy pool_pre_ping")
    pool_recycle: Optional[int] = Field(default=None, ge=0, description="SQLAlchemy pool_recycle timeout in seconds")

    @staticmethod
    def _normalize_query(query: dict[str, str], drivername: str) -> dict[str, str]:
        """Translate psycopg2-style sslmode to asyncpg-style ssl when using asyncpg driver."""
        if "asyncpg" not in drivername or "sslmode" not in query:
            return query
        q = dict(query)
        q["ssl"] = q.pop("sslmode")
        return q

    def build_url(self) -> URL:
        if self.url is not None:
            base = make_url(self.url) if isinstance(self.url, str) else self.url
            overrides = {
                k: v for k, v in {
                    "username": self.username,
                    "password": self.password.get_secret_value() if self.password else None,
                    "host": self.host,
                    "port": self.port,
                    "database": self.database,
                }.items() if v is not None
            }
            merged_query = self._normalize_query({**base.query, **self.query}, base.drivername)
            if merged_query:
                overrides["query"] = merged_query
            return base.set(**overrides) if overrides else base
        # No URL provided — build from parts
        return URL.create(
            "postgresql+asyncpg",
            username=self.username,
            password=self.password.get_secret_value() if self.password else None,
            host=self.host,
            port=self.port,
            database=self.database,
            query=self._normalize_query(self.query, "postgresql+asyncpg"),
        )


class PGSettings(BaseSettings):
    """Minimal settings for loading only PG config — used by alembic and other DB-only contexts.

    Supports the same MM_PG__* environment variables, .env file, and MM_CONFIG_FILE YAML as the
    full Config class, but does not require chat/embedding/model_name to be set.
    Extra keys from YAML or environment are silently ignored.
    """
    model_config = SettingsConfigDict(
        env_prefix='MM_',
        env_nested_delimiter='__',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        env_ignore_empty=True,
    )

    pg: PGConfig = Field(default_factory=PGConfig)

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, **kwargs):
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))


class ChatModelConfig(BaseModel):
    """Configuration for a single chat/LLM model (one provider/endpoint).

    A ``ChatConfig`` holds an ordered list of these; the failover chat model
    tries them in priority order. Each entry carries its own endpoint, thinking
    convention, and sampling params, since providers differ on all three.
    """
    model_name: str = Field(description="Name of the chat model")
    endpoint_type: Literal["tgi", "openai"] = Field(description="Type of endpoint (TGI or OpenAI-compatible)")
    endpoint: str = Field(description="Chat service endpoint URL")
    max_new_tokens: int = Field(default=3072, ge=1, description="Maximum tokens to generate")
    timeout: int = Field(default=900, ge=1, description="Request timeout in seconds")
    api_key: SecretStr | None = Field(default=None, description="API key for authentication (if required)")
    explicit_disable_thinking: bool = Field(default=False, description="Explicitly disable thinking for certain models")

    temperature: Optional[float] = Field(default=None, description="Temperature for models (overrides default if set)")
    top_p: Optional[float] = Field(default=None, description="Top-p (nucleus sampling) for models (overrides default if set)")
    top_k: Optional[int] = Field(default=None, description="Top-k sampling for models (overrides default if set)")
    min_p: Optional[float] = Field(default=None, description="Minimum probability for token sampling (overrides default if set)")
    presence_penalty: Optional[float] = Field(default=None, description="Presence penalty for models (overrides default if set)")
    frequency_penalty: Optional[float] = Field(default=None, description="Frequency penalty for models (overrides default if set)")
    repetition_penalty: Optional[float] = Field(default=None, description="Repetition penalty for models (overrides default if set)")

    # Provider-specific extra_body passthroughs (non-standard OpenAI fields). Left
    # as free-form dicts because their shape varies by provider (e.g. OpenRouter).
    provider: Optional[dict[str, Any]] = Field(
        default=None,
        description="Provider-routing options passed through in extra_body (e.g. OpenRouter's `provider` object).",
    )
    reasoning: Optional[dict[str, Any]] = Field(
        default=None,
        description="Reasoning options passed through in extra_body (provider-specific).",
    )
    chat_template_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description="Extra chat-template kwargs passed through in extra_body. Merged with the "
                    "enable_thinking=False that explicit_disable_thinking adds; keys set here win.",
    )


    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint must be a valid HTTP/HTTPS URL")
        return v.rstrip('/')


class ChatConfig(BaseModel):
    """Configuration for the chat service: an ordered list of models plus the
    failover circuit-breaker policy shared across them.

    At call time models are tried in priority order (index 0 first); a model
    whose breaker is open is skipped until its cooldown expires. See
    ``meeplemate.failover_chat_model``.
    """
    models: list[ChatModelConfig] = Field(
        min_length=1,
        description="Ordered list of chat models to try (index 0 is highest priority)",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="Shared API key applied to any model in `models` that does not set its own "
                    "`api_key`. Lets the model list and the token be configured via separate "
                    "environment variables (e.g. MM_CHAT__MODELS as JSON and MM_CHAT__API_KEY).",
    )
    max_failures: int = Field(
        default=3, ge=1,
        description="Consecutive failures before a model's circuit breaker opens",
    )
    cooldown_seconds: float = Field(
        default=300.0, ge=0,
        description="How long a tripped model is skipped before being retried",
    )

    @model_validator(mode="after")
    def _apply_shared_api_key(self) -> "ChatConfig":
        """Fill in each model's api_key from the shared top-level key when the
        model doesn't specify its own. A per-model api_key always wins."""
        if self.api_key is not None:
            for model in self.models:
                if model.api_key is None:
                    model.api_key = self.api_key
        return self


def build_openai_extra_body(config: ChatModelConfig) -> dict[str, Any]:
    """Assemble the ``extra_body`` for an OpenAI-compatible chat model.

    Carries the non-standard sampling params (``top_k``/``min_p``/
    ``repetition_penalty``) and provider extensions (``provider``/``reasoning``)
    that aren't part of the standard OpenAI schema. ``chat_template_kwargs`` is
    the merge of the ``explicit_disable_thinking`` convenience flag (which adds
    ``enable_thinking=False``) with any explicit ``chat_template_kwargs``; keys
    set explicitly win over the convenience flag.
    """
    extra_body: dict[str, Any] = {}
    for param in ["top_k", "min_p", "repetition_penalty", "provider", "reasoning"]:
        value = getattr(config, param)
        if value is not None:
            extra_body[param] = value

    chat_template_kwargs = {
        **({"enable_thinking": False} if config.explicit_disable_thinking else {}),
        **(config.chat_template_kwargs or {}),
    }
    if chat_template_kwargs:
        extra_body["chat_template_kwargs"] = chat_template_kwargs

    return extra_body


class EmbeddingServiceConfig(BaseModel):
    """Configuration for embedding service."""
    model: str = Field(description="Name/path of the embedding model")
    endpoint: Optional[str] = Field(default=None,description="Embedding service endpoint URL")
    api_key: SecretStr = Field(description="API key for embedding service")
    parallel: Optional[int] = Field(default=None, description="If >1, use parallel encoding with specified number of workers. If 0, use all cores. If None, don't use data-parallel processing.")
    query_instruction: str = Field(default="", description="Instruction prefix for query embeddings")
    embed_instruction: str = Field(default="", description="Instruction prefix for document embeddings")

    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint must be a valid HTTP/HTTPS URL")
        return v.rstrip('/')


class QAChainConfig(BaseModel):
    """Configuration for QA chain parameters."""
    model_config = ConfigDict(extra='allow')

    reword_documents: bool = Field(default=True)
    self_consistency: bool = Field(default=True)
    thread_of_thought: bool = Field(default=True)
    consistency_kwargs: dict[str, Any] = Field(default_factory=lambda: {"samples": 3})


class FirebaseConfig(BaseModel):
    """Firebase Auth configuration for token validation."""
    project_id: str = Field(default="boardbarian", description="Firebase project ID")
    # Supply exactly one of these for the Admin SDK credential:
    service_account_path: Optional[str] = Field(
        default=None,
        description="Path to a service account JSON key file (MM_FIREBASE__SERVICE_ACCOUNT_PATH)",
    )
    service_account_json: Optional[str] = Field(
        default=None,
        description="Base64-encoded service account JSON (MM_FIREBASE__SERVICE_ACCOUNT_JSON)",
    )
    emulator_host: Optional[str] = Field(
        default=None,
        description="Firebase Auth Emulator host (e.g. firebase-emulator:9099). When set, token verification is directed to the emulator (MM_FIREBASE__EMULATOR_HOST)",
    )


class TraceConfig(BaseModel):
    """Configuration for persisting agent run traces.

    Interim persistence of LangChain run traces (issue #53) until a dedicated
    tracing tool (Langfuse/LangSmith) is set up. Selects where completed run
    trees are written, keyed by ``<chat_id>/<message_id>.json.gz``.
    """
    backend: Literal["gcs", "local", "noop"] = Field(
        default="noop",
        description="Trace sink backend: 'gcs' (prod), 'local' (dev/testing), or 'noop' (off). (MM_TRACE__BACKEND)",
    )
    bucket: Optional[str] = Field(
        default=None,
        description="GCS bucket name; required when backend='gcs'. Provisioned out-of-band with its retention/lifecycle rule. (MM_TRACE__BUCKET)",
    )
    prefix: str = Field(
        default="",
        description="Optional key prefix prepended to every GCS object. (MM_TRACE__PREFIX)",
    )
    local_dir: str = Field(
        default="./traces",
        description="Directory for the 'local' backend. (MM_TRACE__LOCAL_DIR)",
    )


class LogConfig(BaseModel):
    """Logging configuration for the API server.

    The API also configures logging at import time from these same environment
    variables (see ``meeplemate.logging_config``), because records emitted while
    ``meeplemate.*`` is still importing predate any ``Config`` instance. This model
    exists so the settings are part of the documented config surface and so the
    values can be re-applied once ``Config`` has resolved them from the ``.env``
    file or a YAML config file, neither of which ``os.environ`` alone would see.
    """
    level: str = Field(
        default="INFO",
        description="Root log level: DEBUG, INFO, WARNING, ERROR, CRITICAL. (MM_LOG__LEVEL)",
    )
    format: Literal["json", "console", "auto"] = Field(
        default="auto",
        description="Log rendering: 'json' for Cloud Run/Cloud Logging, 'console' for "
                    "human-readable colored dev output, or 'auto' to pick json when the "
                    "K_SERVICE env var is set (i.e. running on Cloud Run). (MM_LOG__FORMAT)",
    )
    gcp_project_id: Optional[str] = Field(
        default=None,
        description="GCP project id used to build the logging.googleapis.com/trace field "
                    "as 'projects/<id>/traces/<trace_id>', which is what makes container "
                    "logs group under the Cloud Run request log. When unset, falls back to "
                    "GOOGLE_CLOUD_PROJECT/GCP_PROJECT and then the GCE metadata server; if "
                    "none of those resolve, the trace fields are simply omitted. "
                    "(MM_LOG__GCP_PROJECT_ID)",
    )


class Config(BaseSettings):
    """Main application configuration with environment variable support.

    Configuration priority (highest to lowest):
    1. Environment variables (MM_*)
    2. .env file
    3. YAML config file (if MM_CONFIG_FILE is set)
    4. Default values in model definitions
    """
    model_config = SettingsConfigDict(
        env_prefix='MM_',
        env_nested_delimiter='__',
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',  # Ignore extra env vars like MY_UID, MY_GID from .env
        env_ignore_empty=True,
    )

    pg: PGConfig = Field(default_factory=PGConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    chat: ChatConfig
    embedding: EmbeddingServiceConfig
    model_name: str = Field(description="Primary model name for tokenizer/other purposes")
    qa_chain_config: QAChainConfig = Field(default_factory=QAChainConfig)
    firebase: FirebaseConfig = Field(default_factory=FirebaseConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
    trace: TraceConfig = Field(default_factory=TraceConfig)
    log: LogConfig = Field(default_factory=LogConfig)
    auth_bypass: bool = Field(default=False, description="Skip token validation and use a hardcoded user (MM_AUTH_BYPASS)")
    auth_bypass_user: Optional[str] = Field(
        default=None,
        description='JSON string for bypass user, e.g. {"uid":"dev","email":"dev@local","name":"Dev"} (MM_AUTH_BYPASS_USER)',
    )
    use_lightweight_tokenizer: bool = Field(
        default=False,
        description="Use lightweight tokenizer for token counting instead of full AutoTokenizer. Saves ~130 MB. Do not use for ingest pipeline.",
    )
    use_approximate_tokenizer: bool = Field(
        default=False,
        description="Use character-count heuristic for token budgeting instead of a real tokenizer. "
                    "Saves ~94 MB vs LightweightTokenizer. Accuracy: ~7% mean over-estimate (p95: +15%). "
                    "Safe for context-budget decisions; billing uses actual counts from the OpenAI endpoint. "
                    "Set to false to switch back to LightweightTokenizer.",
    )

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings, **kwargs):
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))




class RulebookDescriptor(TypedDict):
    name: str
    url: str
    document_key: str


class GameManifest(TypedDict):
    name: str
    game_id: str
    rulebooks: Sequence[RulebookDescriptor]


class GameRulesAgentState(MessagesState):
    manifest: GameManifest


class AppServices(TypedDict):
    embedding_model: Embeddings
    vector_store: VectorStore
    docstore: BaseStore
    data_layer: BaseDataLayer
    tokenizer: Any
    checkpointer: BaseCheckpointSaver
    chat_model: BaseChatModel
    game_data_store: BaseStore
    game_version_store: BaseStore
    game_questions_store: BaseStore
    full_page_store: BaseStore
    chunk_search_service: ChunkSearchService
    chatloop_service: ChatLoopService
    qa_service: QAService
    game_service: GameService
    rate_limiter: RateLimiter
    trace_sink: TraceSink


class GameInfoDao:
    def __init__(self, datastore: BaseStore):
        self.datastore = datastore

    async def alist(self) -> AsyncIterator[GameManifest]:
        iterator = await self.datastore.ayield_keys()
        keys = [k async for k in iterator]
        values = await self.datastore.amget(keys)
        for value in values:
            yield cast(GameManifest, value)


class FullPageMdDao:
    def __init__(self, datastore: BaseStore):
        self.datastore = datastore
    
    async def aget(self, game_id: str, document_key: str, page: int) -> Document:
        key = f"{game_id}::{document_key}::{page}"
        data = await self.datastore.amget([key])
        if data is None or len(data) == 0 or data[0] is None:
            raise KeyError(f"Full page markdown not found for key: {key}")
        data = data[0]
        return Document(
            page_content=data["content"],
            metadata=data["metadata"],
        )


def create_app_system(cfg: Config) -> System[AppServices]:

    def build_single_chat_model(config: ChatModelConfig, tokenizer) -> BaseChatModel:
        if config.endpoint_type == "tgi":
            chat_model = load_tgi_chat_model(
                tokenizer=tokenizer,
                endpoint_url=config.endpoint,
                max_new_tokens=config.max_new_tokens,
                timeout=config.timeout,
                do_sample=False,
                temperature=0.01,
            )
        elif config.endpoint_type == "openai":
            api_key = config.api_key.get_secret_value() if config.api_key else "not-needed"

            sampling_kwargs = {}
            for param in ["temperature", "frequency_penalty", "presence_penalty", "top_p"]:
                value = getattr(config, param)
                if value is not None:
                    sampling_kwargs[param] = value

            chat_model = ChatOpenAI(
                model=config.model_name,
                max_tokens=config.max_new_tokens,
                timeout=config.timeout,
                base_url=config.endpoint,
                api_key=api_key,
                streaming=True,
                stream_usage=True,
                **sampling_kwargs,
                extra_body=build_openai_extra_body(config),
            )
        else:
            raise ValueError(f"Unsupported chat endpoint type: {config.endpoint_type}")
        return chat_model

    def build_chat_model(chat_cfg: ChatConfig, tokenizer) -> BaseChatModel:
        models = [build_single_chat_model(m, tokenizer) for m in chat_cfg.models]
        return FailoverChatModel(
            models=models,
            max_failures=chat_cfg.max_failures,
            cooldown_seconds=chat_cfg.cooldown_seconds,
        )

    system = System[AppServices](
        {
            # "_embedding_model": (
            #     factory(OpenAIEmbeddings)(
            #         model=cfg.embedding.model,
            #         base_url=cfg.embedding.endpoint,
            #         api_key=cfg.embedding.api_key.get_secret_value(),
            #         tiktoken_enabled=False,
            #         chunk_size=10,
            #     ),
            #     []
            # ),
            "_embedding_model": (
                factory(FastEmbedEmbeddings)(
                    model_name=cfg.embedding.model,
                    parallel=cfg.embedding.parallel,
                ),
                []
            ),
            "embedding_model": (
                factory(wrap_embeddings_with_instructions)(
                    query_instruction=cfg.embedding.query_instruction,
                    embed_instruction=cfg.embedding.embed_instruction,
                ),
                {
                    "embeddings": "_embedding_model",
                }
            ),
            "vector_store": (
                afactory(PartitionedPGVectorStore.create)(
                    table_name="rules_vectors",
                    schema_name="public",
                    id_column="langchain_id",
                    content_column="content",
                    embedding_column="embedding",
                    metadata_columns=["game_version", "game_id"],
                    metadata_json_column="langchain_metadata",
                    distance_strategy=DistanceStrategy.COSINE_DISTANCE,
                    # No hybrid_search_config: the library's sparse arm builds
                    # its query with plainto_tsquery, which ANDs every term and
                    # so matched nothing for 82% of real queries. Lexical
                    # retrieval now lives in Bm25Searcher over parent chunks.
                    # Note this makes asimilarity_search_with_score return raw
                    # cosine *distance* (lower is better) rather than RRF
                    # scores — search.py accounts for that.
                ),
                {
                    "embedding_service": "embedding_model",
                    "engine": "pg_engine",
                },
            ),
            "docstore": (
                factory(PostgresSerializableStore)(
                    namespace="document_store",
                ),
                {
                    "engine": "async_engine",
                }
            ),
            "tokenizer": (
                factory(
                    load_approximate_tokenizer if cfg.use_approximate_tokenizer
                    else load_lightweight_tokenizer if cfg.use_lightweight_tokenizer
                    else load_tokenizer
                )(**({} if cfg.use_approximate_tokenizer else {"model_name": cfg.model_name})),
                []
            ),
            "chat_model": (
                factory(build_chat_model)(cfg.chat),
                ["tokenizer"]
            ),
            "game_version_store": (
                factory(PostgresJSONStore)(
                    namespace="current_game_version",
                ),
                {
                    "engine": "async_engine",
                }
            ),
            "game_data_store": (
                factory(PostgresJSONStore)(
                    namespace="game_info",
                ),
                                {
                    "engine": "async_engine",
                }
            ),
            "game_questions_store": (
                factory(PostgresJSONStore)(
                    namespace="game_questions",
                ),
                {
                    "engine": "async_engine",
                }
            ),
            "full_page_store": (
                factory(PostgresSerializableStore)(
                    namespace="full_page_store",
                ),
                {
                    "engine": "async_engine",
                }
            ),
            "bm25_searcher": (
                factory(Bm25Searcher)(),
                {"engine": "async_engine"},
            ),
            "bm25_index_builder": (
                factory(Bm25IndexBuilder)(),
                {"engine": "async_engine"},
            ),
            "chunk_search_service_2": (
                factory(build_chunk_search_service_2)(default_token_budget=15_000),
                {
                    "vectorstore": "vector_store",
                    "docstore": "docstore",
                    "tokenizer": "tokenizer",
                    "bm25": "bm25_searcher",
                }
            ),
            "qa_service": (
                factory(build_qa_service)(
                    checkpoint_saver=None,
                ),
                {
                    "chat_model": "chat_model",
                    "full_page_store": "full_page_store",
                    "chunk_search_service": "chunk_search_service_2",
                    "tokenizer": "tokenizer"
                }
            ),
            "chatloop_service": (
                factory(build_chatloop_service)(checkpoint_saver=None),
                {
                    "chat_model": "chat_model",
                    "tokenizer": "tokenizer",
                    "qa_service": "qa_service",
                }
            ),
            "game_service": (
                factory(GameService)(),
                {
                    "data_store": "game_data_store",
                    "version_store": "game_version_store",
                    "questions_store": "game_questions_store",
                }
            ),
            "async_engine": (
                factory(create_async_engine)(
                    cfg.pg.build_url(),
                    pool_size=cfg.pg.pool_size,
                    max_overflow=cfg.pg.max_overflow,
                    **({} if cfg.pg.pool_pre_ping is None else {"pool_pre_ping": cfg.pg.pool_pre_ping}),
                    **({} if cfg.pg.pool_recycle is None else {"pool_recycle": cfg.pg.pool_recycle}),
                ),
                {}
            ),
            "pg_engine": (
                factory(PGEngine.from_engine)(),
                {"engine": "async_engine"},
            ),
            "pg_data_layer": (
                factory(PostgresDataLayer)(),
                {"engine": "async_engine"},
            ),
            "rate_limiter": (
                factory(RateLimiter)(cfg.rate_limit),
                ["pg_data_layer"],
            ),
            "trace_sink": (
                factory(build_trace_sink)(cfg.trace),
                [],
            ),
            "api_deps": (
                factory(ApiDeps)(cors_config=cfg.cors),
                {
                    "chatloop_service": "chatloop_service",
                    "game_service": "game_service",
                    "data_layer": "pg_data_layer",
                    "rate_limiter": "rate_limiter",
                    "trace_sink": "trace_sink",
                }
            )
        }
    )
    return system


