from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Iterator, Literal, Optional, Sequence, Tuple, TypedDict, cast, ContextManager, AsyncContextManager
import os

if TYPE_CHECKING:
    from cassandra_asyncio.cluster import Cluster
    from cassandra.cluster import Session
    from chainlit.data.base import BaseDataLayer
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_postgres import PGEngine
from meeplemate.postgres.vectorstore import PartitionedPGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig, reciprocal_rank_fusion
from langchain_postgres.v2.indexes import DistanceStrategy
import yaml

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.engine import URL, make_url

from pydantic import BaseModel, Field, SecretStr, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
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
from meeplemate.qa import build_qa_chain
from meeplemate.db.repository import PostgresDataLayer

from meeplemate.search import (
    ChunkSearchService, build_chunk_search_service, build_chunk_search_service_2
)
from meeplemate.server.deps import ApiDeps, CorsConfig
from meeplemate.server.rate_limit import RateLimitConfig, RateLimiter


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


class DBConfig(BaseModel):
    """Database configuration for Cassandra cluster."""
    dc: str = Field(default="datacenter1", description="Cassandra datacenter name")
    contact_points: list[str] = Field(default=["cassandra"], description="Cassandra contact points")
    replication_factor: int = Field(default=1, ge=1, description="Replication factor for keyspaces")
    chainlit_keyspace: str = Field(default="chainlit_meeplemate", description="Keyspace for Chainlit data")
    langgraph_keyspace: str = Field(default="meeplemate_checkpoints", description="Keyspace for LangGraph checkpoints")
    create_keyspaces: bool = Field(default=True, description="Whether to create keyspaces on startup")

    @field_validator('contact_points')
    @classmethod
    def validate_contact_points(cls, v):
        if not v:
            raise ValueError("At least one contact point must be specified")
        return v


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


class DataAPIConfig(BaseModel):
    """Configuration for Stargate Data API access."""
    token: SecretStr = Field(description="Data API authentication token")
    endpoint: str = Field(description="Data API endpoint URL")
    namespace: str = Field(default="meeplemate", description="Data API namespace")

    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint must be a valid HTTP/HTTPS URL")
        return v.rstrip('/')


class ChatServiceConfig(BaseModel):
    """Configuration for chat/LLM service."""
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


    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint must be a valid HTTP/HTTPS URL")
        return v.rstrip('/')


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

    # db: DBConfig = Field(default_factory=DBConfig)
    pg: PGConfig = Field(default_factory=PGConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    # data_api: DataAPIConfig
    chat: ChatServiceConfig
    embedding: EmbeddingServiceConfig
    rules_path: Path = Field(default=Path("./data/rules/munchkin_rules/"))
    load_docs: bool = Field(default=False)
    model_name: str = Field(description="Primary model name for tokenizer/other purposes")
    qa_chain_config: QAChainConfig = Field(default_factory=QAChainConfig)
    firebase: FirebaseConfig = Field(default_factory=FirebaseConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    cors: CorsConfig = Field(default_factory=CorsConfig)
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

    @field_validator('rules_path', mode='before')
    @classmethod
    def validate_rules_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v



def create_keyspace(data_api_endpoint:str, data_api_token:str, keyspace:str, replication_factor:int):
    import requests
    url = f"{data_api_endpoint}/v1"
    headers = {
        "TOKEN": data_api_token,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    resp = requests.post(
        url=url,
        headers=headers,
        json={
            "createKeyspace": {
                "name": keyspace,
                "options": {
                    "replication": {
                        "class": "SimpleStrategy",
                        "replication_factor": replication_factor
                    }
                }
            }
        }
    )
    resp.raise_for_status()


def build_vectorstore_cassandra(*, embedding_model: Embeddings, api_endpoint: str, token: str, namespace: str) -> VectorStore:
    from langchain_astradb import AstraDBVectorStore
    from langchain_astradb.utils.astradb import HybridSearchMode
    vector_store = AstraDBVectorStore(
            collection_name="document_vector_mapping",
            embedding=embedding_model,
            api_endpoint=api_endpoint,
            token=token,
            namespace=namespace,
            hybrid_search=HybridSearchMode.OFF,
            bulk_insert_batch_concurrency=1,
        )
    return vector_store


def build_docstore_cassandra(*, api_endpoint, token, namespace) -> BaseStore:
    from meeplemate.cassandra_util import AstraDBSerializableStore
    store = AstraDBSerializableStore(
        collection_name="document_store",
        api_endpoint=api_endpoint,
        token=token,
        namespace=namespace,
    )
    return store


def build_data_store_cassandra(*, api_endpoint, token, namespace, collection_name) -> BaseStore:
    from langchain_astradb import AstraDBStore
    store = AstraDBStore(
        collection_name=collection_name,
        api_endpoint=api_endpoint,
        token=token,
        namespace=namespace,
    )
    return store


def load_docs(rules_path: Path) -> list[Document]:
    from meeplemate.pdf import parse_pdf  # lazy: only used during ingest, not API startup
    rule_docs = []
    for filename in rules_path.glob("*.pdf"):
        print(f"Processing {filename}")
        rule_docs.extend(parse_pdf(filename))
    return rule_docs


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


def build_graph(checkpoint_saver: Optional[BaseCheckpointSaver], chain: Runnable):
    async def call_chain(state: GameRulesAgentState):
        msg = None
        for msg in reversed(state["messages"]):
            if msg.type == "human":
                break
        assert msg is not None

        query = msg.content
        result = await chain.ainvoke(query)
        return {"messages": [result["answer"]]}

    builder = StateGraph(GameRulesAgentState)
    builder.add_node("call_chain", call_chain)
    builder.add_edge(START, "call_chain")
    builder.add_edge("call_chain", END)
    agent_graph = builder.compile(checkpointer=checkpoint_saver) 

    return agent_graph


class AppServices(TypedDict):
    db_cluster: Cluster
    db_session: Session
    embedding_model: Embeddings
    vector_store: VectorStore
    docstore: BaseStore
    retriever: BaseRetriever
    data_layer: BaseDataLayer
    tokenizer: Any
    checkpointer: BaseCheckpointSaver
    chat_model: BaseChatModel
    qa_chain: Runnable
    agent_graph: CompiledStateGraph[GameRulesAgentState, None, GameRulesAgentState, GameRulesAgentState]
    game_data_store: BaseStore
    game_version_store: BaseStore
    full_page_store: BaseStore
    chunk_search_service: ChunkSearchService
    chatloop_service: ChatLoopService
    qa_service: QAService
    game_service: GameService
    rate_limiter: RateLimiter


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

    def _build_retriever(*args, **kwargs):
        from meeplemate.retrievers import build_retriever
        return build_retriever(*args, **kwargs)

    @contextmanager
    def create_session(cluster: Cluster) -> Iterator[Session]:
        with cluster.connect() as session:
            yield session
    
    def build_chat_model(config: ChatServiceConfig, tokenizer) -> BaseChatModel:
        if cfg.chat.endpoint_type == "tgi":
            chat_model = load_tgi_chat_model(
                tokenizer=tokenizer,
                endpoint_url=config.endpoint,
                max_new_tokens=config.max_new_tokens,
                timeout=config.timeout,
                do_sample=False,
                temperature=0.01,
            )
        elif cfg.chat.endpoint_type == "openai":
            api_key = config.api_key.get_secret_value() if config.api_key else "not-needed"

            sampling_kwargs = {}
            for param in ["temperature", "frequency_penalty", "presence_penalty", "top_p"]:
                value = getattr(config, param)
                if value is not None:
                    sampling_kwargs[param] = value

            extra_body_sampling_kwargs = {}
            for param in ["top_k", "min_p", "repetition_penalty"]:
                value = getattr(config, param)
                if value is not None:
                    extra_body_sampling_kwargs[param] = value

            chat_model = ChatOpenAI(
                model=config.model_name,
                max_tokens=config.max_new_tokens,
                timeout=config.timeout,
                base_url=config.endpoint,
                api_key=api_key,
                streaming=True,
                stream_usage=True,
                **sampling_kwargs,
                extra_body={
                    **extra_body_sampling_kwargs,
                    **({
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        }
                    } if cfg.chat.explicit_disable_thinking else {})
                }
            )
        else:
            raise ValueError(f"Unsupported chat endpoint type: {cfg.chat.endpoint_type}")
        return chat_model

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
                    hybrid_search_config=HybridSearchConfig(
                        tsv_column="content_tsv",
                        tsv_lang="pg_catalog.english",
                        fusion_function=reciprocal_rank_fusion,
                        fusion_function_parameters={"rrf_k": 60},
                        primary_top_k=50,
                        secondary_top_k=50,
                    ),
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
            "retriever": (
                factory(_build_retriever)(),
                ["tokenizer", "vector_store"],
                {"docstore": "docstore"}
            ),
            "chat_model": (
                factory(build_chat_model)(cfg.chat),
                ["tokenizer"]
            ),
            "qa_chain": (
                factory(build_qa_chain)(
                    **cfg.qa_chain_config.model_dump()
                ),
                {"chat_model": "chat_model", "retriever": "retriever", "embedding_model": "embedding_model"}
            ),
            # "checkpointer": (
            #     factory(
            #         CassandraSaver,
            #         start=lambda saver: saver.setup(replication_factor=cfg.db.replication_factor)
            #     )(
            #         thread_id_type="uuid",
            #         keyspace=cfg.db.langgraph_keyspace,
            #     ),
            #     {"session": "db_session"}
            # ),
            "agent_graph": (
                factory(build_graph)(checkpoint_saver=None),
                {"chain": "qa_chain"},
            ),
            # "keyspace_creator": (
            #     keyspace_creator(
            #         [
            #             (cfg.data_api.namespace, cfg.db.replication_factor),
            #         ],
            #         data_api_endpoint=cfg.data_api.endpoint,
            #         data_api_token=cfg.data_api.token.get_secret_value(),
            #         create_keyspaces=cfg.db.create_keyspaces,
            #     ),
            #     []
            # ),
            # "data_layer": (
            #     create_data_layer(
            #         storage_client=None,
            #         keyspace=cfg.db.chainlit_keyspace,
            #         replication_factor=cfg.db.replication_factor,
            #     ),
            #     ["db_session"]
            # ),
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
            "full_page_store": (
                factory(PostgresSerializableStore)(
                    namespace="full_page_store",
                ),
                {
                    "engine": "async_engine",
                }
            ),
            "chunk_search_service": (
                factory(build_chunk_search_service)(
                    checkpoint_saver=None,
                ),
                {
                    "chat_model": "chat_model",
                    "retriever": "retriever",
                }
            ),
            "chunk_search_service_2": (
                factory(build_chunk_search_service_2)(default_token_budget=15_000),
                {
                    "vectorstore": "vector_store",
                    "docstore": "docstore",
                    "tokenizer": "tokenizer",
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
            "api_deps": (
                factory(ApiDeps)(cors_config=cfg.cors),
                {
                    "chatloop_service": "chatloop_service",
                    "game_service": "game_service",
                    "data_layer": "pg_data_layer",
                    "rate_limiter": "rate_limiter",
                }
            )
        }
    )
    return system


def create_data_layer(storage_client: Any, keyspace: str, replication_factor: int) -> Callable[[Session], AsyncContextManager[BaseDataLayer]]:
    @asynccontextmanager
    async def _with_data_layer(session: Session) -> AsyncIterator[BaseDataLayer]:
        from chainlit_cassandra_data_layer.data import CassandraDataLayer
        dl = CassandraDataLayer(session=session, storage_client=storage_client, keyspace=keyspace)
        try:
            dl.setup(replication_factor=replication_factor)
            yield dl
        finally:
            await dl.close()

    return _with_data_layer


def keyspace_creator(keyspaces_and_replication: Sequence[Tuple[str, int]], data_api_endpoint: str, data_api_token: str, create_keyspaces: bool) -> Callable[[], ContextManager[None]]:
    @contextmanager
    def _keyspace_creator() -> Iterator[None]:
        if create_keyspaces:
            for keyspace, replication_factor in keyspaces_and_replication:
                create_keyspace(
                    data_api_endpoint=data_api_endpoint,
                    data_api_token=data_api_token,
                    keyspace=keyspace,
                    replication_factor=replication_factor,
                )
        yield
    return _keyspace_creator
