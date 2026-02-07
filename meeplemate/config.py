from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Literal, Sequence, Tuple, TypedDict, cast, ContextManager, AsyncContextManager
import os
import yaml
from dataclasses import dataclass

from pydantic import BaseModel, Field, SecretStr, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_astradb import AstraDBStore, AstraDBVectorStore
from langchain_astradb.utils.astradb import HybridSearchMode
from cassandra_asyncio.cluster import Cluster
from cassandra.cluster import Session
from cassandra.policies import DCAwareRoundRobinPolicy

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

from langgraph_checkpoint_cassandra import CassandraSaver
from chainlit_cassandra_data_layer.data import CassandraDataLayer


from meeplemate.cassandra_util import AstraDBSerializableStore
from meeplemate.chatloop import ChatLoopService, build_chatloop_service
from meeplemate.component_system import System, factory
from meeplemate.ingest.gamepackage import GamePackage
from meeplemate.qa_graph import QAService, build_qa_service
from meeplemate.retrievers import build_retriever
from meeplemate.llm_models import load_tgi_chat_model, load_tokenizer, sentence_transformer_to_hf_embeddings
from meeplemate.pdf import parse_pdf
from meeplemate.qa import build_qa_chain
from chainlit.data.base import BaseDataLayer

from meeplemate.search import (
    ChunkSearchService, build_chunk_search_service, build_chunk_search_service_2
)


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
    max_new_tokens: int = Field(default=1024, ge=1, description="Maximum tokens to generate")
    timeout: int = Field(default=900, ge=1, description="Request timeout in seconds")
    api_key: SecretStr | None = Field(default=None, description="API key for authentication (if required)")
    explicit_disable_thinking: bool = Field(default=False, description="Explicitly disable thinking for certain models")

    @field_validator('endpoint')
    @classmethod
    def validate_endpoint(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Endpoint must be a valid HTTP/HTTPS URL")
        return v.rstrip('/')


class EmbeddingServiceConfig(BaseModel):
    """Configuration for embedding service."""
    model: str = Field(description="Name/path of the embedding model")
    endpoint: str = Field(description="Embedding service endpoint URL")
    api_key: SecretStr = Field(description="API key for embedding service")

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

    db: DBConfig = Field(default_factory=DBConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    data_api: DataAPIConfig
    chat: ChatServiceConfig
    embedding: EmbeddingServiceConfig
    rules_path: Path = Field(default=Path("./data/rules/munchkin_rules/"))
    load_docs: bool = Field(default=False)
    model_name: str = Field(description="Primary model name for tokenizer/other purposes")
    qa_chain_config: QAChainConfig = Field(default_factory=QAChainConfig)

    def __init__(self, **kwargs):
        """Initialize config with support for YAML file loading.

        Checks MM_CONFIG_FILE environment variable for YAML config path.
        YAML values are used as defaults, with environment variables taking precedence.
        """
        # Load YAML config if specified
        config_file = os.environ.get('MM_CONFIG_FILE')
        print(f"Loading configuration from: {config_file}" if config_file else "No YAML config file specified.")
        yaml_config = {}

        if config_file:
            config_path = Path(config_file)
            if config_path.exists():
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f) or {}
            else:
                raise FileNotFoundError(f"Config file not found: {config_file}")

        # Merge YAML config with explicit kwargs (kwargs take precedence)
        # This creates the base that environment variables will override
        merged_config = {**yaml_config, **kwargs}

        super().__init__(**merged_config)

    @field_validator('rules_path', mode='before')
    @classmethod
    def validate_rules_path(cls, v):
        if isinstance(v, str):
            return Path(v)
        return v


@dataclass
class GameService:
    data_store: BaseStore
    version_store: BaseStore
    
    async def get_current_version_for_game(self, game_id: str) -> str | None:
        results = await self.version_store.amget([game_id])
        assert len(results) == 1 and results[0] is not None, "No version found for game_id"
        version = results[0]
        return str(version)
    
    async def get_manifest(self, game_id: str) -> GamePackage | None:
        game_key = await self.get_current_version_for_game(game_id)
        manifest = self.data_store.mget([game_key])[0]
        return manifest


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
    store = AstraDBSerializableStore(
        collection_name="document_store",
        api_endpoint=api_endpoint,
        token=token,
        namespace=namespace,
    )
    return store


def build_data_store_cassandra(*, api_endpoint, token, namespace, collection_name) -> BaseStore:
    store = AstraDBStore(
        collection_name=collection_name,
        api_endpoint=api_endpoint,
        token=token,
        namespace=namespace,
    )
    return store


def load_docs(rules_path: Path) -> list[Document]:
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


def build_graph(checkpoint_saver: BaseCheckpointSaver, chain: Runnable):
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
            chat_model = ChatOpenAI(
                model=config.model_name,
                max_tokens=config.max_new_tokens,
                presence_penalty=1.5,
                temperature=0.7,
                top_p=0.8,
                timeout=config.timeout,
                base_url=config.endpoint,
                api_key=api_key,
                streaming=True,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
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
            "db_cluster": (
                factory(Cluster)(
                    contact_points=cfg.db.contact_points,
                    load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=cfg.db.dc)
                ),
                []
            ),
            "db_session": (
                create_session,
                ["db_cluster"]
            ),
            "embedding_model": (
                factory(OpenAIEmbeddings)(
                    model=cfg.embedding.model,
                    base_url=cfg.embedding.endpoint,
                    api_key=cfg.embedding.api_key.get_secret_value(),
                    tiktoken_enabled=False,
                    chunk_size=10,
                ),
                []
            ),
            "vector_store": (
                factory(build_vectorstore_cassandra)(api_endpoint=cfg.data_api.endpoint, token=cfg.data_api.token.get_secret_value(), namespace=cfg.data_api.namespace),
                {"embedding_model": "embedding_model"},
            ),
            "docstore": (
                factory(AstraDBSerializableStore)(
                    collection_name="document_store",
                    api_endpoint=cfg.data_api.endpoint,
                    token=cfg.data_api.token.get_secret_value(),
                    namespace=cfg.data_api.namespace,
                ),
                []
            ),
            "tokenizer": (
                factory(load_tokenizer)(cfg.model_name),
                []
            ),
            "retriever": (
                factory(build_retriever)(),
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
            "checkpointer": (
                factory(
                    CassandraSaver,
                    start=lambda saver: saver.setup(replication_factor=cfg.db.replication_factor)
                )(
                    thread_id_type="uuid",
                    keyspace=cfg.db.langgraph_keyspace,
                ),
                {"session": "db_session"}
            ),
            "agent_graph": (
                factory(build_graph)(),
                {"checkpoint_saver": "checkpointer", "chain": "qa_chain"},
            ),
            "keyspace_creator": (
                keyspace_creator(
                    [
                        (cfg.data_api.namespace, cfg.db.replication_factor),
                    ],
                    data_api_endpoint=cfg.data_api.endpoint,
                    data_api_token=cfg.data_api.token.get_secret_value(),
                    create_keyspaces=cfg.db.create_keyspaces,
                ),
                []
            ),
            "data_layer": (
                create_data_layer(
                    storage_client=None,
                    keyspace=cfg.db.chainlit_keyspace,
                    replication_factor=cfg.db.replication_factor,
                ),
                ["db_session"]
            ),
            "game_version_store": (
                factory(AstraDBStore)(
                    collection_name="current_game_version",
                    api_endpoint=cfg.data_api.endpoint,
                    token=cfg.data_api.token.get_secret_value(),
                    namespace=cfg.data_api.namespace,
                ),
                []
            ),
            "game_data_store": (
                factory(AstraDBStore)(
                    collection_name="game_info",
                    api_endpoint=cfg.data_api.endpoint,
                    token=cfg.data_api.token.get_secret_value(),
                    namespace=cfg.data_api.namespace,
                ),
                []
            ),
            "full_page_store": (
                factory(AstraDBSerializableStore)(
                    collection_name="full_page_store",
                    api_endpoint=cfg.data_api.endpoint,
                    token=cfg.data_api.token.get_secret_value(),
                    namespace=cfg.data_api.namespace,
                ),
                []
            ),
            "chunk_search_service": (
                factory(build_chunk_search_service)(),
                {
                    "checkpoint_saver": "checkpointer",
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
                factory(build_qa_service)(),
                {
                    "checkpoint_saver": "checkpointer",
                    "chat_model": "chat_model",
                    "full_page_store": "full_page_store",
                    "chunk_search_service": "chunk_search_service_2",
                    "tokenizer": "tokenizer"
                }
            ),
            "chatloop_service": (
                factory(build_chatloop_service)(),
                {
                    "checkpoint_saver": "checkpointer",
                    "chat_model": "chat_model",
                    "qa_service": "qa_service",
                }
            ),
            "game_service": (
                factory(GameService)(),
                {
                    "data_store": "game_data_store",
                    "version_store": "game_version_store",
                }
            )
        }
    )
    return system


def create_data_layer(storage_client: Any, keyspace: str, replication_factor: int) -> Callable[[Session], AsyncContextManager[BaseDataLayer]]:
    @asynccontextmanager
    async def _with_data_layer(session: Session) -> AsyncIterator[BaseDataLayer]:
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


class Services:
    db_cluster: Cluster | None = None
    db_session: Session | None = None
    embedding_model: Embeddings | None = None
    vector_store: VectorStore | None = None
    docstore: BaseStore | None = None
    data_layer: BaseDataLayer | None = None
    tokenizer: Any = None
    chain: Runnable | None = None
    checkpointer: BaseCheckpointSaver | None = None
    _agent_graph: CompiledStateGraph[GameRulesAgentState, None, GameRulesAgentState, GameRulesAgentState] | None = None
    cfg: Config
    stack: AsyncExitStack

    @property
    def agent_graph(self) -> CompiledStateGraph[GameRulesAgentState, None, GameRulesAgentState, GameRulesAgentState]:
        # We do this one lazily to ensure the chainlit contextvar has been
        # populated as the LangchainTracer depends on it.
        if self._agent_graph is not None:
            return self._agent_graph
        assert self.chain is not None
        assert self.checkpointer is not None

        chain = self.chain
        self._agent_graph = build_graph(
            checkpoint_saver=self.checkpointer,
            chain=chain,
        )

        return self._agent_graph

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stack = AsyncExitStack()

    async def start(self):
        # Create the db cluster and session
        self.db_cluster = Cluster(
            contact_points=self.cfg.db.contact_points,
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=self.cfg.db.dc),
        )
        self.db_session = self.stack.enter_context(self.db_cluster.connect())
        self.stack.callback(self.db_session.shutdown)

        # Setup chainlit datalayer
        data_layer = CassandraDataLayer(session=self.db_session, storage_client=None, keyspace=self.cfg.db.chainlit_keyspace)
        data_layer.setup(replication_factor=self.cfg.db.replication_factor)
        self.stack.push_async_callback(data_layer.close)
        self.data_layer = data_layer

        # Setup the checkpointer
        checkpointer = CassandraSaver(
            thread_id_type="uuid",
            keyspace=self.cfg.db.langgraph_keyspace,
            session=self.db_session,
        )
        checkpointer.setup()
        self.checkpointer = checkpointer

        # Create keyspaces for document stores and vector stores
        if self.cfg.db.create_keyspaces:
            create_keyspace(
                data_api_endpoint=self.cfg.data_api.endpoint,
                data_api_token=self.cfg.data_api.token.get_secret_value(),
                keyspace=self.cfg.db.chainlit_keyspace,
                replication_factor=self.cfg.db.replication_factor,
            )
            create_keyspace(
                data_api_endpoint=self.cfg.data_api.endpoint,
                data_api_token=self.cfg.data_api.token.get_secret_value(),
                keyspace=self.cfg.db.langgraph_keyspace,
                replication_factor=self.cfg.db.replication_factor,
            )

        # Load the embedding
        self.embedding_model = OpenAIEmbeddings(
            model=self.cfg.embedding.model,
            base_url=self.cfg.embedding.endpoint,
            api_key=self.cfg.embedding.api_key.get_secret_value(),
            tiktoken_enabled=False
        )
        
        # Setup vector store
        self.vector_store = build_vectorstore_cassandra(
            embedding_model=self.embedding_model,
            api_endpoint=self.cfg.data_api.endpoint,
            token=self.cfg.data_api.token.get_secret_value(),
            namespace=self.cfg.data_api.namespace,
        )

        # Setup doc store
        self.docstore = build_docstore_cassandra(
            api_endpoint=self.cfg.data_api.endpoint,
            token=self.cfg.data_api.token.get_secret_value(),
            namespace=self.cfg.data_api.namespace,
        )

        # Load the tokenizer
        self.tokenizer = load_tokenizer(self.cfg.model_name)

        # Build the retriever
        retriever = build_retriever(self.tokenizer, self.vector_store, docstore=self.docstore)

        # Optionally load documents
        if self.cfg.load_docs:
            rule_docs = load_docs(self.cfg.rules_path)
            retriever.add_documents(rule_docs)

        # Build the chat model
        if self.cfg.chat.endpoint_type == "tgi":
            chat_model = load_tgi_chat_model(
                tokenizer=self.tokenizer,
                endpoint_url=self.cfg.chat.endpoint,
                max_new_tokens=self.cfg.chat.max_new_tokens,
                timeout=self.cfg.chat.timeout,
                do_sample=False,
                temperature=0.01,
            )
        elif self.cfg.chat.endpoint_type == "openai":
            # For OpenAI-compatible endpoints, api_key is required even if not used for auth
            api_key = self.cfg.chat.api_key.get_secret_value() if self.cfg.chat.api_key else "not-needed"
            chat_model = ChatOpenAI(
                model=self.cfg.model_name,
                max_tokens=self.cfg.chat.max_new_tokens,
                temperature=0.0,
                timeout=self.cfg.chat.timeout,
                base_url=self.cfg.chat.endpoint,
                api_key=api_key,
                extra_body={
                    "top_k": 20,
                    "min_p": 0.0,
                    **({
                        "chat_template_kwargs": {
                            "enable_thinking": False,
                        }
                    } if self.cfg.chat.explicit_disable_thinking else {})
                }
            )
        else:
            raise ValueError(f"Unsupported chat endpoint type: {self.cfg.chat.endpoint_type}")

        # Build the qa chain
        self.chain = build_qa_chain(
            chat_model=chat_model,
            retriever=retriever,
            embedding_model=self.embedding_model,
            **self.cfg.qa_chain_config.model_dump()
        )

    async def stop(self):
        await self.stack.aclose()
