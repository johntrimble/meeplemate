from contextlib import AsyncExitStack, ExitStack
from pathlib import Path
from typing import Any, Literal, Mapping, NotRequired, Protocol, TypedDict, runtime_checkable

from langchain_astradb import AstraDBVectorStore
from langchain_astradb.utils.astradb import HybridSearchMode
from cassandra_asyncio.cluster import Cluster
from cassandra.cluster import Session
from cassandra.policies import DCAwareRoundRobinPolicy

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
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

from sentence_transformers import SentenceTransformer

from meeplemate.cassandra_util import AstraDBSerializableStore
from meeplemate.chainlit_utils import LangchainTracer
from meeplemate.retrievers import build_retriever
from meeplemate.llm_models import load_jina_embedding_model, load_tgi_chat_model, load_tokenizer, sentence_transformer_to_hf_embeddings
from meeplemate.pdf import parse_pdf
from meeplemate.qa import build_qa_chain
from chainlit.data.base import BaseDataLayer


class DBConfig(TypedDict):
    dc: str
    contact_points: list[str]
    replication_factor: int
    chainlit_keyspace: str
    langgraph_keyspace: str
    create_keyspaces: bool


class DataAPIConfig(TypedDict):
    token: str
    endpoint: str
    namespace: str


class ChatServiceConfig(TypedDict):
    model_name: str
    endpoint_type: Literal["tgi", "openai"]
    endpoint: str
    max_new_tokens: int
    timeout: int
    api_key: NotRequired[str]
    explicit_disable_thinking: NotRequired[bool]


class EmbeddingServiceConfig(TypedDict):
    model: str
    endpoint: str
    api_key: str


class Config(TypedDict):
    db: DBConfig
    data_api: DataAPIConfig
    chat: ChatServiceConfig
    embedding: EmbeddingServiceConfig
    rules_path: str
    load_docs: bool
    model_name: str
    qa_chain_config: Mapping[str, Any]


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


def buid_vectorstore_cassandra(*, embedding_model: Embeddings, api_endpoint: str, token: str, namespace: str) -> VectorStore:
    vector_store = AstraDBVectorStore(
            collection_name="document_vector_mapping",
            embedding=embedding_model,
            api_endpoint=api_endpoint,
            token=token,
            namespace=namespace,
            hybrid_search=HybridSearchMode.OFF,
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


def load_docs(rules_path: Path) -> list[Document]:
    rule_docs = []
    for filename in rules_path.glob("*.pdf"):
        print(f"Processing {filename}")
        rule_docs.extend(parse_pdf(filename))
    return rule_docs


class GameRulesAgentState(MessagesState):
    pass


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

        chain = self.chain.with_config({"callbacks": [LangchainTracer()]})
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
            contact_points=self.cfg["db"]["contact_points"],
            load_balancing_policy=DCAwareRoundRobinPolicy(local_dc=self.cfg["db"]["dc"]),
        )
        self.db_session = self.stack.enter_context(self.db_cluster.connect())
        self.stack.callback(self.db_session.shutdown)

        # Setup chainlit datalayer
        data_layer = CassandraDataLayer(session=self.db_session, storage_client=None, keyspace=self.cfg["db"]["chainlit_keyspace"])
        data_layer.setup(replication_factor=self.cfg["db"]["replication_factor"])
        self.stack.push_async_callback(data_layer.close)
        self.data_layer = data_layer

        # Setup the checkpointer
        checkpointer = CassandraSaver(
            thread_id_type="uuid",
            keyspace=self.cfg["db"]["langgraph_keyspace"],
            session=self.db_session,
        )
        checkpointer.setup()
        self.checkpointer = checkpointer

        # Create keyspaces for document stores and vector stores
        if self.cfg["db"]["create_keyspaces"]:
            create_keyspace(
                data_api_endpoint=self.cfg["data_api"]["endpoint"],
                data_api_token=self.cfg["data_api"]["token"],
                keyspace=self.cfg["db"]["chainlit_keyspace"],
                replication_factor=self.cfg["db"]["replication_factor"],
            )
            create_keyspace(
                data_api_endpoint=self.cfg["data_api"]["endpoint"],
                data_api_token=self.cfg["data_api"]["token"],
                keyspace=self.cfg["db"]["langgraph_keyspace"],
                replication_factor=self.cfg["db"]["replication_factor"],
            )

        # Load the embedding
        self.embedding_model = OpenAIEmbeddings(
            model=self.cfg["embedding"]["model"],
            base_url=self.cfg["embedding"]["endpoint"],
            api_key=self.cfg["embedding"]["api_key"],
            tiktoken_enabled=False
        )
        
        # Setup vector store
        self.vector_store = buid_vectorstore_cassandra(
            embedding_model=self.embedding_model,
            api_endpoint=self.cfg["data_api"]["endpoint"],
            token=self.cfg["data_api"]["token"],
            namespace=self.cfg["data_api"]["namespace"],
        )

        # Setup doc store
        self.docstore = build_docstore_cassandra(
            api_endpoint=self.cfg["data_api"]["endpoint"],
            token=self.cfg["data_api"]["token"],
            namespace=self.cfg["data_api"]["namespace"],
        )

        # Load the tokenizer
        self.tokenizer = load_tokenizer(self.cfg["model_name"])

        # Build the retriever
        retriever = build_retriever(self.tokenizer, self.vector_store, docstore=self.docstore)

        # Optionally load documents
        if self.cfg["load_docs"]:
            rule_docs = load_docs(Path(self.cfg["rules_path"]))
            retriever.add_documents(rule_docs)

        # Build the chat model
        if self.cfg["chat"]["endpoint_type"] == "tgi":
            chat_model = load_tgi_chat_model(
                tokenizer=self.tokenizer,
                endpoint_url=self.cfg["chat"]["endpoint"],
                max_new_tokens=self.cfg["chat"]["max_new_tokens"],
                timeout=self.cfg["chat"]["timeout"],
                do_sample=False,
                temperature=0.01,
            )
        elif self.cfg["chat"]["endpoint_type"] == "openai":
            chat_model = ChatOpenAI(
                model=self.cfg["model_name"],
                max_tokens=self.cfg["chat"]["max_new_tokens"],
                temperature=0.0,
                timeout=self.cfg["chat"]["timeout"],
                base_url=self.cfg["chat"]["endpoint"],
                api_key=self.cfg["chat"].get("api_key", None),
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False,
                    }
                } if self.cfg["chat"].get("explicit_disable_thinking", False) else {}
            )
        else:
            raise ValueError(f"Unsupported chat endpoint type: {self.cfg['chat']['endpoint_type']}")

        # Build the qa chain
        self.chain = build_qa_chain(
            chat_model=chat_model,
            retriever=retriever,
            embedding_model=self.embedding_model,
            **self.cfg["qa_chain_config"]
        )

    async def stop(self):
        await self.stack.aclose()
