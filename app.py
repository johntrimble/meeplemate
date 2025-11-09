from pathlib import Path
from operator import itemgetter

import chainlit as cl
from chainlit.data.base import BaseDataLayer
from langchain_astradb import AstraDBByteStore, AstraDBStore, AstraDBVectorStore
from langchain_core.output_parsers import StrOutputParser

from chainlit_cassandra_data_layer.data import CassandraDataLayer
from cassandra_asyncio.cluster import Cluster
from cassandra.cluster import Session
from cassandra.policies import DCAwareRoundRobinPolicy
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_astradb.utils.astradb import HybridSearchMode

from meeplemate.cassandra_util import AstraDBSerializableStore

from meeplemate.llm_models import load_tokenizer, load_tgi_chat_model, load_jina_embedding_model, sentence_transformer_to_hf_embeddings
from meeplemate.retrievers import build_retriever
from meeplemate.vectorstores import build_vectorstore_faiss
from meeplemate.qa import build_qa_chain
from meeplemate.pdf import parse_pdf


REPLICATION_FACTOR = 1
ASTRA_VECTOR_STORE_CREATE_KEYSPACE = True
ASTRA_VECTOR_STORE_API_ENDPOINT = "http://data-api:8181"
ASTRA_VECTOR_STORE_TOKEN = "Cassandra:Cg==:Cg=="
ASTRA_VECTOR_STORE_NAMESPACE = "meeplemate"

rules_path = Path("./data/rules/munchkin_rules/")

db_session: Session | None = None
data_layer: CassandraDataLayer | None = None


async def start_data_layer():
    global db_session, data_layer
    cluster = Cluster(
        ['cassandra'],
        load_balancing_policy=DCAwareRoundRobinPolicy(local_dc='datacenter1'),
        # protocol_version=5,
        allow_beta_protocol_version=True,
    )
    db_session = cluster.connect()
    data_layer = CassandraDataLayer(db_session, storage_client=None, keyspace="chainlit_meeplemate")
    data_layer.setup(replication_factor=1)


async def stop_data_layer():
    global db_session, data_layer
    if data_layer is not None:
        try:
            await data_layer.close()
        except Exception as e:
            print(f"Error closing data layer: {e}")
    if db_session is not None:
        try:
            db_session.shutdown()
        except Exception as e:
            print(f"Error shutting down db session: {e}")
    
    db_session = None
    data_layer = None


retriever: BaseRetriever | None = None
chain: Runnable | None = None


def buid_vectorstore_cassandra(*, embedding: Embeddings, api_endpoint: str, token: str, namespace: str) -> VectorStore:
    vector_store = AstraDBVectorStore(
            collection_name="document_vector_mapping",
            embedding=embedding,
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


async def start_chain():
    global chain, retriever

    if ASTRA_VECTOR_STORE_CREATE_KEYSPACE:
        import requests
        url = f"{ASTRA_VECTOR_STORE_API_ENDPOINT}/v1"
        headers = {
            "TOKEN": ASTRA_VECTOR_STORE_TOKEN,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        resp = requests.post(
            url=url,
            headers=headers,
            json={
                "createKeyspace": {
                    "name": ASTRA_VECTOR_STORE_NAMESPACE,
                    "options": {
                        "replication": {
                            "class": "SimpleStrategy",
                            "replication_factor": REPLICATION_FACTOR
                        }
                    }
                }
            }
        )
        resp.raise_for_status()

    model_name='teknium/OpenHermes-2.5-Mistral-7B'
    tokenizer = load_tokenizer(model_name)
    chat_model = load_tgi_chat_model(
        tokenizer=tokenizer,
        # inference_server_url="http://tgi:80",
        endpoint_url="http://tgi:80",
        max_new_tokens=512,
        timeout=900,
        do_sample=False,
        temperature=0.01,
    )

    embedding_model = load_jina_embedding_model()
    hf_embeddings = sentence_transformer_to_hf_embeddings(embedding_model, normalize_embeddings=True)
    db = buid_vectorstore_cassandra(
        embedding=hf_embeddings,
        api_endpoint=ASTRA_VECTOR_STORE_API_ENDPOINT,
        token=ASTRA_VECTOR_STORE_TOKEN,
        namespace=ASTRA_VECTOR_STORE_NAMESPACE,
    )
    docstore = build_docstore_cassandra(
        api_endpoint=ASTRA_VECTOR_STORE_API_ENDPOINT,
        token=ASTRA_VECTOR_STORE_TOKEN,
        namespace=ASTRA_VECTOR_STORE_NAMESPACE,
    )
    # db = build_vectorstore_faiss(hf_embeddings)
    retriever = build_retriever(tokenizer, db, docstore=docstore)
    rule_docs = load_docs()
    retriever.add_documents(rule_docs)

    _chain = build_qa_chain(
        chat_model=chat_model,
        retriever=retriever,
        embedding_model=embedding_model,
        reword_documents=True,
        self_consistency=True,
        thread_of_thought=True,
    )

    chain = (_chain | itemgetter("answer") | StrOutputParser())


async def stop_chain():
    global chain, vector_store_retriever
    chain = None
    vector_store_retriever = None


def get_chain() -> Runnable:
    global chain
    if chain is None:
        raise ValueError("Chain is not initialized.")
    return chain


@cl.on_app_startup
async def startup():
    await start_data_layer()
    await start_chain()


@cl.on_app_shutdown
async def shutdown():
    await stop_chain()
    await stop_data_layer()


@cl.data_layer
def get_data_layer() -> BaseDataLayer:
    global data_layer
    if data_layer is None:
        raise ValueError("Data layer is not initialized.")
    return data_layer


def load_docs():
    rule_docs = []
    for filename in rules_path.glob("*.pdf"):
        print(f"Processing {filename}")
        rule_docs.extend(parse_pdf(filename))
    return rule_docs


@cl.set_chat_profiles
async def chat_profile(user=None):
    return [
        cl.ChatProfile(
            name="GPT-3.5",
            markdown_description="The underlying LLM model is **GPT-3.5**.",
            icon="https://picsum.photos/200",
        ),
        cl.ChatProfile(
            name="GPT-4",
            markdown_description="The underlying LLM model is **GPT-4**.",
            icon="https://picsum.photos/250",
        ),
    ]


async def add_mock_user():
    # Create and set mock user
    mock_user = cl.User(
        identifier="mock_user_001",
        display_name="Mock Developer",
        metadata={"environment": "development"}
    )
    cl.context.session.user = mock_user


@cl.on_chat_start  # this function will be called when the user opens the UI
async def start():
    await add_mock_user()
    game_options = ["Munchkin", "Boss Monster", "Secret Hitler"]
    from chainlit.input_widget import Select
    await cl.ChatSettings(
        inputs=[Select(id="game_select", label="Choose a game", values=game_options, initial_index=0)]
    ).send()
    await cl.Message(content="Welcome to Meeplmate! Pick a game in ⚙️ Chat Settings. The default is Munchkin.").send()


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns: 
        None.
    """
    # thread_id = message.thread_id
    # assert thread_id
    from meeplemate.chainlit_utils import LangchainTracer
    chain = get_chain()
    chain = chain.with_config({"callbacks": [LangchainTracer()]})
    output = await chain.ainvoke(message.content)

    # Send the final answer.
    await cl.Message(content=output).send()
