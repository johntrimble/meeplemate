from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.docstore.in_memory import InMemoryDocstore

def build_vectorstore_faiss(embedding_model:HuggingFaceEmbeddings) -> VectorStore:
    import faiss
    embeddings = embedding_model.embed_documents(["test"])
    embedding_size = len(embeddings[0])
    distance_strategy = DistanceStrategy.EUCLIDEAN_DISTANCE
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embedding_model,
        index,
        InMemoryDocstore(),
        {},
        normalize_L2=False,
        distance_strategy=distance_strategy,
    )
    return vectorstore
