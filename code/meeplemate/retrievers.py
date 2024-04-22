import uuid
from pathlib import Path
from typing import Any, List, Union

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from transformers import (
    PreTrainedTokenizerBase,
)
from langchain.storage import (InMemoryStore, LocalFileStore)
from langchain.retrievers import (ParentDocumentRetriever, MultiVectorRetriever)
from langchain.text_splitter import TextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable
from langchain.schema import Document
from langchain.storage import InMemoryByteStore

from meeplemate.question_generation import build_questions_for_documents_chain

class VectorStoreRetrieverWithTextSplitter(VectorStoreRetriever):
    text_splitter: TextSplitter

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        documents = self.text_splitter.split_documents(documents)
        return super().add_documents(documents, **kwargs)

class VectorStoreRetrieverWithTextSplitter(VectorStoreRetriever):
    text_splitter: TextSplitter

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        documents = self.text_splitter.split_documents(documents)
        return super().add_documents(documents, **kwargs)

RetrieverWithAddDocuments = Union[ParentDocumentRetriever, VectorStoreRetrieverWithTextSplitter]


def build_retriever(
    tokenizer:PreTrainedTokenizerBase,
    vectorstore:VectorStore,
    parent_chunk_size:int=500,
    parent_chunk_overlap:int=50,
    child_chunk_size:int=125,
    child_chunk_overlap:int=12,
    k:int=10,
):
    parent_chunk_size = parent_chunk_size
    parent_chunk_overlap = parent_chunk_overlap
    parent_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap
    )

    search_kwargs={"k": k}

    child_chunk_size = child_chunk_size

    if child_chunk_size > 0:
        child_chunk_overlap = child_chunk_overlap
        child_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap
        )
        store = InMemoryStore()
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs=search_kwargs,
        )
    else:
        retriever = VectorStoreRetrieverWithTextSplitter(
            vectorstore=vectorstore,
            text_splitter=parent_splitter,
            search_kwargs=search_kwargs,
        )

    return retriever


def build_retriever_for_documents(
    tokenizer:PreTrainedTokenizerBase,
    vectorstore:VectorStore,
    documents:List[Document],
    parent_chunk_size:int=500,
    parent_chunk_overlap:int=50,
    child_chunk_size:int=125,
    child_chunk_overlap:int=12,
    k:int=10,
    
):
    retriever = build_retriever(
        tokenizer,
        vectorstore,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        k=k,
    )
    retriever.add_documents(documents)
    return retriever


def build_retriever_with_hypothetical_questions_for_documents(
    tokenizer:PreTrainedTokenizerBase,
    vectorstore:VectorStore,
    chat_model:Runnable,
    documents:List[Document],
    chunk_size:int=500,
    chunk_overlap:int=50,
    k:int=10
):
    search_kwargs={"k": k, "fetch_k": int(k * 1.5)}

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_documents = text_splitter.split_documents(documents)

    store = InMemoryByteStore()
    id_key = "chunk_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs=search_kwargs,
    )

    question_chain = build_questions_for_documents_chain(chat_model)
    hypothetical_questions = question_chain.invoke(split_documents)
    doc_ids = [str(uuid.uuid4()) for _ in split_documents]
    question_docs = []

    for i, question_list in enumerate(hypothetical_questions):
        question_docs.extend(
            [Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in question_list]
        )
    retriever.vectorstore.add_documents(question_docs)
    retriever.docstore.mset(list(zip(doc_ids, split_documents)))
    
    return retriever


def save_faiss_retriever(retriever, outputpath):
    from copy import copy
    import pickle
    retriever = retriever.copy()
    vectorstore = copy(retriever.vectorstore)
    vectorstore.embedding_function = None
    retriever.vectorstore = vectorstore
    pickle.dump(retriever, open(outputpath, "wb"))


def load_faiss_retriever(embedding_model, inputpath):
    import pickle
    retriever = pickle.load(open(inputpath, "rb"))
    retriever.vectorstore.embedding_function = embedding_model
    return retriever
