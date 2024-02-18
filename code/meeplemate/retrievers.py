from typing import Any, List, Union
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from transformers import (
    PreTrainedTokenizerBase,
)
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import TextSplitter
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.schema import Document

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