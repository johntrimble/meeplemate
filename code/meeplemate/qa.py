from typing import Optional
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda

from meeplemate.rag import build_thread_of_thought_rag_chain, build_rag_chain
from meeplemate.reword import build_reword_documents_chain
from meeplemate.consistency import build_universal_consistency_chain


def build_qa_chain(
        chat_model, 
        retriever,
        embedding_model,
        reword_documents=True,
        self_consistency=True,
        document_llm_filter=True,
        thread_of_thought=False,
        sampling_kwargs:Optional[dict]=None,
        consistency_kwargs:Optional[dict]=None,
        limit_number_of_documents:Optional[int]=5,
) -> Runnable:
    if sampling_kwargs is None:
        sampling_kwargs = {"temperature": 0.4, "top_k": 40, "top_p": 0.9}
    if consistency_kwargs is None:
        consistency_kwargs = {}

    if limit_number_of_documents is not None:
        retriever = retriever | RunnableLambda(lambda x: x[:limit_number_of_documents]).with_config({"run_name": "limit-documents"})

    if document_llm_filter:
        from langchain.retrievers.document_compressors import LLMChainFilter
        _filter = LLMChainFilter.from_llm(chat_model)
        retriever = (
            {
                "query": RunnablePassthrough(),
                "documents": RunnablePassthrough() | retriever,
            }
            | RunnableLambda(lambda x: _filter.compress_documents(x["documents"], x["query"])).with_config({"run_name": "filter-documents"})
        )

    sampling_chain = chat_model.bind(**sampling_kwargs)
    rag_sampling_chain = sampling_chain if self_consistency else chat_model
    if thread_of_thought:
        rag_chain = build_thread_of_thought_rag_chain(chat_model, rag_sampling_chain)
    else:
        rag_chain = build_rag_chain(chat_model, rag_sampling_chain)

    if reword_documents:
        reword_chat_chain = sampling_chain if self_consistency else chat_model
        reword_chain = build_reword_documents_chain(reword_chat_chain)
        rag_chain = RunnablePassthrough.assign(documents=itemgetter("documents") | reword_chain) | rag_chain
    if self_consistency:
        rag_chain = build_universal_consistency_chain(embedding_model, rag_chain, chat_model=chat_model, **consistency_kwargs)

    full_chain = (
        {
            "question": RunnablePassthrough(),
            "documents": RunnablePassthrough() | retriever,
        } 
        | rag_chain
    )

    return full_chain
