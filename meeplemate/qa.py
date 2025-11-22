from typing import Any, Dict, Optional
from operator import itemgetter
import re

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import chain
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from meeplemate.rag import build_thread_of_thought_rag_chain, build_rag_chain
from meeplemate.reword import build_reword_documents_chain, build_summarize_chain
from meeplemate.consistency import build_universal_consistency_chain
from langchain_core.language_models.base import BaseLanguageModel, LanguageModelInput
from langchain_classic.output_parsers import BooleanOutputParser


def build_sampling_chain(chat_model:BaseChatModel, **kwargs):
    return chat_model.bind(**kwargs)

    # def chat_model_output_with_logprobs(input: LanguageModelInput, config: RunnableConfig) -> Dict[str, Any]:
    #     llmresult = chat_model.generate_prompt(
    #         [chat_model._convert_input(input)],
    #         callbacks=config.get("callbacks"),
    #         tags=config.get("tags"),
    #         metadata=config.get("metadata"),
    #         run_name=config.get("run_name"),
    #         **kwargs
    #     )
    #     chat_generation = llmresult.generations[0][0]
    #     info = chat_generation.generation_info
    #     if logprobs and "logprobs" in info:
    #         logprobs_dict = info["logprobs"]
    #         _logprobs = logprobs_dict["token_logprobs"]
    #         return {output_key: chat_generation.message, logprobs_output_key: _logprobs}
    #     elif logprobs and "details" in info:
    #         _logprobs = [token["logprob"] for token in info["details"]["tokens"]]
    #         return {
    #             output_key: chat_generation.message, logprobs_output_key: _logprobs}
    #     else:
    #         return {output_key: chat_generation.message}
    
    # return RunnableLambda(chat_model_output_with_logprobs)


class BooleanOutputTrueOnErrorParser(BooleanOutputParser):
    def parse(self, text: str) -> bool:
        """Parse the output of an LLM call to a boolean.

        Args:
            text: output of a language model

        Returns:
            boolean
        """
        regexp = rf"\b({self.true_val}|{self.false_val})\b|^({self.true_val}|{self.false_val})"

        truthy = {
            val.upper() if isinstance(val, str) else val[0].upper()
            for val in re.findall(regexp, text, flags=re.IGNORECASE | re.MULTILINE)
        }
        try:
            if self.true_val.upper() in truthy:
                if self.false_val.upper() in truthy:
                    msg = (
                        f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                        f"in received: {text}."
                    )
                    raise ValueError(msg)
                return True
            if self.false_val.upper() in truthy:
                if self.true_val.upper() in truthy:
                    msg = (
                        f"Ambiguous response. Both {self.true_val} and {self.false_val} "
                        f"in received: {text}."
                    )
                    raise ValueError(msg)
                return False
            # When in doubt, return True
            return True
        except:
            return True

    @property
    def _type(self) -> str:
        """Snake-case string identifier for an output parser type."""
        return "boolean_output_true_on_error_parser"


def build_qa_chain(
        chat_model, 
        retriever,
        embedding_model:Embeddings,
        reword_documents=True,
        self_consistency=True,
        document_llm_filter=True,
        document_llm_compress=False,
        thread_of_thought=False,
        sampling_kwargs:Optional[dict]=None,
        consistency_kwargs:Optional[dict]=None,
        cot_kwargs:Optional[dict]=None,
        limit_number_of_documents:Optional[int]=5,
) -> Runnable:
    if sampling_kwargs is None:
        sampling_kwargs = {"temperature": 0.2}

    if consistency_kwargs is None:
        consistency_kwargs = {}
    
    if cot_kwargs is None:
        cot_kwargs = {}

    if limit_number_of_documents is not None:
        retriever = retriever | RunnableLambda(lambda x: x[:limit_number_of_documents]).with_config({"run_name": "limit-documents"})

    if document_llm_filter:
        from langchain_classic.retrievers.document_compressors.chain_filter import LLMChainFilter, _get_default_chain_prompt
        filter_prompt = _get_default_chain_prompt()
        filter_prompt.output_parser = BooleanOutputTrueOnErrorParser()
        _filter = LLMChainFilter.from_llm(chat_model, prompt=filter_prompt)
        retriever = (
            {
                "query": RunnablePassthrough(),
                "documents": retriever,
            }
            | RunnableLambda(lambda x: _filter.compress_documents(x["documents"], x["query"])).with_config({"run_name": "filter-documents"})
        )

    rag_sampling_chain = build_sampling_chain(chat_model, **sampling_kwargs) if self_consistency else chat_model
    if thread_of_thought:
        rag_chain = build_thread_of_thought_rag_chain(chat_model, rag_sampling_chain, **cot_kwargs)
    else:
        rag_chain = build_rag_chain(chat_model, rag_sampling_chain, **cot_kwargs)

    if reword_documents:
        sampling_chain = chat_model.bind(**sampling_kwargs)
        reword_chat_chain = sampling_chain if self_consistency else chat_model
        reword_chain = RunnablePassthrough.assign(
            documents=itemgetter("documents") | build_reword_documents_chain(reword_chat_chain)
        )

        if document_llm_compress:
            summarization_chain = build_summarize_chain(chat_model)
            summarization_chain = RunnablePassthrough.assign(
                documents=summarization_chain
            )
            reword_chain = reword_chain | summarization_chain

        rag_chain = reword_chain | rag_chain

    if self_consistency:
        rag_chain = build_universal_consistency_chain(embedding_model, rag_chain, chat_model=chat_model, **consistency_kwargs)

    full_chain = (
        {
            # Make default game_name "Munchkin" for backwards compatibility
            # TODO: Do not hardcode this
            "game_name": lambda _: "Munchkin",
            "question": RunnablePassthrough(),
            "documents": retriever,
        }
        | rag_chain
    )

    return full_chain
