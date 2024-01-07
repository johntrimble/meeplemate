from __future__ import annotations

from abc import abstractmethod
import asyncio

from functools import partial
from pathlib import Path
from typing import List, Optional, Any, Dict, Union, cast, final

import torch
from langchain.chat_models.base import SimpleChatModel, BaseChatModel
from langchain.schema import BaseMessage, AIMessage, ChatGeneration, ChatResult, Document
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from langchain.text_splitter import TextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.runnables import RunnableConfig, get_config_list
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.load import dumpd, dumps
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.pipelines import ConversationalPipeline, Conversation

# BaseChatModel does not appear to have any particularly inspired way of
# handling batches. You have a batch of 10 inputs? Great, that will be 10 calls
# to invoke! These calls will happpen in parallel, which is probably okay if you
# are hitting some sort of external API, but it is not great if you have loaded
# a model locally as you'll want to group together these inputs and hand them
# off to the model... in a batch... Also, it would seem to still make sense to
# have some special handling of batches if, for nothing else, to limit the
# amount of concurrency. If some one has a list of 10,000 inputs, do you really
# want to kick off 10,000 concurrent API calls? Maybe the expectation is that
# the client will block as needed to prevent things from getting out of hand. Or
# maybe you can configure the executor with the passed config type? I don't
# know, I'm just a guy trying to give a few inputs to an llm.
#
# At any rate, this new base class adds a place for us to stick some fancy batch
# handling logic.

class ChatModelWithBatchSupport(BaseChatModel):
    def batch(
        self,
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[BaseMessage]:
        if not inputs:
            return []

        config = get_config_list(config, len(inputs))
        max_concurrency = config[0].get("max_concurrency")

        if max_concurrency is None:
            try:
                llm_result = self.generate_prompt(
                    [self._convert_input(input) for input in inputs],
                    callbacks=[c.get("callbacks") for c in config],
                    tags=[c.get("tags") for c in config],
                    metadata=[c.get("metadata") for c in config],
                    run_name=[c.get("run_name") for c in config],
                    **kwargs,
                )
                return [AIMessage(content=g[0].text) for g in llm_result.generations]
            except Exception as e:
                if return_exceptions:
                    return cast(List[BaseMessage], [e for _ in inputs])
                else:
                    raise e
        else:
            batches = [
                inputs[i : i + max_concurrency]
                for i in range(0, len(inputs), max_concurrency)
            ]
            config = [{**c, "max_concurrency": None} for c in config]  # type: ignore[misc]
            return [
                output
                for i, batch in enumerate(batches)
                for output in self.batch(
                    batch,
                    config=config[i * max_concurrency : (i + 1) * max_concurrency],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            ]
    
    async def abatch(
        self,
        inputs: List[LanguageModelInput],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> List[BaseMessage]:
        if not inputs:
            return []
        config = get_config_list(config, len(inputs))
        max_concurrency = config[0].get("max_concurrency")

        if max_concurrency is None:
            try:
                llm_result = await self.agenerate_prompt(
                    [self._convert_input(input) for input in inputs],
                    callbacks=[c.get("callbacks") for c in config],
                    tags=[c.get("tags") for c in config],
                    metadata=[c.get("metadata") for c in config],
                    run_name=[c.get("run_name") for c in config],
                    **kwargs,
                )
                return [AIMessage(content=g[0].text) for g in llm_result.generations]
            except Exception as e:
                if return_exceptions:
                    return cast(List[BaseMessage], [e for _ in inputs])
                else:
                    raise e
        else:
            batches = [
                inputs[i : i + max_concurrency]
                for i in range(0, len(inputs), max_concurrency)
            ]
            config = [{**c, "max_concurrency": None} for c in config]  # type: ignore[misc]
            return [
                output
                for i, batch in enumerate(batches)
                for output in await self.abatch(
                    batch,
                    config=config[i * max_concurrency : (i + 1) * max_concurrency],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            ]

    def generate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Union[Callbacks, List[Callbacks]]] = None,
        *,
        tags: Optional[Union[List[str], List[List[str]]]] = None,
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        run_name: Optional[Union[str, List[str]]] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        # A bunch of code I copied from the base class
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}

                # Create callback managers
        if (
            isinstance(callbacks, list)
            and callbacks
            and (
                isinstance(callbacks[0], (list, BaseCallbackManager))
                or callbacks[0] is None
            )
        ):
            # We've received a list of callbacks args to apply to each input
            assert len(callbacks) == len(messages)
            assert tags is None or (
                isinstance(tags, list) and len(tags) == len(messages)
            )
            assert metadata is None or (
                isinstance(metadata, list) and len(metadata) == len(messages)
            )
            assert run_name is None or (
                isinstance(run_name, list) and len(run_name) == len(messages)
            )
            callbacks = cast(List[Callbacks], callbacks)
            tags_list = cast(List[Optional[List[str]]], tags or ([None] * len(messages)))
            metadata_list = cast(
                List[Optional[Dict[str, Any]]], metadata or ([{}] * len(messages))
            )
            run_name_list = run_name or cast(
                List[Optional[str]], ([None] * len(messages))
            )
            callback_managers = [
                CallbackManager.configure(
                    callback,
                    self.callbacks,
                    self.verbose,
                    tag,
                    self.tags,
                    meta,
                    self.metadata,
                )
                for callback, tag, meta in zip(callbacks, tags_list, metadata_list)
            ]
        else:
            # We've received a single callbacks arg to apply to all inputs
            callback_managers = [
                CallbackManager.configure(
                    cast(Callbacks, callbacks),
                    self.callbacks,
                    self.verbose,
                    cast(List[str], tags),
                    self.tags,
                    cast(Dict[str, Any], metadata),
                    self.metadata,
                )
            ] * len(messages)
            run_name_list = [cast(Optional[str], run_name)] * len(messages)

        run_managers = [
            callback_manager.on_chat_model_start(
                dumpd(self),
                conversation,
                invocation_params=params,
                options=options,
                name=run_name,
                batch_size=len(messages),
            )[0]
            for callback_manager, conversation, run_name in zip(
                callback_managers, messages, run_name_list
            )
        ]

        # This bit is different. Instead of iterating over the `messages` and
        # calling `_generate` on each one like, we pass the whole list to
        # `_generate_with_batch`. This allows us to do whatever fancy pants
        # batching we want to do in the subclass. 
        try:
            results = self._generate_with_batch(
                messages,
                stop=stop,
                run_manager=run_managers[0] if run_managers else None,
                **kwargs
            )
        except BaseException as e:
            for run_manager in run_managers:
                run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise e
        
        # More code I copied from the base class
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        if run_managers:
            run_infos = []
            for manager, flattened_output in zip(run_managers, flattened_outputs):
                manager.on_llm_end(flattened_output)
                run_infos.append(RunInfo(run_id=manager.run_id))
            output.run = run_infos
        return output

    async def agenerate(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        *,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Top Level call"""
        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop}

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        run_managers = await callback_manager.on_chat_model_start(
            dumpd(self),
            messages,
            invocation_params=params,
            options=options,
            name=run_name,
            batch_size=len(messages),
        )
        run_managers = [r[0] for r in run_managers]

        try:
            results = await self._agenerate_with_batch(
                messages,
                stop=stop,
                run_manager=run_managers[0] if run_managers else None,
                **kwargs,
            )   
        except BaseException as e:
            if run_managers:
                await asyncio.gather(
                    *[
                        run_manager.on_llm_error(e, response=LLMResult(generations=[]))
                        for run_manager in run_managers
                    ]
                )
            raise e

        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        await asyncio.gather(
            *[
                run_manager.on_llm_end(flattened_output)
                for run_manager, flattened_output in zip(
                    run_managers, flattened_outputs
                )
            ]
        )
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output

    async def _agenerate_with_batch(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> List[ChatResult]:
        return await asyncio.get_running_loop().run_in_executor(
            None,
            partial(self._generate_with_batch, **kwargs),
            messages,
            stop,
            run_manager
        )

    @abstractmethod
    def _generate_with_batch(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> List[ChatResult]:
        """Run the LLM on the given messages."""
    
    @final
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise Exception("Should not be possible to call this method")


def convert_langchain_messages_to_hf(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    type_to_role = {
        "ai": "assistant",
        "human": "user",
        "system": "system"
    }

    return [
        {
            "role": type_to_role[message.type],
            "content": message.content,
        }
        for message in messages
    ]


def convert_langchain_messages_to_hf_conversation(messages: List[BaseMessage]) -> Conversation:
    messages_hf = convert_langchain_messages_to_hf(messages)
    return Conversation(messages=messages_hf)


class HuggingFaceChatModelWithBatchSupport(ChatModelWithBatchSupport):
    pipeline: ConversationalPipeline
    batch_size: int = 4
    max_new_tokens: int = 512
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.95
    temperature: Optional[float] = 0.8
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = False
    model_kwargs: Dict[str, Any] = {}

    @property
    def _llm_type(self) -> str:
        return "hf-batch-chat"
    
    def _get_pipeline_params(self, **kwargs):
        prop_params = {
            "batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
        }

        token_params = {
            "eos_token_id": self.pipeline.tokenizer.eos_token_id,
            "pad_token_id": self.pipeline.tokenizer.pad_token_id, 
        }

        params = {
            **prop_params,
            **token_params,
            **self.model_kwargs,
            **kwargs,
        }

        if not params.get("do_sample", False):
            # If we have disabled sampling, then we don't need to pass any of
            # the sampling params
            for sampling_param in ["temperature", "top_k", "top_p"]:
                if sampling_param in params:
                    del params[sampling_param]
        
        return params
    
    def _call_pipeline(self, conversations: List[Conversation], **kwargs):
        params = self._get_pipeline_params(**kwargs)
        return self.pipeline(conversations, **params)

    def _generate_with_batch(
        self,
        messages: List[List[BaseMessage]],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> List[ChatResult]:
        conversations = [convert_langchain_messages_to_hf_conversation(conv) for conv in messages]

        results: List[ChatResult] = []
        for i in range(0, len(conversations), self.batch_size):
            batch = conversations[i:i+self.batch_size]
            responses = self._call_pipeline(batch, **kwargs)
            if not isinstance(responses, list):
                responses = [responses]
            for response in responses:
                text = response[-1]["content"]
                if stop:
                    text = enforce_stop_tokens(text, stop)
                results.append(
                    ChatResult(generations=[
                        ChatGeneration(message=AIMessage(content=text))
                    ])
                )

        return results
    
    @classmethod
    def from_model_tokenizer(
        cls,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 4,
        pipeline_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ) -> HuggingFaceChatModelWithBatchSupport:
        _pipeline_kwargs = pipeline_kwargs or {}
        pipeline = ConversationalPipeline(model=model, tokenizer=tokenizer, batch_size=batch_size, **_pipeline_kwargs)
        return cls(pipeline=pipeline, batch_size=batch_size, **kwargs)


class HuggingFaceChatModel(SimpleChatModel):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    generate_kwargs: Dict[str, Any]

    # These mimic the properties in the ChatOpenAI class
    temperature: Optional[float] = None
    n: Optional[int] = None
    max_tokens: Optional[int] = None

    # Other model params
    top_k: Optional[int] = None

    _supports_system_prompt: Optional[bool] = None

    @property
    def supports_system_prompt(self) -> bool:
        if self._supports_system_prompt is None:
            if (
                self.model.name_or_path 
                and self.model.name_or_path.startswith(
                    "mistralai/Mistral-7B-Instruct-"
                )
            ):
                return False
        return True
    
    # @supports_system_prompt.setter
    # def supports_system_prompt(self, value: bool) -> None:
    #     self._supports_system_prompt = value
    
    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        sequences = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        generations = [ChatGeneration(message=AIMessage(content=sequence)) for sequence in sequences]
        return ChatResult(generations=generations)

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        with torch.no_grad():
            type_to_role = {
                "ai": "assistant",
                "human": "user",
                "system": "system"
            }

            messages_hf = [
                {
                    "role": type_to_role[message.type],
                    "content": message.content,
                }
                for message in messages
            ]

            # When using the base mistral model, we will get an error if we try
            # to use the system prompt (presumably because it is not a chat
            # model). To fix this, we will just merge the system prompt with the
            # the first message.
            if not self.supports_system_prompt:
                if messages_hf[0]["role"] == "system":
                    messages_to_combine = [messages_hf.pop(0)]
                    if len(messages_hf) > 0:
                        messages_to_combine.append(messages_hf.pop(0))

                    role = messages_to_combine[-1]["role"]
                    if role == "system":
                        role = "user"

                    combined_message = {
                        "role": role,
                        "content": "\n\n".join(
                            m["content"] for m in messages_to_combine
                        )
                    }

                    messages_hf.insert(0, combined_message)

            input_ids = self.tokenizer.apply_chat_template(
                messages_hf,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.model.device)

            rename_kwargs = {
                "max_tokens": "max_new_tokens",
                "n": "num_return_sequences"
            }

            props_kwargs = {
                "temperature": self.temperature,
                "n": self.n,
                "max_tokens": self.max_tokens,
                "top_k": self.top_k,
            }

            # Ignore property values that are None
            props_kwargs = {k: v for k, v in props_kwargs.items() if v is not None}

            props_kwargs.update(kwargs)

            for old, new in rename_kwargs.items():
                if old in props_kwargs:
                    props_kwargs[new] = props_kwargs.pop(old)

            generate_kwargs = {
                **self.generate_kwargs, 
                **props_kwargs,
                "return_dict_in_generate": True
            }

            if (
                any(key in generate_kwargs for key in ["top_k", "top_p", "temperature", "num_return_sequences"])
            ):
                generate_kwargs["do_sample"] = True
            
            if generate_kwargs.get("temperature") == 0:
                generate_kwargs["do_sample"] = False
                del generate_kwargs["temperature"]

            results = self.model.generate(
                input_ids, 
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id, 
                **generate_kwargs
            )

            sequences = [s[len(input_ids[0]):] for s in results["sequences"]]
            sequences = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
            if stop:
                sequences = [enforce_stop_tokens(s, stop) for s in sequences]

            return sequences


class VectorStoreRetrieverWithTextSplitter(VectorStoreRetriever):
    text_splitter: TextSplitter

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        documents = self.text_splitter.split_documents(documents)
        return super().add_documents(documents, **kwargs)


def has_sentence_end(s):
    s = s.strip()
    enders = ['.', '?', '!', '.)', '?)', '!)', '...']
    return any(s.endswith(ender) for ender in enders)


def fix_dangling_sentences(elements):
    """Fixes sentences that are split across multiple elements"""
    sequence = []
    for el in elements:
        previous = sequence[-1] if sequence else None
        if el.metadata["category"] == "NarrativeText" or has_sentence_end(el.page_content):
            if previous and previous.metadata["category"] == "NarrativeText":
                if not has_sentence_end(previous.page_content):
                    sequence[-1].page_content += " " + el.page_content
                    continue
        sequence.append(el)
    return sequence


def organize_into_sections(elements):
    sections = [[]]
    for element in elements:
        if element.metadata["category"] == "Title":
            sections.append([element])
        else:
            sections[-1].append(element)
    sections = list(filter(None, sections))
    return sections


def sections_to_pages(sections):
    pages = []
    current_page = -1
    for section in sections:
        if section:
            page = section[0].metadata["page_number"]
            if page != current_page:
                pages.append([section])
                current_page = page
            else:
                pages[-1].append(section)
    return pages


def pages_to_documents(pages):
    documents = []
    for page in pages:
        if page:
            page_content = build_string_from_sections(page)
            metadata = page[0][0].metadata
            page_number = metadata.get('page_number', -1)
            source = metadata.get('source', None)
            documents.append(Document(page_content=page_content, metadata={"page_number": page_number, "source": source}))
    return documents


def build_string_from_sections(sections):
    return "\n\n".join(
        "\n".join(element.page_content for element in section)
        for section in sections
    )


def fix_document_metadata(document):
    metadata = document.metadata
    for k in metadata:
        if not isinstance(metadata[k], (str, int, float, bool)):
            metadata[k] = str(metadata[k])
    return document


def parse_pdf(path:Path) -> List[Document]:
    # To parse the PDFs, there are three strategies available: "fast", "hi_res", and
    # "ocr_only". For the PDFs used here, "fast" retrieves a bunch of duplicate text
    # in the wrong order. "hi_res" doesn't handle columns of text well and produces
    # incoherent results. "ocr_only" seems to work reasonably well in this case.
    loader = UnstructuredPDFLoader(path, mode="elements", strategy="ocr_only")
    result = loader.load()
    result = fix_dangling_sentences(result)
    sections = organize_into_sections(result)
    pages = sections_to_pages(sections)
    # text = build_string_from_sections(sections)
    # metadata = loader._get_metadata()
    documents = pages_to_documents(pages)
    documents = [fix_document_metadata(document) for document in documents]
    return documents
