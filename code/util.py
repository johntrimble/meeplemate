from pathlib import Path
from typing import List, Optional, Any, Dict
from langchain.chat_models.base import SimpleChatModel
from langchain.schema import BaseMessage, AIMessage, ChatGeneration, ChatResult, Document
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.text_splitter import TextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


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

            results = self.model.generate(input_ids, pad_token_id=self.tokenizer.pad_token_id, **generate_kwargs)

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
