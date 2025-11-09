from types import MethodType
from typing import Any, Dict, List, Optional, Union, cast
import inspect
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs.chat_result import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_community.chat_models.huggingface import ChatHuggingFace
from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer,
)
import langchain_huggingface.chat_models as hfcm
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from pydantic import Field
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from text_generation.types import Details

import requests
from urllib.parse import urljoin

class HuggingFaceChatModelLocal(ChatHuggingFace):
    """
    For whatever reason, the parent class makes assumptions about certain
    endpoints existing even when running Text Gen Inference locally. This
    class is a workaround for that.
    """

    def _resolve_model_id(self):
        info_endpoint_url = urljoin(self.llm.inference_server_url, "info")
        info = requests.get(info_endpoint_url).json()
        model_id = info.get("model_id")
        if "Nous-Hermes-2-SOLAR-10.7B" in model_id:
            model_id = "NousResearch/Nous-Hermes-2-SOLAR-10.7B"
        elif "OpenHermes-2.5-Mistral-7B" in model_id:
            model_id = "teknium/OpenHermes-2.5-Mistral-7B"
        elif "Nous-Hermes-2-Yi-34B" in model_id:
            model_id = "NousResearch/Nous-Hermes-2-Yi-34B"

        self.model_id = model_id

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        # Call super class method
        chat_result = ChatHuggingFace._to_chat_result(llm_result)

        def attr_or_key(obj, attr, default=None):
            if hasattr(obj, attr):
                return getattr(obj, attr)
            elif hasattr(obj, "__getitem__") and attr in obj:
                return obj[attr]
            else:
                return default
        
        for g in chat_result.generations:
            details = g.generation_info.get("details")
            if details is not None:
                tokens = attr_or_key(details, "tokens") or []
                tokens = [token for token in tokens if not attr_or_key(token,"special", False)]
                token_ids = [attr_or_key(token,"id") for token in tokens]
                token_texts = [attr_or_key(token,"text") for token in tokens]
                token_logprobs = [attr_or_key(token,"logprob") for token in tokens]
                g.message.additional_kwargs["token_ids"] = token_ids
                g.message.additional_kwargs["token_texts"] = token_texts
                g.message.additional_kwargs["token_logprobs"] = token_logprobs
            # del g.generation_info["details"]

        return chat_result


def tgi_details_to_generation_details(tgi_details:Details) -> Dict[str, Any]:
    _tokens = tgi_details.tokens or []
    tokens = [dict(id=token.id, text=token.text, logprob=token.logprob, special=token.special) for token in _tokens]
    return {"tokens": tokens}


# class VLLMOpenAIChatModel(BaseChatModel):
#     llm: Union[BaseLLM]
#     system_message: SystemMessage = SystemMessage(content="""You are a helpful, respectful, and honest assistant.""")
#     tokenizer: Any = None

#     def _generate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         llm_input = self._to_chat_prompt(messages)
#         llm_result = self.llm._generate(
#             prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
#         )
#         return self._to_chat_result(llm_result)

#     async def _agenerate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         llm_input = self._to_chat_prompt(messages)
#         llm_result = await self.llm._agenerate(
#             prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
#         )
#         return self._to_chat_result(llm_result)

#     def _to_chat_prompt(
#         self,
#         messages: List[BaseMessage],
#     ) -> str:
#         """Convert a list of messages into a prompt format expected by wrapped LLM."""
#         if not messages:
#             raise ValueError("at least one HumanMessage must be provided")

#         if not isinstance(messages[-1], HumanMessage):
#             raise ValueError("last message must be a HumanMessage")

#         messages_dicts = [self._to_chatml_format(m) for m in messages]

#         return self.tokenizer.apply_chat_template(
#             messages_dicts, tokenize=False, add_generation_prompt=True
#         )

#     def _to_chatml_format(self, message: BaseMessage) -> dict:
#         """Convert LangChain message to ChatML format."""

#         if isinstance(message, SystemMessage):
#             role = "system"
#         elif isinstance(message, AIMessage):
#             role = "assistant"
#         elif isinstance(message, HumanMessage):
#             role = "user"
#         else:
#             raise ValueError(f"Unknown message type: {type(message)}")

#         return {"role": role, "content": message.content}

#     @staticmethod
#     def _to_chat_result(llm_result: LLMResult) -> ChatResult:
#         chat_generations = []

#         for g in llm_result.generations[0]:
#             chat_generation = ChatGeneration(
#                 message=AIMessage(content=g.text), generation_info=g.generation_info
#             )
#             chat_generations.append(chat_generation)

#         return ChatResult(
#             generations=chat_generations, llm_output=llm_result.llm_output
#         )

#     @property
#     def _llm_type(self) -> str:
#         return "vllm-openai-chat-wrapper"


def get_tgi_model_info(tgi_url: str) -> dict:
    response = requests.get(f"{tgi_url}/info")
    response.raise_for_status()
    return response.json()


def load_tgi_chat_model(**kwargs):
    from langchain_huggingface.chat_models import ChatHuggingFace
    from langchain_huggingface.llms import HuggingFaceEndpoint

    # These keys should be passed to the chat model instead of the LLM model
    chat_model_keys = [
        "system_message",
        "tokenizer",
        "callbacks",
        "callback_manager",
        "tags",
        "metadata",
    ]

    # Instantiate the LLM completion model
    llm_kwargs = {
        k: v 
        for k, v in kwargs.items() 
        if k not in chat_model_keys
    }
    # llm_model = EnhancedHuggingFaceTextGenInference(
    #     **llm_kwargs
    # )
    llm_model = HuggingFaceEndpoint(
        **llm_kwargs
    )

    # Wrap the LLM with the chat model
    chat_model_kwargs = {
        k: v 
        for k, v in kwargs.items() 
        if k in chat_model_keys
    }

    if "tokenizer" not in chat_model_kwargs or chat_model_kwargs["tokenizer"] is None:
        endpoint_url = kwargs["endpoint_url"]
        info = get_tgi_model_info(endpoint_url)
        model_id = info["model_id"]
        tokenizer = load_tokenizer(model_id)
        chat_model_kwargs["tokenizer"] = tokenizer

    # chat_model = HuggingFaceChatModelLocal(
    #     llm=llm_model,
    #     **chat_model_kwargs,
    # )
    chat_model = ChatHuggingFace(
        llm=llm_model,
        **chat_model_kwargs,
    )

    # Patch the chat model if it does not support system prompts
    if not _does_tokenizer_support_system_prompt(chat_model.tokenizer):
        chat_model = monkey_patch_huggingface_chat_model_no_system_prompt(chat_model)

    return chat_model


# def load_vllm_chat_model(inference_server_url, timeout=900, temperature=0, **kwargs):
#     from langchain_community.llms.vllm import VLLMOpenAI

#     chat_model_keys = [
#         "system_message",
#         "tokenizer",
#         "callbacks",
#         "callback_manager",
#         "tags",
#         "metadata",
#     ]

#     llm_kwargs = {
#         k: v
#         for k, v in kwargs.items()
#         if k not in chat_model_keys
#     }
#     llm_model = VLLMOpenAI(
#         model="teknium/OpenHermes-2.5-Mistral-7B",
#         openai_api_key="EMPTY",
#         openai_api_base=inference_server_url,
#         temperature=temperature,
#         timeout=timeout,
#         **llm_kwargs
#     )

#     chat_model_kwargs = {
#         k: v 
#         for k, v in kwargs.items() 
#         if k in chat_model_keys
#     }
#     chat_model = VLLMOpenAIChatModel(
#         llm=llm_model,
#         **chat_model_kwargs,
#     )

#     return chat_model

    # from langchain_community.chat_models.openai import ChatOpenAI
    # return ChatOpenAI(
    #     model="teknium/OpenHermes-2.5-Mistral-7B",
    #     openai_api_key="EMPTY",
    #     openai_api_base=inference_server_url,
    #     temperature=0,
    #     timeout=timeout,
    #     **kwargs,
    # )


def _merge_system_prompt_into_first_user_message(
    messages: List[Dict[str, str]],
    separator: str = "\n\n"
) -> List[Dict[str, str]]:
    """
    Some models do not support a system prompt. This function merges any leading
    system messages into the first user message. The messages are modified
    in place and also returned for convenience.
    """
    # Bail early if there is nothing to do
    if len(messages) < 2:
        return messages

    # Scan for the first non-system message
    idx_first_non_system = None
    for i, message in enumerate(messages):
        if message["role"] != "system":
            idx_first_non_system = i
            break

    # Nothing to do if we only have system messages
    if idx_first_non_system is None:
        return messages
    
    # Nothing to do if the first non-system message is the first message
    if idx_first_non_system == 0:
        return messages
    
    # Nothing to do if the first non-system message is not a user message
    if messages[idx_first_non_system]["role"] != "user":
        return messages
    
    # Gather the leading system messages
    system_messages = messages[:idx_first_non_system]

    # Drop the leading system messages (mutating the list)
    for _ in range(idx_first_non_system):
        messages.pop(0)
    
    # Merge the system messages together and prepend to the first user message
    system_prompt = separator.join([m["content"] for m in system_messages])
    messages[0]["content"] = system_prompt + separator + messages[0]["content"]

    return messages


def _does_tokenizer_support_system_prompt(tokenizer:Any) -> bool:
    # If already monkey patched, then we know the tokenizer does not support
    # system prompts
    if getattr(tokenizer, "_apply_chat_template_monkey_patched", False):
        return False

    # Attempt to apply a chat template with a system prompt. If it works,
    # then we are good. If it raises an error, then we need to merge the
    # system prompt into the first user message.
    try:
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        return True
    except Exception:
        return False


def monkey_patch_huggingface_chat_model_no_system_prompt(chat_model:hfcm.ChatHuggingFace) -> hfcm.ChatHuggingFace:
    original_create_message_dicts = chat_model._create_message_dicts
    def _patched_create_message_dicts(
        self, messages: list[BaseMessage], stop: Optional[list[str]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages_dicts, params = original_create_message_dicts(messages, stop)
        messages_dicts = _merge_system_prompt_into_first_user_message(messages_dicts)
        return messages_dicts, params

    chat_model._create_message_dicts = MethodType(_patched_create_message_dicts, chat_model)
    return chat_model


def maybe_wrap_tokenizer_no_system_prompt(tokenizer:Any) -> Any:
    """
    Some models do not support providing a system prompt. This function checks
    and, if needed, wraps the tokenizer to merge any provided system prompt
    into the user prompt.
    """
    original_apply_chat_template = tokenizer.apply_chat_template

    def _patched_apply_chat_template(
        self,
        *args,
        **kwargs,
    ) -> Any:
        if "conversation" in kwargs:
            conversation = kwargs["conversation"]
        else:
            if len(args) == 0:
                raise ValueError("No conversation provided")
            conversation = args[0]
        
        # If there there are no messages, just delegate the call
        if len(conversation) == 0:
            return original_apply_chat_template(*args, **kwargs)
        
        # Check if the conversation is a batch
        is_batch = isinstance(conversation[0], list)
        if not is_batch:
            conversation = [conversation]
        
        # Process each conversation in the batch
        for messages in conversation:
            # Mutate the messages in place
            _merge_system_prompt_into_first_user_message(messages)

        return original_apply_chat_template(*args, **kwargs)

    if not _does_tokenizer_support_system_prompt(tokenizer):
        tokenizer._apply_chat_template_monkey_patched = True
        tokenizer.apply_chat_template = MethodType(_patched_apply_chat_template, tokenizer)

    return tokenizer


def remove_quantized_suffixes(model_id: str) -> str:
    suffixes = ["-4bit", "-8bit", "-gptq", "-gguf", "-AWQ", "-INT4"]
    
    while True:
        removed_suffix = False
        for suffix in suffixes:
            if model_id.endswith(suffix):
                model_id = model_id[: -len(suffix)]
                removed_suffix = True
        if not removed_suffix:
            break

    return model_id


def matches_cannonical_name(cannonical_name: str, model_id: str) -> bool:
    cannonical_name = remove_quantized_suffixes(cannonical_name)
    model_id = remove_quantized_suffixes(model_id)

    # Exact match
    if cannonical_name == model_id:
        return True

    _, cannon_suffix = cannonical_name.split("/")
    _, model_suffix = model_id.split("/")

    # Only a different repository name
    if cannon_suffix.lower() == model_suffix.lower():
        return True
    
    return False


def load_tokenizer(model_name:str) -> PreTrainedTokenizerBase:
    # Sometimes when models are repackaged they screw up the tokenizer config,
    # so fix it here
    cannonical_tokenizer_names = [
        "teknium/OpenHermes-2.5-Mistral-7B",
        "NousResearch/Nous-Hermes-2-SOLAR-10.7B"
    ]

    for cannonical_name in cannonical_tokenizer_names:
        if matches_cannonical_name(cannonical_name, model_name):
            model_name = cannonical_name
            break

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    if tokenizer.pad_token is None:
        # For some reason, this isn't set in the config. For Mistral, it's just
        # the EOS token (which is the default). However, with OpenHermes, the
        # EOS token is a different token, but the padding token appears to still
        # be </s>:
        #
        # https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B/blob/main/special_tokens_map.json
        #
        # So if it is not set, we just set it explicitly to </s> here.
        tokenizer.pad_token = '</s>'

    # Wrap the tokenizer if it does not support system prompts
    tokenizer = maybe_wrap_tokenizer_no_system_prompt(tokenizer)

    return tokenizer


def load_jina_embedding_model() -> SentenceTransformer:
    """
    Load the Jina SentenceTransformer model for inference on the CPU and without
    gradients.
    """
    model = SentenceTransformer(
        "jinaai/jina-embeddings-v2-base-en",
        trust_remote_code=True,
        device="cpu",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    return model


class SentenceTransformerEmbeddings(Embeddings):
    """
    Wrap the SentenceTransformer model in a class that mimics the HuggingFace
    model so that it can be used in the same way. This code comes from
    langchain HuggingFaceEmbeddings, but adapted to wrap an already instantiated
    model.
    """
    model: SentenceTransformer #: :meta private:
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, model:SentenceTransformer, encode_kwargs:Optional[Dict[str, Any]] = None):
        self.model = model
        self.encode_kwargs = encode_kwargs or {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        texts = list(map(lambda x: x.replace("\n", " "), texts))

        embeddings = self.model.encode(texts, **self.encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

def sentence_transformer_to_hf_embeddings(model: SentenceTransformer, **kwargs) -> HuggingFaceEmbeddings:
    """
    Wrap the SentenceTransformer model so that it can be used as a langchain
    Embeddings model.
    """
    # This isn't really a HuggingFaceEmbeddings, but it has the same interface
    # so we can cast it for convenience
    return cast(
        HuggingFaceEmbeddings, 
        SentenceTransformerEmbeddings(model=model, encode_kwargs=kwargs)
    )
