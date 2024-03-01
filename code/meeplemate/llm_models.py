from typing import Any, Dict, List, Optional, Union
import inspect
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    BaseCallbackManager,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.chat_models import ChatResult, ChatGeneration
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_community.llms.huggingface_text_gen_inference import HuggingFaceTextGenInference
from langchain_community.chat_models.huggingface import ChatHuggingFace
from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from text_generation.types import Details

class HuggingFaceChatModelLocal(ChatHuggingFace):
    """
    For whatever reason, the parent class makes assumptions about certain
    endpoints existing even when running Text Gen Inference locally. This
    class is a workaround for that.
    """
    
    tokenizer: Any
    
    def _resolve_model_id(self):
        pass

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        # Call super class method
        chat_result = ChatHuggingFace._to_chat_result(llm_result)
        
        for g in chat_result.generations:
            details = g.get("generation_info", {}).get("details")
            if details is not None:
                tokens = details.get("tokens", [])
                tokens = [token for token in tokens if not token.get("special", False)]
                token_ids = [t["id"] for t in tokens]
                token_texts = [t["text"] for t in tokens]
                token_logprobs = [t["logprob"] for t in tokens]
                g.message.additional_kwargs["token_ids"] = token_ids
                g.message.additional_kwargs["token_texts"] = token_texts
                g.message.additional_kwargs["token_logprobs"] = token_logprobs
            del g.generation_info["details"]

        return chat_result


def tgi_details_to_generation_details(tgi_details:Details) -> Dict[str, Any]:
    _tokens = tgi_details.tokens or []
    tokens = [dict(id=token.id, text=token.text, logprob=token.logprob, special=token.special) for token in _tokens]
    return {"tokens": tokens}


class EnhancedHuggingFaceTextGenInference(HuggingFaceTextGenInference):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("This defunct method should no longer be called.")
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        raise NotImplementedError("This defunct method should no longer be called.")

    def _call_return_generation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Generation:
        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return Generation(text=completion)

        invocation_params = self._invocation_params(stop, **kwargs)
        res = self.client.generate(prompt, **invocation_params)
        # remove stop sequences from the end of the generated text
        for stop_seq in invocation_params["stop_sequences"]:
            if stop_seq in res.generated_text:
                res.generated_text = res.generated_text[
                    : res.generated_text.index(stop_seq)
                ]
        generation_info = {"details": tgi_details_to_generation_details(res.details)}
        return Generation(text=res.generated_text, generation_info=generation_info)

    async def _acall_return_generation(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Generation:
        if self.streaming:
            completion = ""
            async for chunk in self._astream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return Generation(text=completion)

        invocation_params = self._invocation_params(stop, **kwargs)
        res = await self.async_client.generate(prompt, **invocation_params)
        # remove stop sequences from the end of the generated text
        for stop_seq in invocation_params["stop_sequences"]:
            if stop_seq in res.generated_text:
                res.generated_text = res.generated_text[
                    : res.generated_text.index(stop_seq)
                ]
        generation_info = {"details": res.details}
        return Generation(text=res.generated_text, generation_info=generation_info)
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.
        generations = []
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        for prompt in prompts:
            generation = (
                self._call_return_generation(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else self._call_return_generation(prompt, stop=stop, **kwargs)
            )
            generations.append([generation])
        return LLMResult(generations=generations)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        for prompt in prompts:
            generation = (
                await self._acall_return_generation(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else await self._acall_return_generation(prompt, stop=stop, **kwargs)
            )
            generations.append([generation])
        return LLMResult(generations=generations)


class VLLMOpenAIChatModel(BaseChatModel):
    llm: Union[BaseLLM]
    system_message: SystemMessage = SystemMessage(content="""You are a helpful, respectful, and honest assistant.""")
    tokenizer: Any = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = self.llm._generate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_input = self._to_chat_prompt(messages)
        llm_result = await self.llm._agenerate(
            prompts=[llm_input], stop=stop, run_manager=run_manager, **kwargs
        )
        return self._to_chat_result(llm_result)

    def _to_chat_prompt(
        self,
        messages: List[BaseMessage],
    ) -> str:
        """Convert a list of messages into a prompt format expected by wrapped LLM."""
        if not messages:
            raise ValueError("at least one HumanMessage must be provided")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("last message must be a HumanMessage")

        messages_dicts = [self._to_chatml_format(m) for m in messages]

        return self.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )

    def _to_chatml_format(self, message: BaseMessage) -> dict:
        """Convert LangChain message to ChatML format."""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    @staticmethod
    def _to_chat_result(llm_result: LLMResult) -> ChatResult:
        chat_generations = []

        for g in llm_result.generations[0]:
            chat_generation = ChatGeneration(
                message=AIMessage(content=g.text), generation_info=g.generation_info
            )
            chat_generations.append(chat_generation)

        return ChatResult(
            generations=chat_generations, llm_output=llm_result.llm_output
        )

    @property
    def _llm_type(self) -> str:
        return "vllm-openai-chat-wrapper"



def load_tgi_chat_model(**kwargs):
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
    llm_model = EnhancedHuggingFaceTextGenInference(
        **llm_kwargs
    )

    # Wrap the LLM with the chat model
    chat_model_kwargs = {
        k: v 
        for k, v in kwargs.items() 
        if k in chat_model_keys
    }
    chat_model = HuggingFaceChatModelLocal(
        llm=llm_model,
        **chat_model_kwargs,
    )

    return chat_model


def load_vllm_chat_model(inference_server_url, timeout=900, temperature=0, **kwargs):
    from langchain_community.llms.vllm import VLLMOpenAI

    chat_model_keys = [
        "system_message",
        "tokenizer",
        "callbacks",
        "callback_manager",
        "tags",
        "metadata",
    ]

    llm_kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in chat_model_keys
    }
    llm_model = VLLMOpenAI(
        model="teknium/OpenHermes-2.5-Mistral-7B",
        openai_api_key="EMPTY",
        openai_api_base=inference_server_url,
        temperature=temperature,
        timeout=timeout,
        **llm_kwargs
    )

    chat_model_kwargs = {
        k: v 
        for k, v in kwargs.items() 
        if k in chat_model_keys
    }
    chat_model = VLLMOpenAIChatModel(
        llm=llm_model,
        **chat_model_kwargs,
    )

    return chat_model

    # from langchain_community.chat_models.openai import ChatOpenAI
    # return ChatOpenAI(
    #     model="teknium/OpenHermes-2.5-Mistral-7B",
    #     openai_api_key="EMPTY",
    #     openai_api_base=inference_server_url,
    #     temperature=0,
    #     timeout=timeout,
    #     **kwargs,
    # )


def load_tokenizer(model_name:str) -> PreTrainedTokenizerBase:
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

def sentence_transformer_to_hf_embeddings(model: SentenceTransformer, **kwargs) -> Embeddings:
    """
    Wrap the SentenceTransformer model so that it can be used as a langchain
    Embeddings model.
    """
    # normalize_embeddings=True
    return SentenceTransformerEmbeddings(model=model, encode_kwargs=kwargs)
