from typing import Any, Dict, List, Optional
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

class HuggingFaceChatModelLocal(ChatHuggingFace):
    """
    For whatever reason, the parent class makes assumptions about certain
    endpoints existing even when running Text Gen Inference locally. This
    class is a workaround for that.
    """
    
    tokenizer: Any
    
    def _resolve_model_id(self):
        pass


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
    llm_model = HuggingFaceTextGenInference(
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
