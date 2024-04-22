from pathlib import Path
from typing import Optional
from urllib.parse import urljoin
import click
import requests

from meeplemate.llm_models import (
    load_tgi_chat_model,
    load_jina_embedding_model,
    sentence_transformer_to_hf_embeddings,
    load_tokenizer,
)
from meeplemate.pdf import parse_pdf
from meeplemate.retrievers import (
    build_retriever_for_documents,
    build_retriever_with_hypothetical_questions_for_documents,
    save_faiss_retriever
)
from meeplemate.vectorstores import build_vectorstore_faiss


def get_tokenizer(tgi_endpoint:str):
    if not tgi_endpoint.endswith("/"):
        tgi_endpoint += "/"
    info_endpoint_url = urljoin(tgi_endpoint, "info")
    info = requests.get(info_endpoint_url).json()
    model_id = info.get("model_id")
    if "Nous-Hermes-2-SOLAR-10.7B" in model_id:
        model_id = "NousResearch/Nous-Hermes-2-SOLAR-10.7B"
    elif "OpenHermes-2.5-Mistral-7B" in model_id:
        model_id = "teknium/OpenHermes-2.5-Mistral-7B"
    tokenizer = load_tokenizer(model_id)
    return tokenizer


@click.group()
def cli():
    pass


@cli.command()
@click.option('--type', type=click.Choice(['hypothetical_queries', 'parent_document'], case_sensitive=False), help="The type of retriever to build")
@click.option('--output_file', type=Path, help="The path to save the retriever to")
@click.option('--input_file', type=Path, help="The path to the input file")
@click.option('--tgi_endpoint', type=str, help="The endpoint to use for Text Generation Inference")
@click.option('--tokenizer_id', type=str, help="The id of the tokenizer to use")
def build_retriever(type:str, input_file:Path, output_file:Path, tgi_endpoint:Optional[str], tokenizer_id:Optional[str]):
    if input_file.is_dir():
        # Get all pdfs in the directory
        pdfs = list(input_file.glob("*.pdf"))
    else:
        pdfs = [input_file]
    
    documents = []
    for pdf_path in pdfs:
        documents.extend(parse_pdf(pdf_path))
    
    if tokenizer_id:
        tokenizer = load_tokenizer(tokenizer_id)
    elif tgi_endpoint:
        tokenizer = get_tokenizer(tgi_endpoint)
    else:
        raise ValueError("Either tgi_endpoint or tokenizer_id is required")

    jina_embedding_model = load_jina_embedding_model()
    hf_embedding_model = sentence_transformer_to_hf_embeddings(jina_embedding_model, normalize_embeddings=True)

    if type == "hypothetical_queries":
        assert tgi_endpoint, "tgi_endpoint is required for hypothetical_queries"
        chat_model = load_tgi_chat_model(
            tokenizer=tokenizer,
            inference_server_url=tgi_endpoint,
            max_new_tokens=512,
            timeout=900,
            do_sample=False,
            temperature=0.01,
        )
        db = build_vectorstore_faiss(hf_embedding_model)
        retriever = build_retriever_with_hypothetical_questions_for_documents(
            tokenizer,
            db,
            chat_model,
            documents,
            k=20
        )
    elif type == "parent_document":
        db = build_vectorstore_faiss(hf_embedding_model)
        retriever = build_retriever_for_documents(
            tokenizer,
            db,
            documents,
            k=20
        )
    else:
        raise ValueError(f"Invalid type: {type}")
    
    save_faiss_retriever(retriever, output_file)


if __name__ == "__main__":
    print("hello")
