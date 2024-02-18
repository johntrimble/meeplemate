from pathlib import Path
from operator import itemgetter

import chainlit as cl
from langchain_core.output_parsers import StrOutputParser

from meeplemate.llm_models import load_tokenizer, load_tgi_chat_model, load_jina_embedding_model, sentence_transformer_to_hf_embeddings
from meeplemate.retrievers import build_retriever
from meeplemate.vectorstores import build_vectorstore_faiss
from meeplemate.qa import build_qa_chain
from meeplemate.pdf import parse_pdf

data_path = Path("./munchkin_rules/")


def load_docs():
    rule_docs = []
    for filename in data_path.glob("*.pdf"):
        print(f"Processing {filename}")
        rule_docs.extend(parse_pdf(filename))
    return rule_docs


@cl.cache
def get_chat_chain():
    model_name='teknium/OpenHermes-2.5-Mistral-7B'
    tokenizer = load_tokenizer(model_name)
    chat_model = load_tgi_chat_model(
        tokenizer=tokenizer,
        inference_server_url="http://tgi:80",
        max_new_tokens=512,
        timeout=900,
        do_sample=False,
        temperature=0.01,
    )

    embedding_model = load_jina_embedding_model()
    hf_embeddings = sentence_transformer_to_hf_embeddings(embedding_model, normalize_embeddings=True)
    db = build_vectorstore_faiss(hf_embeddings)
    retriever = build_retriever(tokenizer, db)
    rule_docs = load_docs()
    retriever.add_documents(rule_docs)
    
    chain = build_qa_chain(
        chat_model=chat_model,
        retriever=retriever,
        embedding_model=embedding_model,
        reword_documents=True,
        self_consistency=True,
        thread_of_thought=True,
    )

    return (chain | itemgetter("answer") | StrOutputParser())


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns: 
        None.
    """

    chain = get_chat_chain()
    chain = chain.with_config({"callbacks": [cl.LangchainCallbackHandler()]})
    output = await chain.ainvoke(message.content)

    # Send the final answer.
    await cl.Message(content=output).send()
