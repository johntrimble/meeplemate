from functools import partial
from operator import itemgetter
from typing import Any, Dict

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.schema import Document
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import chain
from langchain_core.prompts import format_document


rewrite_prompt_template = """Reword and improve clarity of the following excerpt from the board game Munchkin, without changing the meaning or the information conveyed. Use Markdown to format the text.

{context}

Rewritten text:"""

REWRITE_PROMPT = PromptTemplate(
    template=rewrite_prompt_template,
    input_variables=["context"],
)

REWRITE_DOCUMENT_PROMPT = PromptTemplate.from_template("""{page_content}""")

def build_document_content_transform_chain(
    transform_chain:Runnable,
    document_prompt:PromptTemplate,
):
    _format_document = partial(format_document, prompt=document_prompt)

    def create_new_document(data):
        old_doc = data["original_document"]
        new_doc = Document(page_content=data["context"], metadata=old_doc.metadata)
        return new_doc

    document_rewrite_chain = (
        RunnablePassthrough()
        | _format_document
        | transform_chain
    )

    chain = (
        {
            "original_document": RunnablePassthrough(),
            "context": document_rewrite_chain
        }
        | RunnableLambda(create_new_document)
    )

    return chain


def build_reword_documents_chain(
    chat_model:Runnable,
    prompt=REWRITE_PROMPT,
    document_prompt=REWRITE_DOCUMENT_PROMPT
) -> Runnable:
    from langchain_core.output_parsers import StrOutputParser
    
    transform_chain = {"context": RunnablePassthrough()} | prompt | chat_model | StrOutputParser()
    document_transform_chain = build_document_content_transform_chain(transform_chain, document_prompt=document_prompt)
    chain = document_transform_chain.map().with_config({"run_name": "reword-documents"})
    return chain


# summarize_prompt = """\
# Summarize the following rules from the board game Munchkin related to the given prompt. Do not change the meaning or the information conveyed. Do not answer the prompt, simply summarize the relevant information. Use Markdown to format the text.

# > Prompt: {question}
# > Rules:
# >>>
# {context}
# >>>
# > Summary (remember do not answer the prompt, simply summarize the relevant information. Use Markdown to format the text.):
# """

summarize_prompt = """\
Given the following question and context, extract any part of the context *AS IS* that is relevant to answer the question.

Remember, *DO NOT* edit the extracted parts of the context.

> Question: {question}
> Context:
>>>
{context}
>>>
Extracted relevant parts:\
"""


SUMMARIZE_PROMPT = PromptTemplate(
    template=summarize_prompt,
    input_variables=["question", "context"],
)



def build_summarize_chain(
    chat_model:Runnable,
    prompt=SUMMARIZE_PROMPT,
    document_prompt=REWRITE_DOCUMENT_PROMPT
) -> Runnable:
    
    from langchain_core.output_parsers import StrOutputParser
    
    @chain
    def get_inputs(input):
        documents = input["documents"]
        question = input["question"]

        inputs = [
            {"question": question, "context": doc.page_content}
            for doc in documents
        ]
        return inputs
    
    @chain
    def create_output(inputs):
        result = []
        for output, original_document in zip(inputs["outputs"], inputs["original_documents"]):
            result.append(
                Document(page_content=output, metadata=original_document.metadata)
            )
        return result


    transform_chain = prompt | chat_model | StrOutputParser()
    summarize_chain = (
        {
            "original_documents": itemgetter("documents"),
            "outputs": get_inputs | transform_chain.map()
        }
        | create_output
    )
    summarize_chain = summarize_chain.with_config({"run_name": "summarize-documents"})
    return summarize_chain
