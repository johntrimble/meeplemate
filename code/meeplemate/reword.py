from functools import partial
from operator import itemgetter

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.schema import Document
from langchain_core.runnables import Runnable
from langchain_core.prompts import format_document


rewrite_prompt_template = """Reword and improve clarity of the following excerpt from the board game Munchkin, without changing the meaning or the information conveyed. Use Markdown to format the text.

> Context:
>>>
{context}
>>>
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