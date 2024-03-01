from operator import itemgetter
from typing import Optional
from functools import partial

from langchain_core.prompts import ChatPromptTemplate, BasePromptTemplate, format_document
from langchain.schema import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.output_parsers import StrOutputParser

system_prompt_template = """\
As an AI Assistant specialized in board game rules, your primary goal is to provide users with precise, understandable explanations and interpretations of game rules. Follow these key principles:

- **Base your responses on the official rulebooks or authoritative sources,** recognizing that these rules hold in all standard situations unless an explicit exception is stated. Avoid assumptions and unofficial variations unless specifically requested by the user.

- **Use clear, simple language** to explain rules. Avoid unnecessary jargon and ensure explanations are accessible to both new and experienced players.

- **Consider the gameplay context** in your interpretations, including game phases, player counts, and specific scenarios that might impact rule application.

- **Highlight rule variants and exceptions** clearly, explaining how they alter standard gameplay and under what circumstances they apply.

- **Attempt to understand user intent,** focusing on the aspect of the rule they might find confusing or the specific information they seek.

Your responses should empower users with a deeper understanding of the game rules, enhancing their gameplay experience.
"""

# document_prompt_template = """---
# NAME: {source}
# PAGE: {page_number}
# PASSAGE:
# {page_content}
# ---"""

document_prompt_template = """{page_content}"""

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(document_prompt_template)

rag_prompt_template = """\
Answer the following board game question based on the given rules from the rulebook. Provide your step-by-step reasoning first followed by the answer. Each step should be a separate bullet point. Remember rules found in a board game rulebook generally hold unless there is an explicit exception.

> Context:
>>>
{context}
>>>
> Question: {question}
> Answer: Let's think step by step. \
"""

DEFAULT_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", rag_prompt_template),
    ]
)

follow_up_prompt_template = """\
Now revise your answer. Be concise and remove any unrelated information to the question.
"""

DEFAULT_ANSWER_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("assistant", "{answer}"),
        ("human", follow_up_prompt_template)
    ]
)

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

def remove_answer_prefix(answer_message):
    """
    Some times the LLM "helpfully" prefixes it's answer with something like
    "Answer:" or "A:". This function removes those prefixes.
    """
    answer = answer_message.content
    prefixes = ["Answer:", "answer:", "> Answer:", "> answer:", "A:", "a:"]
    for prefix in prefixes:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
            break
    
    answer_message.content = answer
    return answer_message

def build_rag_chain(chat_chain, sampling_chain=None, prompt=DEFAULT_RAG_PROMPT, document_prompt=DEFAULT_DOCUMENT_PROMPT, answer_extract_prompt=DEFAULT_ANSWER_EXTRACT_PROMPT):
    """
    Creates a chain that takes a set of documents and a question and returns an
    answer. The returned chain has the following inputs and outputs:

    Inputs:
    - documents: A list of documents to use as context for the question.
    - question: The question to answer.

    Outputs:
    - context: The combined context of the documents.
    - answer: The answer to the question.
    - question: The original question.
    """

    if sampling_chain is None:
        sampling_chain = chat_chain

    followup_chain = (prompt + answer_extract_prompt) | chat_chain | remove_answer_prefix
    followup_chain = followup_chain.with_config({"run_name": "extract-answer"})

    return (
        RunnablePassthrough.assign(
            context=RunnableLambda(itemgetter("documents")) | partial(_combine_documents, document_prompt=document_prompt)
        )
        | RunnablePassthrough.assign(
            answer=RunnablePassthrough() | prompt | sampling_chain
        )
        # If sampling chain return a dict of answer and logprobs, we want to
        # merge it into the main dict. Otherwise, it is just the answer, so 
        # leave it as is.
        | RunnableLambda(lambda x: {**x, **x["answer"]} if isinstance(x["answer"], dict) else x)
        | RunnablePassthrough.assign(answer=(itemgetter("answer") | StrOutputParser()))
        | RunnablePassthrough.assign(
            answer=followup_chain
        )
    ).with_config({"run_name": "rag-chain"})

build_basic_rag_chain = build_rag_chain


thread_of_thought_template = """\
Answer the question based only on the following board game rules. \
Do not use any other information.

> Rules:
>>>
{context}
>>>
> Question: {question}
Walk me through this context in manageable parts step by step, summarizing and \
analyzing as we go. Only afterwards, provide your answer.
"""

DEFAULT_THREAD_OF_THOUGHT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("human", thread_of_thought_template),
    ]
)

# DEFAULT_THEREFORE_PROMPT = ChatPromptTemplate.from_messages(
#     [
#         ("assistant", "{answer}"),
#         ("human", "Now revise your answer. Be concise and remove any unrelated information to the question.")
#     ]
# )
DEFAULT_THEREFORE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("assistant", "{answer}"),
        ("human", "And so the answer is:")
    ]
)

def build_thread_of_thought_rag_chain(
    chat_chain:Runnable,
    sampling_chain=None,
    prompt=DEFAULT_THREAD_OF_THOUGHT_PROMPT,
    document_prompt=DEFAULT_DOCUMENT_PROMPT,
    answer_extract_prompt=DEFAULT_THEREFORE_PROMPT
) -> Runnable:
    return build_rag_chain(
        chat_chain,
        sampling_chain=sampling_chain,
        prompt=prompt,
        document_prompt=document_prompt,
        answer_extract_prompt=answer_extract_prompt
    ).with_config({"run_name": "thread-of-thought-rag-chain"})
