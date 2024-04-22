import re

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

hypothetical_question_template = """\
Given the following document, generate a comprehensive set of questions that \
cover its key points, facts, and themes. Structure each question in a simple, \
numbered list format, beginning with "Q1:", followed by "Q2:", and so on. \
Ensure that each question is concise, directly related to the content of the \
document, and formulated to facilitate document retrieval based on the \
information queried by the question. Start your response with "Generated \
Questions:" and maintain a clear, consistent numbering format for ease of \
parsing.

Document:

{doc}
"""

hypothetical_question_prompt = ChatPromptTemplate.from_template(
    hypothetical_question_template
)

question_regex = re.compile(r"^(?:Q?\d+[.:]\s+)+(.*?)$")


def clean_question_text(text):
    # Remove extra white space
    text = ' '.join(text.split())

    # Remove leading and trailing white space
    text = text.strip()

    return text



@chain
def parse_questions(input):
    questions = []
    lines = input.split("\n")
    for line in lines:
        match = question_regex.match(line)
        if match:
            questions.append(clean_question_text(match.group(1)))
    return questions


def build_questions_for_documents_chain(chat_model, prompt=hypothetical_question_prompt):
    _chain = (
        {"doc": RunnablePassthrough() | RunnableLambda(lambda x: x.page_content)}
        | prompt
        | chat_model
        | StrOutputParser()
        | parse_questions
    )
    _chain = _chain.map()
    return _chain
