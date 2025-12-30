
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from langchain_classic.output_parsers.regex import RegexParser

from meeplemate.ingest.gamepackage import amap, get_document_page_aiter
from meeplemate.ingest.ocr import GamePackage
from meeplemate.util import achain, aenumerate, apairwise, arepeat, atakewhile

from structlog import get_logger


logger = get_logger(__name__)


DOCUMENT_TEMPLATE = """\
<document name="{name}" page="{page_number}">
{content}
</document>
"""

SUMMARY_SYSTEM_TEMPLATE = """\
You are an expert at board game design and rule analysis. Your goal is to create a concise and informative summary of the rules for the board game "{game_name}" based on the provided rulebook content. The summary should help players quickly understand the game's objectives, setup, and key mechanics. You will receive a current summary ond one or more rulebook pages that should be used to update and improve the summary.
"""

SUMMARY_TEMPLATE = """\
## Current Summary

<summary>
{current_summary}
</summary>

## Rulebook Pages

{rulebook_pages}

## Instructions

- Read the current summary and the provided rulebook pages carefully.
- Provide a step-by-step analysis of how to update the summary based on the new information from the rulebook pages.
- Identify any important rules, mechanics, or clarifications that are missing from the current summary.
- Update the summary to include these new details while ensuring it remains concise and easy to understand.
- Maintain a neutral and informative tone throughout the summary.
- Ensure that the summary is well-organized, using headings or bullet points if necessary to enhance readability.
- Avoid adding unnecessary details or overly complex explanations; focus on the most relevant information for players.
- Ignore fluff or non-essential content from the rulebook pages that do not contribute to understanding the game's rules.
- The final summary should be comprehensive yet succinct, providing players with a clear understanding of the game's rules.
- If there is no new information to add, retain and output the current summary as is.
- Keep summary length to a maximum of 500 words.
- Output the updated summary in markdown format.

## Formatting

- Output reasoning steps as a bulleted list.
- Output the summary enclosed within <summary> and </summary> tags.
"""

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM_TEMPLATE),
        ("human", SUMMARY_TEMPLATE),
    ]
)

@dataclass
class GenerateGameReferenceJob:
    output_dir: Path
    gp: GamePackage
    chat_model: BaseChatModel
    tokenizer: Any

    async def run(self):
        # We need to be careful here. We want pairs of consecutive pages to
        # ensure we don't split important context. However, if we just use
        # apairwise, we might not process any pages is there's only one page.

        # Get the page iterator
        page_iter = get_document_page_aiter(self.gp)

        # Convert pages to string content
        document_prompt = PromptTemplate.from_template(
            DOCUMENT_TEMPLATE,
            template_format="f-string",
        )
        def page_to_string(page: Document) -> str:
            s = document_prompt.format(
                name=page.metadata.get("rulebook_name", "Unknown"),
                page_number=page.metadata.get("page_num", -1),
                content=page.page_content,
            )
            return s
        page_iter = amap(page_to_string, page_iter)

        # Have the iterator produce None when exhausted
        page_iter = achain(page_iter, arepeat(None))

        # Now we get pairs
        page_pairs_iter = apairwise(page_iter)

        # Only take pairs while the first element is not None
        # This will give us all non-None pairs for documents with 2+ pages and
        # a single pair of (page, None) for documents with 1 page.
        page_pairs_iter = atakewhile(lambda pair: pair[0] is not None, page_pairs_iter)

        output_parser = RegexParser(
            regex=r".*<summary>([\s\S]*)</summary>.*",
            output_keys=["summary"],
        )

        summary_chain = (
            SUMMARY_PROMPT | self.chat_model | output_parser
        ).with_retry()

        current_summary = ""
        async for page1, page2 in page_pairs_iter:
            assert page1 is not None
            if page2 is None:
                pages_content = page1
            else:
                assert page2 is not None
                pages_content = '\n'.join([page1, page2])
            try:
                output = await summary_chain.ainvoke(
                    {
                        "game_name": self.gp["name"],
                        "current_summary": current_summary,
                        "rulebook_pages": pages_content,
                    }
                )
                current_summary = output["summary"]
                print(f"Updated Summary:\n{current_summary}\n\n\n")
            except:
                logger.exception("Failed to process pages")


EXTRACT_TERMINOLOGY_SYSTEM_TEMPLATE = """\
You are an expert at board game design and rule analysis. Your goal is to extract and define key terminology used in the rules for the board game "{{game_name}}" based on the provided rulebook content. The extracted terminology should help players quickly understand important terms and concepts used in the game's rules. You will receive a current list of terminology definitions and one or more rulebook pages that should be used to update and improve the terminology list. This list will act as both a glossary and an index for the game's rules.
"""

EXTRACT_TERMINOLOGY_TEMPLATE = """\
## Current Terminology

<terminology>
{{current_terminology}}
</terminology>

## Rulebook Pages

{{#documents}}
<document name="{{metadata.rulebook_name}}" page="{{metadata.page_num}}">
{{page_content}}
</document>
{{/documents}}

## Instructions

- Read the current terminology list and the provided rulebook pages carefully.
- Provide a step-by-step analysis of how to update the terminology list based on the new information from the rulebook pages.
- Identify any important terms or concepts that are missing from the current terminology list.
- For each new term, provide a clear and concise definition based on the rulebook content.
- Update the terminology list to include these new definitions while ensuring it remains concise and easy to understand.
- Maintain a neutral and informative tone throughout the terminology list.
- Keep each definition brief and to the point, about one or two sentences.
- Use bullet points for each term and its definition.
- Ignore fluff or non-essential content from the rulebook pages that do not contribute to understanding key terminology.
- The final terminology list should be comprehensive yet succinct, providing players with a clear understanding of important terms used in the game's rules.
- If there is no new terminology to add, retain and output the current terminology list as is.

## Formatting

- Output reasoning steps as a bulleted list.
- Output the terminology list enclosed within <terminology> and </terminology> tags.
"""


EXTRACT_TERMINOLOGY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", EXTRACT_TERMINOLOGY_SYSTEM_TEMPLATE),
        ("human", EXTRACT_TERMINOLOGY_TEMPLATE),
    ],
    template_format="mustache"
)


@dataclass
class ExtractTerminologyJob:
    output_dir: Path
    gp: GamePackage
    chat_model: BaseChatModel
    tokenizer: Any
    summary_tag: str = "terminology"

    async def run(self):
        logger.info("Starting terminology extraction job")

        # We need to be careful here. We want pairs of consecutive pages to
        # ensure we don't split important context. However, if we just use
        # apairwise, we might not process any pages is there's only one page.

        # Get the page iterator
        page_iter = get_document_page_aiter(self.gp)

        # Convert pages to dictionaries for use with the prompt template
        page_iter = amap(Document.model_dump, page_iter)

        # Have the iterator produce None when exhausted
        page_iter = achain(page_iter, arepeat(None))

        # Now we get pairs
        page_pairs_iter = apairwise(page_iter)

        # Only take pairs while the first element is not None
        # This will give us all non-None pairs for documents with 2+ pages and
        # a single pair of (page, None) for documents with 1 page.
        page_pairs_iter = atakewhile(lambda pair: pair[0] is not None, page_pairs_iter)

        # Define summary tag regex, being sure to escape any special characters
        # in the tag name
        escaped_tag = re.escape(self.summary_tag)
        regex_pattern = rf".*<{escaped_tag}>([\s\S]*)</{escaped_tag}>.*"

        output_parser = RegexParser(
            regex=regex_pattern,
            output_keys=["summary"],
        )

        summary_chain = (
            EXTRACT_TERMINOLOGY_PROMPT | self.chat_model | output_parser
        )

        current_summary = ""
        async for idx, (page1, page2) in aenumerate(page_pairs_iter):
            try:
                input = {
                    "game_name": self.gp["name"],
                    "current_terminology": current_summary,
                    "documents": [
                        page for page in [page1, page2] if page is not None
                    ],
                }
                logger.info(f"Processing page pair", idx=idx, input=input)
                output = await summary_chain.ainvoke(input)

                logger.info(f"Received output for page pair", idx=idx, output=output)
                current_summary = output["summary"]
                print(f"Updated Summary:\n{current_summary}\n\n\n")
            except:
                logger.exception("Failed to process pages")

