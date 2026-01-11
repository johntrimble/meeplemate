
import asyncio
from dataclasses import dataclass
import functools
from pathlib import Path
import re
from typing import Any, AsyncIterator, Tuple

from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from langchain_classic.output_parsers.regex import RegexParser

from meeplemate.ingest.gamepackage import amap, get_document_page_aiter, save_manifest, document_keys
from meeplemate.ingest.ocr import GamePackage, aspit
from meeplemate.util import achain, aenumerate, apairwise, arepeat, atakewhile, compose, queue_to_async_iter, sink_into_queue, pipeline, to_async_iter, xf_amap

from structlog import get_logger


logger = get_logger(__name__)


DOCUMENT_TEMPLATE = """\
<document name="{name}" page="{page_number}">
{content}
</document>
"""

SUMMARY_SYSTEM_TEMPLATE = """\
You are an expert at board game design and rule analysis. Your goal is to create a concise and informative summary of the rules for the board game "{{game_name}}" based on the provided rulebook content. The summary should help players quickly understand the game's objectives, setup, and key mechanics. You will receive a current summary ond one or more rulebook pages that should be used to update and improve the summary.
"""

SUMMARY_TEMPLATE = """\
## Current Summary

<summary>
{{current_summary}}
</summary>

## Rulebook Pages

{{#documents}}
<document name="{{metadata.rulebook_name}}" page="{{metadata.page_num}}">
{{page_content}}
</document>
{{/documents}}

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
    ],
    template_format="mustache"
)


short_summary_template = """\
## Current Summary

<summary>
{{#documents}}
{{page_content}}
{{/documents}}
</summary>


## Instructions

- Read the current summary and produce an updated version that is only 2-3 sentences long.
- Provide a step-by-step analysis of how to update the summary based on the new information from the rulebook pages.
- Keep summary length to a maximum of 2-3 sentences.
- Output the updated summary in markdown format.

## Formatting

- Output reasoning steps as a bulleted list.
- Output the summary enclosed within <summary> and </summary> tags.
"""


SHORT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM_TEMPLATE),
        ("human", short_summary_template),
    ],
    template_format="mustache"
)


def game_summary_path(gp: GamePackage) -> Path:
    return gp["path"] / "summary.md"


def rulebook_summary_path(gp: GamePackage, document_key: str) -> Path:
    return gp["path"] / document_key / "summary.md"


def rulebook_short_summary_path(gp: GamePackage, document_key: str) -> Path:
    return gp["path"] / document_key / "summary.short.md"


async def generate_summary_with_refinement(
    chat_model: BaseChatModel,
    prompt: BasePromptTemplate,
    documents: AsyncIterator[Document],
    summary_tag: str = "summary",
):
    # We need to be careful here. We want pairs of consecutive pages to
    # ensure we don't split important context. However, if we just use
    # apairwise, we might not process any pages is there's only one page.

    # Convert pages to dictionaries for use with the prompt template
    page_iter = amap(Document.model_dump, documents)

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
    escaped_tag = re.escape(summary_tag)
    regex_pattern = rf".*<{escaped_tag}>([\s\S]*)</{escaped_tag}>.*"

    output_parser = RegexParser(
        regex=regex_pattern,
        output_keys=["summary"],
    )

    # Run the summary chain on successive page pairs, refining the summary
    # at each step
    summary_chain = (
        prompt | chat_model | output_parser
    )

    current_summary = ""
    async for idx, (page1, page2) in aenumerate(page_pairs_iter):
        try:
            input = {
                "current_summary": current_summary,
                "documents": [
                    page for page in [page1, page2] if page is not None
                ],
            }
            logger.info(f"Processing page pair", idx=idx, input=input)
            output = await summary_chain.ainvoke(input)
            logger.info(f"Received output for page pair", idx=idx, output=output)
            current_summary = output["summary"]
        except:
            logger.exception("Failed to process pages")

    return current_summary.strip()


@dataclass
class GenerateGameReferenceJob:
    output_dir: Path
    gp: GamePackage
    chat_model: BaseChatModel
    tokenizer: Any

    async def run(self):
        async def run_summary_for_document(document_key_wrapped: Tuple[str|None]) -> Tuple[str|None, str]:
            prompt = SUMMARY_PROMPT.partial(game_name=self.gp["name"])

            document_key = document_key_wrapped[0]
            logger.info(f"Starting summary generation for document", document_key=document_key)
            document_iter = get_document_page_aiter(self.gp, document_key=document_key)
            summary = await generate_summary_with_refinement(
                chat_model=self.chat_model,
                prompt=prompt,
                documents=document_iter,
                summary_tag="summary",
            )
            logger.info(f"Finished summary generation for document", document_key=document_key)
            return (document_key, summary)


        async def run_summary_of_summary(item: Tuple[str|None, str]) -> Tuple[str|None, str, str]:
            # We don't generate a short summary for the overall game
            document_key, summary = item
            if document_key is None:
                return (document_key, summary, summary)

            document = Document(
                page_content=summary,
            )
            documents_iter = to_async_iter([document])
            prompt = SHORT_SUMMARY_PROMPT.partial(game_name=self.gp["name"])
            short_summary = await generate_summary_with_refinement(
                chat_model=self.chat_model,
                prompt=prompt,
                documents=documents_iter,
                summary_tag="summary",
            )
            return (document_key, summary, short_summary)


        async def write_summaries(item: Tuple[str|None, str, str]) -> Tuple[str|None, str, str]:
            document_key, summary, short_summary = item
            if document_key is None:
                path = game_summary_path(self.gp)
            else:
                path = rulebook_summary_path(self.gp, document_key)
            logger.info(f"Writing summary to path", path=path)
            path.parent.mkdir(exist_ok=True)
            await aspit(summary, path)

            if document_key is not None:
                short_path = rulebook_short_summary_path(self.gp, document_key)
                logger.info(f"Writing short summary to path", path=short_path)
                short_path.parent.mkdir(exist_ok=True)
                await aspit(short_summary, short_path)
            else:
                short_path = None

            return item


        # Create tasks to run summaries for each document key
        input_queue: asyncio.Queue[Tuple[str|None]] = asyncio.Queue()
        items: list[Tuple[str|None]] = [(document_key, ) for document_key in document_keys(self.gp)]

        # Add one with null document key for game summary
        items.append((None,))

        # Start producing into the queue
        producer_task = asyncio.create_task(
            sink_into_queue(input_queue, items)
        )

        # Generate summaries
        done_queue: asyncio.Queue[Tuple[str|None, str, str]] = asyncio.Queue()
        process_task = asyncio.create_task(
            pipeline(
                done_queue,
                compose(
                    xf_amap(run_summary_for_document),
                    xf_amap(run_summary_of_summary),
                    xf_amap(write_summaries),
                ),
                input_queue,
                concurrency=5,
            )
        )

        # Update the manifest with summaries
        gp = self.gp
        summary_items_iter = queue_to_async_iter(done_queue)
        async for item in summary_items_iter:
            if isinstance(item, Exception):
                raise item
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError(f"Invalid summary item received: {item}")
            document_key, summary, short_summary = item
            # Update the game summary
            if document_key is None:
                gp["summary"] = summary
            else:
                # Update the rulebook summary
                for rulebook in gp["rulebooks"]:
                    if rulebook["document_key"] == document_key:
                        rulebook["summary"] = short_summary
                        break

        # Double check all tasks are complete
        logger.info("Waiting for summary tasks to complete")
        await process_task
        await producer_task

        # Write updated manifest
        logger.info("Writing updated manifest with summaries")
        save_manifest(gp)


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

        # Run the summary chain on successive page pairs, refining the summary
        # at each step
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
