
import asyncio
from dataclasses import dataclass
import functools
from itertools import groupby
from pathlib import Path
import re
from typing import Any, AsyncIterator, Tuple, TypedDict
from unittest.mock import Base

from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

from langchain_classic.output_parsers.regex import RegexParser

from meeplemate.ingest.gamepackage import amap, get_document_page_aiter, get_game_example_questions_path, get_game_presentation_path, get_game_setting_summary_path, save_manifest, document_keys
from meeplemate.ingest.ocr import GamePackage, aspit
from meeplemate.util import achain, aenumerate, apairwise, arepeat, aslurp, aspit_yaml, atakewhile, compose, queue_to_async_iter, sink_into_queue, pipeline, to_async_iter, xf_amap

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

SETTING_SUMMARY_SYSTEM_TEMPLATE = """\
You are an expert at summarizing the genre, world, and setting of a board game based on its rulebook content. Your goal is to create a concise and engaging summary that captures the essence of the game's theme and setting, helping players quickly understand the game's world and what makes it unique. Focus exclusively on narrative, thematic, and atmospheric elements. Never include game mechanics, rules, card effects, turn procedures, or win conditions — even when described in narrative language.
""".strip()

SETTING_PAGE_BY_PAGE_SUMMARY_TEMPLATE = """\
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
- Identify any important details about the game's genre, world, or setting that are missing from the current summary.
- Update the summary to include these new details while ensuring it remains concise and easy to understand.
- Ensure that the summary is well-organized, using headings or bullet points if necessary to enhance readability.
- Avoid adding unnecessary details or overly complex explanations; focus on the most relevant information.
- Focus ONLY on: the game's genre, fictional world, story premise, tone, aesthetic, factions, characters, races, lore, and thematic atmosphere.
- Do NOT include: turn structure, card effects, combat rules, stat modifiers, win conditions, setup procedures, or any game rules — even when they are described using narrative or thematic language.
- The final summary should be comprehensive yet succinct, providing players with a clear understanding of the game's genre, world, and setting.
- If there is no new information to add, retain and output the current summary as is.
- Keep summary length to a maximum of 500 words.
- Output the updated summary in markdown format.

## Formatting

- Output reasoning steps as a bulleted list.
- Output the summary enclosed within <summary> and </summary> tags.
""".strip()

SETTING_PAGE_BY_PAGE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SETTING_SUMMARY_SYSTEM_TEMPLATE),
        ("human", SETTING_PAGE_BY_PAGE_SUMMARY_TEMPLATE),
    ],
    template_format="mustache"
)

SETTING_SUMMARY_OF_SUMMARIES_TEMPLATE = """\
## Current Summary

<summary>
{{current_summary}}
</summary>

## Document Summaries

{{#documents}}
<document name="{{metadata.rulebook_name}}">
{{page_content}}
</document>
{{/documents}}

## Instructions

- Read the current summary and the provided document summaries carefully.
- Provide a step-by-step analysis of how to update the summary based on the new information from the document summaries.
- Identify any important details about the game's genre, world, or setting that are missing from the current summary.
- If the input summaries contain any mechanical content (card effects, turn procedures, stat rules, win conditions), exclude it — only carry forward setting, world, and thematic details.
- Update the summary to include these new details while ensuring it remains concise and easy to understand.
- Focus ONLY on: genre, fictional world, story premise, tone, aesthetic, factions, characters, races, lore, and thematic atmosphere.
- Do NOT include: turn structure, card effects, combat rules, stat modifiers, win conditions, setup procedures, or any game rules — even when described in narrative language.
- Ensure that the summary is well-organized, using headings or bullet points if necessary to enhance readability.
- Avoid adding unnecessary details or overly complex explanations; focus on the most relevant information.
- The final summary should be comprehensive yet succinct, providing players with a clear understanding of the game's genre, world, and setting.
- If there is no new information to add, retain and output the current summary as is.
- Keep summary length to a maximum of 500 words.
- Output the updated summary in markdown format.

## Formatting

- Output reasoning steps as a bulleted list.
- Output the summary enclosed within <summary> and </summary> tags.
""".strip()

SETTING_SUMMARY_OF_SUMMARIES_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SETTING_SUMMARY_SYSTEM_TEMPLATE),
        ("human", SETTING_SUMMARY_OF_SUMMARIES_TEMPLATE),
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
    # apairwise, we might not process any pages if there's only one page.

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


async def aggregate_summaries(
    chat_model: BaseChatModel,
    prompt: BasePromptTemplate,
    documents: list[Document],
    summary_tag: str = "summary",
) -> str:
    escaped_tag = re.escape(summary_tag)
    regex_pattern = rf".*<{escaped_tag}>([\s\S]*)</{escaped_tag}>.*"
    output_parser = RegexParser(
        regex=regex_pattern,
        output_keys=["summary"],
    )
    chain = prompt | chat_model | output_parser
    input = {
        "current_summary": "",
        "documents": [doc.model_dump() for doc in documents],
    }
    output = await chain.ainvoke(input)
    return output["summary"].strip()


async def generate_hierarchical_summary_with_refinement(
    gp: GamePackage,
    chat_model: BaseChatModel,
    summary_prompt: BasePromptTemplate,
    summary_of_summaries_prompt: BasePromptTemplate,
    summary_tag: str = "summary",
):
    doc_keys = list(document_keys(gp))
    tasks = []
    # Generate summaries of each document
    for document_key in doc_keys:
        document_iter = get_document_page_aiter(gp, document_key=document_key)
        task = asyncio.create_task(
            generate_summary_with_refinement(
                chat_model=chat_model,
                prompt=summary_prompt,
                documents=document_iter,
                summary_tag=summary_tag,
            )
        )
        tasks.append(task)
    
    # Raise any exceptions that occurred during summary generation
    raw_summaries = await asyncio.gather(*tasks, return_exceptions=True)
    for document_key, summary in zip(doc_keys, raw_summaries):
        if isinstance(summary, Exception):
            raise summary
    summaries: list[str] = raw_summaries  # type: ignore[assignment]

    # Make Document instances out of summaries
    documents = []
    rulebook_names = {
        rulebook["document_key"]: rulebook["name"]
        for rulebook in gp["rulebooks"]
    }
    for document_key, summary in zip(doc_keys, summaries):
        documents.append(
            Document(
                page_content=summary,
                metadata={
                    "rulebook_name": rulebook_names.get(document_key, ""),
                    "document_key": document_key,
                    "game_name": gp["name"],
                }
            )
        )
    
    # Now generate overall summary — single call to give all documents equal weight
    overall_summary = await aggregate_summaries(
        chat_model=chat_model,
        prompt=summary_of_summaries_prompt,
        documents=documents,
        summary_tag=summary_tag,
    )

    # Build result dictionary
    result = {}
    for document_key, summary in zip(doc_keys, summaries):
        result[document_key] = summary
    result[""] = overall_summary

    return result


@dataclass
class GenerateGameReferenceJob:
    path: Path
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


@dataclass
class SettingSummaryJob:
    path: Path
    gp: GamePackage
    chat_model: BaseChatModel
    tokenizer: Any
    summary_tag: str = "summary"

    async def run(self):
        summaries_dict = await generate_hierarchical_summary_with_refinement(
            gp=self.gp,
            chat_model=self.chat_model,
            summary_prompt=SETTING_PAGE_BY_PAGE_SUMMARY_PROMPT,
            summary_of_summaries_prompt=SETTING_SUMMARY_OF_SUMMARIES_PROMPT,
            summary_tag=self.summary_tag,
        )

        for document_key, summary in summaries_dict.items():
            if document_key == "":
                path = get_game_setting_summary_path(self.gp)
            else:
                path = self.path / document_key / "setting_summary.md"
            logger.info(f"Writing summary to path", path=path)
            path.parent.mkdir(exist_ok=True)
            await aspit(summary, path)


UNICODE_BACKGROUND_COLOR_TEMPLATE = """
Given this board game description:

<description>
{{summary}}
</description>

Give a single unicode character that best encapsulates the setting/background of the game. Also, provide a background color as a hexcode sequence (like what one might use in CSS) upon which the unicode character will be set. Your selections should both convey the setting, while also looking well-coordinated for displaying to a user.

Give your response in the following JSON format:

{
"reasoning": "<... reasoning for the unicode and background color selection... >",
"unicode_character": "<single unicode character>",
"background_color": "#<hexcode of background color"
}
""".strip()

UNICODE_BACKGROUND_COLOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", UNICODE_BACKGROUND_COLOR_TEMPLATE),
    ],
    template_format="mustache"
)


class UnicodeBackgroundOutput(TypedDict):
    reasoning: str
    unicode_character: str
    background_color: str


@dataclass
class PresentationJob:
    path: Path
    gp: GamePackage
    chat_model: BaseChatModel

    async def run(self):
        # Load the summary
        summary_path = get_game_setting_summary_path(self.gp)
        summary = await aslurp(summary_path)

        # Generate unicode character and background color for presentation
        # on the frontend. Use self-consistency to find a good option.
        model_with_structured_output = self.chat_model.with_structured_output(
            UnicodeBackgroundOutput
        )
        chain = UNICODE_BACKGROUND_COLOR_PROMPT | model_with_structured_output
        input = {
            "summary": summary,
        }
        outputs = []
        for i in range(5):
            output = await chain.ainvoke(input)
            outputs.append(output)


        # Select the output with the most common unicode character
        outputs_by_unicode = {}
        for output in outputs:
            unicode_char = output["unicode_character"]
            outputs_by_unicode.setdefault(unicode_char, []).append(output)

        most_common_unicode = max(outputs_by_unicode.items(), key=lambda x: len(x[1]))[0]
        output = outputs_by_unicode[most_common_unicode][0]

        # Save the output
        presentation_path = get_game_presentation_path(self.gp)
        await aspit_yaml(output, presentation_path)


GENERATE_QUESTIONS_TEMPLATE = """
You are helping players of the board game "{{game_name}}" discover what they can ask a rules
assistant.

Here is an overall summary of the game, for context:

<game_summary>
{{game_summary}}
</game_summary>

Focus specifically on the rulebook titled "{{rulebook_name}}". Here is its summary:

<rulebook_summary>
{{rulebook_summary}}
</rulebook_summary>

Brainstorm {{count}} example rules questions grounded in the content of THIS rulebook
("{{rulebook_name}}") — phrased naturally, the way a real player would ask a friend across the
table mid-game (concise, spoken-aloud, not formal). Use the overall game context only to
disambiguate; the questions themselves should be about this rulebook's material.

Requirements:
- Every question must be answerable from this rulebook's actual rules (no theme/lore/trivia).
- Cover a deliberate spread: setup, turn structure, common points of confusion or tricky rule
  interactions, card/ability/component edge cases, and win conditions.
- Favor the kinds of tricky, specific edge cases players actually get stuck on over generic
  "what are the rules?" questions.

Give your response in the following JSON format:

{
"reasoning": "<...brief reasoning about the coverage spread...>",
"questions": ["<question 1>", "<question 2>", ...]
}
""".strip()

GENERATE_QUESTIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", GENERATE_QUESTIONS_TEMPLATE),
    ],
    template_format="mustache",
)


SELECT_QUESTIONS_TEMPLATE = """
You are curating example rules questions to show players of the board game "{{game_name}}" before
they start chatting with a rules assistant. Below is a list of candidate questions.

<candidates>
{{#questions}}
- {{.}}
{{/questions}}
</candidates>

Select the best {{count}} questions to display. When selecting:
- Remove near-duplicates and near-identical phrasings.
- Prefer a diverse mix that showcases the assistant's ability to handle tricky, specific rule
  interactions — not just generic questions.
- Keep phrasing concise and natural, the way a player would actually speak. Lightly clean up
  wording if needed.

Give your response in the following JSON format:

{
"reasoning": "<...brief reasoning about the final selection...>",
"questions": ["<question 1>", "<question 2>", ...]
}
""".strip()

SELECT_QUESTIONS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", SELECT_QUESTIONS_TEMPLATE),
    ],
    template_format="mustache",
)


class GenerateQuestionsOutput(TypedDict):
    reasoning: str
    questions: list[str]


class SelectQuestionsOutput(TypedDict):
    reasoning: str
    questions: list[str]


@dataclass
class ExampleQuestionsJob:
    path: Path
    gp: GamePackage
    chat_model: BaseChatModel
    num_questions: int = 5
    candidates_per_rulebook: int = 10

    async def run(self):
        game_name = self.gp["name"]

        # Overall game summary — used as context for every rulebook. Prefer the on-disk asset,
        # fall back to the summary folded into the manifest.
        game_summary = ""
        summary_path = game_summary_path(self.gp)
        if summary_path.exists():
            game_summary = await aslurp(summary_path)
        if not game_summary.strip():
            game_summary = self.gp.get("summary", "") or ""

        # Collect the per-rulebook summaries. Each rulebook is brainstormed independently so that
        # smaller rulebooks / expansions aren't drowned out by the game-level summary.
        rulebooks: list[tuple[str, str]] = []
        for rulebook in self.gp["rulebooks"]:
            rb_summary_path = rulebook_summary_path(self.gp, rulebook["document_key"])
            if rb_summary_path.exists():
                rb_summary = await aslurp(rb_summary_path)
                if rb_summary.strip():
                    rulebooks.append((rulebook["name"], rb_summary))

        # Fall back to a single pass over the game summary if no per-rulebook summaries exist.
        if not rulebooks:
            if not game_summary.strip():
                logger.warning(
                    "No summaries available for example question generation; skipping",
                    game_id=self.gp["game_id"],
                )
                return
            rulebooks = [(game_name, game_summary)]

        # Prompt 1: brainstorm candidates per rulebook, concurrently.
        generate_chain = GENERATE_QUESTIONS_PROMPT | self.chat_model.with_structured_output(
            GenerateQuestionsOutput
        )
        tasks = [
            generate_chain.ainvoke(
                {
                    "game_name": game_name,
                    "game_summary": game_summary,
                    "rulebook_name": rb_name,
                    "rulebook_summary": rb_summary,
                    "count": self.candidates_per_rulebook,
                }
            )
            for rb_name, rb_summary in rulebooks
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates: list[str] = []
        seen: set[str] = set()
        for (rb_name, _), result in zip(rulebooks, results):
            if isinstance(result, BaseException):
                logger.exception(
                    "Failed to brainstorm questions for rulebook",
                    game_id=self.gp["game_id"],
                    rulebook_name=rb_name,
                    exc_info=result,
                )
                continue
            for q in result["questions"]:
                q = (q or "").strip()
                if q and q.lower() not in seen:
                    seen.add(q.lower())
                    candidates.append(q)

        logger.info("Generated candidate questions", game_id=self.gp["game_id"], candidates=candidates)

        if not candidates:
            logger.warning("No candidate questions generated; skipping", game_id=self.gp["game_id"])
            return

        # Prompt 2: select and polish the best diverse subset across all rulebooks.
        select_chain = SELECT_QUESTIONS_PROMPT | self.chat_model.with_structured_output(
            SelectQuestionsOutput
        )
        selected = await select_chain.ainvoke(
            {"game_name": game_name, "questions": candidates, "count": self.num_questions}
        )
        final_questions = [q.strip() for q in selected["questions"] if q and q.strip()]
        final_questions = final_questions[: self.num_questions]
        logger.info("Selected example questions", game_id=self.gp["game_id"], questions=final_questions)

        # Persist the asset. Import into the DB happens later via the import job.
        example_questions_path = get_game_example_questions_path(self.gp)
        await aspit_yaml({"questions": final_questions}, example_questions_path)
        