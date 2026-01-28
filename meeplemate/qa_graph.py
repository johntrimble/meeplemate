import copy
from dataclasses import dataclass
import json
from shlex import quote
import sys
from typing import Literal, NotRequired, Sequence, Tuple, TypedDict, cast
from weakref import ref
from langchain.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.documents import Document
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.load import Serializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, chain
from langchain_core.stores import BaseStore
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from meeplemate import quote_util
from meeplemate.ingest.gamepackage import Manifest, get_page_id
from meeplemate.search import ChunkSearchService, ChunkSearchServiceInput, CompiledStateGraph
import structlog
from structlog import get_logger

from meeplemate.util import load_template
logger = get_logger()

system_prompt_template = load_template("system_prompt_rules_lawyer.md")
qa_template = load_template("single_question_and_tool_use.md")
answer_template = load_template("structured_rag_answer_addl_questions.md")

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("user", qa_template),
        ("placeholder", "{messages}"),

    ],
    template_format="mustache"
)

class Rulebook(TypedDict):
    rulebook_name: str
    summary: str
    page_count: int


class Page(TypedDict):
    rulebook_name: str
    page: int
    content: str


class Chunk(TypedDict):
    rulebook_name: str
    page: int
    offset: int
    content: str


class ChunkSearchResult(TypedDict):
    relevance_reason: str
    chunk: Chunk


@dataclass
class GameAgentContext:
    manifest: Manifest
    full_page_store: BaseStore[str, Serializable]
    chunk_search_service: ChunkSearchService


@tool(description="Returns a list of rulebooks for the game along with brief summary of each. This information can be used to subsequently retrieve pages using the `retrieve_page` function.")
async def list_rulebooks(runtime: ToolRuntime[GameAgentContext]) -> Sequence[Rulebook]:
    logger.info("list_rulebooks called")
    manifest = runtime.context.manifest
    rulebooks: list[Rulebook] = []
    for descriptor in manifest["rulebooks"]:
        rulebook = Rulebook(
            rulebook_name=descriptor["name"],
            summary=descriptor.get("summary", ""),
            page_count=descriptor["page_count"]
        )
        rulebooks.append(rulebook)
    return rulebooks


@tool(description="Returns a specific page from the specified rulebook.")
async def retrieve_page(rulebook_name: str, page: int, runtime: ToolRuntime[GameAgentContext]) -> Page:
    logger.info("retrieve_page called", rulebook_name=rulebook_name, page=page)
    manifest = runtime.context.manifest
    full_page_store = runtime.context.full_page_store

    document_key = None
    for descriptor in manifest["rulebooks"]:
        if descriptor["name"].strip() == rulebook_name.strip():
            document_key = descriptor["document_key"]
            break

    if document_key is None:
        raise ValueError(f"Could not find rulebook: {rulebook_name}")

    document_id = get_page_id(manifest["game_id"], manifest.get("game_version", ""), document_key, page)
    page_documents: list[Document] = cast(list[Document], await full_page_store.amget([document_id]))
    if len(page_documents) == 0:
        raise ValueError(f"No page found for {rulebook_name} {page}")
    
    page_document: Document = page_documents[0]
    return Page(
        rulebook_name=rulebook_name,
        page=page,
        content=page_document.page_content
    )


def get_chunk_id_tuple(chunk: Chunk) -> Tuple[str, int, int]:
    return (chunk["rulebook_name"], chunk["page"], chunk["offset"])


# TODO: Fix search service to accept multiple queries at once
@tool(description="""Search for chunks, blocks of text from rulebook pages, relevant to the given search terms. Multiple sets of search terms can be provided. Search is performed using RAG against a semantic vector database, and then further refined by an LLM-judge to determine which chunks should be returned. Every returned chunk will have an associated "relevance_reason" indicating why that chunk was relevant to the provided search terms. The chunks will also indicate the rulebook they are from and which page. This information can then be used with `retrieve_page` to retrieve the entire page if desired.""")
async def search_chunks(search_terms: str|list[str], runtime: ToolRuntime[GameAgentContext]) -> Sequence[ChunkSearchResult]:
    logger.info("search_chunks called", search_terms=search_terms)
    if isinstance(search_terms, str):
        search_terms = [search_terms]

    manifest = runtime.context.manifest
    chunk_search_service = runtime.context.chunk_search_service
    seen_chunk_ids: set = set()
    results: list[ChunkSearchResult] = []
    for query in search_terms:
        input = ChunkSearchServiceInput(
            manifest=manifest,
            query=query,
            user_main_query=query
        )

        chunk_search_result = await chunk_search_service.ainvoke(input, config=runtime.config)
        dump_documents(chunk_search_result["chunks"])
        for relevance_result in chunk_search_result["relevance"]["chunks"]:
            reasoning = relevance_result["reasoning"]
            chunk_id = relevance_result["id"]
            is_relevant = relevance_result["is_relevant"]

            if not is_relevant:
                continue

            chunk_document = None
            for chunk in chunk_search_result["chunks"]:
                if chunk.id == chunk_id:
                    chunk_document = chunk
                    break
            if chunk_document is None:
                raise ValueError(f"Missing chunk for chunk ID: {chunk_id}")
            
            metadata = chunk_document.metadata

            page_number: int = -1
            if "page_num" in metadata and isinstance(metadata["page_num"], int):
                page_number = metadata["page_num"]

            start_index: int = -1
            if "start_index" in metadata and isinstance(metadata["start_index"], int):
                start_index = metadata["start_index"]

            rulebook_name: str = "unknown_rulebook"
            if "rulebook_name" in metadata and isinstance(metadata["rulebook_name"], str):
                rulebook_name = metadata["rulebook_name"]

            if not chunk_id in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                results.append(
                    {
                        "relevance_reason": reasoning,
                        "chunk": Chunk(
                            rulebook_name=rulebook_name,
                            page=page_number,
                            offset=start_index,
                            content=chunk_document.page_content
                        )
                    }
                )
                
    return results


class QuoteEntry(TypedDict):
    text: str
    rulebook_name: str
    page: int


class DefinitionEntry(TypedDict):
    term: str
    quotes: list[QuoteEntry]
    defines_term: bool
    clarifying_question: str


class ExceptionEntry(TypedDict):
    reasoning_about_exception: str
    general_rule: str
    exception_rule: str
    quotes: list[QuoteEntry]
    exception_names_general_rule: bool
    quotes_discounting_link: list[QuoteEntry]
    does_exception_apply: bool | Literal["clarification_needed"]
    clarifying_question: str


class QaResponse(TypedDict):
    definitions: list[DefinitionEntry]
    exceptions: list[ExceptionEntry]
    reasoning: str
    final_answer: str
    sufficient_information_to_answer: bool


# class GameAgentInputState(TypedDict):
#     query: str


# class GameAgentOutputState(TypedDict):
#     response: str
#     response_evidence: NotRequired[Sequence[Chunk]]


# class GameAgentOverallState(MessagesState):
#     query: str
#     response: str
#     ready_to_answer: bool
#     response_evidence: NotRequired[Sequence[Chunk]]


def get_all_chunks_from_message_history(messages: list[AnyMessage]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.status == "success":
            try:
                results: list[ChunkSearchResult] = json.loads(message.text)
                for result in results:
                    chunk: Chunk = result["chunk"]
                    chunks.append(chunk)
            except Exception as e:
                logger.error(f"Error extracting chunks from tool message content: {e}")
    return chunks


def dedupe_chunks_in_message_history(messages):
    # Find the tool call ids for search_chunks
    tool_call_ids = set()
    for message in messages:
        tool_calls = getattr(message, "tool_calls", [])
        for tool_call in tool_calls:
            if tool_call["name"] == search_chunks.name:
                tool_call_ids.add(tool_call["id"])
    
    # Go over tool messages in reverse order, and dedupe chunks
    edited_ids: list[str] = []
    seen_chunk_ids: set = set()

    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.tool_call_id in tool_call_ids and message.status == "success":
            try:
                results: list[ChunkSearchResult] = json.loads(message.text)
                deduped_results = []
                edited = False
                for result in results:
                    chunk: Chunk = result["chunk"]
                    chunk_id = get_chunk_id_tuple(chunk)
                    if chunk_id not in seen_chunk_ids:
                        deduped_results.append(result)
                        seen_chunk_ids.add(chunk_id)
                    else:
                        edited = True
                # Update the tool message content
                if edited:
                    message.content = json.dumps(deduped_results)
                    assert message.id is not None
                    edited_ids.append(message.id)
            except Exception as e:
                logger.error(f"Error deduping tool message content: {e}")

    return [m for m in messages if getattr(m, "id", None) in edited_ids]


class QuoteValidationException(ValueError):
    
    def __init__(self, result: "TweakAndValidateQuotesResult"):
        self.result = result
        message = f"Quote validation failed: {len(result.invalid_quotes)} invalid quotes"
        super().__init__(message)


def compile_evidence_from_documents(quote_entries: Sequence[QuoteEntry], documents: Sequence[Chunk]) -> Sequence["Chunk"]:
    quote_entries = copy.deepcopy(quote_entries)

    # Organize documents
    documents_by_rulebook_and_page: dict[tuple[str, int], list[Chunk]] = {}
    for document in documents:
        rulebook_name = document["rulebook_name"]
        page = document["page"]
        key = (rulebook_name, page)
        if not key in documents_by_rulebook_and_page:
            documents_by_rulebook_and_page[key] = []
        documents_by_rulebook_and_page[key].append(document)
    
    # Expand quotes to a paragraph large
    for quote in quote_entries:
        key = (quote["rulebook_name"], quote["page"])
        candidate_documents = documents_by_rulebook_and_page.get(key, [])
        for document in candidate_documents:
            page_content = document["content"]
            match = quote_util.find_quote_with_gaps(page_content, quote["text"])
            if match:
                quote["text"] = quote_util.expand_to_full_paragraphs(page_content, match.matched_text)
                break
    
    # Dedupe quotes
    seen_texts: set = set()
    deduped_quote_entries: list[QuoteEntry] = []
    for quote in quote_entries:
        if quote["text"] not in seen_texts:
            deduped_quote_entries.append(quote)
            seen_texts.add(quote["text"])
    quote_entries = deduped_quote_entries
    
    # Convert quote entries to chunks
    chunks: list[Chunk] = []
    for quote in quote_entries:
        chunk = Chunk(
            rulebook_name=quote["rulebook_name"],
            page=quote["page"],
            offset=-1,
            content=quote["text"]
        )
        chunks.append(chunk)
    
    return chunks


@dataclass
class FixQuoteCitationsResult:
    fixed_text: str
    unfixable_quotes: list[quote_util.ExtractedQuote]
    valid_quotes: list[quote_util.ExtractedQuote]
    referenced_chunks: list[Chunk]


def get_chunks_by_rulebook_and_page(chunks: list[Chunk]) -> dict[tuple[str, int], list[Chunk]]:
    chunks_by_rulebook_and_page: dict[tuple[str, int], list[Chunk]] = {}
    for chunk in chunks:
        key = (chunk["rulebook_name"], chunk["page"])
        if key not in chunks_by_rulebook_and_page:
            chunks_by_rulebook_and_page[key] = []
        chunks_by_rulebook_and_page[key].append(chunk)
    return chunks_by_rulebook_and_page


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    def chunk_id(chunk: Chunk) -> tuple[str, int, str]:
        return (chunk["rulebook_name"], chunk["page"], chunk["content"])

    seen_chunk_ids: set = set()
    deduped_chunks: list[Chunk] = []
    for chunk in chunks:
        id = chunk_id(chunk)
        if id not in seen_chunk_ids:
            deduped_chunks.append(chunk)
            seen_chunk_ids.add(id)
    return deduped_chunks


def format_blockquote_with_inline_citation(quote_text: str, citation_text: str) -> str:
    """Format a blockquote with citation at the end of the last blockquote line.

    Handles both:
    - Blockquotes with no citation: appends citation
    - Blockquotes with citation on separate line: moves citation inline
    - Lazy continuation lines: converts them to proper blockquote lines

    Args:
        quote_text: The blockquote text (may or may not include citation)
        citation_text: The citation text to append (e.g., "(Book, p. 1)")

    Returns:
        Formatted blockquote with citation at end of last '>' line
    """
    # Split into lines
    lines = quote_text.rstrip().split('\n')

    # Find the first blockquote line (line starting with >)
    first_blockquote_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('>'):
            first_blockquote_idx = i
            break

    if first_blockquote_idx == -1:
        # No blockquote line found, fallback to append
        return quote_text.rstrip() + ' ' + citation_text

    # Convert all lines after the first blockquote line to blockquote lines
    # (handle lazy continuation where lines don't start with >)
    formatted_lines = []
    for i, line in enumerate(lines):
        if i < first_blockquote_idx:
            # Lines before the first blockquote
            formatted_lines.append(line)
        elif line.strip().startswith('>'):
            # Already a blockquote line
            formatted_lines.append(line)
        elif line.strip():  # Non-empty line that doesn't start with >
            # Lazy continuation - add > prefix
            formatted_lines.append('> ' + line)
        else:
            # Empty line - keep as is
            formatted_lines.append(line)

    # Find the last non-empty line
    last_content_idx = -1
    for i in range(len(formatted_lines) - 1, -1, -1):
        if formatted_lines[i].strip():
            last_content_idx = i
            break

    if last_content_idx == -1:
        # No content found, fallback
        return quote_text.rstrip() + ' ' + citation_text

    # Append citation to the last non-empty line
    formatted_lines[last_content_idx] = formatted_lines[last_content_idx].rstrip() + ' ' + citation_text

    # Return all lines up to and including the last content line
    return '\n'.join(formatted_lines[:last_content_idx + 1])


def should_reformat_blockquote_citation(quote_text: str, citation_start_index: int) -> bool:
    """Check if a blockquote has its citation on a separate line.

    Args:
        quote_text: The full extracted quote text including citation
        citation_start_index: Index where citation starts in quote_text

    Returns:
        True if citation is on a separate line (needs reformatting)
    """
    if citation_start_index <= 0:
        return False

    text_before_citation = quote_text[:citation_start_index]
    # Citation is on separate line if there's a blank line (\n\n) before it
    return '\n\n' in text_before_citation or text_before_citation.rstrip() != text_before_citation.rstrip('\n')


def fix_quote_citations_in_text(text: str, chunks: list[Chunk]) -> FixQuoteCitationsResult:
    # Track referenced chunks
    referenced_chunks: list[Chunk] = []

    # Make it easier to lookup chunks by rulebook and page
    chunks_by_rulebook_and_page = get_chunks_by_rulebook_and_page(chunks)

    # Find the quotes in the text
    unfixable_quotes: list[quote_util.ExtractedQuote] = []
    valid_or_fixed_quotes: list[quote_util.ExtractedQuote] = []
    quotes_in_text = quote_util.find_quotes_in_text(text)

    # We need to go in reverse order of appearance to not mess up indices
    # Sort quotes by their position in the text
    quotes_in_text.sort(key=lambda q: q["start_index"], reverse=True)

    for quote_info in quotes_in_text:
        citation = quote_info["citation"]

        # Maybe the citation is right?
        citation_correct = False
        if citation:
            citation_key = (citation["ref_name"], int(citation["page"]))
            if citation_key in chunks_by_rulebook_and_page:
                candidate_chunks = chunks_by_rulebook_and_page[citation_key]
                for candidate_chunk in candidate_chunks:
                    page_content = candidate_chunk["content"]
                    m = quote_util.find_quote_with_gaps(page_content, quote_info["quote"])
                    if m:
                        # Citation is correct, move to next quote
                        citation_correct = True
                        referenced_chunks.append(candidate_chunk)

                        # Reformat blockquotes with citations on separate lines
                        if quote_info["quote_type"] == "blockquote" and citation and should_reformat_blockquote_citation(quote_info["text"], citation["start_index"]):
                            # Get quote text without citation, then reformat
                            text_before_citation = quote_info["text"][:citation["start_index"]]
                            reformatted_quote = format_blockquote_with_inline_citation(text_before_citation, citation["text"])

                            # Replace in overall text
                            text = text[:quote_info["start_index"]] + reformatted_quote + text[quote_info["end_index"]:]

                            # Update quote_info with new indices
                            new_cit_start = len(reformatted_quote) - len(citation["text"])
                            valid_or_fixed_quotes.append({
                                "text": reformatted_quote,
                                "quote": quote_info["quote"],
                                "quote_type": quote_info["quote_type"],
                                "start_index": quote_info["start_index"],
                                "end_index": quote_info["start_index"] + len(reformatted_quote),
                                "citation": {
                                    "text": citation["text"],
                                    "ref_name": citation["ref_name"],
                                    "page": citation["page"],
                                    "start_index": new_cit_start,
                                    "end_index": len(reformatted_quote)
                                }
                            })
                        else:
                            # Citation already inline or not a blockquote, add as-is
                            valid_or_fixed_quotes.append(quote_info)

        if citation_correct:
            continue

        # Well, the citation is wrong. Let's try to find the right one
        correct_citation = None
        for (rulebook_name, page), candidate_chunks in chunks_by_rulebook_and_page.items():
            for candidate_chunk in candidate_chunks:
                page_content = candidate_chunk["content"]
                m = quote_util.find_quote_with_gaps(page_content, quote_info["quote"])
                if m:
                    correct_citation = {
                        "ref_name": rulebook_name,
                        "page": page
                    }
                    referenced_chunks.append(candidate_chunk)
            if correct_citation:
                break
        
        # If we found the correct citation, update the text
        if correct_citation:
            fixed_citation_text = f'({correct_citation["ref_name"]}, p. {correct_citation["page"]})'
            # If there is a citation, replace it
            if citation:
                # Get text before old citation
                text_before_citation = quote_info["text"][:citation["start_index"]]

                # For blockquotes, format with citation inline; for inline quotes, just append
                if quote_info["quote_type"] == "blockquote":
                    new_quote_text = format_blockquote_with_inline_citation(text_before_citation, fixed_citation_text)
                else:
                    new_quote_text = text_before_citation.rstrip() + ' ' + fixed_citation_text

                # Update overall text
                text = text[:quote_info["start_index"]] + new_quote_text + text[quote_info["end_index"]:]

                # Add to valid quotes
                new_cit_start = len(new_quote_text) - len(fixed_citation_text)
                valid_or_fixed_quotes.append({
                    "text": new_quote_text,
                    "quote": quote_info["quote"],
                    "quote_type": quote_info["quote_type"],
                    "start_index": quote_info["start_index"],
                    "end_index": quote_info["start_index"] + len(new_quote_text),
                    "citation": {
                        "text": fixed_citation_text,
                        "ref_name": correct_citation["ref_name"],
                        "page": str(correct_citation["page"]),
                        "start_index": new_cit_start,
                        "end_index": len(new_quote_text)
                    }
                })

            else:
                # No citation exists, insert one
                # For blockquotes, format with citation inline; for inline quotes, append
                if quote_info["quote_type"] == "blockquote":
                    new_quote_text = format_blockquote_with_inline_citation(quote_info["text"], fixed_citation_text)
                else:
                    new_quote_text = quote_info["text"] + " " + fixed_citation_text

                # Update overall text
                text = text[:quote_info["start_index"]] + new_quote_text + text[quote_info["end_index"]:]

                # Add to valid quotes
                new_cit_start = len(new_quote_text) - len(fixed_citation_text)
                valid_or_fixed_quotes.append({
                    "text": new_quote_text,
                    "quote": quote_info["quote"],
                    "quote_type": quote_info["quote_type"],
                    "start_index": quote_info["start_index"],
                    "end_index": quote_info["start_index"] + len(new_quote_text),
                    "citation": {
                        "text": fixed_citation_text,
                        "ref_name": correct_citation["ref_name"],
                        "page": str(correct_citation["page"]),
                        "start_index": new_cit_start,
                        "end_index": len(new_quote_text)
                    }
                })
        else:
            # If we couldn't find a correct citation, just report it as missing
            unfixable_quotes.append(quote_info)

    # Remove standalone citations that might have been left over after fixing quotes
    # Pattern: citations on their own line(s), typically after blockquotes
    # This handles the case where LLMs put a single citation at the end covering multiple quotes
    # We need to be careful not to remove citations that are part of blockquotes
    import re

    # Find all citation positions that were part of valid quotes
    protected_ranges = set()
    for q in valid_or_fixed_quotes:
        if q["citation"]:
            # Protect the range where this citation appears in the text
            cit_start = q["start_index"] + q["citation"]["start_index"]
            cit_end = q["start_index"] + q["citation"]["end_index"]
            for i in range(cit_start, cit_end):
                protected_ranges.add(i)

    # Find and remove standalone citations that aren't protected
    standalone_citation_pattern = re.compile(
        r'\n\s*\n\s*(\([^)]+,?\s*pg?[.]\s*[0-9]+\))\s*(?=\n|$)',
        re.MULTILINE
    )

    # Actually, we want to remove non-protected ones, so:
    matches_to_remove = []
    for match in standalone_citation_pattern.finditer(text):
        is_protected = any(i in protected_ranges for i in range(match.start(), match.end()))
        if not is_protected:
            matches_to_remove.append(match)

    # Remove in reverse order to preserve indices
    for match in reversed(matches_to_remove):
        text = text[:match.start()] + text[match.end():]

    return FixQuoteCitationsResult(
        fixed_text=text,
        unfixable_quotes=unfixable_quotes,
        referenced_chunks=dedupe_chunks(referenced_chunks),
        valid_quotes=valid_or_fixed_quotes
    )


@dataclass
class TweakAndValidateQuotesResult:
    revised_response: QaResponse
    invalid_quotes: list[QuoteEntry]
    valid_quotes: list[QuoteEntry]
    chunks_referenced: list[Chunk]

    @property
    def valid(self) -> bool:
        return len(self.invalid_quotes) == 0


def tweak_and_validate_quotes_response(response: QaResponse, chunks: list[Chunk]) -> TweakAndValidateQuotesResult:
    # Clone the QaResponse to avoid mutating the input
    response = copy.deepcopy(response)

    # Keep track of all chunks referenced in the response
    referenced_chunks: list[Chunk] = []

    # Keep track of all quotes
    valid_quotes: list[QuoteEntry] = []

    chunks_by_rulebook_and_page = get_chunks_by_rulebook_and_page(chunks)

    # Check and fix quotes in the final answer
    result = fix_quote_citations_in_text(response["final_answer"], chunks)
    response["final_answer"] = result.fixed_text
    referenced_chunks.extend(result.referenced_chunks)

    for _quote in result.valid_quotes:
        valid_quotes.append(
            QuoteEntry(
                text=_quote["quote"],
                rulebook_name=_quote["citation"]["ref_name"] if _quote["citation"] else "",
                page=int(_quote["citation"]["page"]) if _quote["citation"] else -1
            )
        )

    # Check the final answer for invalid quotes
    invalid_final_answer_quotes: list[QuoteEntry] = []
    for extracted_quote in result.unfixable_quotes:
        citation = extracted_quote["citation"]
        if not citation:
            citation = {"ref_name": "", "page": -1}
        quote_entry: QuoteEntry = {
            "text": extracted_quote["quote"],
            "rulebook_name": citation["ref_name"],
            "page": int(citation["page"])
        }
        invalid_final_answer_quotes.append(quote_entry)
    
    def _check_quote(quote: QuoteEntry) -> bool:
        key = (quote["rulebook_name"], quote["page"])
        if key not in chunks_by_rulebook_and_page:
            return False
        candidate_chunks = chunks_by_rulebook_and_page[key]
        return_value = False
        for candidate_chunk in candidate_chunks:
            page_content = candidate_chunk["content"]
            m = quote_util.find_quote_with_gaps(page_content, quote["text"])
            if m:
                return_value = True
                referenced_chunks.append(candidate_chunk)
        return return_value
    
    def _find_chunk_with_quote(text: str) -> Chunk | None:
        found_chunk = None
        for chunk in chunks:
            m = quote_util.find_quote_with_gaps(chunk["content"], text)
            if m:
                found_chunk = chunk
                referenced_chunks.append(chunk)
        return found_chunk
    
    def check_quotes_in_list(quotes: list[QuoteEntry]) -> list[QuoteEntry]:
        invalid_quotes: list[QuoteEntry] = []
        for quote in quotes:
            if not _check_quote(quote):
                chunk_with_quote = _find_chunk_with_quote(quote["text"])
                if chunk_with_quote:
                    # Fix the quote to have the right rulebook and page
                    quote["rulebook_name"] = chunk_with_quote["rulebook_name"]
                    quote["page"] = chunk_with_quote["page"]
                    valid_quotes.append(quote)
                else:
                    # Record invalid quote
                    invalid_quotes.append(quote)
            else:
                valid_quotes.append(quote)
        return invalid_quotes

    # Validate definitions
    invalid_definition_quotes: list[QuoteEntry] = []
    for definition in response["definitions"]:
        invalid_definition_quotes.extend(check_quotes_in_list(definition["quotes"]))

    # Validate exceptions
    invalid_exception_quotes: list[QuoteEntry] = []
    for exception in response["exceptions"]:
        invalid_exception_quotes.extend(check_quotes_in_list(exception["quotes"]))
    
    # Validate quotes discounting links
    invalid_discounting_link_quotes: list[QuoteEntry] = []
    for exception in response["exceptions"]:
        invalid_discounting_link_quotes.extend(check_quotes_in_list(exception["quotes_discounting_link"]))
    
    return TweakAndValidateQuotesResult(
        revised_response=response,
        invalid_quotes=invalid_definition_quotes + invalid_exception_quotes + invalid_discounting_link_quotes + invalid_final_answer_quotes,
        chunks_referenced=dedupe_chunks(referenced_chunks),
        valid_quotes=valid_quotes,
    )


def get_run_id_from_config(config: RunnableConfig | None) -> str:
    """Extract run_id from config or callback manager, returning 'unknown' if not found."""
    if not config:
        return "unknown"

    # First try to get run_id directly from config
    run_id = config.get("run_id")
    if run_id:
        return str(run_id)

    # If not in config, try to get it from the callback manager
    callbacks = config.get("callbacks")
    if callbacks is not None:
        # Callback managers have a parent_run_id attribute
        parent_run_id = getattr(callbacks, "parent_run_id", None)
        if parent_run_id:
            return str(parent_run_id)
        # Some callback managers might have run_id directly
        callback_run_id = getattr(callbacks, "run_id", None)
        if callback_run_id:
            return str(callback_run_id)

    return "unknown"


def dump_documents(documents: list[Document]):
    print("===== Dumping Documents =====")
    for document in documents:
        print(document.metadata['rulebook_name'], document.metadata['page_num'], document.page_content[:100].replace("\n", " "))
    print("===== End Dump =====")


def dump_chunks(chunks: list[Chunk]):
    print("===== Dumping Chunks =====")
    for chunk in chunks:
        print(chunk['rulebook_name'], chunk['page'], chunk["content"][:100].replace("\n", " "))
    print("===== End Dump =====")


class GameAgentInputState(MessagesState):
    query: str
    """The user's query"""
    recursion_depth: int
    """The current recursion depth"""
    evidence: list[Chunk]


class GameAgentOutputState(MessagesState):
    response: str
    """The answer to the user's query"""
    evidence: list[Chunk]
    """The evidence chunks supporting the answer"""


class ClarifyingQA(TypedDict):
    question: str
    """A clarifying question about the original question asked"""
    answer: str
    """The answer to the clarifying question"""
    evidence: list[Chunk]
    """The evidence chunks supporting the answer"""


class GameAgentOverallState(GameAgentInputState, GameAgentOutputState):
    clarifying_questions: list[ClarifyingQA]
    analysis: QaResponse
    rounds_clarification: int
    """The number of rounds of clarification performed"""
    validation_attempts: int
    """Track how many times we've attempted to fix/validate quotes"""


def build_game_agent_graph(checkpoint_saver: BaseCheckpointSaver, chat_model: BaseChatModel, qa_prompt: ChatPromptTemplate) -> CompiledStateGraph[GameAgentOverallState, GameAgentContext, GameAgentInputState, GameAgentOutputState]:
    tools = [list_rulebooks, retrieve_page, search_chunks]
    tool_node = ToolNode(tools)

    async def retrieve_data(state: GameAgentInputState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        """Retrieves relevant chunks for the given query."""
        # chunk_search_service = runtime.context.chunk_search_service
        manifest = runtime.context.manifest

        if len(state["messages"]) == 0:
            chat_model_with_tools = chat_model.bind_tools(tools, tool_choice="any")
        else:
            chat_model_with_tools = chat_model.bind_tools(tools)
        
        chain = qa_prompt | chat_model_with_tools

        input = {
            "game_summary": manifest.get("summary", ""),
            "game_name": manifest["name"],
            "query": state["query"],
            "messages": state["messages"],
        }

        logger.info("Invoking LLM with input", message_count=len(state["messages"]))
        try:
            final_message = await chain.ainvoke(input, config=config)
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            for m in state["messages"]:
                m.pretty_print()
            raise

        return {
            "messages": [final_message]
        }

    async def dedupe_chunks(state: GameAgentOverallState) -> dict:
        """Dedupe chunks in the message history"""
        messages = state["messages"]
        message_edits = dedupe_chunks_in_message_history(messages)
        return {
            "messages": message_edits
        }
    
    def get_evidence(state: GameAgentOverallState) -> list[Chunk]:
        # Gather the context we've collected so far
        documents = []
        # If we've already acquired evidence, use that.
        if "evidence" in state and state["evidence"]:
            documents.extend(state["evidence"])
        else:
            # Otherwise, extract from message history
            documents.extend(get_all_chunks_from_message_history(state["messages"]))
        # Add evidence from clarifying questions too
        for qa in state.get("clarifying_questions", []):
            documents.extend(qa.get("evidence", []))
        return documents

    async def analyze_evidence(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        """Take all the input chunks, determines which are relevant to the query, produces clarifying questions if needed, and produces an answer with evidence."""
        manifest = runtime.context.manifest

        # Gather the context we've collected so far
        documents = get_evidence(state)

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_template),
                ("user", answer_template),

            ],
            template_format="mustache"
        )

        answer_chat_model = chat_model.with_structured_output(
            QaResponse, include_raw=True
        ).bind(
            temperature=0.3, top_p=0.6
        )

        chain = answer_prompt | answer_chat_model
        chain = chain.with_config(run_name="game_agent_answer_chain")

        input = dict(
            # game_summary=manifest.get("summary", ""),
            game_summary="",
            game_name=manifest["name"],
            documents=documents,
            query=state["query"],
            clarifying_questions_and_answers=state.get("clarifying_questions", []),
        )

        result = await chain.ainvoke(input, config=config)
        assert isinstance(result, dict)
        parsed = result["parsed"]

        return {
            "analysis": parsed,
        }

    async def validate_analysis(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        """Validates the analysis output, ensuring all quotes are valid."""
        # Ensure every quote reference in the response is valid
        analysis = state["analysis"]
        chunks = get_evidence(state)
        quote_validation_result = tweak_and_validate_quotes_response(analysis, chunks)
        analysis_updated = (analysis != quote_validation_result.revised_response)

        # This is a pretty big thing to log, only do it if things changed
        if analysis_updated:
            logger.info("QAResponse before and after tweak", before=analysis, after=quote_validation_result.revised_response, invalid_quote_count=len(quote_validation_result.invalid_quotes))

        analysis = quote_validation_result.revised_response

        # Log as warning every invalid quote
        if not quote_validation_result.valid:
            for invalid_quote in quote_validation_result.invalid_quotes:
                logger.warning(
                    "Invalid quote detected in answer",
                    text=invalid_quote["text"],
                    rulebook_name=invalid_quote["rulebook_name"],
                    page=invalid_quote["page"],
                )

        # Print all clarifying questions
        for definition in analysis["definitions"]:
            if definition["clarifying_question"]:
                logger.info("Clarifying question from definition", question=definition["clarifying_question"])
        for exception in analysis["exceptions"]:
            if exception["clarifying_question"]:
                logger.info("Clarifying question from exception", question=exception["clarifying_question"])

        output = {}

        # Include updated analysis if it changed
        if analysis_updated:
            output["analysis"] = analysis

        # Track invalid quotes
        output["invalid_quotes"] = quote_validation_result.invalid_quotes

        if not quote_validation_result.valid:
            # Increment validation attempts counter since we had invalid quotes
            output["validation_attempts"] = state.get("validation_attempts", 0) + 1
            return output

        # Okay, everything is valid

        # Reset validation attempts
        output["validation_attempts"] = 0

        # Build the evidence
        #
        # This is based off the quotes that were validated in the response. The
        # hope is that these quotes are sufficient to support the answer.
        evidence = compile_evidence_from_documents(
            quote_validation_result.valid_quotes,
            quote_validation_result.chunks_referenced
        )

        # Add evidence to output
        output["evidence"] = evidence

        return output

    async def address_clarifying_questions(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        """If there are clarifying questions, ask them and incorporate the answers."""
        analysis = state["analysis"]

        # Collect questions to ask
        inputs: list[GameAgentInputState] = []
        for definition in analysis["definitions"]:
            question = definition.get("clarifying_question", "")
            if question:
                game_agent_input: GameAgentInputState = {
                    "query": question,
                    "recursion_depth": state.get("recursion_depth", 0) + 1,
                    "messages": [],
                    "evidence": []
                }
                inputs.append(game_agent_input)

        for exception in analysis["exceptions"]:
            question = exception.get("clarifying_question", "")
            if question:
                game_agent_input: GameAgentInputState = {
                    "query": question,
                    "recursion_depth": state.get("recursion_depth", 0) + 1,
                    "messages": [],
                    "evidence": [],
                }
                inputs.append(game_agent_input)

        # If for some reason we do not have any questions, return early
        if len(inputs) == 0:
            return {}

        # Ask the questions using the game agent graph
        responses = await agent.abatch(
            inputs, 
            context=runtime.context,
            config=config
        )

        # Update the state with the answers
        clarifying_qas: list[ClarifyingQA] = []
        for i, response in enumerate(responses):
            answer = response["response"]
            question = inputs[i]["query"]
            clarifying_qas.append(
                ClarifyingQA(
                    question=question,
                    answer=answer,
                    evidence=response["evidence"]
                )
            )

        return {
            "clarifying_questions": clarifying_qas,
            "rounds_clarification": state.get("rounds_clarification", 0) + 1
        }

    async def produce_response(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        """Produce the final answer based on the analysis and clarifying questions."""
        evidence = state.get("evidence", [])
        logger.info("Producing response", answer=state["analysis"], evidence=evidence)
        return {
            "response": state["analysis"]["final_answer"],
            "evidence": evidence,
        }

    def check_response_status(state: GameAgentOverallState) -> Literal["analyze_evidence", "address_clarifying_questions", "produce_response"]:
        analysis = state["analysis"]

        # If the analysis isn't valid, try analyze_evidence again
        invalid_quotes = state.get("invalid_quotes", [])
        if len(invalid_quotes) > 0:
            # If we are over max attempts, just produce response
            if state.get("validation_attempts", 0) >= 3:
                return "produce_response"
            
            # We aren't over our max attempts, so retry analysis
            return "analyze_evidence"
        
        # If there are clarifying questions to ask, go to address_clarifying_questions
        questions_to_answer = False
        sufficient_information = True
        analysis = state.get("analysis")
        if analysis:
            sufficient_information = analysis["sufficient_information_to_answer"]
            for definition in analysis["definitions"]:
                if definition.get("clarifying_question", ""):
                    questions_to_answer = True
            for exception in analysis["exceptions"]:
                if exception.get("clarifying_question", ""):
                    questions_to_answer = True

        at_recursion_limit = state.get("recursion_depth", 0) >= 2

        if questions_to_answer and not at_recursion_limit and not sufficient_information:
            return "address_clarifying_questions"
        
        # Otherwise, we're done
        return "produce_response"
    
    def select_start_node(state: GameAgentInputState) -> Literal["analyze_evidence", "retrieve_data"]:
        # If provided evidence up-front, start with analyze_evidence
        if state.get("evidence"):
            logger.info("Starting with provided evidence, skipping data retrieval")
            return "analyze_evidence"
        
        # Always start with retrieve_data
        logger.info("Starting with data retrieval")
        return "retrieve_data"

    graph = StateGraph(
        state_schema=GameAgentOverallState,
        input_schema=GameAgentInputState,
        output_schema=GameAgentOutputState,
        context_schema=GameAgentContext,
    )

    graph.add_node("tool_node", tool_node)
    graph.add_node("analyze_evidence", analyze_evidence)
    graph.add_node("validate_analysis", validate_analysis)
    graph.add_node("address_clarifying_questions", address_clarifying_questions)
    graph.add_node("produce_response", produce_response)
    graph.add_node("retrieve_data", retrieve_data)
    graph.add_node("dedupe_chunks", dedupe_chunks)

    graph.add_conditional_edges(START, select_start_node)
    graph.add_edge("retrieve_data", "tool_node")
    # TODO: As of right now, we only do one round of tool calls to retrieve data
    # before moving on to analysis. We should consider looping back to retrieve
    # more data if needed.
    graph.add_edge("tool_node", "dedupe_chunks")
    graph.add_edge("dedupe_chunks", "analyze_evidence")
    graph.add_edge("analyze_evidence", "validate_analysis")
    graph.add_conditional_edges("validate_analysis", check_response_status)
    graph.add_edge("address_clarifying_questions", "analyze_evidence")
    graph.add_edge("produce_response", END)

    # Compile the agent
    agent = graph.compile(checkpointer=checkpoint_saver)
    return agent


# def build_game_agent_graph(checkpoint_saver: BaseCheckpointSaver, chat_model: BaseChatModel, qa_prompt: ChatPromptTemplate=qa_prompt) -> CompiledStateGraph[GameAgentOverallState, GameAgentContext, GameAgentInputState, GameAgentOutputState]:
#     tools = [list_rulebooks, retrieve_page, search_chunks]
#     tool_node = ToolNode(tools)
    
#     async def check_answer_progress(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
#         assert runtime is not None
#         manifest = runtime.context.manifest

#         # Set up structlog context with the run ID - this will automatically
#         # add run_id to all log calls within this function and any functions it calls
#         run_id = get_run_id_from_config(config)
#         with structlog.contextvars.bound_contextvars(run_id=run_id):
#             chunks = get_all_chunks_from_message_history(state["messages"])

#             # Print all the chunks found
#             dump_chunks(chunks)

#             answer_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", system_prompt_template),
#                     ("user", answer_template),

#                 ],
#                 template_format="mustache"
#             )

#             answer_chat_model = chat_model.with_structured_output(
#                 QaResponse, include_raw=True
#             ).bind(
#                 logprobs=True, top_logprobs=2, temperature=0.3, top_p=0.6
#             )

#             chain = answer_prompt | answer_chat_model
#             chain = chain.with_config(run_name="game_agent_answer_chain")

#             input = dict(
#                 # game_summary=manifest.get("summary", ""),
#                 game_summary="",
#                 game_name=manifest["name"],
#                 documents=chunks,
#                 query=state["query"]
#             )

#             quote_validation_result = None
#             for _ in range(3):
#                 result = await chain.ainvoke(input, config=config)
#                 assert isinstance(result, dict)
#                 parsed = result["parsed"]
#                 raw = result["raw"]

#                 # Ensure every quote reference in the response is valid
#                 quote_validation_result = tweak_and_validate_quotes_response(parsed, chunks)

#                 # This is a pretty big thing to log, only do it if things changed
#                 if parsed != quote_validation_result.revised_response:
#                     logger.info("QAResponse before and after tweak", before=parsed, after=quote_validation_result.revised_response, invalid_quote_count=len(quote_validation_result.invalid_quotes))
        
#                 parsed = quote_validation_result.revised_response

#                 # Log as warning every invalid quote
#                 if not quote_validation_result.valid:
#                     for invalid_quote in quote_validation_result.invalid_quotes:
#                         logger.warning(
#                             "Invalid quote detected in answer",
#                             text=invalid_quote["text"],
#                             rulebook_name=invalid_quote["rulebook_name"],
#                             page=invalid_quote["page"],
#                         )
                
#                 # Print all clarifying questions
#                 for definition in parsed["definitions"]:
#                     if definition["clarifying_question"]:
#                         logger.info("Clarifying question from definition", question=definition["clarifying_question"])
#                 for exception in parsed["exceptions"]:
#                     if exception["clarifying_question"]:
#                         logger.info("Clarifying question from exception", question=exception["clarifying_question"])

#                 if quote_validation_result.valid:
#                     evidence: list[QuoteEntry] = quote_validation_result.valid_quotes
#                     return {
#                         "response": parsed["final_answer"],
#                         "messages": [raw],
#                         "response_evidence": compile_evidence_from_documents(
#                             evidence,
#                             quote_validation_result.chunks_referenced
#                         ),
#                     }

#             assert quote_validation_result is not None

#             if not quote_validation_result.valid:
#                 response = QaResponse(
#                     definitions=[],
#                     exceptions=[],
#                     reasoning="",
#                     final_answer="I'm sorry, but I was unable to provide a valid answer with correct citations based on the provided rulebooks.",
#                     sufficient_information_to_answer=False
#                 )
#                 new_message = AIMessage(content=json.dumps(response, indent=2))
#                 return {
#                     "response": response["final_answer"],
#                     "messages": state["messages"] + [new_message],
#                     "evidence": [],
#                 }
 

#     async def llm_call(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
#         manifest = runtime.context.manifest

#         if len(state["messages"]) == 0:
#             chat_model_with_tools = chat_model.bind_tools(tools, tool_choice="any")
#         else:
#             chat_model_with_tools = chat_model.bind_tools(tools)

#         chain = qa_prompt | chat_model_with_tools

#         input = {
#             "game_summary": manifest.get("summary", ""),
#             "game_name": manifest["name"],
#             "query": state["query"],
#             "messages": state["messages"],
#         }

#         logger.info("Invoking LLM with input", message_count=len(state["messages"]))
#         try:
#             final_message = await chain.ainvoke(input, config=config)
#         except Exception as e:
#             logger.error(f"Error invoking LLM: {e}")
#             for m in state["messages"]:
#                 m.pretty_print()
#             raise

#         return {
#             "messages": [final_message]
#         }
    
#     async def dedupe_chunks(state: GameAgentOverallState) -> dict:
#         """Dedupe chunks in the message history"""
#         messages = state["messages"]
#         message_edits = dedupe_chunks_in_message_history(messages)
#         return {
#             "messages": message_edits
#         }
    
#     async def should_continue(state: GameAgentOverallState) -> Literal["tool_node", "ask_to_answer", "response"]:
#         """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

#         messages = state["messages"]
#         last_message = messages[-1]

#         # If the LLM makes a tool call, then perform an action
#         if getattr(last_message, "tool_calls", None):
#             return "tool_node"

#         if not state.get("ready_to_answer", False):
#             return "ask_to_answer"

#         # Otherwise, we stop (reply to the user)
#         return "response"
    
#     async def check_answer_edge(state: GameAgentOverallState) -> Literal["__end__", "dedupe_chunks"]:
#         """Determine if we should dedupe chunks or end the workflow"""

#         if state.get("response"):
#             return "__end__"
#         else:
#             return "dedupe_chunks"
    
#     # Build workflow
#     agent_builder = StateGraph(
#         GameAgentOverallState,
#         context_schema=GameAgentContext,
#         input_schema=GameAgentInputState,
#         output_schema=GameAgentOutputState
#     )

#     # Add nodes
#     # NOTE: The node name "llm_call" is used by the QA service for streaming filtering.
#     # If you rename this node, update the streaming logic in build_qa_service.
#     agent_builder.add_node("llm_call", llm_call)
#     agent_builder.add_node("tool_node", tool_node)
#     agent_builder.add_node("dedupe_chunks", dedupe_chunks)
#     agent_builder.add_node("check_answer_progress", check_answer_progress)
#     # agent_builder.add_node("response", populate_response)

#     # Add edges to connect nodes
#     agent_builder.add_edge(START, "llm_call")
#     agent_builder.add_edge("llm_call", "tool_node")
#     agent_builder.add_edge("tool_node", "dedupe_chunks")
#     agent_builder.add_edge("dedupe_chunks", "check_answer_progress")
#     agent_builder.add_edge("check_answer_progress", END)
#     # agent_builder.add_edge("response", END)

#     # Compile the agent
#     agent = agent_builder.compile(checkpointer=checkpoint_saver)

#     return agent


class QAServiceInput(GameAgentInputState):
    manifest: Manifest


QAService = Runnable[QAServiceInput, GameAgentOutputState]


def build_qa_service(
    checkpoint_saver: BaseCheckpointSaver,
    chat_model: BaseChatModel,
    full_page_store: BaseStore[str, Serializable],
    chunk_search_service: ChunkSearchService,
    qa_prompt: ChatPromptTemplate=qa_prompt,
) -> QAService:
    agent_graph = build_game_agent_graph(
        checkpoint_saver,
        chat_model,
        qa_prompt
    )

    # Create a custom Runnable that properly handles both streaming and non-streaming
    from langchain_core.runnables import Runnable

    @chain
    async def qa_service(input: QAServiceInput, config: RunnableConfig | None = None, **kwargs) -> GameAgentOutputState:
        context = GameAgentContext(
            manifest=input["manifest"],
            full_page_store=full_page_store,
            chunk_search_service=chunk_search_service,
        )

        result = await agent_graph.ainvoke(
            cast(GameAgentInputState, {k: v for k, v in input.items() if k != "manifest"}),
            context=context,
            config=config,
            **kwargs
        )
        return cast(GameAgentOutputState, result)

    return qa_service