import asyncio
import bisect
import copy
from dataclasses import dataclass, field
import re
import json
from typing import Annotated, Any, List, Literal, NotRequired, Optional, Sequence, Tuple, TypedDict, Union, cast
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.documents import Document
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_core.load import Serializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable,  RunnableConfig, chain
from langchain_core.stores import BaseStore
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode

from meeplemate import quote_util
from meeplemate.ingest.gamepackage import Manifest, get_page_id
from meeplemate.search import ChunkSearchService, ChunkSearchServiceInput, CompiledStateGraph
from meeplemate.stream_events import StepEvent, AnalyzedUserQueryEvent, SubquestionAnsweredEvent, UserQueryAnsweredEvent
from structlog import get_logger

from meeplemate.util import load_template
logger = get_logger(__name__)

REFINEMENT_PARTITION_NUMBER = 4

system_prompt_template = load_template("system_prompt_rules_lawyer.md")
format_answer_system_prompt_template = load_template("system_prompt_format_answer.md")
qa_template = load_template("single_question_and_tool_use.md")
answer_template = load_template("structured_rag_answer_addl_questions.md")
markdown_format_response_template = load_template("markdown_format_response.md")
query_documents_guidelines_template = load_template("query_documents_guide_system_prompt.md")
retrieve_documents_template = load_template("examine_user_query_retrieve_docs.md")

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
    page: str
    start_index: int
    end_index: int
    content: str


class ChunkSearchResult(TypedDict):
    relevance_reason: str
    chunk: Chunk


@dataclass
class GameAgentContext:
    manifest: Manifest
    full_page_store: BaseStore[str, Serializable]
    chunk_search_service: ChunkSearchService
    chat_model: BaseChatModel


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


def get_chunk_id_tuple(chunk: Chunk) -> Tuple[str, str, int]:
    return (chunk["rulebook_name"], chunk["page"], chunk["start_index"])


@dataclass
class ContextWithSearchChunkService:
    manifest: Manifest
    chunk_search_service: ChunkSearchService


# Token budget constants for context window management
MAX_CONTEXT_SIZE = 29_000
MAX_RESPONSE_TOKENS = 10_024
BUDGET_BUFFER = 1_000


def _build_answer_prompt():
    """Build the answer prompt template. Shared between retrieve_data (for budget
    calculation) and answer_question (for the actual LLM call) so they stay in sync."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", query_documents_guidelines_template),
            ("user", "Consider the user's query. Provide a step-by-step reasoning process concerning the user's query, then answer the user's query. Your answer MUST directly address the user's exact question as phrased. If the user asks 'Do X need to do Y?', your answer must start with 'Yes, X must do Y' or 'No, X do not need to do Y' — not a reframing like 'No, X do not have immunity from Y'."),
        ],
        template_format="mustache"
    )


# TODO: Fix search service to accept multiple queries at once
@tool(description="""Search for relevant rules from game rulebooks. Uses semantic vector search - the search understands meaning, not just keywords.
Best practices for queries:
- Use separate queries for distinct rules or mechanics (e.g. use ["What are the rules for firing a bow?", "How does movement work?"] instead of ["Can I fire a bow after moving?"])
- Use one query per rule or mechanic you are looking for
- Do not provide multiple phrasings of the same rule or mechanic
- Avoid very generic terms like 'mechanics', 'interactions', etc. in search queries
- When using multiple queries, put them in order of importance (most important first)
- Limit to 5 queries""")
async def search_chunks(search_queries: list[str], runtime: ToolRuntime[ContextWithSearchChunkService]) -> Sequence[ChunkSearchResult]:
    token_budget = None

    # Get the tokens used from the graph state
    if "tokens_used" in runtime.state:
        tokens_used_so_far = runtime.state["tokens_used"]
        token_budget = MAX_CONTEXT_SIZE - tokens_used_so_far - MAX_RESPONSE_TOKENS - BUDGET_BUFFER

    logger.info("search_chunks called", token_budget=token_budget, search_terms=search_queries)
    if isinstance(search_queries, str):
        search_queries = [search_queries]

    # TODO: Fix this hack. We keep searching on these terms needlessly
    search_queries = [term for term in search_queries if term not in ['interaction', 'mechanic', 'mechanics', 'relationship']]

    # Hard cap: the tool description says "Limit to 5 queries" but the model sometimes ignores it
    search_queries = search_queries[:5]

    # Get the query from the graph context
    user_query = runtime.state["query"]

    manifest = runtime.context.manifest
    chunk_search_service = runtime.context.chunk_search_service
    seen_chunk_ids: set = set()
    results: list[ChunkSearchResult] = []
    # for query in search_queries:
    if True:
        input = ChunkSearchServiceInput(
            manifest=manifest,
            query=search_queries,
            user_main_query=user_query
        )

        if token_budget is not None:
            input["token_budget"] = token_budget

        chunk_search_result = await chunk_search_service.ainvoke(input, config=runtime.config)
        logger.info("Retrieved results", query=user_query, relevant_count=len([c for c in chunk_search_result["relevance"]["chunks"] if c["is_relevant"]]), total_retrieved_count=len(chunk_search_result["relevance"]["chunks"]))
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

            page_number: str = ""
            if "page_num" in metadata:
                page_number = str(metadata["page_num"])

            start_index: int = -1
            if "start_index" in metadata and isinstance(metadata["start_index"], int):
                start_index = metadata["start_index"]

            end_index: int = -1
            if "end_index" in metadata and isinstance(metadata["end_index"], int):
                end_index = metadata["end_index"]

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
                            start_index=start_index,
                            end_index=end_index,
                            content=chunk_document.page_content
                        )
                    }
                )
    logger.info("search_chunks returning", result_count=len(results))
    return results


class QuoteEntry(TypedDict):
    """A quote from a rulebook with its citation information"""
    text: Annotated[str, ..., "Verbatim quote from the rulebook"]
    rulebook_name: Annotated[str, ..., "Name of the rulebook this quote comes from"]
    page: Annotated[str, ..., "Page number where this quote appears"]


def get_all_chunks_from_message_history(messages: list[AnyMessage]) -> list[Chunk]:
    # Find the tool call ids for search_chunks
    tool_call_ids = set()
    for message in messages:
        tool_calls = getattr(message, "tool_calls", [])
        for tool_call in tool_calls:
            if tool_call["name"] == search_chunks.name:
                tool_call_ids.add(tool_call["id"])

    chunks: list[Chunk] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id in tool_call_ids and message.status == "success":
            # Skip empty messages - they cannot contain valid search results
            # This can happen when search_chunks returns 0 results due to serialization issues
            if not message.text or not message.text.strip():
                logger.warning("Skipping ToolMessage with empty content", tool_call_id=message.tool_call_id)
                continue
            try:
                results: list[ChunkSearchResult] = json.loads(message.text)
                for result in results:
                    chunk: Chunk = result["chunk"]
                    chunks.append(chunk)
            except Exception:
                logger.exception(
                    "Error extracting chunks from tool message",
                    tool_call_id=message.tool_call_id,
                    content_chars=len(message.text or ""),
                )
                # The raw tool message is a whole serialized chunk-search payload.
                logger.debug("Offending message content", content=message.text)

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
            # Skip empty messages - they cannot contain valid search results
            # This can happen when search_chunks returns 0 results due to serialization issues
            if not message.text or not message.text.strip():
                logger.warning("Skipping ToolMessage with empty content during deduplication", tool_call_id=message.tool_call_id)
                continue
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
            except Exception:
                logger.exception("Error deduping tool message content")

    return [m for m in messages if getattr(m, "id", None) in edited_ids]


def create_chunks_for_quotes(quote_entries: Sequence[QuoteEntry], documents: Sequence[Chunk]) -> Sequence[Chunk|None]:
    # Organize documents
    documents_by_rulebook_and_page: dict[tuple[str, str], list[Chunk]] = {}
    for document in documents:
        rulebook_name = document["rulebook_name"]
        page = document["page"]
        key = (rulebook_name, page)
        if not key in documents_by_rulebook_and_page:
            documents_by_rulebook_and_page[key] = []
        documents_by_rulebook_and_page[key].append(document)

    result_chunks: list[Chunk|None] = []
    for i, quote in enumerate(quote_entries):
        key = (quote["rulebook_name"], quote["page"])
        candidate_documents = documents_by_rulebook_and_page.get(key, [])
        chunk = None
        for document in candidate_documents:
            page_content = document["content"]
            match = quote_util.find_quote_with_gaps(page_content, quote["text"])
            if match:
                # start and end relative to the chunk
                start_index, end_index = match.start, match.end

                # start and end relative to the document
                start_index += document["start_index"]
                end_index += document["start_index"]

                chunk = Chunk(
                    rulebook_name=document["rulebook_name"],
                    page=document["page"],
                    start_index=start_index,
                    end_index=end_index,
                    content=match.matched_text
                )
                break
        result_chunks.append(chunk)
    return result_chunks


def compile_evidence_from_documents(quote_entries: Sequence[QuoteEntry], documents: Sequence[Chunk]) -> Sequence["Chunk"]:
    quote_entries = copy.deepcopy(quote_entries)

    # Organize documents
    documents_by_rulebook_and_page: dict[tuple[str, str], list[Chunk]] = {}
    for document in documents:
        rulebook_name = document["rulebook_name"]
        page = document["page"]
        key = (rulebook_name, page)
        if not key in documents_by_rulebook_and_page:
            documents_by_rulebook_and_page[key] = []
        documents_by_rulebook_and_page[key].append(document)

    # Match quotes against documents, expand to full paragraphs, and track positions
    match_info: dict[int, tuple[str, int, int]] = {}  # quote index -> (content, start_index, end_index)

    for i, quote in enumerate(quote_entries):
        key = (quote["rulebook_name"], quote["page"])
        candidate_documents = documents_by_rulebook_and_page.get(key, [])
        for document in candidate_documents:
            page_content = document["content"]
            match = quote_util.find_quote_with_gaps(page_content, quote["text"])
            if match:
                expanded = quote_util.expand_to_full_paragraphs(page_content, match.matched_text)
                content = expanded if len(expanded) > len(quote["text"]) else match.matched_text
                pos = page_content.find(content)
                doc_start = document["start_index"]
                if doc_start >= 0 and pos >= 0:
                    abs_start = doc_start + pos
                    abs_end = abs_start + len(content)
                else:
                    abs_start, abs_end = -1, -1
                quote["text"] = content
                match_info[i] = (content, abs_start, abs_end)
                break

    # Dedupe quotes
    seen_texts: set = set()
    deduped: list[tuple[QuoteEntry, int]] = []  # (quote, original index)
    for i, quote in enumerate(quote_entries):
        if quote["text"] not in seen_texts:
            deduped.append((quote, i))
            seen_texts.add(quote["text"])

    # Convert to chunks, using match info when available
    chunks: list[Chunk] = []
    for quote, orig_idx in deduped:
        if orig_idx in match_info:
            content, start_index, end_index = match_info[orig_idx]
        else:
            content, start_index, end_index = quote["text"], -1, -1
        chunk = Chunk(
            rulebook_name=quote["rulebook_name"],
            page=quote["page"],
            start_index=start_index,
            end_index=end_index,
            content=content,
        )
        chunks.append(chunk)
    
    chunks = dedupe_chunks(chunks)

    return chunks


@dataclass
class FixQuoteCitationsResult:
    fixed_text: str
    unfixable_quotes: list[quote_util.ExtractedQuote]
    valid_quotes: list[quote_util.ExtractedQuote]
    referenced_chunks: list[Chunk]
    segments: list = field(default_factory=list)  # list[Segment]; default keeps old callers compiling


@dataclass
class RulebookIndex:
    combined: str        # all chunk contents joined with "\n\n"
    chunks: list[Chunk]  # ordered by start_index (unindexed appended last)
    offsets: list[int]   # combined-string start offset for each chunk


def build_rulebook_index(chunks: list[Chunk]) -> RulebookIndex:
    indexed = sorted([c for c in chunks if c["start_index"] >= 0], key=lambda c: c["start_index"])
    unindexed = [c for c in chunks if c["start_index"] < 0]
    ordered = indexed + unindexed

    combined = ""
    offsets: list[int] = []
    for chunk in ordered:
        offsets.append(len(combined))
        combined += chunk["content"] + "\n\n"

    return RulebookIndex(combined=combined, chunks=ordered, offsets=offsets)


def find_chunk_for_offset(offset: int, index: RulebookIndex) -> Chunk:
    """Return the chunk that owns the given character offset in the combined string."""
    idx = bisect.bisect_right(index.offsets, offset) - 1
    return index.chunks[max(0, idx)]


def find_chunks_for_span(start: int, end: int, index: RulebookIndex) -> list[Chunk]:
    """Return all chunks whose content overlaps with [start, end) in the combined string."""
    result = []
    for i, chunk in enumerate(index.chunks):
        chunk_start = index.offsets[i]
        chunk_end = chunk_start + len(chunk["content"])
        if chunk_start < end and chunk_end > start:
            result.append(chunk)
    return result


@dataclass
class QuoteMatch:
    matched_text: str              # literal document text at the match span
    source_chunk: Chunk            # chunk providing citation (rulebook_name, page)
    referenced_chunks: list[Chunk] # all chunks overlapping the match span
    match_ratio: float             # normalised score from find_quote_with_gaps (0.0–1.0)
    matched_span: Chunk | None = None  # exact span of match in source-document coordinates


@dataclass
class LocatedQuote:
    quote: quote_util.ExtractedQuote
    match: QuoteMatch | None        # high-confidence match (≥92); None = unverified
    hint_match: QuoteMatch | None = None  # low-confidence match used only for chunk extraction

    @property
    def is_verified(self) -> bool:
        return self.match is not None


@dataclass
class QuoteReplacement:
    start_index: int   # position in original text
    end_index: int     # position in original text
    replacement: str   # text to splice in
    is_verified: bool
    quote_type: str    # "blockquote" or "inline" — from ExtractedQuote


@dataclass
class PlainText:
    text: str


@dataclass(eq=False)
class QuoteSegment:
    original_text: str    # the quote's raw text in the original response
    located: LocatedQuote  # carries is_verified, quote_type, match (with matched_span)
    formatted_text: str   # result of format_quote (the replacement text)


Segment = Union[PlainText, QuoteSegment]


def get_chunks_by_rulebook_and_page(chunks: list[Chunk]) -> dict[tuple[str, str], list[Chunk]]:
    chunks_by_rulebook_and_page: dict[tuple[str, str], list[Chunk]] = {}
    for chunk in chunks:
        key = (chunk["rulebook_name"], chunk["page"])
        if key not in chunks_by_rulebook_and_page:
            chunks_by_rulebook_and_page[key] = []
        chunks_by_rulebook_and_page[key].append(chunk)
    return chunks_by_rulebook_and_page


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    # Group by (rulebook_name, page), preserving first-seen order
    groups: dict[tuple[str, str], list[Chunk]] = {}
    group_order: list[tuple[str, str]] = []
    for chunk in chunks:
        key = (chunk["rulebook_name"], chunk["page"])
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append(chunk)

    result: list[Chunk] = []
    for key in group_order:
        group = groups[key]
        indexed = [c for c in group if c["start_index"] >= 0]
        unindexed = [c for c in group if c["start_index"] < 0]

        # Merge overlapping/adjacent indexed chunks
        if indexed:
            indexed.sort(key=lambda c: c["start_index"])
            merged: list[Chunk] = [indexed[0].copy()]
            for chunk in indexed[1:]:
                prev = merged[-1]
                if chunk["start_index"] <= prev["end_index"]:  # overlap or adjacent
                    if chunk["end_index"] > prev["end_index"]:
                        overlap = prev["end_index"] - chunk["start_index"]
                        prev["content"] += chunk["content"][overlap:]
                        prev["end_index"] = chunk["end_index"]
                    # else: fully contained, skip
                else:
                    merged.append(chunk.copy())
            result.extend(merged)

        # Dedup unindexed by content
        seen: set[str] = set()
        for chunk in unindexed:
            if chunk["content"] not in seen:
                result.append(chunk)
                seen.add(chunk["content"])

    return result


def quote_entry_to_blockquote(quote_entry: QuoteEntry) -> str:
    blockquote = ""

    # Prefix each line with `>`
    for line in quote_entry["text"].splitlines():
        blockquote += f"> {line}\n"
    
    # Put citation at the end of the blockquote
    blockquote += f"> \n> ({quote_entry['rulebook_name']}, p. {quote_entry['page']})"

    return blockquote


def build_table_htmls_regex() -> re.Pattern:
    tags_list = ["table", "tr", "td", "th", "caption", "colgroup", "col", "tbody", "thead", "tfoot"]
    tags = "|".join(tags_list)
    gt = fr'{re.escape("&gt;")}|{re.escape(">")}'
    lt = fr'{re.escape("&lt;")}|{re.escape("<")}'
    # Match any attributes after the tag name up to the closing gt, note the closing gt can be escaped or unescaped
    attributes = r'(?:\s+\w+(?:\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s"\'=<>`&]+))?)*'
    elements = rf'({lt})(?P<tag_internal>/?({tags})({attributes}\s*))({gt})'
    pattern = re.compile(
        elements, re.DOTALL
    )
    return pattern


table_htmls_regex:re.Pattern = build_table_htmls_regex()

def unescape_table_html(text: str) -> str:
    # This is a bit hacky, so only do it if there is an escaped table tag
    if "&lt;table&gt;" not in text:
        return text

    # Replace all occurrences of table_htmls_regex with <tag_internal>, removing
    # the escaping
    def replacer(match):
        return f"<{match.group('tag_internal')}>"
    return table_htmls_regex.sub(replacer, text)


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
        Formatted blockquote with citation on its own line after a blank blockquote separator
    """
    # Unescape table HTML if present. We need to do this because the LLM will
    # sometimes "helpfully" escape the table HTML in the documents when quoting
    quote_text = unescape_table_html(quote_text)

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
        return quote_text.rstrip() + '\n> \n> ' + citation_text

    # Convert all lines after the first blockquote line to blockquote lines
    # (handle lazy continuation where lines don't start with >)
    formatted_lines = []
    for i, line in enumerate(lines):
        if i < first_blockquote_idx:
            # Lines before the first blockquote
            formatted_lines.append(line)
        elif line.strip().startswith('>'):
            # Already a blockquote line — strip any leading indentation (LLMs sometimes
            # indent blockquotes inside numbered lists; we normalise that here)
            formatted_lines.append(line.lstrip(' \t'))
        elif line.strip():  # Non-empty line that doesn't start with >
            # Lazy continuation - add > prefix
            formatted_lines.append('> ' + line)
        else:
            # Empty line - keep as is
            formatted_lines.append(line)

    # When a citation was in its own separate blockquote (> quote\n\n> (citation)),
    # text_before_citation ends with blank blockquote lines followed by a bare '>'.
    # Strip all trailing blank blockquote lines (both empty strings and bare '>'
    # lines, whose .strip() is '>' not '') since they're extraction artefacts.
    if formatted_lines and formatted_lines[-1].strip() == '>':
        while formatted_lines and formatted_lines[-1].strip() in ('', '>'):
            formatted_lines.pop()

    # Find the last non-empty line
    last_content_idx = -1
    for i in range(len(formatted_lines) - 1, -1, -1):
        if formatted_lines[i].strip():
            last_content_idx = i
            break

    if last_content_idx == -1:
        # No content found, fallback
        return quote_text.rstrip() + '\n> \n> ' + citation_text

    # Append citation on its own line after a blank blockquote separator
    result_lines = formatted_lines[:last_content_idx + 1]
    result_lines.append('> ')
    result_lines.append('> ' + citation_text)

    return '\n'.join(result_lines)


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


def locate_quotes(
    quotes: list[quote_util.ExtractedQuote],
    chunks: list[Chunk],
) -> list[LocatedQuote]:
    """Stage 2: Map each extracted quote to its source chunk(s) via fuzzy matching.

    For each quote, searches the per-rulebook combined text using find_quote_with_gaps.
    Quotes with ellipsis are allowed a wider match span (10× vs 3× quote length) to
    accommodate intentional gaps between passages.
    """
    chunks_by_rulebook: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        chunks_by_rulebook.setdefault(chunk["rulebook_name"], []).append(chunk)
    rulebook_indices: dict[str, RulebookIndex] = {
        name: build_rulebook_index(rb_chunks)
        for name, rb_chunks in chunks_by_rulebook.items()
    }

    located: list[LocatedQuote] = []
    for quote in quotes:
        quote_text = quote["quote"]
        match: QuoteMatch | None = None
        for index in rulebook_indices.values():
            m = quote_util.find_quote_with_gaps(index.combined, quote_text)
            if m:
                start_chunk_idx = max(0, bisect.bisect_right(index.offsets, m.start) - 1)
                source_chunk = index.chunks[start_chunk_idx]
                ref_chunks = find_chunks_for_span(m.start, m.end, index)
                chunk_start_in_combined = index.offsets[start_chunk_idx]
                span_start = source_chunk["start_index"] + (m.start - chunk_start_in_combined)
                end_chunk_idx = max(0, bisect.bisect_right(index.offsets, m.end - 1) - 1)
                end_chunk = index.chunks[end_chunk_idx]
                span_end = end_chunk["start_index"] + (m.end - index.offsets[end_chunk_idx])
                matched_span = Chunk(
                    rulebook_name=source_chunk["rulebook_name"],
                    page=source_chunk["page"],
                    start_index=span_start,
                    end_index=span_end,
                    content=m.matched_text,
                )
                match = QuoteMatch(
                    matched_text=m.matched_text,
                    source_chunk=source_chunk,
                    referenced_chunks=ref_chunks,
                    match_ratio=m.score / 100.0,
                    matched_span=matched_span,
                )
                break

        hint_match: QuoteMatch | None = None
        if match is None:
            for index in rulebook_indices.values():
                m = quote_util.find_quote_with_gaps(index.combined, quote_text, min_score=50)
                if m:
                    start_chunk_idx = max(0, bisect.bisect_right(index.offsets, m.start) - 1)
                    source_chunk = index.chunks[start_chunk_idx]
                    ref_chunks = find_chunks_for_span(m.start, m.end, index)
                    chunk_start_in_combined = index.offsets[start_chunk_idx]
                    span_start = source_chunk["start_index"] + (m.start - chunk_start_in_combined)
                    end_chunk_idx = max(0, bisect.bisect_right(index.offsets, m.end - 1) - 1)
                    end_chunk = index.chunks[end_chunk_idx]
                    span_end = end_chunk["start_index"] + (m.end - index.offsets[end_chunk_idx])
                    matched_span = Chunk(
                        rulebook_name=source_chunk["rulebook_name"],
                        page=source_chunk["page"],
                        start_index=span_start,
                        end_index=span_end,
                        content=m.matched_text,
                    )
                    hint_match = QuoteMatch(
                        matched_text=m.matched_text,
                        source_chunk=source_chunk,
                        referenced_chunks=ref_chunks,
                        match_ratio=m.score / 100.0,
                        matched_span=matched_span,
                    )
                    break

        located.append(LocatedQuote(quote=quote, match=match, hint_match=hint_match))

    return located


def format_quote(located: LocatedQuote, *, strip_invalid_blockquotes: bool = False) -> QuoteReplacement:
    """Stage 3: Determine the replacement text for a single quote.

    Unverified quotes are returned unchanged (identity replacement), except when
    strip_invalid_blockquotes=True, in which case unverified blockquotes have
    their '> ' markers stripped so they become plain prose. Verified quotes have
    their citations corrected and blockquotes reformatted inline.
    This is the single place to add further formatting changes (e.g. a
    data-verified wrapper div).
    """
    quote = located.quote
    original_text = quote["text"]

    if not located.is_verified:
        if quote["quote_type"] == "blockquote":
            if strip_invalid_blockquotes:
                replacement = quote_util.strip_blockquote_markers_and_quotes(original_text)
            else:
                replacement = re.sub(r'^[ \t]+(>)', r'\1', original_text, flags=re.MULTILINE)
        else:
            replacement = original_text
        return QuoteReplacement(
            start_index=quote["start_index"],
            end_index=quote["end_index"],
            replacement=replacement,
            is_verified=False,
            quote_type=quote["quote_type"],
        )

    assert located.match is not None
    source_chunk = located.match.source_chunk
    citation = quote["citation"]
    fixed_citation_text = f'({source_chunk["rulebook_name"]}, p. {source_chunk["page"]})'

    citation_correct = (
        citation is not None
        and citation["ref_name"] == source_chunk["rulebook_name"]
        and citation["page"] == source_chunk["page"]
    )

    if citation_correct:
        if quote["quote_type"] == "blockquote" and should_reformat_blockquote_citation(original_text, citation["start_index"]):
            text_before_citation = original_text[:citation["start_index"]]
            replacement = format_blockquote_with_inline_citation(text_before_citation, citation["text"])
        else:
            replacement = original_text
    elif citation is not None:
        text_before_citation = original_text[:citation["start_index"]]
        if quote["quote_type"] == "blockquote":
            replacement = format_blockquote_with_inline_citation(text_before_citation, fixed_citation_text)
        else:
            replacement = text_before_citation.rstrip() + ' ' + fixed_citation_text
    else:
        # No citation at all — insert one
        if quote["quote_type"] == "blockquote":
            replacement = format_blockquote_with_inline_citation(original_text, fixed_citation_text)
        else:
            replacement = original_text + ' ' + fixed_citation_text

    return QuoteReplacement(
        start_index=quote["start_index"],
        end_index=quote["end_index"],
        replacement=replacement,
        is_verified=True,
        quote_type=quote["quote_type"],
    )


def apply_replacements(text: str, replacements: list[QuoteReplacement]) -> str:
    """Stage 4: Apply quote replacements to the text.

    Processes in reverse index order so that earlier positions are not
    invalidated by changes at later positions.
    """
    for r in sorted(replacements, key=lambda r: r.start_index, reverse=True):
        text = text[:r.start_index] + r.replacement + text[r.end_index:]
    return text


# ── Segment model ─────────────────────────────────────────────────────────────


def build_segments(text: str, located: list[LocatedQuote], *, strip_invalid_blockquotes: bool = False) -> list[Segment]:
    """Split text into alternating PlainText/QuoteSegment based on quote positions.

    Each QuoteSegment carries the full LocatedQuote metadata and its
    formatted replacement text, so downstream operations do not need to
    re-run fuzzy matching.
    """
    result: list[Segment] = []
    cursor = 0
    for lq in sorted(located, key=lambda lq: lq.quote["start_index"]):
        start = lq.quote["start_index"]
        end = lq.quote["end_index"]
        if start > cursor:
            result.append(PlainText(text[cursor:start]))
        result.append(QuoteSegment(
            original_text=lq.quote["text"],
            located=lq,
            formatted_text=format_quote(lq, strip_invalid_blockquotes=strip_invalid_blockquotes).replacement,
        ))
        cursor = end
    if cursor < len(text):
        result.append(PlainText(text[cursor:]))
    return result


def materialize(segments: list[Segment], *, wrap_verified: bool = True) -> str:
    """Produce final text from a segment list.

    When wrap_verified=True, verified blockquotes are wrapped in
    <div data-quote-status="verified"> markers as the very last step before the
    string is formed, so no downstream function ever sees partially-wrapped text.
    """
    parts: list[str] = []
    for seg in segments:
        if isinstance(seg, PlainText):
            parts.append(seg.text)
        else:
            is_verified_bq = (
                seg.located.is_verified
                and seg.located.quote["quote_type"] == "blockquote"
            )
            if wrap_verified and is_verified_bq:
                parts.append(
                    f'<div data-quote-status="verified"></div>\n\n{seg.formatted_text}'
                )
            else:
                parts.append(seg.formatted_text)
    return "".join(parts)


def are_segments_adjacent(
    segments: list[Segment], seg_a: QuoteSegment, seg_b: QuoteSegment
) -> bool:
    """True when everything between seg_a and seg_b is whitespace-only PlainText.

    Replaces quote_util.are_blockquotes_adjacent for the deduplication step —
    purely structural, immune to whatever has been materialised between the quotes.
    """
    idx_a = next(i for i, s in enumerate(segments) if s is seg_a)
    idx_b = next(i for i, s in enumerate(segments) if s is seg_b)
    if idx_a > idx_b:
        idx_a, idx_b = idx_b, idx_a
    between = segments[idx_a + 1 : idx_b]
    return all(isinstance(s, PlainText) and s.text.strip() == "" for s in between)


def remove_quote_segments(
    segments: list[Segment], segs_to_remove: set
) -> list[Segment]:
    """Return a new segment list with the given QuoteSegments removed.

    Adjacent PlainText runs left by the removal are merged so the invariant
    (no two consecutive PlainTexts) is preserved.
    """
    result: list[Segment] = []
    for seg in segments:
        if isinstance(seg, QuoteSegment) and seg in segs_to_remove:
            continue
        if result and isinstance(result[-1], PlainText) and isinstance(seg, PlainText):
            result[-1] = PlainText(result[-1].text + seg.text)
        else:
            result.append(seg)
    return result


def fix_quote_citations_in_text(text: str, chunks: list[Chunk], *, strip_invalid_blockquotes: bool = False) -> FixQuoteCitationsResult:
    original_text = text
    quotes = quote_util.find_quotes_in_text(text)
    located = locate_quotes(quotes, chunks)
    parsed = build_segments(text, located, strip_invalid_blockquotes=strip_invalid_blockquotes)
    fixed_text = materialize(parsed, wrap_verified=False)

    if original_text != fixed_text:
        logger.debug("Fixed quote citations in text", text=original_text, fixed_text=fixed_text)

    unfixable_quotes = [lq.quote for lq in located if not lq.is_verified]
    valid_quotes = [lq.quote for lq in located if lq.is_verified]
    referenced_chunks = dedupe_chunks([
        chunk
        for lq in located
        for m in [lq.match or lq.hint_match]
        if m
        for chunk in m.referenced_chunks
    ])

    if unfixable_quotes:
        logger.warning(
            "Some quotes could not be verified and fixed",
            invalid_quotes=[q["text"] for q in unfixable_quotes],
            chunk_count=len(chunks),
            text_chars=len(text),
        )
        # The chunk set and full answer are what you actually need to diagnose this,
        # but they are far too large for INFO/WARNING in production.
        logger.debug(
            "Unverified quote context",
            chunks=chunks,
            text=text,
        )

    return FixQuoteCitationsResult(
        fixed_text=fixed_text,
        segments=parsed,
        unfixable_quotes=unfixable_quotes,
        referenced_chunks=referenced_chunks,
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


class ClarifyingQA(TypedDict):
    question: str
    """A clarifying question about the original question asked"""
    answer: str
    """The answer to the clarifying question"""
    evidence: list[Chunk]
    """The evidence chunks supporting the answer"""


class GameAgentInputState(MessagesState):
    query: str
    """The user's query"""
    recursion_depth: int
    """The current recursion depth"""
    evidence: list[Chunk]
    clarifying_questions: NotRequired[list[ClarifyingQA]]
    filter_invalid_quotes: NotRequired[bool]
    """When True, invalid quotes are silently stripped from the response rather than
    retried. Used for subquestion calls so hallucinated quotes cannot propagate into
    the final answer's validation corpus."""


class GameAgentOutputState(MessagesState):
    response: str
    """The answer to the user's query"""
    evidence: list[Chunk]
    """The evidence chunks supporting the answer"""
    valid: bool


class GameAgentOverallState(GameAgentInputState, GameAgentOutputState):
    rounds_clarification: int
    """The number of rounds of clarification performed"""
    validation_attempts: int
    """Track how many times we've attempted to fix/validate quotes"""
    refine_current_partition: NotRequired[int]
    """If set, indicates which partition index is being refined currently"""
    refine_saved_evidence: NotRequired[list[Chunk]]
    """If set, the evidence saved before refinement started"""
    answer: str
    reasoning: str
    invalid_quotes: list[QuoteEntry]
    format_attempts: int
    """Track how many times we've attempted format_answer with quote validation"""
    tokens_used: NotRequired[int]
    referenced_evidence: NotRequired[list[Chunk]]
    """The subset of `evidence` the current response actually quotes.

    Kept separate from `evidence` so that narrowing to "what the answer cited" does
    not shrink the corpus the later nodes still need. `format_answer` and
    `validate_and_fix_response` both read the full set via `get_evidence`; a quote
    can only be verified or repaired against a document that is still in hand.
    Quote validation and repair happen in `validate_and_fix_response`. By the time
    `provide_response` runs that loop has finished, so that is where the narrowed set
    becomes the graph's `evidence` output."""


class GetEvidenceInput(TypedDict):
    messages: list[AnyMessage]
    evidence: list[Chunk]


def get_evidence(state: GetEvidenceInput) -> list[Chunk]:
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
    
    documents = dedupe_chunks(documents)

    return documents

def sort_chunks(chunks: list[Chunk], gp: Manifest) -> list[Chunk]:
    # Sort chunks by rulebook priority and document position (start_index)
    rulebook_order = [rulebook["name"] for rulebook in gp["rulebooks"]]
    rulebook_priority = {name: index for index, name in enumerate(rulebook_order)}
    def chunk_sort_key(chunk: Chunk) -> tuple[int, int]:
        rulebook_name = chunk["rulebook_name"]
        start_index = chunk["start_index"]
        priority = rulebook_priority.get(rulebook_name, len(rulebook_priority))
        return (priority, start_index)
    sorted_chunks = sorted(chunks, key=chunk_sort_key)
    return sorted_chunks


class QuestionAnalysisOverallState(MessagesState):
    documents: list[Chunk]
    reasoning: str
    query: str
    analysis: str
    subquestions: list[str]
    classification: Literal["SIMPLE", "COMPLEX"]
    tokens_used: NotRequired[int]


def create_question_analysis_state(query, **kwargs) -> QuestionAnalysisOverallState:
    state = QuestionAnalysisOverallState(
        documents=[],
        messages=[],
        reasoning="",
        query=query,
        analysis="",
        subquestions=[],
        classification="SIMPLE",
    )
    state.update(**kwargs)
    return state


@dataclass
class QuestionAnalysisContext:
    manifest: Manifest
    chunk_search_service: ChunkSearchService


# analyze_user_query_template = """\
# Examine the user query in light of the provided documents. What is the user asking? Do not answer the question, just explain what the question is.
# """

# analyze_user_query_retrieval_template = """\
# Examine the user query. What is the user asking? Do not answer the question, just explain what the question is. Enumerate the names of all the rules involved in the user's question. You can lookup information for rules questions by using the `search_chunks` tool.
# """
analyze_user_query_template = load_template("analyze_question.md")

list_essential_rule_interactions_template = """\
Some rule interactions are transitive (X has Y which interacts with Z) others are direct (Y interacts with Z). We care about the direct rule interactions (X has Y and Y interacts with Z) essential to answering the user's query. Given the user's query and the explanation of what the user is asking, identify the direct rule interactions that are the crux of what the user is asking. Ensure to list interactions that might be exceptions to a general rule. List them in the format "Y interacts with Z" where Y and Z are the names of rules (no citations required here). Provide no more than 5. Do not include parenthetical elements or commentary. Provide the output as a json list of strings.
"""

short_system_prompt_template = """\
You are an expert Rules Lawyer specializing in boardgame rules. Being "technically correct" is your highest aspiration. You believe in "the rules as written" above all else, because the rules are not merely words on a page, they are devine truth. You are sensitive to even the slimmest nuances in wording, and you always interpret the rules in the most literal way possible. You never make assumptions or inferences beyond what is explicitly written in the rules, because that would be the greatest of heresies. You have a keen eye for detail, and you always notice even the smallest distinctions in wording that others might overlook.

{{#documents.0}}
## Documents

<documents>
{{#documents}}
<document rulebook_name="{{rulebook_name}}" page="{{page}}" start_index="{{start_index}}">
{{content}}
</document>
{{/documents}}
</documents>
{{/documents.0}}

{{#query}}
## User query

<query>
{{query}}
</query>
{{/query}}
"""


class EssentialRuleInteractionResponse(TypedDict):
    essential_rule_interactions: Annotated[list[str], ..., "list of essential direct rule interactions needed to answer the query. Format each as 'Y interacts with Z' where Y and Z are the names of rules. No citations required."]


class Subquestion(TypedDict):
    subquestion: Annotated[str, ..., "An independently answerable subquestion targeting a distinct rule or rule interaction"]
    explanation: Annotated[str, ..., "An explanation as to how an answer to this subquestion helps address the user's original query."]


class RelevantMechanic(TypedDict):
    name: Annotated[str, ..., "Name of the game mechanic, rule concept, or category relevant to answering the question"]
    in_question: Annotated[bool, ..., "True if this mechanic is explicitly named in the user's question; False if it was introduced by the retrieved documents"]


class QuestionAnalysis(TypedDict):
    mechanics: Annotated[list[RelevantMechanic], ..., "All game mechanics relevant to answering the question. Fill this in before determining classification."]
    classification: Annotated[Literal["SIMPLE", "COMPLEX"], ..., "COMPLEX if any entry in 'mechanics' has in_question=False — the relationship between that unlisted mechanic and the question's mechanics is unresolved and must be researched. SIMPLE only if every mechanic has in_question=True AND a direct answer passage exists in the retrieved documents."]
    explanation: Annotated[str, ..., "Explanation of what the user is asking and the key rules and rule interactions involved in the user's query. Be concise — 2 to 4 sentences maximum."]
    subquestions: Annotated[list[Subquestion], ..., "List of 2-5 independently answerable subquestions for COMPLEX queries. Empty list for SIMPLE queries."]


def build_analyze_question_graph(
    checkpoint_saver: Optional[BaseCheckpointSaver],
    chat_model: BaseChatModel,
    tokenizer: Any
) -> CompiledStateGraph[QuestionAnalysisOverallState, QuestionAnalysisContext, QuestionAnalysisOverallState, QuestionAnalysisOverallState]:
    tools = [search_chunks]
    tool_node = ToolNode(tools)

    async def retrieve_data(state: QuestionAnalysisOverallState, *, runtime: Runtime[QuestionAnalysisContext], config: RunnableConfig|None = None) -> dict:
        retrieval_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", short_system_prompt_template),
                ("user", analyze_user_query_template),
            ],
            template_format="mustache"
        )

        input = {
            "retrieval": True,
            "game_summary": False,
            "documents": [],
            "query": state["query"],
        }

        chat_model_with_tools = chat_model.bind_tools(tools, tool_choice="any")
        chain = retrieval_prompt | chat_model_with_tools

        message = await chain.ainvoke(input, config=config)

        tokens_used = calculate_tokens_used(tokenizer, retrieval_prompt, input)

        return {
            "tokens_used": tokens_used,
            "messages": [message]
        }

    async def analyze_question(state: QuestionAnalysisOverallState, *, runtime: Runtime[QuestionAnalysisContext], config: RunnableConfig|None = None) -> dict:
        retrieval_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", short_system_prompt_template),
                ("user", analyze_user_query_template),
            ],
            template_format="mustache"
        )

        # Pull documents from message history
        chunks = get_all_chunks_from_message_history(state["messages"])
        chunks = sort_chunks(chunks, runtime.context.manifest)

        input = {
            "retrieval": False,
            "game_summary": False,
            "query": state["query"],
            "documents": chunks,
            "clarifying_questions_and_answers": [],
            "messages": state["messages"],
        }

        chat_model_with_structured = chat_model.with_structured_output(QuestionAnalysis, include_raw=True)
        chain = retrieval_prompt | chat_model_with_structured

        result = await chain.ainvoke(input, config=config)
        assert isinstance(result, dict)
        question_analysis = result["parsed"]
        message = result["raw"]

        tokens_used = calculate_tokens_used(tokenizer, retrieval_prompt, input)
        for sub in question_analysis["subquestions"]:
            logger.info("Subquestion identified", subquestion=sub["subquestion"], explanation=sub["explanation"])

        subquestions = [sub["subquestion"] for sub in question_analysis["subquestions"]]
        classification = question_analysis["classification"]

        # Fall back to SIMPLE if classified COMPLEX but no subquestions were produced
        if classification == "COMPLEX" and not subquestions:
            logger.warning("Classified as COMPLEX but no subquestions generated, falling back to SIMPLE", query=state["query"])
            classification = "SIMPLE"

        return {
            "analysis": question_analysis["explanation"],
            "subquestions": subquestions,
            "classification": classification,
            "tokens_used": tokens_used,
            # TODO: We should return something that indicates the importance of
            # each chunk. The caller might want to use these in its context, but
            # have a different budget for how many chunks to include.
            "documents": chunks,
            "messages": [message],
        }

    async def consolidate_search_calls(state: MessagesState) -> dict:
        message = state["messages"][-1]
        tool_calls = getattr(message, "tool_calls", [])
        # Copy the list so that we don't edit the original
        tool_calls = copy.deepcopy(tool_calls)
        # Find indices of search_chunks calls
        search_chunks_indices = [i for i, call in enumerate(tool_calls) if call["name"] == "search_chunks"]

        # Bail early if we don't have multiple search calls
        if len(search_chunks_indices) <= 1:
            return {}
        
        # Okay, we have multiple search calls, let's collect the queries
        queries = []
        for index in search_chunks_indices:
            tool_call = tool_calls[index]
            query = tool_call["args"]["search_queries"]
            if isinstance(query, str):
                queries.append(query)
            elif isinstance(query, list):
                queries.extend(query)
            else:
                logger.warning("Unexpected query format in search_chunks tool call", query=query)
        
        # Remove all but the first search_chunks call
        for index in reversed(search_chunks_indices[1:]):
            del tool_calls[index]
        
        # Update the first search_chunks call to include all queries
        first_index = search_chunks_indices[0]
        tool_calls[first_index]["args"]["search_queries"] = queries

        # Update the message with the consolidated tool calls
        setattr(message, "tool_calls", tool_calls)

        # Now return an update for the specific message
        return {
            # Should match on ID and replace the existing message
            "messages": [message]
        }


    async def produce_response(state: QuestionAnalysisOverallState, *, runtime: Runtime[QuestionAnalysisContext], config: RunnableConfig|None = None) -> dict:
        return {}
    

    graph = StateGraph(
        state_schema=QuestionAnalysisOverallState,
        context_schema=QuestionAnalysisContext,
    )

    # Nodes (tool_node only handles search_chunks)
    graph.add_node("tool_node", tool_node)
    graph.add_node("consolidate_search_calls", consolidate_search_calls)
    graph.add_node("retrieve_data", retrieve_data)
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("produce_response", produce_response)
    graph.add_node("dedupe_chunks", dedupe_chunks_node)

    # Edges
    graph.add_edge(START, "retrieve_data")
    graph.add_edge("retrieve_data", "consolidate_search_calls")
    graph.add_edge("consolidate_search_calls", "tool_node")
    graph.add_edge("tool_node", "dedupe_chunks")
    graph.add_edge("dedupe_chunks", "analyze_question")
    graph.add_edge("analyze_question", "produce_response")
    graph.add_edge("produce_response", END)

    agent = graph.compile(checkpointer=checkpoint_saver)
    return agent


class CoordinationInputState(TypedDict):
    query: str


class CoordinationOutputState(TypedDict):
    response: str
    evidence: list[Chunk]
    clarifying_questions: list[ClarifyingQA]


class CoordinationOverallState(TypedDict):
    query: str
    question_analysis: str
    subquestions: list[str]
    clarifying_questions: list[ClarifyingQA]
    response: str
    evidence: list[Chunk]
    classification: Literal["SIMPLE", "COMPLEX"]

def build_coordinating_agent_graph(
    checkpoint_saver: Optional[BaseCheckpointSaver],
    analyze_question_agent: CompiledStateGraph[QuestionAnalysisOverallState, QuestionAnalysisContext, QuestionAnalysisOverallState, QuestionAnalysisOverallState],
    game_agent: CompiledStateGraph[GameAgentOverallState, GameAgentContext, GameAgentInputState, GameAgentOutputState],
) -> CompiledStateGraph[CoordinationOverallState, GameAgentContext, CoordinationInputState, CoordinationOutputState]:
    
    async def analyze_question(state: CoordinationInputState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        writer = get_stream_writer()
        writer(StepEvent(type="mm_step", description="Analyzing user query"))

        logger.info("Analyzing question", query=state["query"])
        state_analyze_question = create_question_analysis_state(query=state["query"])
        context_analyze_question = QuestionAnalysisContext(
            manifest=runtime.context.manifest,
            chunk_search_service=runtime.context.chunk_search_service,
        )
        result = await analyze_question_agent.ainvoke(
            state_analyze_question,
            context=context_analyze_question,
            config=config
        )

        writer(AnalyzedUserQueryEvent(type="mm_user_query_analysis", analysis=result["analysis"], classification=result["classification"]))

        return {
            "question_analysis": result["analysis"],
            "subquestions": result["subquestions"],
            "evidence": result["documents"],
            "classification": result["classification"],
        }


    class AskSubquestionsInputState(TypedDict):
        question_analysis: str
        subquestions: list[str]


    async def ask_simple_question(state: CoordinationOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        # Reuse the evidence analyze_question already retrieved. Without this the
        # game agent re-runs retrieval with near-identical queries and can come back
        # with a different chunk set, dropping evidence the analysis stage had in hand.
        input = {
            "query": state["query"],
            "evidence": state.get("evidence", []),
        }

        writer = get_stream_writer()
        writer(StepEvent(type="mm_step", description="Answering user query"))

        response = await game_agent.ainvoke(
            input,
            context=runtime.context,
            config=config
        )

        return {
            "response": response["response"],
            "evidence": response["evidence"],
        }


    async def ask_subquestions(state: AskSubquestionsInputState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        subquestions = state["subquestions"]

        writer = get_stream_writer()
        writer(StepEvent(type="mm_step", description="Breaking down into subquestions"))

        inputs = []
        for question in subquestions:
            logger.info("Asking subquestion", question=question)
            input = {
                "query": question,
                "filter_invalid_quotes": True,
            }
            inputs.append(input)
        
        async def result_with_input(coroutine, input):
            result = await coroutine
            return result, input

        tasks = []
        for input in inputs:
            task = asyncio.create_task(
                result_with_input(
                    game_agent.ainvoke(
                        input,
                        context=runtime.context,
                        config=config
                    ),
                    input
                )
            )
            tasks.append(task)

        subquestions_answers: list[ClarifyingQA] = []
        for future in asyncio.as_completed(tasks):
            response, input = await future
            writer(SubquestionAnsweredEvent(type="mm_subquestion_answered", question=input["query"], answer=response["response"], valid=response["valid"]))

            if not response["valid"]:
                logger.error("Subquestion answer was not valid", question=input["query"], answer_chars=len(response["response"] or ""))
                logger.debug("Invalid subquestion answer", question=input["query"], answer=response["response"])
                continue

            subquestions_answers.append(
                ClarifyingQA(
                    question=input["query"],
                    answer=response["response"],
                    evidence=response["evidence"],
                )
            )

        return {
            "clarifying_questions": subquestions_answers
        }
    
    class CombineSubanswersInputState(TypedDict):
        query: str
        clarifying_questions: list[ClarifyingQA]

    async def combine_subanswers(state: CombineSubanswersInputState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        # Collect evidence from the subquestion answers
        evidence = []
        for qa in state.get("clarifying_questions", []):
            evidence.extend(qa["evidence"])
        
        evidence = dedupe_chunks(evidence)

        writer = get_stream_writer()
        writer(StepEvent(type="mm_step", description="Answering user query"))

        input = {
            "query": state["query"],
            "clarifying_questions": state.get("clarifying_questions", []),
            "evidence": evidence
        }
        response = await game_agent.ainvoke(
            input,
            context=runtime.context,
            config=config
        )

        return {
            "response": response["response"],
            "evidence": response["evidence"],
        }

    async def produce_response(state: CoordinationOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        writer = get_stream_writer()
        writer(UserQueryAnsweredEvent(type="mm_user_query_answered", answer=state["response"], query=state["query"]))
        return {}
    
    async def route_by_classification(state: CoordinationOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> Literal["ask_simple_question", "ask_subquestions"]:
        if state["classification"] == "SIMPLE":
            return "ask_simple_question"
        else:
            return "ask_subquestions"

    graph = StateGraph(
        state_schema=CoordinationOverallState,
        input_schema=CoordinationInputState,
        output_schema=CoordinationOutputState,
        context_schema=GameAgentContext,
    )

    graph.add_node("analyze_question", analyze_question)
    graph.add_node("ask_subquestions", ask_subquestions)
    graph.add_node("combine_subanswers", combine_subanswers)
    graph.add_node("produce_response", produce_response)
    graph.add_node("ask_simple_question", ask_simple_question)

    graph.add_edge(START, "analyze_question")
    graph.add_conditional_edges("analyze_question", route_by_classification)
    graph.add_edge("ask_subquestions", "combine_subanswers")
    graph.add_edge("combine_subanswers", "produce_response")
    graph.add_edge("ask_simple_question", "produce_response")
    graph.add_edge("produce_response", END)

    agent = graph.compile(checkpointer=checkpoint_saver)
    return agent


async def dedupe_chunks_node(state: MessagesState) -> dict:
    """Dedupe chunks in the message history"""
    messages = state["messages"]
    message_edits = dedupe_chunks_in_message_history(messages)
    return {
        "messages": message_edits
    }


def calculate_tokens_used(tokenizer: Any,prompt: ChatPromptTemplate, input: dict) -> int:
    messages_formatted = prompt.format_messages(**input)

    role_map = role_map = {"human": "user", "ai": "assistant", "system": "system"}
    message_dicts = [
        {"role": role_map.get(m.type, m.type), "content": m.text} for m in messages_formatted
    ]

    prompt_tokens = len(
        tokenizer.apply_chat_template(
            message_dicts, tokenize=True, add_generation_prompt=True
        )
    )

    return prompt_tokens


def are_chunks_overlapping(chunk1: Chunk|None, chunk2: Chunk|None) -> bool:
    if chunk1 is None or chunk2 is None:
        return False

    if chunk1["rulebook_name"] != chunk2["rulebook_name"]:
        return False
    if chunk1["page"] != chunk2["page"]:
        return False

    if chunk1["start_index"] > chunk2["start_index"]:
        chunk1, chunk2 = chunk2, chunk1

    # Now chunk1 starts before chunk2, so we just need to check if chunk1 ends after chunk2 starts
    return chunk1["end_index"] > chunk2["start_index"]


def extracted_quote_to_quote_entry(extracted_quote: quote_util.ExtractedQuote) -> QuoteEntry:
    return QuoteEntry(
        text=extracted_quote["quote"],
        rulebook_name=extracted_quote["citation"]["ref_name"] if extracted_quote["citation"] else "",
        page=extracted_quote["citation"]["page"] if extracted_quote["citation"] else ""
    )

class FixQuotesResult(TypedDict):
    reasoning: str
    fixable: bool
    fixed_quote: QuoteEntry

class FixQuoteInput(TypedDict):
    invalid_quote: QuoteEntry
    documents: list[Chunk]

def is_usable_fix_quote_result(result: Any) -> bool:
    """True if a fix-quote result carries everything we need to apply the fix.

    The structured-output parser repairs *partial* JSON, so a generation that
    runs away mid-object still yields a dict, just one missing its later keys.
    Callers treat an unusable result as "not fixed" rather than indexing into it.
    """
    if not isinstance(result, dict) or not result.get("fixable"):
        return False
    fixed_quote = result.get("fixed_quote")
    return isinstance(fixed_quote, dict) and all(
        key in fixed_quote for key in ("text", "rulebook_name", "page")
    )


def build_fix_quote_chain(chat_model: BaseChatModel) -> Runnable[FixQuoteInput,FixQuotesResult]:
    system_prompt_with_documents = load_template('system_prompt_with_documents.md')
    fix_quotes = load_template('fix_invalid_quotes.md')

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_with_documents),
            ("human", fix_quotes)
        ],
        template_format="mustache"
    )

    # max_tokens has to go through with_structured_output, binding it on the model
    # first is silently discarded, since with_structured_output rebinds from scratch.
    # A repair is a short JSON object; without a cap a hopeless input (e.g. an empty
    # document set) makes the model ramble until it hits the model-level limit.
    chain = prompt | chat_model.with_structured_output(FixQuotesResult, max_tokens=1024)

    chain = chain.with_config(
        run_name="fix_quote_chain",
    )

    return cast(Runnable[FixQuoteInput, FixQuotesResult], chain)


class ValidateAndFixResponseInput(TypedDict):
    response: str
    evidence: list[Chunk]
    messages: list[AnyMessage]
    validation_attempts: int


@dataclass
class ValidateAndFixResponseContext:
    chat_model: BaseChatModel
    manifest: Manifest


class ValidateAndFixResponseOutput(TypedDict):
    response: str
    referenced_evidence: NotRequired[list[Chunk]]
    invalid_quotes: list[QuoteEntry]
    validation_attempts: int


def _strip_invalid_quotes(fix_result: FixQuoteCitationsResult, documents: list[Chunk]) -> ValidateAndFixResponseOutput:
    """Remove unverified quote segments from the response and return a clean result.

    Used when filter_invalid_quotes=True (subquestion mode) so that hallucinated quotes
    are silently dropped rather than retried or propagated to the final answer.
    """
    invalid_segs = {
        seg for seg in fix_result.segments
        if isinstance(seg, QuoteSegment) and not seg.located.is_verified
    }
    cleaned = remove_quote_segments(fix_result.segments, invalid_segs)
    # Use referenced_chunks directly, it already includes chunks recovered via hint_match
    # from low-confidence (paraphrase) matches, not just verified quotes.
    return {
        "response": materialize(cleaned, wrap_verified=True),
        "referenced_evidence": fix_result.referenced_chunks,
        "invalid_quotes": [],
        "validation_attempts": 0,
    }


async def validate_and_fix_response(state: ValidateAndFixResponseInput, *, runtime: Runtime[Any], config: RunnableConfig|None = None) -> ValidateAndFixResponseOutput:
    # Get required services
    chat_model = runtime.context.chat_model

    # Get the current answer and related documents
    response = state["response"]
    documents = get_evidence(state)
    documents = sort_chunks(documents, runtime.context.manifest)

    if state.get("filter_invalid_quotes", False):
        # Exclude synthetic chunks (start_index == -1) built from unverified quotes in
        # earlier subquestion answers. They are not real source-document spans and must
        # not be used as evidence when validating quotes in this response.
        documents = [d for d in documents if d.get("start_index", -1) != -1]

    # Do simple citation fixes and validate the response
    fix_result = fix_quote_citations_in_text(response, documents)
    fixed_response = fix_result.fixed_text

    # Get the quotes from the fixed response
    valid_extracted, invalid_extracted = fix_result.valid_quotes, fix_result.unfixable_quotes

    # If nothing is invalid, we are good to return
    if len(invalid_extracted) == 0:
        # All quotes valid, build evidence from the quotes in the response
        valid_quote_entries = [extracted_quote_to_quote_entry(vq) for vq in valid_extracted]
        evidence = compile_evidence_from_documents(valid_quote_entries, documents)
        return {
            "response": materialize(fix_result.segments, wrap_verified=True),
            "referenced_evidence": list(evidence),
            "invalid_quotes": [],
            "validation_attempts": 0,
        }

    # Okay, something is still busted with the response
    invalid_quote_entries = [
        extracted_quote_to_quote_entry(iq)
        for iq in invalid_extracted
    ]

    # Formatting inline quotes correctly can be tricky, so we'll only
    # attempt to repair blockquotes for now
    if any(q["quote_type"] not in ["blockquote"] for q in invalid_extracted):
        if state.get("filter_invalid_quotes", False):
            return _strip_invalid_quotes(fix_result, documents)
        return {
            "response": materialize(fix_result.segments, wrap_verified=True),
            "referenced_evidence": fix_result.referenced_chunks,
            "invalid_quotes": invalid_quote_entries,
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    # Okay, we have bad quotes, but they are all blockquotes, there is hope!

    # ...unless there is nothing to match them against. The repair chain can only
    # return a verbatim passage from the documents, so an empty set makes every
    # quote unfixable by construction. Skipping saves an LLM call per quote, and
    # avoids handing the model a hopeless task it tends to ramble on.
    if not documents:
        logger.info(
            "Skipping quote repair, no documents to match against",
            invalid_quote_count=len(invalid_extracted),
        )
        if state.get("filter_invalid_quotes", False):
            return _strip_invalid_quotes(fix_result, documents)
        return {
            "response": materialize(fix_result.segments, wrap_verified=True),
            "referenced_evidence": fix_result.referenced_chunks,
            "invalid_quotes": invalid_quote_entries,
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    # Order the extracted blockquotes. This allows us to index the
    # blockquotes by order of appearance.
    fixed_response = fixed_response
    all_extracted_ordered = quote_util.find_quotes_in_text(fixed_response)
    all_extracted_ordered = [eq for eq in all_extracted_ordered if eq["quote_type"] == "blockquote"]
    all_extracted_ordered.sort(key=lambda eq: eq["start_index"])
    total_blockquotes = len(all_extracted_ordered)

    invalid_indices = []
    invalid_index_to_fixed = {}
    for idx, eq in enumerate(all_extracted_ordered):
        for invalid_eq in invalid_extracted:
            if eq["quote"] == invalid_eq["quote"]:
                invalid_indices.append(idx)


    # Use LLM to attempt to fix each invalid quote by looking at our
    # documents
    chain = build_fix_quote_chain(chat_model)

    inputs: list[FixQuoteInput] = []
    inputs_to_indices: list[int] = []
    for idx in invalid_indices:
        iq = extracted_quote_to_quote_entry(all_extracted_ordered[idx])
        inputs.append(
            FixQuoteInput(
                invalid_quote=iq,
                documents=documents
            )
        )
        inputs_to_indices.append(idx)
    
    # return_exceptions so one bad repair cannot abort the whole request. A
    # generation that hits the token limit raises LengthFinishReasonError out of
    # the structured-output parser rather than returning partial content; that is
    # a failed repair, not a failed request, and is_usable_fix_quote_result already
    # rejects anything that is not a well-formed dict.
    fix_quote_results: list[FixQuotesResult | BaseException] = await chain.abatch(
        inputs,
        config=config,
        return_exceptions=True,
    )

    # RunnableSequence.abatch catches BaseException, not Exception, so with
    # return_exceptions it also captures cancellation, Ctrl-C and interpreter exit.
    # None of those are repair failures: swallowing a CancelledError would keep the
    # request running after the caller has gone away. Re-raise anything that is not
    # a normal error and let genuine failures through as values.
    for result in fix_quote_results:
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result

    for idx, result in zip(invalid_indices, fix_quote_results):
        invalid_index_to_fixed[idx] = result

    # Bail if we were unable to fix all the quotes. A result the model never
    # finished counts as unfixed, same as one it declared unfixable, either way
    # we can't apply it, and the caller retries.
    all_fixed = all(is_usable_fix_quote_result(result) for result in fix_quote_results)
    if not all_fixed:
        errors = [r for r in fix_quote_results if isinstance(r, BaseException)]
        logger.info(
            "Unable to fix all invalid quotes",
            quote_count=len(fix_quote_results),
            unfixable_count=sum(
                1 for r in fix_quote_results if isinstance(r, dict) and r.get("fixable") is False
            ),
            incomplete_count=sum(
                1 for r in fix_quote_results
                if not is_usable_fix_quote_result(r)
                and not isinstance(r, BaseException)
                and not (isinstance(r, dict) and r.get("fixable") is False)
            ),
            error_count=len(errors),
            errors=[repr(e) for e in errors],
        )
        if state.get("filter_invalid_quotes", False):
            return _strip_invalid_quotes(fix_result, documents)
        return {
            "response": materialize(fix_result.segments, wrap_verified=True),
            "referenced_evidence": fix_result.referenced_chunks,
            "invalid_quotes": invalid_quote_entries,
            "validation_attempts": state.get("validation_attempts", 0) + 1,
        }

    # Okay, let's apply the fixes to the response
    
    # Iterate invalid quotes in reverse order of appearance. This is so that
    # as we replace text we don't mess up the start_index of the remaining
    # invalid quotes that we need to fix.
    for idx in sorted(invalid_indices, reverse=True):
        invalid_quote = all_extracted_ordered[idx]
        fix_result = invalid_index_to_fixed[idx]
        assert fix_result["fixable"], "We should have bailed if any quote was unfixable"
        fixed_quote = fix_result["fixed_quote"]
        # Format the fixed quote as a blockquote with the correct citation
        fixed_blockquote = quote_entry_to_blockquote(fixed_quote)
        # Replace the invalid quote in the response with the fixed blockquote
        start_index = invalid_quote["start_index"]
        end_index = invalid_quote["end_index"]
        fixed_response = (
            fixed_response[:start_index]
            + fixed_blockquote
            + fixed_response[end_index:]
        )

    # Okay, we have applied the fixes, now let's re-validate
    fix_result2 = fix_quote_citations_in_text(fixed_response, documents)
    fixed_response = fix_result2.fixed_text

    if len(fix_result2.unfixable_quotes) > 0:
        # Okay, it is still busted, time to bail
        logger.info("Response still invalid after attempting fixes")
        for iq in fix_result2.unfixable_quotes:
            logger.info("Unfixable quote", quote=iq["quote"])
        return {
            "response": materialize(fix_result2.segments, wrap_verified=True),
            "referenced_evidence": fix_result2.referenced_chunks,
            "invalid_quotes": [
                extracted_quote_to_quote_entry(iq) for iq in fix_result2.unfixable_quotes
            ],
            "validation_attempts": state.get("validation_attempts", 0) + 1
        }

    # Build an ordered list of blockquote segments from the second pass.
    # This replaces re-parsing + create_chunks_for_quotes, the segment list
    # already carries all the metadata we need.
    ordered_bq_segs: list[QuoteSegment] = [
        s for s in fix_result2.segments
        if isinstance(s, QuoteSegment) and s.located.quote["quote_type"] == "blockquote"
    ]

    assert len(ordered_bq_segs) == total_blockquotes, (
        "We should not have changed the number of blockquotes in the response, "
        "only fixed their formatting and citations. If this assertion fails, we need "
        "to add logic to handle the case where the number of blockquotes changes, "
        "since that can affect the indices of the quotes in the response and how we apply fixes."
    )

    # The LLM will often output multiple consecutive blockquotes that point
    # out essentially the same rule in multiple places in the documents.
    # Sometimes when fixing the quotes, we end up with duplicates of the
    # same quote adjacent to each other in the response. We can drop any
    # fixed quotes that introduce such duplication.

    # Get the consecutive groups of blockquotes, purely structural check
    # (immune to whatever text is materialised between the quotes).
    consecutive_blockquote_groups: List[List[int]] = []
    current_group: List[int] = []
    for index, seg in enumerate(ordered_bq_segs):
        if not current_group:
            current_group.append(index)
        else:
            prev_seg = ordered_bq_segs[current_group[-1]]
            if are_segments_adjacent(fix_result2.segments, prev_seg, seg):
                current_group.append(index)
            else:
                consecutive_blockquote_groups.append(current_group)
                current_group = [index]
    if current_group:
        consecutive_blockquote_groups.append(current_group)

    segs_to_remove: set[QuoteSegment] = set()
    for group in consecutive_blockquote_groups:
        overlapping_pairs: List[tuple[int, int]] = []
        for i in group:
            if ordered_bq_segs[i] in segs_to_remove:
                continue
            for j in group:
                if ordered_bq_segs[j] in segs_to_remove:
                    continue
                if i == j:
                    continue
                # We only care about pairs where one of our fixed quotes is
                # implicated
                if i not in invalid_indices and j not in invalid_indices:
                    continue

                # Overlap is determined by the match span in the source document
                # (not just chunk identity — two quotes from the same chunk but
                # different non-overlapping spans should not be treated as overlapping).
                match_i = ordered_bq_segs[i].located.match
                match_j = ordered_bq_segs[j].located.match
                span_i = match_i.matched_span if match_i else None
                span_j = match_j.matched_span if match_j else None
                if are_chunks_overlapping(span_i, span_j):
                    overlapping_pairs.append((i, j))

        for i, j in overlapping_pairs:
            seg_i = ordered_bq_segs[i]
            seg_j = ordered_bq_segs[j]
            # If we've already decided to remove one of these, skip
            if seg_i in segs_to_remove or seg_j in segs_to_remove:
                continue

            # We have two blockquotes that are adjacent and point to overlapping
            # spans in the source document. This likely means they quote the
            # same rule, so we can remove one of them. If only one was fixed
            # by the LLM, remove that one since it introduced the duplication.
            if i in invalid_indices and j in invalid_indices:
                # Keep the one that starts earlier in the source document
                span_i = seg_i.located.match.matched_span if seg_i.located.match else None
                span_j = seg_j.located.match.matched_span if seg_j.located.match else None
                if span_i and span_j and span_i["start_index"] < span_j["start_index"]:
                    segs_to_remove.add(seg_j)
                else:
                    segs_to_remove.add(seg_i)
            elif j in invalid_indices:
                segs_to_remove.add(seg_j)
            elif i in invalid_indices:
                segs_to_remove.add(seg_i)

    # Remove duplicate blockquotes and materialise the final response.
    # Wrapping is applied in the same materialisation step, so there is no
    # second fuzzy-match pass for wrapping.
    final_segments = remove_quote_segments(fix_result2.segments, segs_to_remove)
    # In the process of fixing the quotes, we may have introduced extra
    # blank lines, so collapse those in the final string.
    response_str = re.sub(r'\n\s*\n', '\n\n', materialize(final_segments, wrap_verified=True))

    # We should have a fixed response now. Yay! Let's compile the evidence.
    quote_entries = [
        extracted_quote_to_quote_entry(eq) for eq in fix_result2.valid_quotes
    ]
    evidence = compile_evidence_from_documents(quote_entries, documents)
    return {
        "response": response_str,
        "referenced_evidence": list(evidence),
        "invalid_quotes": [],
        "validation_attempts": 0,
    }


def build_question_answer_graph(
    checkpoint_saver: Optional[BaseCheckpointSaver],
    chat_model: BaseChatModel,
    tokenizer: Any,
) -> CompiledStateGraph[GameAgentOverallState, GameAgentContext, GameAgentInputState, GameAgentOutputState]:
    
    tools = [search_chunks]
    tool_node = ToolNode(tools)
    
    async def retrieve_data(state: GameAgentInputState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        logger.info("Retrieving data for query", query=state["query"])
        manifest = runtime.context.manifest
        retrieve_data_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", query_documents_guidelines_template),
                ("user", retrieve_documents_template),
            ],
            template_format="mustache"
        )
        chat_model_with_tools = chat_model.bind_tools(tools, tool_choice="any")
        chain = retrieve_data_prompt | chat_model_with_tools

        # Collect evidence chunks from clarifying questions, since
        # get_evidence() will add these to the documents list in answer_question.
        clarifying_evidence = []
        for qa in state.get("clarifying_questions", []):
            clarifying_evidence.extend(qa.get("evidence", []))

        input = {
            "game_summary": False,
            "game_name": runtime.context.manifest["name"],
            "query": state["query"],
            "documents": clarifying_evidence,
            "clarifying_questions_and_answers": state.get("clarifying_questions", []),
        }

        # Compute the token budget against the ANSWER prompt (not the retrieval
        # prompt), since that's where the retrieved documents will ultimately be
        # consumed. This accounts for both the clarifying Q&A text AND their
        # evidence chunks, which get_evidence() merges into the documents list.
        answer_overhead = calculate_tokens_used(tokenizer, _build_answer_prompt(), input)

        message = await chain.ainvoke(input, config=config)

        return {
            "messages": [message],
            "tokens_used": answer_overhead
        }

    def extract_reasoning_and_answer(response: str) -> dict:
        reasoning_start = response.find("<reasoning>")
        # Find last occurrence of </reasoning>
        reasoning_end = response.rfind("</reasoning>")

        if reasoning_start != -1 and reasoning_end != -1:
            reasoning = response[reasoning_start + len("<reasoning>"):reasoning_end].strip()
            answer = response[reasoning_end + len("</reasoning>"):].strip()
        else:
            reasoning = ""
            answer = response.strip()
        return {
            "reasoning": reasoning,
            "answer": answer
        }

    async def answer_question(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        logger.info("Answering question", query=state["query"])
        manifest = runtime.context.manifest
        documents = get_evidence(state)
        documents = sort_chunks(documents, manifest)
        # dump_chunks(documents)

        answer_prompt = _build_answer_prompt()

        input = dict(
            # No need for a game summary here, we will have plenty of context
            # from the documents
            game_summary=False,
            game_name=manifest["name"],
            documents=documents,
            query=state["query"],
            clarifying_questions_and_answers=state.get("clarifying_questions", [])
        )

        chain = answer_prompt | chat_model
        chain = chain.with_config(run_name="qa_graph_answer_chain")
        result = await chain.ainvoke(input, config=config)
        fix_result = fix_quote_citations_in_text(result.text, documents, strip_invalid_blockquotes=True)
        extracted = extract_reasoning_and_answer(fix_result.fixed_text)
        referenced = fix_result.referenced_chunks
        logger.info("Answer question result", answer_chars=len(extracted["answer"] or ""), document_count=len(documents), valid_quotes=len(fix_result.valid_quotes), invalid_quotes=len(fix_result.unfixable_quotes), referenced_chunks=len(referenced))
        logger.debug("Answer question text", answer=extracted["answer"])
        return {
            "answer": extracted["answer"],
            "reasoning": extracted["reasoning"],
            # Deliberately not "evidence": narrowing to the cited chunks here would
            # take the rest of the corpus away from format_answer and the validation
            # loop, which still have to verify and repair quotes against it. If this
            # answer's quotes were all hallucinated, `referenced` is empty, and
            # overwriting `evidence` with it used to leave validation with nothing to
            # match against, so the quote could never be fixed or dropped.
            "referenced_evidence": referenced,
        }
    
    def get_quotes(fix_quote_result: FixQuoteCitationsResult) -> Tuple[list[QuoteEntry], list[QuoteEntry]]:
        valid_quotes: list[QuoteEntry] = []
        invalid_quotes: list[QuoteEntry] = []

        for vq in fix_quote_result.valid_quotes:
            valid_quotes.append(
                QuoteEntry(
                    text=vq["quote"],
                    rulebook_name=vq["citation"]["ref_name"] if vq["citation"] else "",
                    page=vq["citation"]["page"] if vq["citation"] else ""
                )
            )

        for iq in fix_quote_result.unfixable_quotes:
            citation = iq["citation"]
            if not citation:
                citation = {"ref_name": "", "page": ""}
            invalid_quotes.append(
                QuoteEntry(
                    text=iq["quote"],
                    rulebook_name=citation["ref_name"],
                    page=citation["page"]
                )
            )

        return valid_quotes, invalid_quotes
        

    async def format_answer(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        format_attempts = state.get("format_attempts", 0)

        # If we've exhausted all attempts, return an error response
        if format_attempts >= 5:
            logger.error("format_answer failed after 5 attempts, returning error response")
            return {
                "response": "I was unable to generate a response with valid quotes. Please try again.",
                # The canned response quotes nothing, so nothing is referenced. Clear
                # the narrowed set rather than `evidence`, which the validation loop
                # still needs, and which provide_response falls back to.
                "referenced_evidence": [],
                "invalid_quotes": state.get("invalid_quotes", []),
                "format_attempts": format_attempts,
            }

        documents = get_evidence(state)
        documents = sort_chunks(documents, runtime.context.manifest)
        manifest = runtime.context.manifest

        # TODO: The prompt might now be too long to fit into the context window,
        # this is because we budgeted for retrieval without considering the size
        # of the answer + reasoning (since it hadn't been generated yet)
        format_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", format_answer_system_prompt_template),
                ("user", "Consider the user's query. Provide a step-by-step reasoning process concerning the user's query, then answer the user's query."),
                ("assistant", "{{{answer}}}"),
                ("user", markdown_format_response_template),
            ],
            template_format="mustache"
        )

        input = dict(
            game_summary=False,
            game_name=manifest["name"],
            documents=[],
            query=state["query"],
            reasoning=state["reasoning"],
            answer=state["answer"],
        )

        chain = format_prompt | chat_model
        chain = chain.with_config(run_name="format_answer_chain")
        result = await chain.ainvoke(input, config=config)
        extracted = extract_reasoning_and_answer(result.text)

        return {
            "response": extracted["answer"],
        }


    async def provide_response(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        # Validation is finished by the time we get here, so this is where the
        # corpus collapses to just the chunks the response actually quotes. Fall
        # back to the full set when nothing was cited (an answer with no quotes, or
        # one whose quotes never verified), returning no evidence at all would
        # leave the caller unable to show any source.
        evidence = state.get("referenced_evidence") or state.get("evidence", [])
        if not evidence:
            evidence = get_evidence(state)

        return {
            "response": state["response"],
            "evidence": evidence,
            "valid": len(state.get("invalid_quotes", [])) == 0,
        }

    def check_validation_result(state: GameAgentOverallState) -> Literal["format_answer", "provide_response"]:
        invalid_quotes = state.get("invalid_quotes", [])
        validation_attempts = state.get("validation_attempts", 0)
        if invalid_quotes:
            documents = get_evidence(state)
            logger.info("Response contains invalid quotes", invalid_quote_count=len(invalid_quotes), validation_attempts=validation_attempts, query=state["query"], document_count=len(documents))
            # This node re-runs up to 5 times per request, so the evidence set and the
            # full response are DEBUG-only — at INFO they dominate the log volume.
            logger.debug("Invalid quote context", invalid_quotes=invalid_quotes, documents=documents, response=state["response"])
        if invalid_quotes and validation_attempts < 5:
            return "format_answer"
        return "provide_response"
    
    def select_start_node(state: GameAgentInputState) -> Literal["answer_question", "retrieve_data"]:
        # If provided evidence up-front, skip retrieval
        if state.get("evidence"):
            logger.info("Starting with provided evidence, skipping data retrieval")
            return "answer_question"

        logger.info("Starting with data retrieval")
        return "retrieve_data"

    graph = StateGraph(
        state_schema=GameAgentOverallState,
        input_schema=GameAgentInputState,
        output_schema=GameAgentOutputState,
        context_schema=GameAgentContext,
    )

    graph.add_node("retrieve_data", retrieve_data)
    graph.add_node("dedupe_chunks", dedupe_chunks_node)
    graph.add_node("answer_question", answer_question)
    # graph.add_node("validate_answer", validate_answer)
    graph.add_node("format_answer", format_answer)
    graph.add_node("validate_and_fix_response", validate_and_fix_response)
    graph.add_node("provide_response", provide_response)
    graph.add_node("tool_node", tool_node)

    graph.add_conditional_edges(START, select_start_node)
    graph.add_edge("retrieve_data", "tool_node")
    graph.add_edge("tool_node", "dedupe_chunks")
    graph.add_edge("dedupe_chunks", "answer_question")
    graph.add_edge("answer_question", "format_answer")
    graph.add_edge("format_answer", "validate_and_fix_response")
    
    # graph.add_edge("answer_question", "validate_answer")
    graph.add_conditional_edges("validate_and_fix_response", check_validation_result)
    graph.add_edge("provide_response", END)

    # Compile the agent
    agent = graph.compile(checkpointer=checkpoint_saver)
    return agent


class QAServiceInput(GameAgentInputState):
    manifest: Manifest


QAService = Runnable[QAServiceInput, GameAgentOutputState]


def build_qa_service(
    checkpoint_saver: Optional[BaseCheckpointSaver],
    chat_model: BaseChatModel,
    full_page_store: BaseStore[str, Serializable],
    chunk_search_service: ChunkSearchService,
    qa_prompt: ChatPromptTemplate=qa_prompt,
    tokenizer: Any = None,
) -> QAService:

    analyze_graph = build_analyze_question_graph(
        checkpoint_saver,
        chat_model,
        tokenizer
    )

    qa_graph = build_question_answer_graph(
        checkpoint_saver,
        chat_model,
        tokenizer
    )

    coord_graph = build_coordinating_agent_graph(
        checkpoint_saver,
        analyze_question_agent=analyze_graph,
        game_agent=qa_graph,
    )

    # Create a custom Runnable that properly handles both streaming and non-streaming
    from langchain_core.runnables import Runnable

    @chain
    async def qa_service(input: QAServiceInput, config: RunnableConfig | None = None, **kwargs) -> GameAgentOutputState:
        context = GameAgentContext(
            manifest=input["manifest"],
            full_page_store=full_page_store,
            chunk_search_service=chunk_search_service,
            chat_model=chat_model,
        )

        result = await coord_graph.ainvoke(
            cast(GameAgentInputState, {k: v for k, v in input.items() if k != "manifest"}),
            context=context,
            config=config,
            **kwargs
        )
        return cast(GameAgentOutputState, result)

    return qa_service
