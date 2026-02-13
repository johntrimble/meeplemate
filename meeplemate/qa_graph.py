import copy
from dataclasses import dataclass
import json
from re import sub
from typing import Annotated, Any, Literal, NotRequired, Sequence, Tuple, TypedDict, cast
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.documents import Document
from langchain.tools import ToolRuntime, tool
from langchain_core.tools import Tool
from langchain_core.language_models import BaseChatModel
from langchain_core.load import Serializable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable,  RunnableConfig, chain
from langchain_core.stores import BaseStore
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode

from meeplemate import quote_util
from meeplemate.ingest.gamepackage import Manifest, get_page_id
from meeplemate.search import ChunkSearchService, ChunkSearchServiceInput, CompiledStateGraph
from structlog import get_logger

from meeplemate.util import load_template, serialize_typeddict
logger = get_logger()

REFINEMENT_PARTITION_NUMBER = 4

system_prompt_template = load_template("system_prompt_rules_lawyer.md")
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
            ("user", "Consider the user's query. Provide a step-by-step reasoning process concerning the user's query inside <reasoning> </reasoning> tags. Then answer the user's query."),
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
    return results


class QuoteEntry(TypedDict):
    """A quote from a rulebook with its citation information"""
    text: Annotated[str, ..., "Verbatim quote from the rulebook"]
    rulebook_name: Annotated[str, ..., "Name of the rulebook this quote comes from"]
    page: Annotated[str, ..., "Page number where this quote appears"]


class DefinitionEntry(TypedDict):
    """A term and its definition status with supporting quotes"""
    term: Annotated[str, ..., "The term being defined"]
    quotes: Annotated[list[QuoteEntry], ..., "List of quotes that define or relate to this term"]
    defines_term: Annotated[bool, ..., "Whether the provided quotes contain a clear definition of the term"]
    clarifying_question: Annotated[str, ..., "If defines_term is false, a clarifying question to ask; otherwise, leave empty"]


# New nested types for exceptions
class ExplicitNamingCheck(TypedDict):
    """Step 2: Check whether the exception explicitly names the target mechanic"""
    does_exception_name_target: Annotated[bool, ..., "Whether the exception explicitly names the target mechanic"]
    explanation: Annotated[str, ..., "Brief explanation of the naming check result"]


class RelationshipCheck(TypedDict):
    """Step 3: Check for relationship statements linking the exception to the target mechanic"""
    relationship_exists: Annotated[bool | Literal["unclear"], ..., "Whether a relationship between mechanics exists (true/false/'unclear')"]
    quotes: Annotated[list[QuoteEntry], ..., "Quotes showing the relationship between mechanics"]
    explanation: Annotated[str, ..., "Explanation of the relationship or lack thereof"]


class SeparationCheck(TypedDict):
    """Step 4: Check for separation statements that prevent the exception from applying"""
    separation_exists: Annotated[bool, ..., "Whether the mechanics are explicitly separated in the rules"]
    quotes: Annotated[list[QuoteEntry], ..., "Quotes showing separation between mechanics"]
    explanation: Annotated[str, ..., "Explanation of the separation or lack thereof"]


# New types for top-level fields
class IdentifiedMechanics(TypedDict):
    """Game mechanics"""
    primary_mechanics: Annotated[list[str], ..., "The primary game mechanics involved in the question"]
    secondary_mechanics: Annotated[list[str], ..., "Other mechanics mentioned or implied that might affect the primary mechanics"]
    reasoning: Annotated[str, ..., "Brief explanation of why these mechanics were identified and how they relate to each other"]


class RelationshipStatement(TypedDict):
    """A statement about how two mechanics relate to each other"""
    mechanics: Annotated[list[str], ..., "List of two mechanics whose relationship is being described"]
    relationship_type: Annotated[Literal["separate", "same", "subset", "other"], ..., "Type of relationship: 'separate', 'same', 'subset', or 'other'"]
    quotes: Annotated[list[QuoteEntry], ..., "Quotes establishing the relationship between these mechanics"]
    interpretation: Annotated[str, ..., "What this relationship means for answering the user's query"]


class GeneralRule(TypedDict):
    """A general rule governing a game mechanic"""
    mechanic: Annotated[str, ..., "The game mechanic this rule governs"]
    quotes: Annotated[list[QuoteEntry], ..., "Quotes stating the general rule"]
    summary: Annotated[str, ..., "Brief summary of what the rule states"]


class ExceptionEntry(TypedDict):
    """An exception that might override general rules, analyzed using the 4-step test"""
    exception_source: Annotated[str, ..., "Where the exception comes from (card name, ability name, etc.)"]
    exception_scope_language: Annotated[str, ..., "Exact language describing what the exception affects"]
    target_mechanic: Annotated[str, ..., "The mechanic in the user's query being tested against this exception"]
    step1_scope_analysis: Annotated[str, ..., "Analysis of what language the exception uses to describe its scope"]
    step2_explicit_naming: ExplicitNamingCheck
    step3_relationship_check: RelationshipCheck
    step4_separation_check: SeparationCheck
    does_exception_apply: Annotated[bool | Literal["clarification_needed"], ..., "Whether this exception applies to the target mechanic (true/false/'clarification_needed')"]
    precedence_level: Annotated[str, ..., "Precedence level from rule #10: level 1-5"]
    clarifying_question: Annotated[str, ..., "If clarification is needed, a question to ask; otherwise, leave empty"]


class QaResponse(TypedDict):
    """Rules analysis and answer structure"""
    reasoning: Annotated[str, ..., "Step-by-step reasoning process using bullet points"]
    identified_mechanics: IdentifiedMechanics
    # relationship_statements: Annotated[list[RelationshipStatement], ..., "List of relationship statements between mechanics found in the documents"]
    general_rules: Annotated[list[GeneralRule], ..., "List of general rules governing the mechanics in question"]
    definitions: Annotated[list[DefinitionEntry], ..., "List of term definitions found in or missing from the documents. Do not include definitions for things already defined under general_rules."]
    exceptions: Annotated[list[ExceptionEntry], ..., "List of exceptions that might apply to the situation"]
    precedence_analysis: Annotated[str, ..., "If multiple rules apply, explanation of which takes precedence and why (using rule #10)"]
    final_answer: Annotated[str, ..., "Free-form markdown text following all citation requirements. Must follow rule #1 for document-first, quote-first answering, use blockquotes instead of inline quotes, and include citations in the form (Rulebook name, p. X). Do NOT refer to rule interpretation criteria names (e.g. 'Rule #10') in the final answer."]
    sufficient_information_to_answer: Annotated[bool, ..., "Whether there is sufficient information in the documents to answer the query"]


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
    original_text = text

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
            citation_key = (citation["ref_name"], citation["page"])
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
                        break  # Stop after first matching chunk to avoid stale-index rewrites

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
        r'\n\s*\n\s*(\([^)]+,?\s*pg?[.]\s*[0-9]+\))[^\S\n]*(?=\n|$)',
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

    if original_text != text:
        logger.info("Fixed quote citations in text", text=original_text, fixed_text=text)

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
                page=_quote["citation"]["page"] if _quote["citation"] else ""
            )
        )

    # Check the final answer for invalid quotes
    invalid_final_answer_quotes: list[QuoteEntry] = []
    for extracted_quote in result.unfixable_quotes:
        citation = extracted_quote["citation"]
        if not citation:
            citation = {"ref_name": "", "page": ""}
        quote_entry: QuoteEntry = {
            "text": extracted_quote["quote"],
            "rulebook_name": citation["ref_name"],
            "page": citation["page"]
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

    # Validate relationship_statements
    invalid_relationship_quotes: list[QuoteEntry] = []
    if "relationship_statements" in response:
        for statement in response["relationship_statements"]:
            invalid_relationship_quotes.extend(check_quotes_in_list(statement["quotes"]))

    # Validate general_rules
    invalid_general_rule_quotes: list[QuoteEntry] = []
    for rule in response["general_rules"]:
        invalid_general_rule_quotes.extend(check_quotes_in_list(rule["quotes"]))

    # Validate exceptions (new nested structure)
    invalid_exception_quotes: list[QuoteEntry] = []
    for exception in response["exceptions"]:
        # step3_relationship_check contains quotes
        invalid_exception_quotes.extend(
            check_quotes_in_list(exception["step3_relationship_check"]["quotes"])
        )
        # step4_separation_check contains quotes
        invalid_exception_quotes.extend(
            check_quotes_in_list(exception["step4_separation_check"]["quotes"])
        )

    return TweakAndValidateQuotesResult(
        revised_response=response,
        invalid_quotes=(
            invalid_definition_quotes +
            invalid_relationship_quotes +
            invalid_general_rule_quotes +
            invalid_exception_quotes +
            invalid_final_answer_quotes
        ),
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


class GameAgentOutputState(MessagesState):
    response: str
    """The answer to the user's query"""
    evidence: list[Chunk]
    """The evidence chunks supporting the answer"""
    valid: bool


class GameAgentOverallState(GameAgentInputState, GameAgentOutputState):
    analysis: QaResponse
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
    subquestion: Annotated[str, ..., "A self-contained subquestion"]
    explanation: Annotated[str, ..., "An explanation as to how an answer to this subquestion helps address the user's original query."]
    

class QuestionAnalysis(TypedDict):
    explanation: Annotated[str, ..., "Explanation of what the user asking and the key rules and rule interactions involved in the user's query. Free form markdown text."]
    subquestions: Annotated[list[Subquestion], ..., "List of 1-3 subquestions and explanations of their relevance"]
    classification: Literal["SIMPLE", "COMPLEX"]


def build_analyze_question_graph(
    checkpoint_saver: BaseCheckpointSaver,
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

        return {
            "analysis": question_analysis["explanation"],
            "subquestions": [sub["subquestion"] for sub in question_analysis["subquestions"]],
            "classification": question_analysis["classification"],
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
            query = tool_call["args"]["query"]
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
        tool_calls[first_index]["args"]["query"] = queries

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
    checkpoint_saver: BaseCheckpointSaver,
    analyze_question_agent: CompiledStateGraph[QuestionAnalysisOverallState, QuestionAnalysisContext, QuestionAnalysisOverallState, QuestionAnalysisOverallState],
    game_agent: CompiledStateGraph[GameAgentOverallState, GameAgentContext, GameAgentInputState, GameAgentOutputState],
) -> CompiledStateGraph[CoordinationOverallState, GameAgentContext, CoordinationInputState, CoordinationOutputState]:
    
    async def analyze_question(state: CoordinationInputState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
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
        input = {
            "query": state["query"],
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


    async def ask_subquestions(state: AskSubquestionsInputState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        subquestions = state["subquestions"]

        inputs = []
        for question in subquestions:
            logger.info("Asking subquestion", question=question)
            input = {
                "query": question,
            }
            inputs.append(input)
        
        responses: list[GameAgentOutputState] = await game_agent.abatch(
            inputs,
            context=runtime.context,
            config=config
        )

        subquestions_answers: list[ClarifyingQA] = []
        for question, response in zip(subquestions, responses):
            if not response["valid"]:
                logger.error("Subquestion answer was not valid", question=question, answer=response["response"])
                continue

            subquestions_answers.append(
                ClarifyingQA(
                    question=question,
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


def build_question_answer_graph(
    checkpoint_saver: BaseCheckpointSaver,
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
        extracted = extract_reasoning_and_answer(result.text)
    
        logger.info("Answer question result", answer=extracted["answer"], document_count=len(documents))
        print("Reasoning:\n", extracted["reasoning"])
        print("Answer:\n", extracted["answer"])
        return {
            "answer": extracted["answer"],
            "reasoning": extracted["reasoning"],
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

    async def validate_answer(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        documents = get_evidence(state)
        reasoning = state["reasoning"]
        answer = state["answer"]

        # validate all citations in reasoning and fix if possible
        reasoning_result = fix_quote_citations_in_text(reasoning, documents)
        reasoning_valid_quotes, reasoning_invalid_quotes = get_quotes(reasoning_result)

        # validate all citations in answer and fix if possible
        answer_result = fix_quote_citations_in_text(answer, documents)
        answer_valid_quotes, answer_invalid_quotes = get_quotes(answer_result)

        if len(reasoning_invalid_quotes) > 0 or len(answer_invalid_quotes) > 0:
            if state.get("validation_attempts", 0) < 4:
                logger.warning(
                    "Invalid quotes found, will attempt to re-answer the question",
                    invalid_quotes=reasoning_invalid_quotes + answer_invalid_quotes,
                    attempt=state.get("validation_attempts", 0) + 1
                )
            else:
                logger.error(
                    "Invalid quotes found but maximum validation attempts reached, returning error response",
                    invalid_quotes=reasoning_invalid_quotes + answer_invalid_quotes,
                    attempt=state.get("validation_attempts", 0),
                    documents=documents,
                )
            logger.info(
                "Invalid quotes found during validation",
                invalid_quotes=reasoning_invalid_quotes + answer_invalid_quotes,
                attempt=state.get("validation_attempts", 0)
            )
            return {
                "validation_attempts": state.get("validation_attempts", 0) + 1,
                "invalid_quotes": reasoning_invalid_quotes + answer_invalid_quotes,
            }
        
        evidence = compile_evidence_from_documents(
            reasoning_valid_quotes + answer_valid_quotes,
            documents
        )

        return {
            "validation_attempts": 0,
            "invalid_quotes": [],
            "answer": answer_result.fixed_text,
            "reasoning": reasoning_result.fixed_text,
            # "evidence": evidence,
        }

    async def format_answer(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        format_attempts = state.get("format_attempts", 0)

        # If we've exhausted all attempts, return an error response
        if format_attempts >= 5:
            logger.error("format_answer failed after 5 attempts, returning error response")
            return {
                "response": "I was unable to generate a response with valid quotes. Please try again.",
                "evidence": [],
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
                ("system", system_prompt_template),
                ("user", "Consider the user's query. Provide a step-by-step reasoning process concerning the user's query inside <reasoning> </reasoning> tags. Then answer the user's query."),
                ("assistant", "<reasoning>{{reasoning}}</reasoning>\n{{answer}}"),
                ("user", markdown_format_response_template),
            ],
            template_format="mustache"
        )

        input = dict(
            game_summary="",
            game_name=manifest["name"],
            documents=documents,
            query=state["query"],
            reasoning=state["reasoning"],
            answer=state["answer"],
        )

        chain = format_prompt | chat_model
        chain = chain.with_config(run_name="format_answer_chain")
        result = await chain.ainvoke(input, config=config)

        # Validate quotes in the formatted response
        fix_result = fix_quote_citations_in_text(result.text, documents)
        valid_quotes, invalid_quotes = get_quotes(fix_result)

        if invalid_quotes:
            for iq in invalid_quotes:
                logger.warning(
                    "Invalid quote in formatted answer",
                    answer=result.text,
                    text=iq["text"],
                    rulebook_name=iq["rulebook_name"],
                    page=iq["page"],
                    attempt=format_attempts + 1,
                    documents=documents,
                )
            return {
                "response": fix_result.fixed_text,
                "invalid_quotes": invalid_quotes,
                "format_attempts": format_attempts + 1,
            }

        # All quotes valid — build evidence from the quotes in the response
        evidence = compile_evidence_from_documents(valid_quotes, documents)

        return {
            "response": fix_result.fixed_text,
            "evidence": list(evidence),
            "invalid_quotes": [],
            "format_attempts": 0,
        }

    async def provide_response(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
        evidence = state.get("evidence", [])
        if not evidence:
            evidence = get_evidence(state)

        return {
            "response": state["response"],
            "evidence": evidence,
            "valid": len(state.get("invalid_quotes", [])) == 0,
        }

    async def check_validation_result(state: GameAgentOverallState) -> Literal["format_answer", "answer_question"]:
        if state.get("invalid_quotes", []) and state.get("validation_attempts", 0) < 5:
            return "answer_question"
        return "format_answer"

    def check_format_result(state: GameAgentOverallState) -> Literal["format_answer", "provide_response"]:
        invalid_quotes = state.get("invalid_quotes", [])
        format_attempts = state.get("format_attempts", 0)
        if invalid_quotes and 0 < format_attempts < 5:
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
    graph.add_node("validate_answer", validate_answer)
    graph.add_node("format_answer", format_answer)
    graph.add_node("provide_response", provide_response)
    graph.add_node("tool_node", tool_node)

    graph.add_conditional_edges(START, select_start_node)
    graph.add_edge("retrieve_data", "tool_node")
    graph.add_edge("tool_node", "dedupe_chunks")
    graph.add_edge("dedupe_chunks", "answer_question")
    graph.add_edge("answer_question", "format_answer")
    # graph.add_edge("answer_question", "validate_answer")
    # graph.add_conditional_edges("validate_answer", check_validation_result)
    graph.add_conditional_edges("format_answer", check_format_result)
    graph.add_edge("provide_response", END)

    # Compile the agent
    agent = graph.compile(checkpointer=checkpoint_saver)
    return agent
    

class QAServiceInput(GameAgentInputState):
    manifest: Manifest


QAService = Runnable[QAServiceInput, GameAgentOutputState]


def build_qa_service(
    checkpoint_saver: BaseCheckpointSaver,
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
        )

        result = await coord_graph.ainvoke(
            cast(GameAgentInputState, {k: v for k, v in input.items() if k != "manifest"}),
            context=context,
            config=config,
            **kwargs
        )
        return cast(GameAgentOutputState, result)

    return qa_service
