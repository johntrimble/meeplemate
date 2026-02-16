from dataclasses import dataclass
from operator import itemgetter
from typing import Annotated, Any, Literal, NotRequired, Sequence, TypedDict, List, cast
from langchain.messages import AIMessage
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_classic.schema.runnable import ConfigurableField
from langchain_classic.schema.vectorstore import VectorStoreRetriever
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever, RetrieverInput, RetrieverOutput
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, RunnablePassthrough, RunnableSerializable, chain
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.runnables.config import patch_config

from meeplemate.ingest.gamepackage import Manifest
from meeplemate.util import select_keys, slugify
from structlog import get_logger

logger = get_logger(__name__)

# Per-chunk token overhead to account for metadata (rulebook_name, page, start_index,
# end_index, relevance_reason) and dict structure when chunks are serialized into tool
# messages. Without this, the token budget only counts page_content, causing the actual
# prompt to exceed the context window.
PER_CHUNK_OVERHEAD = 75

system_prompt_template = """\
You are an expert Rules Lawyer specializing in boardgame rules. Being "technically correct" is your highest aspiration. You believe in "the rules as written" above all else, because the rules are not merely words on a page, they are devine truth. You are sensitive to even the slimmest nuances in wording, and you always interpret the rules in the most literal way possible. You never make assumptions or inferences beyond what is explicitly written in the rules, because that would be the greatest of heresis.

The game summary is non-authoritative context. Use it only to orient yourself; do not use it as evidence for a rules conclusion.

<game_summary>
{{game_summary}}
</game_summary>
"""


select_relevant_results_template = """\
Given the following chunks taken from the game rules, determine which chunks are relevant to answering the user query and why or why not. A chunk is considered relevant if it pertains to any aspect of the user query, either directly answering it or providing necessary context to understand the answer.

## Chunks

<chunks>
{{#chunks}}
<chunk id="{{id}}" name="{{metadata.rulebook_name}}" page="{{metadata.page_num}}" offset="{{metadata.start_index}}">
{{page_content}}
</chunk>
{{/chunks}}
</chunks>

## User Query

<query>
{{query}}
</query>

## Instructions

- Carefully read the user query and each chunk, leveraging the game summary to understand the context.
- Determine the relevance of each chunk to the user query.
- Explain your reasoning for each chunk's relevance or irrelevance in detail.
- Each explanation should consider:
  - Whether the chunk directly addresses the topic of the user query.
  - If the chunk provides necessary background information that aids in understanding the answer.
  - The specificity of the chunk in relation to the user's needs.

## Response Format

Provide your response in JSON format as follows (excluding backticks):

```json
{
    "chunks": [
        {
            "id": "<chunk_id>",
            "reasoning": "<detailed_explanation>",
            "is_relevant": true/false
        },
        // ... repeat for each chunk ...
    ]
}
```
"""


resolve_response_errors_template = """\
The response you provided has some errors. Return a corrected version of the response with the below errors fixed.

## Errors

<errors>
{{#errors}}
<error>
{{.}}
</error>
{{/errors}}
</errors>

## Response Format

Provide your response in JSON format as follows (excluding backticks):

```json
{
    "chunks": [
        {
            "id": "<chunk_id>",
            "reasoning": "<detailed_explanation>",
            "is_relevant": true/false
        },
        // ... repeat for each chunk ...
    ]
}
```
"""


SELECT_RELEVANT_RESULTS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("user", select_relevant_results_template),
    ],
    template_format="mustache"
)


SELECT_RELEVANT_RESULTS_ERRORS_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("user", select_relevant_results_template),
        ("placeholder", "{ai_response}"),
        ("user", resolve_response_errors_template),
    ],
    template_format="mustache"
)


class ChunkRelevanceResult(TypedDict):
    id: Annotated[str, ..., "The unique identifier of the chunk."]
    reasoning: Annotated[str, ..., "A detailed explanation of why the chunk is relevant or not relevant to the user query."]
    is_relevant: Annotated[bool, ..., "Indicates whether the chunk is relevant (true) or not relevant (false) to the user query."]


class ChunksRelevanceResults(TypedDict):
    chunks: Annotated[list[ChunkRelevanceResult], ..., "A list of chunk relevance results."]


class ChunkSearchOverallState(TypedDict):
    game_id: str
    game_version: str
    game_name: str
    game_summary: str
    chunks: list[Document]
    relevance: ChunksRelevanceResults
    errors: list[str]
    attempts: int
    max_attempts: int
    query: str
    user_main_query: str
    ai_response: AIMessage


class ChunkSearchInputState(TypedDict):
    game_id: str
    game_version: str
    game_name: str
    game_summary: str
    query: str
    user_main_query: str


class ChunkSearchOutputState(TypedDict):
    chunks: list[Document]
    relevance: ChunksRelevanceResults
    tokens_used: int


def add_chunk_ids(documents: list[Document]) -> list[Document]:
    for doc in documents:
        chunk_id = get_chunk_id(doc)
        doc.id = chunk_id
    return documents


def get_chunk_id(doc: Document | dict) -> str:
    if isinstance(doc, Document):
        metadata = doc.metadata
    else:
        metadata = doc.get("metadata", {})
    rulebook_name = metadata.get("rulebook_name", "unknown_rulebook")
    page_number = metadata.get("page_ordinal", "unknown_page")
    start_index = metadata.get("start_index", "unknown_start_index")
    chunk_id = f"{slugify(rulebook_name)}::{page_number}::{start_index}"
    return chunk_id


def prepare_documents_for_prompt(documents: list[Document]) -> list[dict]:
    prepared_docs = []
    for i, doc in enumerate(documents):
        data = doc.model_dump()
        prepared_docs.append(data)
    return prepared_docs


def validate_relevance_results(documents: list[Document], relevance_results: ChunksRelevanceResults) -> Sequence[str]:
    errors: list[str] = []

    # Do we have exactly one result per chunk?
    for document in documents:
        chunk_id = document.id
        matching_results = [result for result in relevance_results["chunks"] if result["id"] == chunk_id]
        if len(matching_results) == 0:
            errors.append(f"Expected exactly one result per chunk. No relevance result found for chunk ID: {chunk_id}.")
        if len(matching_results) > 1:
            errors.append(f"Expected exactly one result per chunk. Multiple relevance results found for chunk ID: {chunk_id}.")

    # Ensure we don't have any extra results
    for result in relevance_results["chunks"]:
        chunk_id = result["id"]
        matching_documents = [doc for doc in documents if doc.id == chunk_id]
        if len(matching_documents) == 0:
            errors.append(f"Relevance result found for unknown chunk ID: {chunk_id}.")
    return errors    


def build_chunk_search_graph(
    checkpoint_saver: BaseCheckpointSaver,
    chat_model: BaseChatModel,
    retriever: MultiVectorRetriever|VectorStoreRetriever,
    select_chunks_prompt: ChatPromptTemplate=SELECT_RELEVANT_RESULTS_PROMPT,
    select_chunks_errors_prompt: ChatPromptTemplate=SELECT_RELEVANT_RESULTS_ERRORS_PROMPT,
    max_attempts=5
) -> CompiledStateGraph[ChunkSearchOverallState, None, ChunkSearchInputState, ChunkSearchOutputState]:

    # Make the search_kwargs configurable so that we can set the game_id in the
    # filter expression
    search_kwargs = dict(retriever.search_kwargs)
    dynamic_retriever = retriever.configurable_fields(
        search_kwargs=ConfigurableField(id="retriever_search_kwargs")
    )


    class RetrieveChunksInput(TypedDict):
        query: str
        game_id: str
        game_version: str
    

    class RetrieveChunksOutput(TypedDict):
        chunks: list[Document]


    async def retrieve_chunks(state: RetrieveChunksInput, config: RunnableConfig) -> RetrieveChunksOutput:
        # Setup config to filter by game_id
        config = patch_config(
            config,
            configurable={
                "retriever_search_kwargs": {
                    **search_kwargs,
                    "filter": {"game_id": state["game_id"], "game_version": state["game_version"]},
                },
            }
        )

        chain = ( dynamic_retriever | add_chunk_ids )
        document_chunks = await chain.ainvoke(state["query"], config=config)
        return {"chunks": document_chunks}
    

    class DetermineChunkRelevanceInput(TypedDict):
        query: str
        game_name: str
        game_summary: str
        chunks: list[Document]


    class DetermineChunkRelevanceOutput(TypedDict):
        ai_response: AIMessage
        relevance: NotRequired[ChunksRelevanceResults]
        errors: list[str]


    async def determine_chunk_relevance(state: DetermineChunkRelevanceInput) -> DetermineChunkRelevanceOutput:
        _chat_model_structured_output = chat_model.with_structured_output(ChunksRelevanceResults, method="json_schema", include_raw=True).bind(streaming=False)
        chain = (
            RunnablePassthrough.assign(
                chunks=RunnableLambda(itemgetter("chunks")) | prepare_documents_for_prompt,
            )
            | select_chunks_prompt
            | _chat_model_structured_output
        ).with_config(run_name="determine_chunk_relevance")

        # Execute the chain
        input = select_keys(state, DetermineChunkRelevanceInput)
        result = await chain.ainvoke(input)
        
        # Pick apart the result
        assert isinstance(result, dict), "Expected result to be a dict since `include_raw=True` used."
        base_message = result["raw"]
        parsed: ChunksRelevanceResults | None = result["parsed"] if "parsed" in result else None
        error = result["parsing_error"] if "parsing_error" in result else None

        # Build output
        output: DetermineChunkRelevanceOutput = {"errors": [], "ai_response": base_message}
        if error:
            output["errors"].append(str(error))
        
        if parsed:
            output["relevance"] = parsed
            # Validate the results to make sure the LLM actually analyzed
            # every chunk
            output["errors"].extend(validate_relevance_results(state["chunks"], parsed))

        return output


    class ResolveErrorsInput(TypedDict):
        query: str
        game_name: str
        game_summary: str
        chunks: list[Document]
        errors: list[str]
        attempts: NotRequired[int]
        ai_response: AIMessage


    class ResolveErrorsOutput(TypedDict):
        relevance: NotRequired[ChunksRelevanceResults]
        errors: list[str]
        attempts: int


    async def resolve_errors(state: ResolveErrorsInput) -> ResolveErrorsOutput:
        assert state["errors"], "Expected errors to be present in the state and not empty."
        _chat_model_structured_output = chat_model.with_structured_output(ChunksRelevanceResults, method="json_schema", include_raw=True).bind(streaming=False)

        # Build the input
        input = {
            "query": state["query"],
            "game_name": state["game_name"],
            "game_summary": state["game_summary"],
            "chunks": state["chunks"],
            "errors": state["errors"],
            "ai_response": [state["ai_response"]],
        }

        chain = (
            RunnablePassthrough.assign(
                chunks=RunnableLambda(itemgetter("chunks")) | prepare_documents_for_prompt,
            )
            | select_chunks_errors_prompt
            | _chat_model_structured_output
        )

        # Execute the chain
        result = await chain.ainvoke(input)
        
        # Pick apart the result
        assert isinstance(result, dict), "Expected result to be a dict since `include_raw=True` used."
        parsed: ChunksRelevanceResults | None = result["parsed"] if "parsed" in result else None
        error = result["parsing_error"] if "parsing_error" in result else None

        # Build output
        output: ResolveErrorsOutput = {"errors": [], "attempts": state.get("attempts", 0) + 1}
        if error:
            output["errors"].append(str(error))
        
        if parsed:
            output["relevance"] = parsed
            # Validate the results to make sure the LLM actually analyzed
            # every chunk
            output["errors"].extend(validate_relevance_results(state["chunks"], parsed))

        return output


    class SkipSelectionInput(TypedDict):
        chunks: list[Document]


    class SkipSelectionOutput(TypedDict):
        relevance: ChunksRelevanceResults


    async def skip_selection(state: SkipSelectionInput) -> SkipSelectionOutput:
        # For some reason we couldn't get the LLM to analyze the chunks properly.
        # Just assume everything is relevant.
        relevance_results: ChunksRelevanceResults = {
            "chunks": [
                ChunkRelevanceResult(
                    id=get_chunk_id(doc),
                    reasoning="Assumed relevant since chunk relevance selection was skipped.",
                    is_relevant=True,
                )
                for doc in state["chunks"]
            ]
        }
        
        return {"relevance": relevance_results}


    class CheckResponseErrorsInput(TypedDict):
        attempts: NotRequired[int]
        errors: NotRequired[list[str]]


    async def check_response_errors(state: CheckResponseErrorsInput) -> Literal["resolve_errors", "skip_selection", "__end__"]:
        if "errors" in state and state["errors"]:
            if state.get("attempts", 0) < max_attempts:
                return "resolve_errors"
            else:
                return "skip_selection"
        return "__end__"


    builder = StateGraph(
        ChunkSearchOverallState,
        input_schema=ChunkSearchInputState,
        output_schema=ChunkSearchOutputState
    )

    # Add nodes
    builder.add_node("retrieve", retrieve_chunks)
    builder.add_node("determine_relevance", determine_chunk_relevance)
    builder.add_node("resolve_errors", resolve_errors)
    builder.add_node("skip_selection", skip_selection)

    # Add edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "determine_relevance")
    builder.add_conditional_edges("determine_relevance", check_response_errors)
    builder.add_conditional_edges("resolve_errors", check_response_errors)
    builder.add_edge("skip_selection", END)

    graph = builder.compile(checkpointer=checkpoint_saver)
    return graph 


class ChunkSearchServiceInput(TypedDict):
    manifest: Manifest
    query: str|list[str]
    user_main_query: NotRequired[str]
    token_budget: NotRequired[int]


ChunkSearchService = Runnable[ChunkSearchServiceInput, ChunkSearchOutputState]


def build_chunk_search_service(
    checkpoint_saver: BaseCheckpointSaver,
    chat_model: BaseChatModel,
    retriever: MultiVectorRetriever|VectorStoreRetriever,
    max_attempts=5
) -> Runnable[ChunkSearchServiceInput, ChunkSearchOutputState]:
    graph = build_chunk_search_graph(
        checkpoint_saver=checkpoint_saver,
        chat_model=chat_model,
        retriever=retriever,
        max_attempts=max_attempts
    )

    @chain
    async def chain_func(input: ChunkSearchServiceInput, config: RunnableConfig|None = None) -> ChunkSearchOutputState:
        if isinstance(input["query"], list):
            # Raise a value error
            raise ValueError("Expected query to be a string, but got a list. Please provide a single query string.")
    
        graph_input: ChunkSearchInputState = {
            "game_id": input["manifest"]["game_id"],
            "game_version": input["manifest"].get("game_version", ""),
            "game_name": input["manifest"]["name"],
            "game_summary": input["manifest"].get("summary", ""),
            "query": input["query"],
            "user_main_query": input.get("user_main_query") or input["query"],
        }
        output = await graph.ainvoke(input=graph_input, config=config)
        
        return cast(ChunkSearchOutputState, output)

    return chain_func


def find_cutoff_adaptive_k(scores, post_k_buffer=5, find_gap_within_top_percent=0.9):
    """
    From "Efficient Context Selection for Long-Context QA: No Tuning, No
    Iteration, Just Adaptive-k" by Taguchi et al. 2025
    https://arxiv.org/abs/2506.08479
    """
    if not scores:
        return -1
    
    # Assert scores are sorted in descending order
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), "Scores must be sorted in descending order"

    # Compute score deltas. We use these to find the biggest drop-off point
    score_deltas = [scores[i] - scores[i+1] for i in range(len(scores)-1)]

    # To guard against k landing amongst low relevance docs, only consider the
    # top percentage of scores
    cutoff_index = int(len(scores) * find_gap_within_top_percent)
    relevant_deltas = score_deltas[:cutoff_index]

    # Find the index of the maximum delta
    max_delta_index = relevant_deltas.index(max(relevant_deltas))

    # Now add the post_k_buffer to ensure we don't cut off too early
    adaptive_k = max_delta_index + post_k_buffer + 1

    return adaptive_k


def build_chunk_search_service_2(
    vectorstore: VectorStore,
    docstore: BaseStore[str, Document],
    tokenizer: Any,
    default_token_budget: int = 13000,
):
    @chain
    def chain_func(input: ChunkSearchServiceInput) -> ChunkSearchOutputState:
        budget = input.get("token_budget", default_token_budget)

        query = input["query"]
        if isinstance(query, str):
            query = [query]
        
        # Assert we have a game version
        assert "game_version" in input["manifest"], "Game version is required in the manifest"
        
        manifest = input["manifest"]
        game_id = manifest["game_id"]
        game_version = manifest["game_version"]

        filter = {"game_id": game_id, "game_version": game_version}
        logger.info("Vectorstore search", game_id=game_id, game_version=game_version, filter=filter, query_count=len(query))

        retrieved_results = []

        for q in query:
            # Step 1: Retrieve a large set of potentially relevant chunks using
            # the vectorstore
            docs_and_scores = vectorstore.similarity_search_with_relevance_scores(
                q,
                filter=filter,
                k=50
            )
            logger.info("Vectorstore returned", query=q, result_count=len(docs_and_scores))

            # Step 2: Use the adaptive k algorithm to find the cutoff point in
            # the retrieved results
            scores = [score for _, score in docs_and_scores]
            docs = [doc for doc, _ in docs_and_scores]
            adaptive_k = find_cutoff_adaptive_k(scores)
            selected_docs = docs[:adaptive_k]
            logger.info("Adaptive k selection", query=q, raw_count=len(docs), adaptive_k=adaptive_k, selected_count=len(selected_docs))

            # Step 3: Add the selected chunks to the overall results
            retrieved_results.extend(selected_docs)

        # Step 4: Get parent chunks
        parent_doc_ids = []
        for doc in retrieved_results:
            parent_doc_ids.append(doc.metadata.get("doc_id"))
        # Make parent_doc_ids unique keeping first occurrence
        seen = set()
        unique_parent_doc_ids = []
        for doc_id in parent_doc_ids:
            if doc_id not in seen:
                unique_parent_doc_ids.append(doc_id)
                seen.add(doc_id)
        parent_docs = docstore.mget(unique_parent_doc_ids)
        # Log warning for missing docs
        for doc_id, doc in zip(unique_parent_doc_ids, parent_docs):
            if doc is None:
                logger.warning(f"Parent document with ID not found in docstore.", doc_id=doc_id)
        parent_docs = [doc for doc in parent_docs if doc is not None]

        # Step 5: Enforce the token budget
        docs_in_budget = []
        remaining_budget = budget
        budget_used = 0
        for doc in parent_docs:
            doc_tokens = len(tokenizer.encode(doc.page_content)) + PER_CHUNK_OVERHEAD

            if doc_tokens <= remaining_budget:
                docs_in_budget.append(doc)
                remaining_budget -= doc_tokens
                budget_used += doc_tokens
            else:
                # We've hit the token budget, so we stop adding more documents
                break
        
        # Step 6: Sort the final documents by rulebook order
        rulebook_priority = {
            rulebook["name"]: i
            for i, rulebook in enumerate(manifest["rulebooks"])
        }
        docs_in_budget.sort(
            key=lambda doc: (
                rulebook_priority.get(doc.metadata.get("rulebook_name", ""), float("inf")),
                doc.metadata.get("page_ordinal", float("inf")),
                doc.metadata.get("start_index", float("inf"))
            )
        )

        # Step 7: Add chunk IDs to the documents
        docs_in_budget = add_chunk_ids(docs_in_budget)

        # Step 8: Create result object
        relevance_results: ChunksRelevanceResults = {
            "chunks": [
                ChunkRelevanceResult(
                    id=get_chunk_id(doc),
                    reasoning="Selected based on vector similarity and adaptive k cutoff.",
                    is_relevant=True,
                )
                for doc in docs_in_budget
            ]
        }

        logger.info("Used tokens", query=input["query"], total_retrieved_count=len(docs_in_budget), tokens_used=budget_used)

        output: ChunkSearchOutputState = {
            "chunks": docs_in_budget,
            "relevance": relevance_results,
            "tokens_used": budget_used,
        }
        return output

    return chain_func
