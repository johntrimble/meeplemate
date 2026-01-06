from dataclasses import dataclass
from typing import Literal, Sequence, TypedDict, cast
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

from meeplemate.ingest.gamepackage import Manifest, get_page_id
from meeplemate.search import ChunkSearchInputState, ChunkSearchOutputState, ChunkSearchOverallState, ChunkSearchService, ChunkSearchServiceInput, CompiledStateGraph
from structlog import get_logger
logger = get_logger()


system_prompt_template = """\
You are an expert on the board game {{game_name}}. Your task is to assist users by providing clear and accurate explanations of the game's rules. Use the following game rules summary to better understand the game and questions users may have:

<game_summary>
{{game_summary}}
</game_summary>
"""

qa_template = """\
Read the user query below carefully. Your goal is to provide a detailed and accurate answer based on the official rules of the game {{game_name}}.

## User Query

<query>
{{query}}
</query>

## Instructions

- Analyze the user query step by step, breakind down the components of the question.
- Gather relevant rules and information by using the provided tools: `list_rulebooks`, `retrieve_page`, and `search_chunks`.
- Base your responses on the official rulebooks or authoritative sources, recognizing that these rules hold in all standard situations unless an explicit exception is stated. Avoid assumptions and unofficial variations unless specifically requested by the user.
- Consider the gameplay context in your interpretations, including game phases, player counts, and specific scenarios that might impact rule application.
- Highlight rule variants and exceptions clearly, explaining how they alter standard gameplay and under what circumstances they apply.
- Attempt to understand user intent, focusing on the aspect of the rule they might find confusing or the specific information they seek.

## Response Format

Respond using markdown. Start with a step-by-step reasoning process, using bullet points for each step. Always quote the relevant rule or section from the rulebook that supports your reasoning. After your reasoning, provide a clear and concise answer to the user's questons
"""

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

    document_id = get_page_id(manifest["game_id"], document_key, page)
    page_documents: list[Document] = cast(list[Document], await full_page_store.amget([document_id]))
    if len(page_documents) == 0:
        raise ValueError(f"No page found for {rulebook_name} {page}")
    
    page_document: Document = page_documents[0]
    return Page(
        rulebook_name=rulebook_name,
        page=page,
        content=page_document.page_content
    )


@tool(description="""Search for chunks, blocks of text from rulebook pages, relevant to the given query string. Search is performed using a semantic vector search, and then further refined by an LLM-judge to determine which chunks should be returned. Every returned chunk will have an associated "relevance_reason" indicating why that chunk was relevant to the provided query. The chunks will also indicate the rulebook they are from and which page. This information can then be used with `retrieve_page` to retrieve the entire page if desired.""")
async def search_chunks(query: str, runtime: ToolRuntime[GameAgentContext]) -> Sequence[ChunkSearchResult]:
    logger.info("search_chunks called", query=query)
    manifest = runtime.context.manifest
    chunk_search_service = runtime.context.chunk_search_service

    input = ChunkSearchServiceInput(
        manifest=manifest,
        query=query,
        user_main_query=query
    )

    chunk_search_result = await chunk_search_service.ainvoke(input, config=runtime.config)

    results: list[ChunkSearchResult] = []
    for relevance_result in chunk_search_result["relevance"]["chunks"]:
        reasoning = relevance_result["reasoning"]
        chunk_id = relevance_result["id"]
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


class GameAgentInputState(TypedDict):
    query: str


class GameAgentOutputState(TypedDict):
    response: str


class GameAgentOverallState(MessagesState):
    query: str
    response: str


def build_game_agent_graph(checkpoint_saver: BaseCheckpointSaver, chat_model: BaseChatModel, qa_prompt: ChatPromptTemplate=qa_prompt) -> CompiledStateGraph[GameAgentOverallState, GameAgentContext, GameAgentInputState, GameAgentOutputState]:
    tools = [list_rulebooks, retrieve_page, search_chunks]
    tool_node = ToolNode(tools)

    async def llm_call(state: GameAgentOverallState, *, runtime: Runtime[GameAgentContext], config: RunnableConfig|None = None) -> dict:
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

        logger.info("Inoking LLM with input", message_count=len(state["messages"]))

        message = await chain.ainvoke(input, config=config)

        return {
            "messages": [message]
        }
    
    async def should_continue(state: GameAgentOverallState) -> Literal["tool_node", "response"]:
        """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, then perform an action
        if getattr(last_message, "tool_calls", None):
            return "tool_node"

        # Otherwise, we stop (reply to the user)
        return "response"
    
    async def populate_response(state: GameAgentOverallState) -> dict:
        """Extract the final response from the messages"""
        messages = state["messages"]
        last_message = messages[-1]
        return {
            "response": last_message.content
        }
    
    # Build workflow
    agent_builder = StateGraph(
        GameAgentOverallState,
        context_schema=GameAgentContext,
        input_schema=GameAgentInputState,
        output_schema=GameAgentOutputState
    )

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("response", populate_response)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue
    )
    agent_builder.add_edge("tool_node", "llm_call")
    agent_builder.add_edge("response", END)

    # Compile the agent
    agent = agent_builder.compile(checkpointer=checkpoint_saver)

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
) -> QAService:
    agent_graph = build_game_agent_graph(
        checkpoint_saver,
        chat_model,
        qa_prompt
    )

    @chain
    async def chain_func(input: QAServiceInput, config: RunnableConfig|None = None) -> GameAgentOutputState:
        context = GameAgentContext(
            manifest=input["manifest"],
            full_page_store=full_page_store,
            chunk_search_service=chunk_search_service,
        )
        agent_result = await agent_graph.ainvoke(
            {
                "query": input["query"]
            },
            context=context,
            config=config
        )
        return cast(GameAgentOutputState, agent_result)
    
    return chain_func
