from dataclasses import dataclass
import operator
from typing import Annotated, Any, AsyncIterator, Protocol, TypedDict, cast, runtime_checkable
from backoff import runtime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, RemoveMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, chain, patch_config
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from meeplemate.ingest.gamepackage import Manifest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.runtime import Runtime

from meeplemate.qa_graph import GameAgentContext, GameAgentInputState, GameAgentOutputState, GameAgentOverallState, QAService, QAServiceInput
from meeplemate.stream_events import RefinedUserQueryEvent, StepEvent


system_prompt_template = """\
You are an expert on the board game {{game_name}}. Your task is to assist users by providing clear and accurate explanations of the game's rules. Use the following game rules summary to better understand the game and questions users may have:

<game_summary>
{{game_summary}}
</game_summary>
"""

refine_query_template = """\
Analyze the user query below. Determine what the user is specifically asking about based on the conversation history, game summary, and rules already retrieved. Produce a refined query that is clear and specific, focusing on the exact information needed to answer the user's question. The refined query should be fully self contained, not relying on the past conversation history to be understood. If the user query is already specific enough and self-contained, return it as is.

## User Query

<query>
{{query}}
</query>
"""

REFINE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_template),
        ("placeholder", "{messages}"),
        ("user", refine_query_template),
    ],
    template_format="mustache"
)


@dataclass
class ChatLoopContext:
    manifest: Manifest
    tokenizer: Any


class ChatLoopInputState(MessagesState):
    pass


class ChatLoopOutputState(MessagesState):
    refined_queries: Annotated[dict[str, str], operator.or_]


class ChatLoopState(ChatLoopInputState, ChatLoopOutputState):
    pass


class RefinedQuery(TypedDict):
    refined_query: str


def build_chatloop_graph(checkpoint_saver: BaseCheckpointSaver, chat_model: BaseChatModel, qa_service: QAService, refine_prompt: ChatPromptTemplate=REFINE_QUESTION_PROMPT) -> CompiledStateGraph[ChatLoopState, ChatLoopContext, ChatLoopInputState, ChatLoopOutputState]:

    async def compress_messages(state: ChatLoopState, *, runtime: Runtime[ChatLoopContext]) -> dict:
        new_messages = trim_messages(
            state["messages"],
            strategy="last",
            allow_partial=False,
            include_system=True,
            start_on="human",
            token_counter=count_tokens_approximately,
            max_tokens=10_000,
        )

        # If we didn't make any changes, return an empty dict to avoid
        # unnecessary state updates and graph executions
        if len(new_messages) == len(state["messages"]):
            return {}

        output = {}

        # Delete any old messages not in new_messages
        old_message_ids = {msg.id for msg in state["messages"]}
        new_message_ids = {msg.id for msg in new_messages}
        removed_message_ids = old_message_ids - new_message_ids
        removed_messages = [RemoveMessage(id=msg_id) for msg_id in removed_message_ids if msg_id]
        output["messages"] = removed_messages

        # Get rid of old refined queries for removed messages
        refined_queries = state["refined_queries"]
        for msg_id in removed_message_ids:
            if msg_id in refined_queries:
                refined_queries.pop(msg_id, None)
                if "refined_queries" not in output:
                    output["refined_queries"] = refined_queries

        return output
    
    async def refine_query(state: ChatLoopState, *, runtime: Runtime[ChatLoopContext]) -> dict:
        writer = get_stream_writer()
        writer(StepEvent(type="mm_step", description="Refining query"))
        manifest = runtime.context.manifest

        current_message = state["messages"][-1]
        previous_messages = state["messages"][:-1]

        chain = (refine_prompt | chat_model.with_structured_output(RefinedQuery))
        input = {
            "game_name": manifest["name"],
            "game_summary": manifest.get("summary", "No summary available."),
            "messages": previous_messages,
            "query": current_message.content,
        }
        result: RefinedQuery = cast(RefinedQuery, await chain.ainvoke(input=input))
        writer(RefinedUserQueryEvent(type="mm_refined_user_query", description=result["refined_query"]))
        return {
            "refined_queries": {current_message.id: result["refined_query"]}
        }

    async def respond_to_query(state: ChatLoopState, *, runtime: Runtime[ChatLoopContext]) -> dict:
        last_id = state["messages"][-1].id
        if last_id is None:
            raise ValueError("Last message must have an ID to look up refined query.")
        refined_query = state["refined_queries"][last_id]

        input: QAServiceInput  = {
            "messages": [],
            "recursion_depth": 0,
            "evidence": [],
            "query": refined_query,
            "manifest": runtime.context.manifest,
        }

        # For streaming to work through nested services, we need to use ainvoke here
        # The streaming happens at a lower level (in the QA service's astream method)
        # When the outer chatloop service calls astream with stream_mode="messages",
        # it will capture the message chunks from the inner QA graph
        result: GameAgentOutputState = await qa_service.ainvoke(input=input)
        response_text = result["response"]
        response = AIMessage(content=response_text)

        return {
            "messages": [response]
        }

    builder = StateGraph(
        ChatLoopState,
        context_schema=ChatLoopContext,
        input_schema=ChatLoopInputState,
        output_schema=ChatLoopOutputState,
    )

    # Add nodes
    builder.add_node("refine_query", refine_query)
    builder.add_node("respond_to_query", respond_to_query)
    builder.add_node("compress_messages", compress_messages)

    # Add edges
    builder.add_edge(START, "refine_query")
    builder.add_edge("refine_query", "compress_messages")
    builder.add_edge("compress_messages", "respond_to_query")
    builder.add_edge("respond_to_query", END)

    graph = builder.compile(checkpointer=checkpoint_saver)
    return graph


class ChatLoopServiceInput(MessagesState):
    manifest: Manifest
    thread_id: str


@runtime_checkable
class ChatLoopService(Protocol):
    def astream_response(self, input: ChatLoopServiceInput, config: RunnableConfig | None = None) -> Any: # AsyncIterator[AIMessageChunk]:
        """Asynchronously streams response chunks based on the input.

        Args:
            input: The input data for the chat loop service.
            config: Optional configuration for the runnable.

        Yields:
            Chunks of the response as they are generated.
        """
        ...

    def astream(self, input: ChatLoopServiceInput, *args, config: RunnableConfig | None = None, **kwargs) -> AsyncIterator[dict[str, Any] | Any]:
        ...


def build_chatloop_service(checkpoint_saver: BaseCheckpointSaver, chat_model: BaseChatModel, tokenizer: Any, qa_service: QAService, refine_prompt: ChatPromptTemplate=REFINE_QUESTION_PROMPT) -> ChatLoopService:
    agent_graph = build_chatloop_graph(
        checkpoint_saver,
        chat_model,
        qa_service,
        refine_prompt=refine_prompt,
    )

    class _ChatLoopService(ChatLoopService):
        def __init__(self, graph: CompiledStateGraph[ChatLoopState, ChatLoopContext, ChatLoopInputState, ChatLoopOutputState]):
            self.graph = graph

        async def astream(self, input: ChatLoopServiceInput, *args, config: RunnableConfig | None = None, **kwargs) -> AsyncIterator[dict[str, Any] | Any]:
            thread_config = {"thread_id": input["thread_id"]}
            config = patch_config(config, configurable=thread_config)
            context = ChatLoopContext(manifest=input["manifest"], tokenizer=tokenizer)
            graph_input: ChatLoopInputState = {"messages": input["messages"]}

            async for value in self.graph.astream(
                graph_input,
                *args,
                context=context,
                config=config,
                **kwargs
            ):
                yield value


        async def astream_response(self, input: ChatLoopServiceInput, config: RunnableConfig | None = None) -> AsyncIterator[AIMessageChunk]:
            thread_config = {"thread_id": input["thread_id"]}
            config = patch_config(config, configurable=thread_config)
            context = ChatLoopContext(manifest=input["manifest"], tokenizer=tokenizer)
            graph_input: ChatLoopInputState = {"messages": input["messages"]}

            result = await self.graph.ainvoke(
                graph_input,
                context=context,
                config=config
            )

            yield AIMessageChunk(content=result["messages"][-1].content)

            # # It's actually the QA service that produces the streaming messages,
            # # so we need to filter the graph output to yield only those
            # # messages here.
            # async for item in self.graph.astream(
            #     graph_input,
            #     context=context,
            #     stream_mode="messages",
            #     subgraphs=True,
            #     config=config
            # ):
            #     namespace, (message, metadata) = item
            #     # Check the namespace
            #     if len(namespace) != 1 or not namespace[0].startswith("respond_to_query:"):
            #         continue

            #     # Ignore tool calls
            #     if getattr(message, "tool_calls", None) or getattr(message, "invalid_tool_calls", None):
            #         continue

            #     # Ignore tool messages
            #     if isinstance(message, ToolMessage):
            #         continue

            #     # Only include messages from the llm_call node
            #     if not metadata.get("langgraph_node") == "llm_call":
            #         continue

            #     # Skip empty messages
            #     if not message.content:
            #         continue

            #     if isinstance(message, AIMessageChunk):
            #         yield message

    return _ChatLoopService(agent_graph)