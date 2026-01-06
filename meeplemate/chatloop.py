from dataclasses import dataclass
import operator
from typing import Annotated, TypedDict, cast
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda, chain, patch_config
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph

from meeplemate.ingest.gamepackage import Manifest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.runtime import Runtime

from meeplemate.qa_graph import GameAgentContext, GameAgentInputState, GameAgentOutputState, GameAgentOverallState, QAService, QAServiceInput


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


class ChatLoopInputState(MessagesState):
    pass


class ChatLoopOutputState(MessagesState):
    refined_queries: Annotated[dict[str, str], operator.or_]


class ChatLoopState(ChatLoopInputState, ChatLoopOutputState):
    pass


class RefinedQuery(TypedDict):
    refined_query: str


def build_chatloop_graph(checkpoint_saver: BaseCheckpointSaver, chat_model: BaseChatModel, qa_service: QAService, refine_prompt: ChatPromptTemplate=REFINE_QUESTION_PROMPT):
    
    async def refine_query(state: ChatLoopState, *, runtime: Runtime[ChatLoopContext]) -> dict:
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
        return {
            "refined_queries": {current_message.id: result["refined_query"]}
        }

    async def respond_to_query(state: ChatLoopState, *, runtime: Runtime[ChatLoopContext]) -> dict:
        last_id = state["messages"][-1].id
        if last_id is None:
            raise ValueError("Last message must have an ID to look up refined query.")
        refined_query = state["refined_queries"][last_id]

        input: QAServiceInput  = {
            "query": refined_query,
            "manifest": runtime.context.manifest,
        }
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

    # Add edges
    builder.add_edge(START, "refine_query")
    builder.add_edge("refine_query", "respond_to_query")
    builder.add_edge("respond_to_query", END)

    graph = builder.compile(checkpointer=checkpoint_saver)
    return graph


class ChatLoopServiceInput(MessagesState):
    manifest: Manifest
    thread_id: str


ChatLoopService = Runnable[ChatLoopServiceInput, ChatLoopOutputState]

def build_chatloop_service(checkpoint_saver: BaseCheckpointSaver, chat_model: BaseChatModel, qa_service: QAService, refine_prompt: ChatPromptTemplate=REFINE_QUESTION_PROMPT) -> ChatLoopService:
    agent_graph = build_chatloop_graph(
        checkpoint_saver,
        chat_model,
        qa_service,
        refine_prompt=refine_prompt,
    )

    async def chain_func(input: ChatLoopServiceInput, *, config: RunnableConfig|None = None) -> ChatLoopOutputState:
        thread_config = {"thread_id": input["thread_id"]}
        config = patch_config(config, configurable=thread_config)
        context = ChatLoopContext(
            manifest=input["manifest"],
        )
        graph_input: ChatLoopInputState = {
            "messages": input["messages"],
        }
        output = await agent_graph.ainvoke(input=graph_input, context=context, config=config)
        return cast(ChatLoopOutputState, output)

    return RunnableLambda(chain_func, name="ChatLoopService")