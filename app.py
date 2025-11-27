import chainlit as cl
from chainlit.data.base import BaseDataLayer
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import AnyMessage
from langchain_core.runnables import RunnableConfig

from meeplemate.config import Config, GameRulesAgentState, Services

settings: Config = {
    "db": {
        "dc": "datacenter1",
        "contact_points": ["cassandra"],
        "replication_factor": 1,
        "chainlit_keyspace": "chainlit_meeplemate",
        "langgraph_keyspace": "meeplemate_checkpoints",
        "create_keyspaces": True,
    },
    "data_api": {
        "token": "Cassandra:Cg==:Cg==",
        "endpoint": "http://data-api:8181",
        "namespace": "meeplemate",
    },
    "chat": {
        "model_name": "Qwen/Qwen3-32B",
        "endpoint_type": "openai",
        "endpoint": "http://vllm:8000/v1",
        "explicit_disable_thinking": True,
        "max_new_tokens": 1024,
        "timeout": 900,
        "api_key": "your_openai_api_key_here",
    },
    "embedding": {
        "model": "jinaai/jina-embeddings-v2-base-en",
        "endpoint": "http://tei:80/v1",
        "api_key": "dummy",
    },
    "rules_path": "./data/rules/munchkin_rules/",
    "load_docs": False,
    "model_name": "Qwen/Qwen3-32B",
    "qa_chain_config": {
        "reword_documents": True,
        "self_consistency": True,
        "thread_of_thought": True,
        "consistency_kwargs": {
            "samples": 3,
        }
    }
}

services: Services = Services(settings)


@cl.on_app_startup
async def startup():
    global services
    await services.start()


@cl.on_app_shutdown
async def shutdown():
    global services
    await services.stop()


@cl.data_layer
def get_data_layer() -> BaseDataLayer:
    assert services.data_layer is not None
    return services.data_layer


async def add_mock_user():
    # Create and set mock user
    mock_user = cl.User(
        identifier="mock_user_001",
        display_name="Mock Developer",
        metadata={"environment": "development"}
    )
    cl.context.session.user = mock_user


@cl.on_chat_start
async def start():
    await add_mock_user()
    game_options = ["Munchkin", "Boss Monster", "Secret Hitler"]
    from chainlit.input_widget import Select
    await cl.ChatSettings(
        inputs=[Select(id="game_select", label="Choose a game", values=game_options, initial_index=0)]
    ).send()
    await cl.Message(content="Welcome to Meeplmate! Pick a game in ⚙️ Chat Settings. The default is Munchkin.").send()


@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns: 
        None.
    """
    agent_graph = services.agent_graph

    assert agent_graph is not None

    thread_id = message.thread_id
    assert thread_id
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    messages: list[AnyMessage] = [HumanMessage(content=message.content)]
    input: GameRulesAgentState  = {"messages": messages}
    output = await agent_graph.ainvoke(input=input, config=config)

    # Send the final answer.
    await cl.Message(content=output["messages"][-1].content).send()
