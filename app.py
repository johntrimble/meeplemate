import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Sequence, cast
from urllib.parse import unquote

import chainlit as cl
import chainlit.socket as cl_socket
import chainlit.auth as cl_auth
from chainlit.user import PersistedUser

from chainlit.data.base import BaseDataLayer
from langchain_classic.schema import output
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import AnyMessage
from langchain_core.runnables import RunnableConfig

from meeplemate.chatloop import ChatLoopServiceInput
from meeplemate.component_system import StartedSystem, System, astart_system, astop_system
from meeplemate.config import AppServices, Config, GameManifest, GameRulesAgentState, create_app_system
from meeplemate.chainlit_utils import LangchainTracer
from chainlit.types import ThreadDict

from meeplemate.ingest.gamepackage import Manifest


@cl.password_auth_callback
async def auth(username: str, password: str) -> cl.User | None:
    if username == "dev" and password == "dev":
        # Returning a User marks the login as successful
        return cl.User(identifier="dev@local", display_name="Dev User")
    return None


# Populated by `async def startup()``
started_system: StartedSystem[AppServices] | None = None


def services() -> AppServices:
    global started_system
    assert started_system is not None
    return started_system.system_map


@cl.on_app_startup
async def startup():
    global started_system

    # Load configuration from YAML file (if MM_CONFIG_FILE is set), .env file, 
    # and environment variables
    # Priority: env vars > .env > YAML config > defaults
    settings: Config = Config()

    # Create the system
    system: System = create_app_system(settings)

    # Start the system and save a reference globally so that we can pull
    # services from it and stop it on shutdown
    started_system = await astart_system(system)


@cl.on_app_shutdown
async def shutdown():
    global started_system
    assert started_system is not None
    await astop_system(started_system)


@cl.data_layer
def get_data_layer() -> BaseDataLayer:
    dl = services()["data_layer"]
    assert dl is not None
    return dl


async def add_mock_user():
    # Create and set mock user
    mock_user = cl.User(
        identifier="mock_user_001",
        display_name="Mock Developer",
        metadata={"environment": "development"}
    )
    cl.context.session.user = mock_user


@cl.action_callback("game_select")
async def on_action(action: cl.Action):
    game_id = action.payload.get("game_id")
    assert isinstance(game_id, str), "game_id should be a string"
    set_current_game_id(game_id)
    game = await get_current_game()
    assert game is not None, "Selected game should exist"
    # Acknowledge the action
    await cl.Message(content=f"You've selection {game['name']}").send()


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    await cl.Message(content=f"You've changed settings {settings}").send()


async def get_all_games() -> Sequence[dict]:
    # Get all the games we support
    data_store = services()["game_data_store"]
    # pylance struggles with the types here
    keys_iter = cast(AsyncIterator[str], data_store.ayield_keys())
    game_ids = [id async for id in keys_iter]
    games = await data_store.amget(game_ids)
    games = cast(Sequence[dict], games)
    return games


def get_current_game_id() -> str | None:
    meta: dict = cast(dict, cl.user_session.get("thread_meta", {}))
    return meta.get("game_id")


def set_current_game_id(game_id: str):
    meta: dict = cast(dict, cl.user_session.get("thread_meta", {}))
    meta["game_id"] = game_id
    cl.user_session.set("thread_meta", meta)


async def maybe_set_thread_name(name: str):
    thread_id = cl.context.session.thread_id
    if not thread_id:
        return
    
    meta: dict = cast(dict, cl.user_session.get("thread_meta", {}))
    is_named = meta.get("is_named", False)
    
    # Only set the thread name once
    if is_named:
        return
    
    dl = get_data_layer()
    if dl:
        await dl.update_thread(
            thread_id=thread_id,
            name=name,
        )
        meta["is_named"] = True
        cl.user_session.set("thread_meta", meta)


async def get_current_game() -> Manifest | None:
    game_id = get_current_game_id()
    if game_id is None:
        return None
    data_store = services()["game_data_store"]
    game = await data_store.amget([game_id])
    if len(game) > 0:
        game = game[0]
    else:
        game = None

    return cast(Manifest, game) if game is not None else None


async def maybe_prompt_user_select_game():
    meta: dict = cast(dict, cl.user_session.get("thread_meta", {}))
    if "game_id" in meta:
        return False # game already selected

    actions = None

    if last := cl.user_session.get("select_game_message"):
        actions = last.actions
        await last.remove_actions()
        cl.user_session.set("select_game_message", None)

    if actions is None:
        # Prompt the user to select a game
        games = await get_all_games()
        # game_items = {game['name']: game["game_id"] for game in games if game is not None}
        actions = [
            cl.Action(
                name="game_select",
                icon="gamepad",
                payload={"game_id": game["game_id"]},
                label=game["name"]
            ) for game in games if game is not None
        ]

    # We use the "on_chat_start" step to prevent the frontend from showing the
    # feedback buttons for this message. We want to save the realestate for the
    # game selection buttons.
    async with cl.Step(name="on_chat_start", type="run"):
        msg = cl.Message(content="Please select one of the following games to continue:", actions=actions)
        await msg.send()
    cl.user_session.set("select_game_message", msg)
    return True

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # optional: restore any custom state; empty is fine
    pass

@cl.on_chat_start
async def start():
    await maybe_prompt_user_select_game()


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
    if await maybe_prompt_user_select_game():
        return

    game = await get_current_game()
    assert game is not None, "Game should be selected by this point"

    # await maybe_set_thread_name(game["name"])

    chatloop_service = services()["chatloop_service"]
    assert chatloop_service is not None

    thread_id = message.thread_id
    assert thread_id

    tracer = LangchainTracer(stream_final_answer=True)
    config: RunnableConfig = {
        "callbacks": [tracer],
    }

    input: ChatLoopServiceInput = ChatLoopServiceInput(
        messages=[HumanMessage(content=message.content)],
        manifest=game,
        thread_id=thread_id,
    )

    output = await chatloop_service.ainvoke(input=input, config=config)

    # Old graph code here for simple RAG
    # agent_graph = services()["agent_graph"]
    # assert agent_graph is not None

    # thread_id = message.thread_id
    # assert thread_id

    # tracer = LangchainTracer(stream_final_answer=True)
    # config: RunnableConfig = {
    #     "configurable": {"thread_id": thread_id},
    #     "callbacks": [tracer],
    # }

    # messages: list[AnyMessage] = [HumanMessage(content=message.content)]
    # input: GameRulesAgentState  = {"messages": messages, "manifest": game}
    # output = await agent_graph.ainvoke(input=input, config=config)

    # Send the final answer.
    await cl.Message(content=output["messages"][-1].text).send()
