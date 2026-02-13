from typing import AsyncIterator, Sequence, cast

import chainlit as cl

from chainlit.data.base import BaseDataLayer
from langchain_core.callbacks import Callbacks
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from meeplemate.chatloop import ChatLoopServiceInput
from meeplemate.component_system import StartedSystem, System, astart_system, astop_system
from meeplemate.config import AppServices, Config, create_app_system
from chainlit.types import ThreadDict

from chainlit.langchain.callbacks import LangchainTracer

from meeplemate.ingest.gamepackage import Manifest, get_game_key_for_id_version
from meeplemate.util import aenumerate


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


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    await cl.Message(content=f"You've changed settings {settings}").send()


@cl.action_callback("game_select")
async def on_action(action: cl.Action):
    game_id = action.payload.get("game_id")
    game_version = action.payload.get("game_version", "")
    assert isinstance(game_id, str), "game_id should be a string"
    assert isinstance(game_version, str), "game_version should be a string"
    set_current_game_id_version(game_id, game_version)
    game = await get_current_game()
    assert game is not None, "Selected game should exist"
    # Acknowledge the action
    await cl.Message(content=f"Your selection {game['name']}").send()
    await maybe_set_thread_name(game["name"])


async def get_all_games() -> Sequence[dict]:
    # Get the current versions of all games we support
    version_store = services()["game_version_store"]
    game_keys_iter = cast(AsyncIterator[str], version_store.ayield_keys())
    game_ids = [id async for id in game_keys_iter]
    game_current_keys = await version_store.amget(game_ids)

    # Get the game data for the current version of each game
    game_data_store = services()["game_data_store"]
    games = await game_data_store.amget(game_current_keys)
    games = cast(Sequence[dict], games)
    return games


def get_current_game_id_version() -> tuple[str | None, str | None]:
    meta: dict = cast(dict, cl.user_session.get("thread_meta", {}))
    return meta.get("game_id"), meta.get("game_version")


def set_current_game_id_version(game_id: str, game_version: str):
    meta: dict = cast(dict, cl.user_session.get("thread_meta", {}))
    meta["game_id"] = game_id
    meta["game_version"] = game_version
    cl.user_session.set("thread_meta", meta)


async def get_current_version_for_game(game_id: str) -> str:
    version_store = services()["game_version_store"]
    results = await version_store.amget([game_id])
    assert len(results) == 1 and results[0] is not None, "No version found for game_id"
    version = results[0]
    return str(version)


async def get_game(game_id: str, game_version: str|None = None) -> Manifest | None:
    if game_id is None:
        return None

    # Use the latest version if not provided
    if not game_version:
        game_version = await get_current_version_for_game(game_id)
    
    # Construct the game key
    game_key = get_game_key_for_id_version(game_id, game_version)

    data_store = services()["game_data_store"]

    # Fetch the game manifest
    manifest = data_store.mget([game_key])[0]

    # If the manifest not found, fallback to latest version
    if manifest is None:
        game_version = await get_current_version_for_game(game_id)
        game_key = get_game_key_for_id_version(game_id, game_version)
        manifest = data_store.mget([game_key])[0]

    # Return the manifest if found
    return cast(Manifest, manifest) if manifest is not None else None


async def get_current_game() -> Manifest | None:
    # Get the current game ID and version from the user session
    game_id, game_version = get_current_game_id_version()

    # Bail if no game is selected
    if game_id is None:
        return None
    
    # Fetch the game manifest
    manifest = await get_game(game_id, game_version)

    # If the version is different than requested, update the session
    if manifest is not None:
        actual_version = manifest.get("game_version", "")
        if game_version != actual_version:
            set_current_game_id_version(game_id, actual_version)
    
    return manifest


async def maybe_set_thread_name(name: str):
    thread_id = cl.context.session.thread_id
    if not thread_id:
        return
    
    meta: dict = cast(dict, cl.user_session.get("thread_meta", {}))
    is_named = meta.get("is_named", False)
    
    # Only set the thread name once
    if is_named:
        return
    
    print(f"Setting thread name to {name} for thread {thread_id}")
    await cl.context.emitter.init_thread(name)

    meta["is_named"] = True
    cl.user_session.set("thread_meta", meta)


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
                payload={"game_id": game["game_id"], "game_version": game.get("game_version", "")},
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
    # Limit message size to prevent overly large requests
    MAX_MESSAGE_LENGTH = 5_000
    if len(message.content) > MAX_MESSAGE_LENGTH:
        await cl.Message(
            content=f"⚠️ Message too long! Please keep your message under {MAX_MESSAGE_LENGTH} characters. (Current: {len(message.content)} characters)"
        ).send()
        return

    if await maybe_prompt_user_select_game():
        return

    game = await get_current_game()
    assert game is not None, "Game should be selected by this point"

    chatloop_service = services()["chatloop_service"]
    assert chatloop_service is not None

    thread_id = message.thread_id
    assert thread_id

    # Keep LangchainTracer for step visibility (non-streaming)
    callbacks: Callbacks = []
    callbacks.append(LangchainTracer())

    config: RunnableConfig = {
        "callbacks": callbacks,
    }

    input: ChatLoopServiceInput = ChatLoopServiceInput(
        messages=[HumanMessage(content=message.content)],
        manifest=game,
        thread_id=thread_id,
    )

    # Create Chainlit message for streaming
    streaming_msg = cl.Message(content="")

    async for index, chunk in aenumerate(chatloop_service.astream_response(input=input, config=config)):
        # Extract token content from the chunk
        # The service already filters for final answer only
        if hasattr(chunk, 'content') and chunk.content:
            await streaming_msg.stream_token(chunk.content)
    await streaming_msg.send()
