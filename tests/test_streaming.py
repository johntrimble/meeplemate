"""
Test streaming functionality through the chatloop_service stack.

This test mocks all dependencies (LLM, retriever, stores) and verifies that:
1. Streaming works end-to-end from chatloop_service through to the LLM
2. The current implementation yields at least the final result
3. After refactoring, tokens stream incrementally
"""

import pytest
from typing import AsyncIterator, Any, Sequence, Callable
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.stores import InMemoryStore
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from meeplemate.chatloop import build_chatloop_service, ChatLoopServiceInput
from meeplemate.qa_graph import build_qa_service
from meeplemate.search import build_chunk_search_service, ChunkSearchService
from meeplemate.ingest.gamepackage import Manifest


def create_fake_chat_model(response_text: str = "The answer to your question is forty-two."):
    """
    Create a fake streaming chat model that returns the given response word-by-word.

    Uses LangChain's built-in FakeListChatModel as a base but overrides _astream
    to simulate token-by-token streaming.
    """
    class FakeStreamingChatModel(FakeListChatModel):
        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            """Override to stream word-by-word instead of all at once."""
            from langchain_core.outputs import ChatGenerationChunk

            # Get the response from the parent class
            response_index = self.i
            response = self.responses[response_index % len(self.responses)]
            self.i += 1

            # Split into words and stream them
            words = response.split()
            for i, word in enumerate(words):
                # Add space before word (except first one)
                content = word if i == 0 else f" {word}"
                chunk = AIMessageChunk(content=content)
                yield ChatGenerationChunk(message=chunk)

        def with_structured_output(self, schema, **kwargs):
            """Mock implementation of with_structured_output."""
            async def generate_structured(input_val):
                """Generate a simple structured response."""
                # For the RefinedQuery schema, just return the refined_query field
                if hasattr(schema, '__annotations__') and 'refined_query' in schema.__annotations__:
                    # Extract the user query from messages
                    if isinstance(input_val, list):
                        messages = input_val
                    elif hasattr(input_val, 'messages'):
                        messages = input_val.messages
                    elif hasattr(input_val, 'to_messages'):
                        messages = input_val.to_messages()
                    else:
                        messages = []

                    user_message = next((m for m in reversed(messages) if hasattr(m, 'content')), None)
                    query = user_message.content if user_message else "test query"

                    return {"refined_query": query}

                return {}

            return RunnableLambda(generate_structured)

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            """Mock implementation of bind_tools - just return self."""
            return self

    return FakeStreamingChatModel(responses=[response_text])


class FakeChunkSearchService:
    """Mock chunk search service that returns empty results."""

    async def ainvoke(self, input: dict, config: RunnableConfig | None = None) -> dict:
        """Return empty search results."""
        return {
            "chunks": [],
            "query": input.get("query", ""),
        }

    async def astream(self, input: dict, config: RunnableConfig | None = None) -> AsyncIterator[dict]:
        """Stream empty search results."""
        yield {
            "chunks": [],
            "query": input.get("query", ""),
        }


@pytest.fixture
def fake_manifest() -> Manifest:
    """Create a minimal fake game manifest."""
    return {
        "name": "Test Game",
        "summary": "A test game for unit testing",
    }


@pytest.fixture
def fake_stores():
    """Create fake in-memory stores."""
    full_page_store = InMemoryStore()
    game_data_store = InMemoryStore()
    return full_page_store, game_data_store


@pytest.fixture
def fake_chat_model():
    """Create a fake streaming chat model."""
    return create_fake_chat_model(
        response_text="The answer to your question is forty-two."
    )


@pytest.fixture
def fake_chunk_search_service() -> ChunkSearchService:
    """Create a fake chunk search service."""
    return FakeChunkSearchService()


@pytest.fixture
def checkpoint_saver():
    """Create an in-memory checkpoint saver."""
    return MemorySaver()


if __name__ == "__main__":
    # Allow running tests directly with: python tests/test_streaming.py
    import asyncio

    print("Running streaming tests...")
    print("=" * 60)

    # Create fixtures
    manifest = {
        "name": "Test Game",
        "summary": "A test game",
    }

    stores = (InMemoryStore(), InMemoryStore())
    chat_model = create_fake_chat_model()
    chunk_search = FakeChunkSearchService()
    checkpointer = MemorySaver()

    # Run tests
    asyncio.run(test_chatloop_service_streaming_current_implementation(
        chat_model, chunk_search, stores, manifest, checkpointer
    ))

    asyncio.run(test_chatloop_service_ainvoke_still_works(
        chat_model, chunk_search, stores, manifest, checkpointer
    ))

    asyncio.run(test_qa_service_streaming_directly(
        chat_model, chunk_search, stores, manifest, checkpointer
    ))

    print("\n" + "=" * 60)
    print("All current implementation tests passed!")
    print("\nNOTE: The refactored implementation test is skipped.")
    print("Unskip it after completing the async generator refactor.")
