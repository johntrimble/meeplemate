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

from langchain_core.messages import AIMessage, HumanMessage, AIMessageChunk, ToolCall
from langchain_core.runnables import RunnableConfig
from langchain_core.stores import InMemoryStore
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from meeplemate.chatloop import build_chatloop_service, ChatLoopServiceInput
from meeplemate.qa_graph import build_qa_service
from meeplemate.search import build_chunk_search_service, ChunkSearchService
from meeplemate.ingest.gamepackage import Manifest


class FakeStreamingChatModel(BaseChatModel):
    """
    A fake chat model that simulates token-by-token streaming.

    This is crucial for testing - it simulates what a real LLM does when streaming.
    """

    # Define as Pydantic fields since BaseChatModel is a Pydantic model
    response_text: str = "This is a test response."
    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake_streaming_chat_model"

    def _generate(self, *args, **kwargs):
        """Sync generation - not used in our async tests."""
        raise NotImplementedError("Use async methods")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        """Async generation without streaming."""
        self.call_count += 1

        # Simulate a response
        message = AIMessage(content=self.response_text)

        from langchain_core.outputs import ChatGeneration, ChatResult
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _astream(
        self,
        messages,
        stop=None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """
        Stream the response token-by-token.

        This simulates real LLM streaming behavior.
        """
        self.call_count += 1

        # Split response into "tokens" (words for simplicity)
        tokens = self.response_text.split()

        for i, token in enumerate(tokens):
            # Add space before token (except first one)
            content = token if i == 0 else f" {token}"

            # Yield token as AIMessageChunk
            yield AIMessageChunk(content=content)

    def with_structured_output(self, schema, **kwargs):
        """
        Mock implementation of with_structured_output.

        Returns a simple mock that generates structured output based on the schema.
        """
        from langchain_core.runnables import RunnableLambda

        async def generate_structured(input_val):
            """Generate a simple structured response."""
            # For the RefinedQuery schema, just return the refined_query field
            if hasattr(schema, '__annotations__') and 'refined_query' in schema.__annotations__:
                # Extract the user query from messages
                if isinstance(input_val, list):
                    messages = input_val
                elif hasattr(input_val, 'messages'):
                    # ChatPromptValue has a messages attribute
                    messages = input_val.messages
                elif hasattr(input_val, 'to_messages'):
                    messages = input_val.to_messages()
                else:
                    messages = []

                user_message = next((m for m in reversed(messages) if hasattr(m, 'content')), None)
                query = user_message.content if user_message else "test query"

                # Return structured output
                return {"refined_query": query}

            # Fallback: return empty dict
            return {}

        return RunnableLambda(generate_structured)

    def bind_tools(
        self,
        tools: Sequence[dict | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ):
        """
        Mock implementation of bind_tools.

        Returns self since we're just testing streaming, not tool calling.
        In our test, we'll make the fake model not make any tool calls.
        """
        # Return a version of this model that won't make tool calls
        # This simplifies testing - we just want to test streaming
        return self


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
def fake_chat_model() -> FakeStreamingChatModel:
    """Create a fake streaming chat model."""
    return FakeStreamingChatModel(
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


@pytest.mark.asyncio
async def test_chatloop_service_streaming_current_implementation(
    fake_chat_model,
    fake_chunk_search_service,
    fake_stores,
    fake_manifest,
    checkpoint_saver,
):
    """
    Test that astream() works on chatloop_service with CURRENT implementation.

    Current behavior: Should yield at least the final result (all at once).
    This test verifies the plumbing is correct before refactoring.
    """
    full_page_store, game_data_store = fake_stores

    # Build the QA service
    qa_service = build_qa_service(
        checkpoint_saver=checkpoint_saver,
        chat_model=fake_chat_model,
        full_page_store=full_page_store,
        chunk_search_service=fake_chunk_search_service,
    )

    # Build the chatloop service
    chatloop_service = build_chatloop_service(
        checkpoint_saver=checkpoint_saver,
        chat_model=fake_chat_model,
        qa_service=qa_service,
    )

    # Create input
    input_data: ChatLoopServiceInput = {
        "messages": [HumanMessage(content="What is the meaning of life?")],
        "manifest": fake_manifest,
        "thread_id": "test-thread-123",
    }

    # Collect streamed chunks
    chunks = []
    async for chunk in chatloop_service.astream(input=input_data, config=None):
        chunks.append(chunk)

    # Assertions for CURRENT implementation
    print(f"\n=== Current Implementation Test ===")
    print(f"Number of chunks yielded: {len(chunks)}")
    print(f"Chunks: {chunks}")

    # Current implementation should yield at least 1 chunk (the final result)
    assert len(chunks) >= 1, "Should yield at least the final result"

    # The last chunk should contain the final messages
    final_chunk = chunks[-1]
    assert "messages" in final_chunk, "Final chunk should contain messages"
    assert len(final_chunk["messages"]) > 0, "Should have at least one message"

    # The last message should be an AI message with our expected content
    last_message = final_chunk["messages"][-1]
    assert isinstance(last_message, AIMessage), "Last message should be AIMessage"
    assert "forty-two" in last_message.content.lower(), "Should contain expected response"

    print(f"✓ Current implementation yields final result correctly")
    print(f"✓ Final message content: {last_message.content}")


@pytest.mark.asyncio
async def test_chatloop_service_ainvoke_still_works(
    fake_chat_model,
    fake_chunk_search_service,
    fake_stores,
    fake_manifest,
    checkpoint_saver,
):
    """
    Test that ainvoke() still works (backwards compatibility).

    This ensures existing code using ainvoke() continues to work.
    """
    full_page_store, game_data_store = fake_stores

    # Build services
    qa_service = build_qa_service(
        checkpoint_saver=checkpoint_saver,
        chat_model=fake_chat_model,
        full_page_store=full_page_store,
        chunk_search_service=fake_chunk_search_service,
    )

    chatloop_service = build_chatloop_service(
        checkpoint_saver=checkpoint_saver,
        chat_model=fake_chat_model,
        qa_service=qa_service,
    )

    # Create input
    input_data: ChatLoopServiceInput = {
        "messages": [HumanMessage(content="What is the answer?")],
        "manifest": fake_manifest,
        "thread_id": "test-thread-456",
    }

    # Invoke (not stream)
    result = await chatloop_service.ainvoke(input=input_data, config=None)

    # Assertions
    print(f"\n=== ainvoke() Backwards Compatibility Test ===")
    assert "messages" in result, "Result should contain messages"
    assert len(result["messages"]) > 0, "Should have at least one message"

    last_message = result["messages"][-1]
    assert isinstance(last_message, AIMessage), "Last message should be AIMessage"
    assert "forty-two" in last_message.content.lower(), "Should contain expected response"

    print(f"✓ ainvoke() works correctly")
    print(f"✓ Result: {last_message.content}")


@pytest.mark.asyncio
@pytest.mark.skip(reason="This test is for AFTER refactoring - skip for now")
async def test_chatloop_service_streaming_refactored_implementation(
    fake_chat_model,
    fake_chunk_search_service,
    fake_stores,
    fake_manifest,
    checkpoint_saver,
):
    """
    Test that astream() streams tokens incrementally AFTER refactoring.

    This test should PASS after implementing the async generator refactor.
    Expected behavior: Multiple chunks with incremental token content.

    SKIP THIS TEST INITIALLY - it will fail with current implementation.
    Unskip after refactoring is complete.
    """
    full_page_store, game_data_store = fake_stores

    # Build services
    qa_service = build_qa_service(
        checkpoint_saver=checkpoint_saver,
        chat_model=fake_chat_model,
        full_page_store=full_page_store,
        chunk_search_service=fake_chunk_search_service,
    )

    chatloop_service = build_chatloop_service(
        checkpoint_saver=checkpoint_saver,
        chat_model=fake_chat_model,
        qa_service=qa_service,
    )

    # Create input
    input_data: ChatLoopServiceInput = {
        "messages": [HumanMessage(content="Stream this response please?")],
        "manifest": fake_manifest,
        "thread_id": "test-thread-789",
    }

    # Collect streamed chunks
    chunks = []
    token_chunks = []

    async for chunk in chatloop_service.astream(input=input_data, config=None):
        chunks.append(chunk)

        # Track chunks that contain content (tokens)
        if hasattr(chunk, 'content') and chunk.content:
            token_chunks.append(chunk.content)

    # Assertions for REFACTORED implementation
    print(f"\n=== Refactored Implementation Test ===")
    print(f"Total chunks yielded: {len(chunks)}")
    print(f"Token chunks: {token_chunks}")

    # Should yield MULTIPLE chunks (one per token)
    assert len(token_chunks) > 1, (
        f"Should yield multiple token chunks, got {len(token_chunks)}. "
        f"Expected streaming of individual tokens."
    )

    # Reconstruct full message from tokens
    full_text = "".join(token_chunks)
    assert "forty-two" in full_text.lower(), "Reconstructed text should match expected"

    # Verify incremental streaming (each chunk should be small)
    for i, token in enumerate(token_chunks):
        print(f"  Token {i}: '{token}'")
        # Each token should be relatively small (not the full response)
        assert len(token) < len(full_text), (
            f"Token {i} should be smaller than full response"
        )

    print(f"✓ Refactored implementation streams tokens incrementally")
    print(f"✓ Full reconstructed text: {full_text}")


@pytest.mark.asyncio
async def test_qa_service_streaming_directly(
    fake_chat_model,
    fake_chunk_search_service,
    fake_stores,
    fake_manifest,
    checkpoint_saver,
):
    """
    Test streaming on qa_service directly (inner layer).

    This isolates testing to just the QA service without the chatloop wrapper.
    """
    full_page_store, game_data_store = fake_stores

    # Build only the QA service
    qa_service = build_qa_service(
        checkpoint_saver=checkpoint_saver,
        chat_model=fake_chat_model,
        full_page_store=full_page_store,
        chunk_search_service=fake_chunk_search_service,
    )

    # Create input for QA service
    from meeplemate.qa_graph import QAServiceInput
    input_data: QAServiceInput = {
        "query": "What is the answer to life?",
        "manifest": fake_manifest,
    }

    # Stream from QA service (need thread_id in config for checkpointer)
    config = {"configurable": {"thread_id": "test-qa-direct-123"}}
    chunks = []
    async for chunk in qa_service.astream(input=input_data, config=config):
        chunks.append(chunk)

    print(f"\n=== QA Service Direct Streaming Test ===")
    print(f"Number of chunks from QA service: {len(chunks)}")

    # Current implementation should yield at least the final result
    assert len(chunks) >= 1, "QA service should yield at least one chunk"

    # Final chunk should have a response
    final_chunk = chunks[-1]
    if isinstance(final_chunk, dict):
        assert "response" in final_chunk, "Final chunk should contain response"
        print(f"✓ QA service response: {final_chunk.get('response', 'N/A')}")
    else:
        # Might be a message chunk after refactoring
        print(f"✓ QA service yielded chunk: {final_chunk}")


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
    chat_model = FakeStreamingChatModel()
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
