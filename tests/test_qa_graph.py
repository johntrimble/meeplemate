from langchain.messages import ToolMessage
from langchain_core.messages.content import ToolCall
from meeplemate.qa_graph import Chunk, ChunkSearchResult, dedupe_chunks_in_message_history, get_chunk_id_tuple
from langchain_core.messages import AIMessage, BaseMessage
from typing import List
import json

def test_dedupe_chunks_in_message_history():
    result1 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 1",
            page=1,
            offset=3,
            content="This is chunk 1",   
        ),
        relevance_reason="Relevant to the query",
    )
    result2 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 1",
            page=2,
            offset=5,
            content="This is chunk 2",   
        ),
        relevance_reason="Also relevant to the query",
    )
    result3 = ChunkSearchResult(
        chunk=Chunk(
            rulebook_name="Rulebook 2",
            page=1,
            offset=0,
            content="This is chunk 3",   
        ),
        relevance_reason="Somewhat relevant to the query",
    )

    tool_messages = [
        ToolMessage(
            content=json.dumps([result1, result2]),
            tool_call_id="7",
            status="success",
            id="tool_msg_1"
        ),
        ToolMessage(
            content=json.dumps([result2, result3]),
            tool_call_id="8",
            status="success",
            id="tool_msg_2"
        ),
    ]

    ai_message = AIMessage(
        content="",
        tool_calls=[
            ToolCall(name="search_chunks", args={}, id="7", type="tool_call"),
            ToolCall(name="search_chunks", args={}, id="8", type="tool_call")
        ],
        metadata={"related_tool_messages": tool_messages},
    )
    

    message_edits = dedupe_chunks_in_message_history([ai_message] + tool_messages)
    assert len(message_edits) == 1
    message_edit = message_edits[0]
    assert message_edit.id == "tool_msg_1"
    deduped_results = json.loads(message_edit.content)
    assert len(deduped_results) == 1
    assert get_chunk_id_tuple(deduped_results[0]["chunk"]) == get_chunk_id_tuple(result1["chunk"])

    
