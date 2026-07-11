"""Unit tests for TokenCountingCallback."""
from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from meeplemate.server.rate_limit import TokenCountingCallback

# on_llm_end receives an LLMResult, whose `.generations` is nested (list[list]).


def _llm_result(prompt_tokens: int, completion_tokens: int) -> LLMResult:
    """Non-streaming shape: usage in llm_output["token_usage"]."""
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
        },
    )


def _llm_result_streaming(input_tokens: int, output_tokens: int) -> LLMResult:
    """Streaming shape: llm_output empty, usage on message.usage_metadata."""
    message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
    return LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=None)


def _llm_result_no_usage() -> LLMResult:
    return LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="hi"))]],
        llm_output={},
    )


def test_accumulates_tokens_with_output_multiplier() -> None:
    """Completion tokens are weighted by output_token_multiplier; prompt tokens are not."""
    cb = TokenCountingCallback(estimated=500, output_token_multiplier=4)
    cb.on_llm_end(_llm_result(100, 200))   # 100 + 200*4 = 900
    cb.on_llm_end(_llm_result(50, 75))     # 50 + 75*4 = 350
    assert cb.tokens == 1_250


def test_default_multiplier_is_four() -> None:
    """Default output_token_multiplier is 4."""
    cb = TokenCountingCallback(estimated=0)
    cb.on_llm_end(_llm_result(0, 10))
    assert cb.tokens == 40


def test_multiplier_of_one_counts_all_tokens_equally() -> None:
    cb = TokenCountingCallback(estimated=0, output_token_multiplier=1)
    cb.on_llm_end(_llm_result(100, 200))
    assert cb.tokens == 300


def test_falls_back_to_estimate_when_no_usage_info() -> None:
    """Returns estimated value when LLMResult has no token usage."""
    cb = TokenCountingCallback(estimated=2_000)
    cb.on_llm_end(_llm_result_no_usage())
    assert cb.tokens == 2_000


def test_real_count_takes_priority_over_estimate() -> None:
    """Once real usage is seen, the estimate is ignored."""
    cb = TokenCountingCallback(estimated=9_999, output_token_multiplier=1)
    cb.on_llm_end(_llm_result(10, 20))
    assert cb.tokens == 30


def test_zero_estimated_with_no_usage_returns_zero() -> None:
    cb = TokenCountingCallback(estimated=0)
    cb.on_llm_end(_llm_result_no_usage())
    assert cb.tokens == 0


def test_counts_streaming_usage_metadata() -> None:
    """Streaming responses (empty llm_output) are counted via usage_metadata."""
    cb = TokenCountingCallback(estimated=9_999, output_token_multiplier=4)
    cb.on_llm_end(_llm_result_streaming(input_tokens=23, output_tokens=447))
    assert cb._has_real_count is True
    assert cb.tokens == 23 + 447 * 4  # 1811


def test_streaming_and_nonstreaming_accumulate_together() -> None:
    cb = TokenCountingCallback(estimated=0, output_token_multiplier=1)
    cb.on_llm_end(_llm_result(10, 20))              # 30 via llm_output
    cb.on_llm_end(_llm_result_streaming(5, 15))     # 20 via usage_metadata
    assert cb.tokens == 50


def test_llm_output_takes_precedence_over_usage_metadata() -> None:
    """When both are present, llm_output is used and usage_metadata is not double-counted."""
    message = AIMessage(
        content="hi",
        usage_metadata={"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998},
    )
    result = LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 20}},
    )
    cb = TokenCountingCallback(estimated=0, output_token_multiplier=1)
    cb.on_llm_end(result)
    assert cb.tokens == 30  # from llm_output only, not 1998 + 30
