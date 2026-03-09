"""Unit tests for TokenCountingCallback."""
from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.outputs import ChatGeneration, ChatResult

from meeplemate.server.rate_limit import TokenCountingCallback


def _llm_result(prompt_tokens: int, completion_tokens: int) -> ChatResult:
    result = MagicMock(spec=ChatResult)
    result.generations = [[MagicMock(spec=ChatGeneration)]]
    result.llm_output = {
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    }
    return result


def _llm_result_no_usage() -> ChatResult:
    result = MagicMock(spec=ChatResult)
    result.generations = [[MagicMock(spec=ChatGeneration)]]
    result.llm_output = {}
    return result


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
