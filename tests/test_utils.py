import pytest
from meeplemate import util

# Test atee
@pytest.mark.asyncio
async def test_atee():
    l = list(range(5))
    iterator = util.to_async_iter(l)
    a, b, c = util.atee(iterator, 3)
    result_a = [item async for item in a]
    result_b = [item async for item in b]
    result_c = [item async for item in c]
    assert result_a == [0, 1, 2, 3, 4]
    assert result_b == [0, 1, 2, 3, 4]
    assert result_c == [0, 1, 2, 3, 4]


# Test amap
@pytest.mark.asyncio
async def test_amap():
    l1 = list(range(5))
    l2 = list(range(5, 10))
    iterator1 = util.to_async_iter(l1)
    iterator2 = util.to_async_iter(l2)
    result = [item async for item in util.amap(lambda x, y: x + y, iterator1, iterator2)]
    assert result == [5, 7, 9, 11, 13]

# Test apairwise
@pytest.mark.asyncio
async def test_apairwise():
    l = [1, 2, 3, 4]
    iterator = util.to_async_iter(l)
    result = [item async for item in util.apairwise(iterator)]
    assert result == [(1, 2), (2, 3), (3, 4)]


# Test aenumerate
@pytest.mark.asyncio
async def test_aenumerate():
    l = ['a', 'b', 'c']
    iterator = util.to_async_iter(l)
    result = [item async for item in util.aenumerate(iterator, start=1)]
    assert result == [(1, 'a'), (2, 'b'), (3, 'c')]


# Test reorder_dict_by_typeddict and serialize_typeddict
import json
from meeplemate.qa_graph import (
    QaResponse, IdentifiedMechanics, GeneralRule, QuoteEntry,
    DefinitionEntry, ExceptionEntry
)


def test_reorder_dict_preserves_all_keys():
    """Verify reordered dict has same keys/values as original (CRITICAL)."""
    response = {
        "final_answer": "Test answer",
        "extra_llm_field": "This wasn't in the TypedDict!",  # Extra key
        "reasoning": "Test reasoning",
        "identified_mechanics": {
            "reasoning": "Mechanic reasoning",
            "secondary_mechanics": ["mech2"],
            "primary_mechanics": ["mech1"],
        },
        "general_rules": [],
        "definitions": [],
        "exceptions": [],
        "precedence_analysis": "",
        "sufficient_information_to_answer": True,
        "another_extra": 42,  # Another extra key
    }

    reordered = util.reorder_dict_by_typeddict(response, QaResponse)

    # CRITICAL: Reordered dict should equal original (same keys/values)
    assert reordered == response
    assert set(reordered.keys()) == set(response.keys())


def test_reorder_dict_field_order():
    """Verify field order: TypedDict fields first, extra fields last."""
    response = {
        "final_answer": "Test answer",
        "extra_field": "Extra",  # Should be last
        "reasoning": "Test reasoning",
        "identified_mechanics": {
            "reasoning": "Mechanic reasoning",
            "secondary_mechanics": ["mech2"],
            "primary_mechanics": ["mech1"],  # Out of order
        },
        "general_rules": [],
        "definitions": [],
        "exceptions": [],
        "precedence_analysis": "",
        "sufficient_information_to_answer": True,
    }

    reordered = util.reorder_dict_by_typeddict(response, QaResponse)

    # Verify TypedDict fields come first in definition order
    keys = list(reordered.keys())
    expected_typeddict_order = [
        "identified_mechanics",
        "general_rules",
        "definitions",
        "exceptions",
        "precedence_analysis",
        "reasoning",
        "final_answer",
        "sufficient_information_to_answer"
    ]

    # TypedDict fields should be first, in order
    for i, expected_key in enumerate(expected_typeddict_order):
        assert keys[i] == expected_key

    # Extra fields should be last
    assert keys[-1] == "extra_field"

    # Verify nested IdentifiedMechanics order
    nested_keys = list(reordered["identified_mechanics"].keys())
    assert nested_keys == [
        "primary_mechanics",
        "secondary_mechanics",
        "reasoning"
    ]


def test_serialize_typeddict_with_extra_keys():
    """Verify JSON serialization preserves order including extra keys."""
    response = {
        "final_answer": "Test",
        "extra_key": "value",
        "reasoning": "Steps",
        "identified_mechanics": {
            "primary_mechanics": [],
            "secondary_mechanics": [],
            "reasoning": ""
        },
        "general_rules": [],
        "definitions": [],
        "exceptions": [],
        "precedence_analysis": "",
        "sufficient_information_to_answer": True,
    }

    json_str = util.serialize_typeddict(response, QaResponse)
    parsed = json.loads(json_str)

    # Verify all keys present
    assert set(parsed.keys()) == set(response.keys())

    # Verify TypedDict fields first, extra keys last
    keys = list(parsed.keys())
    assert keys[0] == "identified_mechanics"
    assert keys[-1] == "extra_key"


def test_multiple_extra_keys_sorted():
    """Verify extra keys are sorted alphabetically."""
    response = {
        "final_answer": "Test",
        "zebra_extra": "z",
        "alpha_extra": "a",
        "reasoning": "Steps",
        "identified_mechanics": {
            "primary_mechanics": [],
            "secondary_mechanics": [],
            "reasoning": ""
        },
        "general_rules": [],
        "definitions": [],
        "exceptions": [],
        "precedence_analysis": "",
        "sufficient_information_to_answer": True,
    }

    reordered = util.reorder_dict_by_typeddict(response, QaResponse)
    keys = list(reordered.keys())

    # Last two keys should be extra keys in alphabetical order
    assert keys[-2] == "alpha_extra"
    assert keys[-1] == "zebra_extra"


def test_with_mock_llm_response():
    """Test with a response that has fields in arbitrary order."""
    # Simulate LLM output with fields out of order
    llm_response = {
        "sufficient_information_to_answer": True,
        "final_answer": "Test answer with citation (Rulebook, p. 1)",
        "reasoning": "- Step 1\n- Step 2",
        "identified_mechanics": {
            "secondary_mechanics": ["mechanic2"],
            "reasoning": "These mechanics are relevant because...",
            "primary_mechanics": ["mechanic1"],  # Out of order!
        },
        "precedence_analysis": "Rule X takes precedence",
        "exceptions": [],
        "definitions": [],
        "general_rules": [],
    }

    json_str = util.serialize_typeddict(llm_response, QaResponse)
    parsed = json.loads(json_str)

    # Should be reordered to match TypedDict definition
    assert list(parsed.keys())[0] == "identified_mechanics"
    assert list(parsed.keys())[-1] == "sufficient_information_to_answer"
