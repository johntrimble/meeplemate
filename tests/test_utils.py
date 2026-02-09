from typing import TypedDict
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


class ReorderTypedDictNested(TypedDict):
    x: int
    y: str


class ReorderTypedDict(TypedDict):
    a: int
    b: str
    c: dict[str, ReorderTypedDictNested]
    d: float


def test_reorder_dict_field_order():
    """Verify field order: TypedDict fields first, extra fields last."""
    response = {
        "extra_field": "Extra",
        "d": 1.1,
        "a": 2,
        "b": "test",
        "c": {
            "foo": {"y": "yes", "x": 10},
            "bar": {"x": 20, "y": "no"},
        }
    }

    reordered = util.reorder_dict_by_typeddict(response, ReorderTypedDict)
    assert isinstance(reordered, dict)

    # Verify TypedDict fields come first in definition order
    keys = list(reordered.keys())
    expected_typeddict_order = [
        "a",
        "b",
        "c",
        "d"
    ]

    # TypedDict fields should be first, in order
    for i, expected_key in enumerate(expected_typeddict_order):
        assert keys[i] == expected_key

    # Extra fields should be last
    assert keys[-1] == "extra_field"


def test_serialize_typeddict_with_extra_keys():
    """Verify JSON serialization preserves order including extra keys."""
    response = {
        "extra_field": "Extra",
        "d": 1.1,
        "a": 2,
        "b": "test",
        "c": {
            "foo": {"y": "yes", "x": 10},
            "bar": {"x": 20, "y": "no"},
        }
    }

    json_str = util.serialize_typeddict(response, ReorderTypedDict)
    parsed = json.loads(json_str)

    # Verify all keys present
    assert set(parsed.keys()) == set(response.keys())

    # Verify TypedDict fields first, extra keys last
    keys = list(parsed.keys())
    assert keys[0] == "a"
    assert keys[-1] == "extra_field"
