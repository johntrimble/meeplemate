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
