"""Tests for grid-search run-directory naming (must stay under FS limits)."""

from meeplemate.eval.grid_search import (
    GridSearchE2ERunner,
    _longest_common_dotted_prefix,
    _MAX_RUN_NAME_LEN,
)

# _hyperparameters_to_group_run_name does not use self, so call it unbound.
_name = GridSearchE2ERunner._hyperparameters_to_group_run_name


def _full_grid_combo():
    return {
        "chat.models.0.model_name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "chat.models.0.temperature": 0.7,
        "chat.models.0.top_p": 0.8,
        "chat.models.0.top_k": -1,
        "chat.models.0.min_p": 0.0,
        "chat.models.0.presence_penalty": 0.0,
        "chat.models.0.frequency_penalty": 0.0,
        "chat.models.0.repetition_penalty": 1.0,
    }


def test_common_prefix_stripped():
    assert _longest_common_dotted_prefix(
        ["chat.models.0.temperature", "chat.models.0.top_p"]
    ) == "chat.models.0."
    # Mixed indices keep the disambiguating index
    assert _longest_common_dotted_prefix(
        ["chat.models.0.temperature", "chat.models.1.temperature"]
    ) == "chat.models."
    # No shared prefix
    assert _longest_common_dotted_prefix(["a.b", "c.d"]) == ""


def test_run_name_under_filesystem_limit():
    name = _name(None, _full_grid_combo())
    assert len(name) < 255  # single path component limit
    assert "chat_models_0_" not in name  # redundant prefix removed
    assert "temperature_0_7" in name     # leaf key/value preserved


def test_run_name_is_deterministic():
    combo = _full_grid_combo()
    assert _name(None, combo) == _name(None, combo)


def test_distinct_combos_get_distinct_names():
    a = _full_grid_combo()
    b = {**a, "chat.models.0.temperature": 0.9}
    assert _name(None, a) != _name(None, b)


def test_pathologically_long_name_is_truncated_and_hashed():
    # A grid with a huge value must still yield a short, unique, deterministic name.
    combo = {f"chat.models.0.param_{i}": "x" * 50 for i in range(20)}
    name = _name(None, combo)
    assert len(name) <= _MAX_RUN_NAME_LEN
    assert name == _name(None, combo)  # deterministic
    # A different combo hashes differently
    other = {**combo, "chat.models.0.param_0": "y" * 50}
    assert name != _name(None, other)
