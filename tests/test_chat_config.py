"""Tests for the multi-model chat config schema and env-var overrides."""

import json
import os

import pytest
import yaml

from meeplemate.config import ChatConfig, ChatModelConfig, Config, build_openai_extra_body


def _model(**overrides):
    base = {"model_name": "m", "endpoint_type": "openai", "endpoint": "http://x/v1"}
    base.update(overrides)
    return ChatModelConfig(**base)


def _write_yaml(tmp_path, chat: dict):
    doc = {
        "model_name": "test-model",
        "embedding": {"model": "BAAI/bge-small-en-v1.5", "api_key": "dummy"},
        "chat": chat,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _isolate_env(monkeypatch):
    # Drop any ambient MM_CHAT__* / MM_CONFIG_FILE so tests are deterministic.
    for key in list(os.environ):
        if key.startswith("MM_CHAT__") or key == "MM_CONFIG_FILE":
            monkeypatch.delenv(key, raising=False)


def test_chatconfig_defaults():
    c = ChatConfig.model_validate(
        {"models": [{"model_name": "m", "endpoint_type": "openai", "endpoint": "http://x/v1"}]}
    )
    assert c.max_failures == 3
    assert c.cooldown_seconds == 300.0
    assert len(c.models) == 1


def test_chatconfig_requires_at_least_one_model():
    with pytest.raises(Exception):
        ChatConfig.model_validate({"models": []})


def test_yaml_multi_model_list_parses(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    chat = {
        "max_failures": 5,
        "cooldown_seconds": 42,
        "models": [
            {
                "model_name": "primary",
                "endpoint_type": "openai",
                "endpoint": "http://primary/v1",
                "min_p": 0.1,
            },
            {
                "model_name": "fallback",
                "endpoint_type": "openai",
                "endpoint": "http://fallback/v1",
            },
        ],
    }
    monkeypatch.setenv("MM_CONFIG_FILE", str(_write_yaml(tmp_path, chat)))

    cfg = Config()
    assert cfg.chat.max_failures == 5
    assert cfg.chat.cooldown_seconds == 42
    assert [m.endpoint for m in cfg.chat.models] == ["http://primary/v1", "http://fallback/v1"]
    assert cfg.chat.models[0].min_p == 0.1


def test_env_json_blob_overrides_yaml_list(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    chat = {
        "models": [
            {"model_name": "yaml-only", "endpoint_type": "openai", "endpoint": "http://yaml/v1"}
        ]
    }
    monkeypatch.setenv("MM_CONFIG_FILE", str(_write_yaml(tmp_path, chat)))
    monkeypatch.setenv(
        "MM_CHAT__MODELS",
        json.dumps(
            [
                {"model_name": "env-a", "endpoint_type": "openai", "endpoint": "http://env-a/v1"},
                {"model_name": "env-b", "endpoint_type": "openai", "endpoint": "http://env-b/v1"},
            ]
        ),
    )
    monkeypatch.setenv("MM_CHAT__MAX_FAILURES", "7")

    cfg = Config()
    # Env replaces the whole list (lists do not deep-merge across sources).
    assert [m.model_name for m in cfg.chat.models] == ["env-a", "env-b"]
    assert cfg.chat.max_failures == 7


def test_shared_api_key_fills_models_and_per_model_wins():
    c = ChatConfig.model_validate(
        {
            "api_key": "shared",
            "models": [
                {"model_name": "a", "endpoint_type": "openai", "endpoint": "http://a/v1"},
                {
                    "model_name": "b",
                    "endpoint_type": "openai",
                    "endpoint": "http://b/v1",
                    "api_key": "own",
                },
            ],
        }
    )
    assert c.models[0].api_key.get_secret_value() == "shared"  # inherited
    assert c.models[1].api_key.get_secret_value() == "own"     # per-model wins


def test_no_shared_api_key_leaves_models_unset():
    c = ChatConfig.model_validate(
        {"models": [{"model_name": "a", "endpoint_type": "openai", "endpoint": "http://a/v1"}]}
    )
    assert c.models[0].api_key is None


def test_env_models_json_plus_separate_shared_api_key(tmp_path, monkeypatch):
    # The production pattern: model list as one JSON env var, token as another.
    _isolate_env(monkeypatch)
    doc = {"model_name": "test", "embedding": {"model": "m", "api_key": "dummy"}}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("MM_CONFIG_FILE", str(path))
    monkeypatch.setenv(
        "MM_CHAT__MODELS",
        json.dumps(
            [
                {"model_name": "a", "endpoint_type": "openai", "endpoint": "http://a/v1"},
                {"model_name": "b", "endpoint_type": "openai", "endpoint": "http://b/v1"},
            ]
        ),
    )
    monkeypatch.setenv("MM_CHAT__API_KEY", "prod-token")

    cfg = Config()
    assert cfg.chat.api_key.get_secret_value() == "prod-token"
    assert [m.api_key.get_secret_value() for m in cfg.chat.models] == ["prod-token", "prod-token"]


def test_apply_hyperparameters_to_config_targets_model_index(tmp_path, monkeypatch):
    # The grid search sweeps the primary model's sampling params via dotted,
    # index-aware paths (e.g. "chat.models.0.temperature").
    from meeplemate.eval.grid_search import E2ERunnerWithHyperparameters

    _isolate_env(monkeypatch)
    chat = {
        "models": [
            {"model_name": "primary", "endpoint_type": "openai", "endpoint": "http://p/v1"}
        ]
    }
    monkeypatch.setenv("MM_CONFIG_FILE", str(_write_yaml(tmp_path, chat)))
    cfg = Config()

    hyperparameters = {
        "chat.models.0.temperature": 0.3,
        "chat.models.0.min_p": 0.05,
        "chat.models.0.model_name": "swept-model",
        "chat.max_failures": 9,
    }
    # apply_hyperparameters_to_config does not use self, so call unbound.
    out = E2ERunnerWithHyperparameters.apply_hyperparameters_to_config(
        None, cfg, hyperparameters
    )
    assert out.chat.models[0].temperature == 0.3
    assert out.chat.models[0].min_p == 0.05
    assert out.chat.models[0].model_name == "swept-model"
    assert out.chat.max_failures == 9


def test_extra_body_carries_provider_extensions():
    body = build_openai_extra_body(
        _model(min_p=0.1, top_k=20, provider={"order": ["a", "b"]}, reasoning={"effort": "low"})
    )
    assert body["min_p"] == 0.1
    assert body["top_k"] == 20
    assert body["provider"] == {"order": ["a", "b"]}
    assert body["reasoning"] == {"effort": "low"}
    assert "chat_template_kwargs" not in body  # none requested


def test_extra_body_explicit_disable_thinking_only():
    body = build_openai_extra_body(_model(explicit_disable_thinking=True))
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_extra_body_merges_chat_template_kwargs_with_disable_thinking():
    body = build_openai_extra_body(
        _model(explicit_disable_thinking=True, chat_template_kwargs={"foo": "bar"})
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "foo": "bar"}


def test_extra_body_explicit_chat_template_kwargs_wins_over_convenience_flag():
    # An explicit enable_thinking key overrides the value the flag would set.
    body = build_openai_extra_body(
        _model(explicit_disable_thinking=True, chat_template_kwargs={"enable_thinking": True})
    )
    assert body["chat_template_kwargs"] == {"enable_thinking": True}


def test_extra_body_empty_when_nothing_set():
    assert build_openai_extra_body(_model()) == {}


def test_new_fields_configurable_via_env_json(tmp_path, monkeypatch):
    _isolate_env(monkeypatch)
    doc = {"model_name": "test", "embedding": {"model": "m", "api_key": "dummy"}}
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(doc))
    monkeypatch.setenv("MM_CONFIG_FILE", str(path))
    monkeypatch.setenv(
        "MM_CHAT__MODELS",
        json.dumps(
            [
                {
                    "model_name": "a",
                    "endpoint_type": "openai",
                    "endpoint": "http://a/v1",
                    "provider": {"order": ["x"]},
                    "reasoning": {"effort": "high"},
                }
            ]
        ),
    )
    cfg = Config()
    assert cfg.chat.models[0].provider == {"order": ["x"]}
    assert cfg.chat.models[0].reasoning == {"effort": "high"}
