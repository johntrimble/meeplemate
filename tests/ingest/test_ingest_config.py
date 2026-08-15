"""Ingest settings: the defaults are a data contract, and overrides must reach them.

Two separate concerns:

1. The defaults reproduce what the pipeline ran with when these values were
   literals in `ocr.py`. Changing one silently changes OCR output, so they are
   pinned here.
2. The override mechanism works — a config file and an environment variable both
   reach the nested ingest models. That is tested against a purpose-built fixture
   rather than the real config files, so the test does not depend on (or dictate)
   what those files happen to contain.
"""
import os
from pathlib import Path

import pytest

from meeplemate.config import Config, IngestConfig

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "ingest_overrides.yaml"
FIXTURE_NO_INGEST = FIXTURES / "ingest_absent.yaml"


@pytest.fixture
def clean_env(monkeypatch):
    """Strip ambient MM_* variables so a developer's shell cannot skew results."""
    for key in list(os.environ):
        if key.startswith("MM_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def build_config() -> Config:
    """Build a Config from the fixture file alone.

    `_env_file=None` disables the dotenv source. Clearing MM_* environment
    variables is not enough on its own: pydantic-settings reads `.env` through a
    separate source, so a checked-out `.env` would quietly supply fields the
    fixture omits — which is exactly how these tests came to pass on a machine
    that had one and fail on a machine that did not.
    """
    return Config(_env_file=None)


def test_defaults_match_the_previously_hardcoded_values():
    """These were literals in ocr.py before the refactor. They are the values the
    rulebooks currently on disk were produced with."""
    cfg = IngestConfig()

    assert cfg.ocr.model == "deepseek-ai/DeepSeek-OCR-2"
    assert cfg.ocr.base_url == "http://vllm-deepseek-ocr:8000/v1"
    assert cfg.ocr.temperature == 0.0
    assert cfg.ocr.max_tokens == 2048
    assert cfg.ocr.extra_body["skip_special_tokens"] is False
    assert cfg.ocr.extra_body["vllm_xargs"] == {
        "ngram_size": 30,
        "window_size": 90,
        "whitelist_token_ids": [128821, 128822],
    }

    assert cfg.page_number_ocr.model == "glm-ocr"
    assert cfg.page_number_ocr.base_url == "http://vllm-glm-ocr:8080/v1"

    # Reproduce the page images already on disk.
    assert (cfg.render.dpi, cfg.render.max_image_size) == (300, 2000)

    # Chunk sizes that build-chunks used to hardcode.
    assert (cfg.chunk_size, cfg.chunk_overlap) == (500, 50)
    assert (cfg.child_chunk_size, cfg.child_chunk_overlap) == (125, 12)


def test_config_file_overrides_reach_the_nested_models(clean_env):
    clean_env.setenv("MM_CONFIG_FILE", str(FIXTURE))
    ingest = build_config().ingest

    assert (ingest.chunk_size, ingest.chunk_overlap) == (321, 21)
    assert (ingest.child_chunk_size, ingest.child_chunk_overlap) == (87, 7)
    assert ingest.max_ocr_workers == 9

    assert (ingest.render.dpi, ingest.render.max_image_size, ingest.render.pdf_page_chunk) == (150, 900, 3)

    assert ingest.ocr.model == "test/other-ocr"
    assert ingest.ocr.base_url == "http://ocr.test/v1"
    assert ingest.ocr.temperature == 0.4
    assert ingest.ocr.max_tokens == 512
    assert ingest.ocr.extra_body == {"vllm_xargs": {"ngram_size": 11}}

    assert ingest.page_number_ocr.model == "test/other-pagenum"
    assert ingest.page_number_ocr.base_url == "http://pagenum.test/v1"


def test_unspecified_values_fall_back_to_defaults(clean_env):
    """The fixture omits ocr.api_key and ocr.timeout."""
    clean_env.setenv("MM_CONFIG_FILE", str(FIXTURE))
    ingest = build_config().ingest
    assert ingest.ocr.api_key == "EMPTY"
    assert ingest.ocr.timeout == 3600


def test_environment_beats_the_config_file(clean_env):
    """Env vars sit above YAML in the source chain, so a one-off run can retarget
    a model without editing a checked-in file."""
    clean_env.setenv("MM_CONFIG_FILE", str(FIXTURE))
    clean_env.setenv("MM_INGEST__OCR__MODEL", "test/env-wins")
    clean_env.setenv("MM_INGEST__RENDER__DPI", "72")

    ingest = build_config().ingest
    assert ingest.ocr.model == "test/env-wins"
    assert ingest.render.dpi == 72
    # Untouched keys still come from the file.
    assert ingest.ocr.base_url == "http://ocr.test/v1"


def test_config_without_an_ingest_key_yields_the_defaults(clean_env):
    """config-dev.yaml has no `ingest:` block, so this is the live dev path."""
    clean_env.setenv("MM_CONFIG_FILE", str(FIXTURE_NO_INGEST))
    assert build_config().ingest == IngestConfig()
