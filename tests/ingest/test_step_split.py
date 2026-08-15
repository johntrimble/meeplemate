"""The render / ocr / build-text split: each step writes only its own artifacts,
and each refuses to run before the step that produces its input.

Neither test needs poppler or a model: render's PDF iterator and the OCR call
are both stubbed, so what is under test is each job's orchestration and the
paths it writes.
"""
import asyncio

import pytest

from meeplemate.ingest import ocr as ocr_module
from meeplemate.ingest import render as render_module
from meeplemate.ingest.errors import MissingStepInput
from meeplemate.ingest.gamepackage import (
    get_page,
    page_raw_md_path,
    page_structured_path,
)
from meeplemate.ingest.layout import PackageLayout
from meeplemate.ingest.ocr import BuildTextJob, OcrJob
from meeplemate.ingest.render import RenderJob


class StubImage:
    """Stands in for a PIL image; records where it was asked to save."""

    def __init__(self, page_num: int):
        self.page_num = page_num


@pytest.fixture
def stub_render(monkeypatch):
    """Replace PDF rasterisation with three synthetic pages per document."""

    async def fake_iter(pdf_path, chunk_size=4, dpi=300):
        for page_num in range(3):
            yield page_num, StubImage(page_num)

    def fake_save(image, output_path, max_size):
        output_path.write_bytes(b"\x89PNG " + str(image.page_num).encode())
        return output_path

    monkeypatch.setattr(render_module, "pdf_page_images_iter", fake_iter)
    monkeypatch.setattr(render_module, "postprocess_and_save", fake_save)


def test_render_writes_only_page_images(package_with_pdfs, watch, stub_render):
    asyncio.run(RenderJob(path=package_with_pdfs["path"]).run())

    touched = watch.touched()
    assert touched, "render wrote nothing"
    assert all(p.endswith(".png") for p in touched), sorted(touched)
    layout = PackageLayout(package_with_pdfs["path"])
    for rulebook in package_with_pdfs["rulebooks"]:
        for page_num in range(3):
            assert layout.page_image(rulebook["document_key"], page_num).exists()


def test_render_requires_a_source_pdf(package):
    """render before init-game-package names the step to run, not a traceback."""
    with pytest.raises(MissingStepInput) as excinfo:
        asyncio.run(RenderJob(path=package["path"]).run())
    assert "init-game-package" in str(excinfo.value)


@pytest.fixture
def stub_ocr_model(monkeypatch):
    seen_models = []

    async def fake_ocr(client, image_path, model_config):
        seen_models.append(model_config.model)
        return f"# Page\n\nText from {image_path.name}."

    monkeypatch.setattr(ocr_module, "image_to_deepseek_ocr", fake_ocr)
    return seen_models


def test_ocr_writes_only_raw_md_and_structured(package_with_images, watch, stub_ocr_model):
    asyncio.run(OcrJob(path=package_with_images["path"], ocr_client=object()).run())

    touched = watch.touched()
    assert touched, "ocr wrote nothing"
    assert all(
        p.endswith(".raw.md") or p.endswith(".structured.json") for p in touched
    ), sorted(touched)

    page = get_page(package_with_images, "d1", 0)
    assert page_raw_md_path(page).exists()
    assert page_structured_path(page).exists()

    # The job passes its configured model down to the endpoint call.
    assert set(stub_ocr_model) == {"deepseek-ai/DeepSeek-OCR-2"}


def test_ocr_requires_rendered_images(package):
    """ocr before render names the step to run."""
    with pytest.raises(MissingStepInput) as excinfo:
        asyncio.run(OcrJob(path=package["path"], ocr_client=object()).run())
    assert "render" in str(excinfo.value)


def test_build_text_requires_ocr_output(package):
    for structured in package["path"].rglob("*.structured.json"):
        structured.unlink()
    with pytest.raises(MissingStepInput) as excinfo:
        asyncio.run(BuildTextJob(path=package["path"]).run())
    assert "ocr" in str(excinfo.value)
