"""Render rulebook PDF pages to PNG images.

Split out of `ocr.py` so the deterministic, CPU-bound render step is independent
of the stochastic, GPU-bound OCR step: re-running OCR should not re-render 2.4 GB
of images, and changing the render settings should not imply new OCR output.

Keeping pdf2image and PIL in this module also leaves `ocr.py` importable in
environments without poppler, which is what CI has.
"""
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Tuple

from PIL import Image
from structlog import get_logger

from meeplemate.ingest.errors import MissingStepInput
from meeplemate.ingest.gamepackage import GamePackage, layout_for, load_game_package

logger = get_logger(__name__)


def get_page_count(pdf_path: Path) -> int:
    from pdf2image import pdfinfo_from_path

    info = pdfinfo_from_path(str(pdf_path))
    return int(info.get("Pages", 0))


def maybe_resize_image(image, max_size: int | None):
    if max_size is None:
        return image
    size = image.size
    if max(size) < max_size:
        return image
    ratio = max_size / max(size)
    new_width = int(image.width * ratio)
    new_height = int(image.height * ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


def postprocess_and_save(image, output_path: Path, max_size: int | None) -> Path:
    image = image.convert("RGB")
    image = maybe_resize_image(image, max_size)
    image.save(output_path, format="PNG")
    return output_path


async def pdf_page_images_iter(
    pdf_path: Path, chunk_size: int = 4, dpi: int = 300
) -> AsyncIterator[Tuple[int, Image.Image]]:
    """
    Yield (page_num, image) in order, loading the PDF in small chunks to keep memory bounded.
    """
    from pdf2image import convert_from_path

    page_count = get_page_count(pdf_path)
    for start in range(1, page_count + 1, chunk_size):
        end = min(start + chunk_size - 1, page_count)
        images = await asyncio.to_thread(
            convert_from_path,
            str(pdf_path),
            dpi=dpi,
            first_page=start,
            last_page=end,
        )
        for idx, image in enumerate(images):
            # Use zero-based page numbering for stable filenames/order
            yield start + idx - 1, image


@dataclass
class RenderJob:
    """Render every page of every rulebook to a PNG."""

    path: Path
    _gp: GamePackage | None = None
    max_size: int | None = 2000
    pdf_page_chunk: int = 10
    dpi: int = 300

    @property
    def gp(self) -> GamePackage:
        if self._gp is None:
            self._gp = load_game_package(self.path)
        return self._gp

    async def run(self) -> None:
        layout = layout_for(self.gp)
        for rulebook in self.gp["rulebooks"]:
            document_key = rulebook["document_key"]
            pdf_path = layout.raw_document(rulebook["path"])
            if not pdf_path.exists():
                raise MissingStepInput(
                    what=f"No source PDF at {pdf_path}",
                    run_step=f"init-game-package <source> {self.path}",
                )

            logger.info("Rendering rulebook", document_key=document_key, pdf=str(pdf_path))
            async for page_num, image in pdf_page_images_iter(
                pdf_path, chunk_size=self.pdf_page_chunk, dpi=self.dpi
            ):
                image_path = layout.page_image(document_key, page_num)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(
                    postprocess_and_save, image, image_path, self.max_size
                )
